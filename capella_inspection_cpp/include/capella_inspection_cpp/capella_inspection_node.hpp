// capella_inspection_node.hpp

#ifndef CAPELLA_INSPECTION_NODE_HPP_
#define CAPELLA_INSPECTION_NODE_HPP_

// ======================== ROS 2 核心 ========================
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/callback_group.hpp>
#include <rclcpp/qos.hpp>

// ======================== TF2 ========================
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// ======================== ROS 消息类型 ========================
#include <std_msgs/msg/bool.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <builtin_interfaces/msg/duration.hpp>
#include <builtin_interfaces/msg/time.hpp>

// ======================== 自定义消息 ========================
#include <capella_ros_msg/msg/recognitions.hpp>
#include <capella_ros_msg/msg/single_recognition.hpp>

// ======================== OpenCV / cv_bridge ========================
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

// ======================== ONNX Runtime（YOLO 推理引擎） ========================
#include <onnxruntime_cxx_api.h>

// ======================== C++ 标准库 ========================
#include <string>
#include <vector>
#include <array>
#include <map>
#include <deque>
#include <memory>
#include <optional>
#include <cmath>
#include <functional>
#include <random>


namespace capella_inspection
{

// ---------------------------------------------------------------------------
// YOLO 检测结果：一个检测框的全部信息
// ---------------------------------------------------------------------------
struct Detection
{
    int class_id;                // 类别 ID
    float confidence;            // 置信度
    float x_center, y_center;   // 框中心（像素坐标）
    float width, height;         // 框宽高（像素）

    // 左上 / 右下角（像素坐标，整数）
    int x1() const { return static_cast<int>(x_center - width  / 2.0f); }
    int y1() const { return static_cast<int>(y_center - height / 2.0f); }
    int x2() const { return static_cast<int>(x_center + width  / 2.0f); }
    int y2() const { return static_cast<int>(y_center + height / 2.0f); }
};

// ---------------------------------------------------------------------------
// 二维向量的简单别名，用于射线计算
// ---------------------------------------------------------------------------
using Vec2d = std::array<double, 2>;

// ---------------------------------------------------------------------------
// YoloDetector —— 封装 ONNX Runtime 推理的辅助类
// ---------------------------------------------------------------------------
class YoloDetector
{
public:
    /// 构造时加载 .onnx 模型，use_cuda 控制是否启用 CUDA EP
    explicit YoloDetector(const std::string & model_path, bool use_cuda = true);

    /// 对一张 BGR cv::Mat 做推理，返回所有检测框
    /// conf_threshold: 置信度阈值（对应 Python 版的 conf=0.65）
    /// iou_threshold:  NMS 的 IoU 阈值
    std::vector<Detection> detect(const cv::Mat & bgr_image,
                                  float conf_threshold = 0.65f,
                                  float iou_threshold  = 0.45f);

private:
    // ---- ONNX Runtime 核心对象 ----
    Ort::Env            env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;

    // ---- 模型输入输出的元信息 ----
    std::vector<std::string>  input_names_;
    std::vector<std::string>  output_names_;
    // 保存 c_str 指针供 Run() 使用
    std::vector<const char *> input_names_cstr_;
    std::vector<const char *> output_names_cstr_;

    int64_t input_h_ = 640;   // 模型期望的输入高度
    int64_t input_w_ = 640;   // 模型期望的输入宽度

    // ---- 前处理 / 后处理 ----

    /// letterbox resize + normalize + HWC→CHW，输出 float 向量
    std::vector<float> preprocess(const cv::Mat & bgr_image,
                                  float & scale, int & pad_x, int & pad_y);

    /// 解码网络输出张量 → Detection 列表（含 NMS）
    std::vector<Detection> postprocess(const Ort::Value & output_tensor,
                                       float conf_threshold,
                                       float iou_threshold,
                                       float scale, int pad_x, int pad_y,
                                       int orig_w, int orig_h);

    /// 标准 IoU 计算（NMS 用）
    static float compute_iou(const Detection & a, const Detection & b);

    /// 非极大值抑制
    static std::vector<Detection> nms(std::vector<Detection> & dets, float iou_threshold);
};

// ---------------------------------------------------------------------------
// CapellaInspectionNode —— ROS 2 节点主体
// ---------------------------------------------------------------------------
class CapellaInspectionNode : public rclcpp::Node
{
public:
    explicit CapellaInspectionNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~CapellaInspectionNode() override = default;

private:
    // ======================== 回调函数 ========================

    /// /inspections_ctr 话题回调，控制节点开关
    void control_callback(const std_msgs::msg::Bool::SharedPtr msg);

    /// 图像话题回调（多个相机共用同一个函数签名，通过绑定参数区分）
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg,
                        const std::string & topic_name,
                        const std::string & tf_name,
                        double horizontal_fov);

    // ======================== 射线几何工具 ========================

    /// 判断两条二维射线是否存在公共部分（相交或共线重合）
    bool rays_have_common_part(const Vec2d & ray1, const Vec2d & origin1,
                               const Vec2d & ray2, const Vec2d & origin2,
                               double eps = 0.1) const;

    // ======================== 可视化 ========================

    /// 在 RViz 中发布一条射线箭头 Marker
    void publish_ray_marker(const builtin_interfaces::msg::Time & stamp,
                            const Vec2d & origin,
                            const Vec2d & direction,
                            int marker_id,
                            const std::array<float, 3> & color);

    /// 把二维数组转成 geometry_msgs::msg::Point（z = 0）
    static geometry_msgs::msg::Point to_point_msg(const Vec2d & v);

    // ======================== UUID 工具 ========================

    /// 生成一个随机 UUID 字符串（8-4-4-4-12 格式）
    static std::string generate_uuid();

    // ======================== 参数 ========================

    std::vector<std::string> camera_topics_;
    std::vector<std::string> camera_tf_names_;
    std::vector<double>      horizontal_fovs_;
    int                      max_publish_per_uuid_;
    double                   process_interval_sec_;

    // ======================== 运行状态 ========================

    bool           is_active_              = false;
    bool           first_frame_processed_  = false;
    rclcpp::Time   last_process_time_;

    // ======================== 射线追踪状态 ========================

    std::optional<Vec2d> last_ray_origin_;
    std::optional<Vec2d> last_ray_direction_;

    /// 最近 N 条射线的历史记录（origin, direction）
    std::deque<std::pair<Vec2d, Vec2d>> ray_history_;
    int max_ray_history_ = 7;

    // ======================== UUID 管理 ========================

    std::string              img_uuid_;
    std::map<std::string, int> uuid_publish_count_;
    static constexpr int     MAX_UUID_HISTORY = 10;

    // ======================== YOLO 推理 ========================

    std::unique_ptr<YoloDetector> yolo_detector_;

    // ======================== ROS 通信对象 ========================

    // 多相机图像订阅（每个相机一个订阅 + 一个独立回调组）
    std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> image_subs_;
    std::vector<rclcpp::CallbackGroup::SharedPtr>                         callback_groups_;

    // 控制话题订阅
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr ctr_sub_;

    // 发布者
    rclcpp::Publisher<capella_ros_msg::msg::Recognitions>::SharedPtr      result_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr         marker_pub_;

    // ======================== TF ========================

    std::shared_ptr<tf2_ros::Buffer>            tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

}  // namespace capella_inspection

#endif  // CAPELLA_INSPECTION_NODE_HPP_