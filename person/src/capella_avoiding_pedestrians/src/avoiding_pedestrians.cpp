#include "capella_avoiding_pedestrians/avoiding_pedestrians.hpp"

#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <opencv2/imgcodecs.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/utils.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>

using namespace std::chrono_literals;

namespace
{
class OpenCvYoloTracker : public YoloTracker
{
public:
    OpenCvYoloTracker(const std::string &model_path, const rclcpp::Logger &logger)
        : logger_(logger)
    {
        try {
            net_ = cv::dnn::readNet(model_path);
            if (net_.empty()) {
                RCLCPP_ERROR(logger_, "Failed to load YOLO model: %s", model_path.c_str());
                yolo_enabled_ = false;
                return;
            }
            yolo_enabled_ = true;
            RCLCPP_INFO(logger_, "YOLO model loaded: %s", model_path.c_str());

            // 写死 CPU，CUDA 后端对 YOLOv8 有 NaN bug
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            RCLCPP_INFO(logger_, "[YOLO] CPU backend enabled");

        } catch (const std::exception &e) {
            RCLCPP_ERROR(logger_, "Exception while loading YOLO model: %s", e.what());
            yolo_enabled_ = false;
        }
    }

    std::vector<TrackItem> track(const cv::Mat &frame) override
    {
        std::vector<TrackItem> tracks;
        if (!yolo_enabled_ || frame.empty()) return tracks;

        constexpr int input_size = 640;
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0,
                               cv::Size(input_size, input_size),
                               cv::Scalar(), true, false);
        net_.setInput(blob);

        std::vector<cv::Mat> outputs;
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());
        if (outputs.empty()) return tracks;

        const cv::Mat &out = outputs[0];
        if (out.dims != 3) return tracks;

        // YOLOv8 输出 [1, 84, 8400] → 转置为 [8400, 84]
        cv::Mat det;
        if (out.size[1] < out.size[2]) {
            cv::Mat raw(out.size[1], out.size[2], CV_32F,
                        const_cast<float *>(out.ptr<float>()));
            cv::transpose(raw, det);
        } else {
            det = cv::Mat(out.size[1], out.size[2], CV_32F,
                          const_cast<float *>(out.ptr<float>())).clone();
        }

        const int class_count = det.cols - 4;
        if (class_count <= 0) return tracks;

        const float x_scale = static_cast<float>(frame.cols) / static_cast<float>(input_size);
        const float y_scale = static_cast<float>(frame.rows) / static_cast<float>(input_size);

        // ========== 改动2: 收集候选框，最后统一做 NMS ==========
        std::vector<cv::Rect> boxes;
        std::vector<float>    confidences;
        std::vector<int>      class_ids;

        for (int i = 0; i < det.rows; ++i) {
            const float *row = det.ptr<float>(i);
            if (!row) continue;

            // 找最高分类别
            int best_class = -1;
            float best_score = 0.0F;
            for (int c = 0; c < class_count; ++c) {
                const float score = row[4 + c];
                if (score > best_score) {
                    best_score = score;
                    best_class = c;
                }
            }

            // 只保留 person (class 0)，低阈值先过滤
            if (best_class != 0 || best_score < 0.25F) continue;

            const float cx = row[0] * x_scale;
            const float cy = row[1] * y_scale;
            const float w  = row[2] * x_scale;
            const float h  = row[3] * y_scale;

            const int x1 = static_cast<int>(cx - 0.5F * w);
            const int y1 = static_cast<int>(cy - 0.5F * h);
            const int bw = static_cast<int>(w);
            const int bh = static_cast<int>(h);

            boxes.emplace_back(x1, y1, bw, bh);
            confidences.push_back(best_score);
            class_ids.push_back(best_class);
        }

        // NMS 去除重叠框
        std::vector<int> indices;
        if (!boxes.empty()) {
            cv::dnn::NMSBoxes(boxes, confidences, 0.25F, 0.45F, indices);
        }

        // 只保留 NMS 后的结果
        tracks.reserve(indices.size());
        for (int idx : indices) {
            TrackItem item;
            item.class_id   = class_ids[idx];
            item.confidence = confidences[idx];
            item.bbox.x      = static_cast<float>(boxes[idx].x);
            item.bbox.y      = static_cast<float>(boxes[idx].y);
            item.bbox.width   = static_cast<float>(boxes[idx].width);
            item.bbox.height  = static_cast<float>(boxes[idx].height);
            tracks.push_back(item);
        }

        return tracks;
    }

private:
    rclcpp::Logger logger_;
    rclcpp::Clock clock_{RCL_SYSTEM_TIME};
    cv::dnn::Net net_;
    bool yolo_enabled_{false};
};
}  // namespace

BehaviorDetectionNode::BehaviorDetectionNode() : Node("behavior_detection_node")
{
    this->declare_parameter<double>("global_search_distance", 5.0);  // 全局路径搜索距离
    this->declare_parameter<double>("pedestrian_distance_threshold", 1.0); // 行人距离阈值
    this->declare_parameter<double>("local_search_distance", 5.0);  // 局部路径搜索距离
    this->declare_parameter<double>("avoid_hold_seconds", 3.0); // 避让行人的保持时间
    this->declare_parameter<double>("person_conf_threshold", 0.6); // 行人检测置信度阈值
    local_search_distance_ = this->get_parameter("local_search_distance").as_double();
    avoid_hold_seconds_ = this->get_parameter("avoid_hold_seconds").as_double();
    person_conf_threshold_ = this->get_parameter("person_conf_threshold").as_double();
    //默认的模型的位置
    const std::string default_model_path =
        (std::filesystem::path(__FILE__).parent_path() / "car8.onnx").string();

    this->declare_parameter<std::string>("yolo_model_path", default_model_path);
    //默认的雷达和相机话题
    this->declare_parameter<std::string>("scan_topic_name_front", "/front_scan");
    this->declare_parameter<std::string>("camera_topic", "/rgb_camera_front/compressed");
    this->declare_parameter<std::string>("rgb_camera_frame", "front_camera_color_frame");

    const std::string scan_topic = this->get_parameter("scan_topic_name_front").as_string();
    const std::string camera_topic = this->get_parameter("camera_topic").as_string();
    // ros2 run ...pkg ...node --ros-args -p yolo_model_path:=/...model.pt
    const std::string yolo_model_path = this->get_parameter("yolo_model_path").as_string();
    //暂时是用的onnx，不用pt
    if (std::filesystem::path(yolo_model_path).extension() == ".pt") {
        RCLCPP_WARN(
            this->get_logger(),
            "Configured YOLO model is .pt (%s). Current C++ tracker uses OpenCV DNN and expects ONNX; please convert .pt to .onnx.",
            yolo_model_path.c_str());
    }
    yolo_ = std::make_shared<OpenCvYoloTracker>(yolo_model_path, this->get_logger());
    laser_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    global_plan_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    local_poses_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    //qos类型
    auto qos_transient_local = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable();
    //警报话题
    pub_avoiding_ = this->create_publisher<std_msgs::msg::Bool>("is_avoiding_pedestrians", qos_transient_local);
    pub_annotated_image_ = this->create_publisher<sensor_msgs::msg::Image>("/rgb_camera_front/annotated_image", 10);
    pub_pedestrians_map_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/pedestrians/map", 10);
    pub_pedestrians_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/pedestrians/map_markers", 10);
    
    sub_camera_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
        camera_topic,
        rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::CompressedImage::SharedPtr msg)
        {
            this->imageCallback(msg);
        }); //相机回调

    auto laser_sub_options = rclcpp::SubscriptionOptions();
    laser_sub_options.callback_group = laser_callback_group_;
    sub_laser_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        scan_topic,
        rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::LaserScan::SharedPtr msg)
        {
            this->laserCallback(msg);
        },
        laser_sub_options); //激光回调

    auto global_plan_sub_options = rclcpp::SubscriptionOptions();
    global_plan_sub_options.callback_group = global_plan_callback_group_;
    sub_global_plan_ = this->create_subscription<nav_msgs::msg::Path>(
        "teb_global_plan",
        10,
        [this](const nav_msgs::msg::Path::SharedPtr msg)
        {
            this->globalPlanCallback(msg);
        },
        global_plan_sub_options);    //全局地图回调

    auto local_poses_sub_options = rclcpp::SubscriptionOptions();
    local_poses_sub_options.callback_group = local_poses_callback_group_;
    sub_local_poses_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
        "teb_poses",
        //rclcpp::SensorDataQoS(),   // 如果有订阅但是不触发回调就改成这个
        10,
        [this](const geometry_msgs::msg::PoseArray::SharedPtr msg)
        {
            this->localPosesCallback(msg);
        },
        local_poses_sub_options);   //局部回调

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    timer_ = this->create_wall_timer(
        100ms,
        [this]()
        {
            this->timerCallback();
        });

    // RCLCPP_INFO(this->get_logger(), "Behavior Detection Node initialized.");
    // RCLCPP_INFO(this->get_logger(), "capella_avoiding_pedestrians is ready");
}

cv::Mat BehaviorDetectionNode::decodeCompressedImage(
    const sensor_msgs::msg::CompressedImage::SharedPtr &msg,
    rclcpp::Time &out_stamp)
{
    out_stamp = msg->header.stamp;
    // 将 msg->data (std::vector<uint8_t>) 转为 cv::Mat 后解码
    cv::Mat img = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
    if (img.empty()) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            2000,
            "[decodeCompressedImage] cv::imdecode failed, compressed data may be corrupted");
    }
    return img;
}

void BehaviorDetectionNode::imageCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
{
    static bool callback_logged_once = false;
    if (!callback_logged_once) {
        RCLCPP_INFO(this->get_logger(), "imageCallback triggered");
        callback_logged_once = true;
    }
    if (!msg) return;
    if (msg->data.empty()) return;
    // 跳帧：每5帧推理一次
    static int frame_skip_counter = 0;
    if (++frame_skip_counter % 5 != 0) return;

    // 解码压缩图像
    rclcpp::Time img_stamp;
    cv::Mat frame = decodeCompressedImage(msg, img_stamp);
    if (frame.empty()) {
        return;
    }
    static bool decoded_logged_once = false;
    if (!decoded_logged_once) {
        RCLCPP_INFO(this->get_logger(), "[imageCallback] Compressed image decoded successfully, size=%dx%d",
            frame.cols, frame.rows);
        decoded_logged_once = true;
    }
    // YOLO检测人体框
    std::vector<TrackItem> tracks;
    if (!runYoloTrack(frame, tracks)) return;
    // 只在框数量变化（0->非0 或 非0->0）时打印日志
    const bool current_has_boxes = !tracks.empty();
    if (current_has_boxes != last_yolo_has_boxes_) {
        if (current_has_boxes) {
            RCLCPP_INFO(this->get_logger(), "YOLO detected %zu boxes", tracks.size());
        } else {
            RCLCPP_INFO(this->get_logger(), "YOLO no detection (0 boxes)");
        }
        last_yolo_has_boxes_ = current_has_boxes;
    }
    // 只在有行人框时才打印
    const bool frame_has_person = std::any_of(tracks.begin(), tracks.end(),
        [this](const TrackItem &t) {
            return t.class_id == 0 && t.confidence >= static_cast<float>(person_conf_threshold_);
        });
    if (frame_has_person) {
        RCLCPP_INFO_THROTTLE(
            this->get_logger(), *this->get_clock(), 500,
            "[imageCallback] person box(es) found, total YOLO boxes=%zu", tracks.size());
    }

    // 获取最新的激光雷达数据
    sensor_msgs::msg::LaserScan::SharedPtr latest_scan;
    {
        std::lock_guard<std::mutex> lock(laser_queue_mutex_);
        if (!laser_queue_.empty()) {
            latest_scan = laser_queue_.back().second;
        }
    }

    // 获取激光雷达到相机坐标系的完整TF变换
    const std::string rgb_camera_frame = this->get_parameter("rgb_camera_frame").as_string();
    geometry_msgs::msg::TransformStamped tf_cam_laser;
    bool have_tf = false;
    if (latest_scan && !rgb_camera_frame.empty()) {
        try {
            tf_cam_laser = tf_buffer_->lookupTransform(
                rgb_camera_frame,
                latest_scan->header.frame_id,
                tf2::TimePointZero,
                tf2::durationFromSec(0.05));
            have_tf = true;
        } catch (const tf2::TransformException &e) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "[TF] lookupTransform failed: %s (cam='%s', laser='%s')",
                e.what(), rgb_camera_frame.c_str(), latest_scan->header.frame_id.c_str());
        }
    }

    // 将所有有效激光点批量变换到相机坐标系
    // 相机body坐标系约定：X前、Y左
    struct LaserCamPt { float xc; float yc; float r; float theta_l; };
    std::vector<LaserCamPt> laser_cam_pts;
    if (latest_scan && have_tf) {
        const auto &q = tf_cam_laser.transform.rotation;
        const auto &t = tf_cam_laser.transform.translation;
        // 由四元数构造旋转矩阵
        const float qx = static_cast<float>(q.x);
        const float qy = static_cast<float>(q.y);
        const float qz = static_cast<float>(q.z);
        const float qw = static_cast<float>(q.w);
        const float R00 = 1.0F - 2.0F*(qy*qy + qz*qz);
        const float R01 = 2.0F*(qx*qy - qz*qw);
        const float R10 = 2.0F*(qx*qy + qz*qw);
        const float R11 = 1.0F - 2.0F*(qx*qx + qz*qz);
        const float tx = static_cast<float>(t.x);
        const float ty = static_cast<float>(t.y);
        const int n = static_cast<int>(latest_scan->ranges.size());
        laser_cam_pts.reserve(n); //预分配内存的方法
        for (int i = 0; i < n; ++i) {
            const float r = latest_scan->ranges[i];
            if (!std::isfinite(r) || r < latest_scan->range_min || r > latest_scan->range_max) continue;
            const float theta_l = latest_scan->angle_min + i * latest_scan->angle_increment;
            const float xl = r * std::cos(theta_l);
            const float yl = r * std::sin(theta_l);
            const float xc = R00*xl + R01*yl + tx;
            const float yc = R10*xl + R11*yl + ty;
            laser_cam_pts.push_back({xc, yc, r, theta_l});
        }
        RCLCPP_DEBUG(this->get_logger(), "[TF] cam=%s laser=%s valid_pts=%zu",
            rgb_camera_frame.c_str(), latest_scan->header.frame_id.c_str(), laser_cam_pts.size());
    }

    cv::Mat annotated = frame.clone();
    DetectionResult result;   // imageCallback的检测结果，包含行人坐标和激光信息
    result.detected = false;
    result.stamp = msg->header.stamp;
    if (latest_scan) {
        result.laser_frame_id = latest_scan->header.frame_id;
        result.laser_stamp = latest_scan->header.stamp;
    }

    for (const auto &det : tracks) {
        if (det.class_id != 0) continue;
        if (det.confidence < static_cast<float>(person_conf_threshold_)) continue;

        // 由检测框左右边界像素坐标计算相机水平角度范围
        // 角度约定：图像中心为0，左正右负
        const float img_w = static_cast<float>(frame.cols);
        const float theta1 = (0.5F - det.bbox.x / img_w) * static_cast<float>(h_fov_rad_);
        const float theta2 = (0.5F - (det.bbox.x + det.bbox.width) / img_w) * static_cast<float>(h_fov_rad_);
        const float theta_min = std::min(theta1, theta2);
        const float theta_max = std::max(theta1, theta2);

        // 在相机坐标系角度范围内搜索最近的激光点
        float distance = std::numeric_limits<float>::quiet_NaN();
        float best_theta_l = 0.0F;  // 最近点在激光坐标系中的角度
        float best_r = 0.0F;        // 最近点在激光坐标系中的距离
        if (!laser_cam_pts.empty()) {
            float min_dist2d = std::numeric_limits<float>::max();
            for (const auto &lp : laser_cam_pts) {
                const float pt_angle = std::atan2(lp.yc, lp.xc);
                const float dist2d = std::hypot(lp.xc, lp.yc);
                if (pt_angle >= theta_min && pt_angle <= theta_max && dist2d > 0.3F) {
                    if (dist2d < min_dist2d) {
                        min_dist2d = dist2d;
                        distance = dist2d;
                        best_theta_l = lp.theta_l;
                        best_r = lp.r;
                    }
                }
            }
        } else if (!latest_scan) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "[laser] no scan data available");
        } else if (!have_tf) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "[laser] TF unavailable, cannot map laser to camera frame");
        } else {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "[laser] no valid laser pts in bbox angle range [%.3f, %.3f]rad conf=%.2f",
                theta_min, theta_max, det.confidence);
        }

        if (!std::isnan(distance)) {
            // 用激光坐标系原始极坐标还原笛卡尔位置，保持坐标系一致
            geometry_msgs::msg::Point p;
            p.x = best_r * std::cos(best_theta_l);
            p.y = best_r * std::sin(best_theta_l);
            p.z = 0.0;
            result.pedestrians_laser.push_back(p);
            result.detected = true;
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                "[laser] matched! dist=%.2fm bbox_angle=[%.3f,%.3f]rad conf=%.2f",
                distance, theta_min, theta_max, det.confidence);
        }

        // 绘制标注框，显示置信度和距离
        const int x_raw = static_cast<int>(std::lround(det.bbox.x));
        const int y_raw = static_cast<int>(std::lround(det.bbox.y));
        const int w_raw = std::max(1, static_cast<int>(std::lround(det.bbox.width)));
        const int h_raw = std::max(1, static_cast<int>(std::lround(det.bbox.height)));
        const int x = std::max(0, std::min(x_raw, annotated.cols - 1));
        const int y = std::max(0, std::min(y_raw, annotated.rows - 1));
        const int w = std::min(w_raw, std::max(1, annotated.cols - x));
        const int h = std::min(h_raw, std::max(1, annotated.rows - y));
        cv::rectangle(annotated, cv::Rect(x, y, w, h), cv::Scalar(0, 0, 255), 2);

        std::ostringstream label_stream;
        label_stream << std::fixed << std::setprecision(2) << det.confidence;
        if (!std::isnan(distance)) {
            label_stream << " " << std::fixed << std::setprecision(1) << distance << "m";
        }
        cv::putText(
            annotated,
            label_stream.str(),
            cv::Point(x, std::max(15, y - 5)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 0, 255),
            1);
    }

    // 只要有人被YOLO检测到都发布标注图像
    const bool has_any_person = std::any_of(tracks.begin(), tracks.end(),
        [this](const TrackItem &t) {
            return t.class_id == 0 && t.confidence >= static_cast<float>(person_conf_threshold_);
        });
    if (has_any_person && pub_annotated_image_) {
        auto annotated_msg = sensor_msgs::msg::Image();
        annotated_msg.header = msg->header;
        annotated_msg.height = static_cast<uint32_t>(annotated.rows);
        annotated_msg.width = static_cast<uint32_t>(annotated.cols);
        annotated_msg.encoding = sensor_msgs::image_encodings::BGR8;
        annotated_msg.is_bigendian = 0;
        annotated_msg.step = static_cast<sensor_msgs::msg::Image::_step_type>(annotated.cols * annotated.elemSize());
        const size_t image_bytes = static_cast<size_t>(annotated_msg.step) * static_cast<size_t>(annotated.rows);
        annotated_msg.data.assign(annotated.data, annotated.data + image_bytes);
        pub_annotated_image_->publish(annotated_msg);
    }

    {
        std::lock_guard<std::mutex> lock(detection_mutex_);
        detection_result_ = std::move(result);
    }
}

bool BehaviorDetectionNode::runYoloTrack(const cv::Mat &frame, std::vector<TrackItem> &tracks)
{
    if (!yolo_) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            2000,
            "YOLO tracker is not initialized.");
        return false;
    }

    try {
        tracks = yolo_->track(frame); // 运行YOLO跟踪的
        return true;
    } catch (const std::exception &e) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            2000,
            "YOLO inference failed: %s",
            e.what());
        return false;
    } catch (...) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            2000,
            "YOLO inference failed with unknown exception.");
        return false;
    }
}

void BehaviorDetectionNode::laserCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
    if (!msg) return;

    const double ts = static_cast<double>(msg->header.stamp.sec)
                    + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;

    std::lock_guard<std::mutex> lock(laser_queue_mutex_);
    laser_queue_.emplace_back(ts, msg);

    while (laser_queue_.size() > laser_queue_max_size_) {
        laser_queue_.pop_front();
    }
}
void BehaviorDetectionNode::globalPlanCallback(const nav_msgs::msg::Path::SharedPtr msg)
{
    if (!msg) return;
    last_global_plan_ = msg;  // 保存全局路径
    
    static bool once = true;
    if (once) {
        RCLCPP_INFO(this->get_logger(), "globalPlanCallback: received path with %zu poses", 
            msg->poses.size());
        once = false;
    }
}

void BehaviorDetectionNode::localPosesCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
{
    if (!msg) return;
    last_local_poses_ = msg;

    static bool once = true;
    if (once) {
        RCLCPP_INFO(this->get_logger(), "localPosesCallback: received poses with %zu poses at %d.%09d",
            msg->poses.size(), msg->header.stamp.sec, msg->header.stamp.nanosec);
        RCLCPP_INFO(this->get_logger(), "get local poses plan");
        once = false;
    }
}

void BehaviorDetectionNode::timerCallback()
{
    bool trigger_now = false;  // 默认没检测到人

    if (last_global_plan_ || last_local_poses_) {
        DetectionResult det;
        {
            std::lock_guard<std::mutex> lock(detection_mutex_);
            det = detection_result_;
        }
        // imageCallback中已完成激光测距，直接使用激光坐标系下的行人坐标
        if (det.detected && !det.pedestrians_laser.empty() && !det.laser_frame_id.empty()) {

            // 将行人从激光坐标系转换到map坐标系
            // 使用 rclcpp::Time(0) 查询最新可用变换，避免相机/激光时间戳不一致导致TF失败
            std::vector<geometry_msgs::msg::PointStamped> pedestrians_map;
            pedestrians_map.reserve(det.pedestrians_laser.size());
            for (const auto &pt_laser : det.pedestrians_laser) {
                geometry_msgs::msg::PointStamped p_in;
                p_in.header.frame_id = det.laser_frame_id;
                p_in.header.stamp = rclcpp::Time(0, 0, RCL_ROS_TIME);  // 使用最新可用变换
                p_in.point = pt_laser;
                try {
                    geometry_msgs::msg::PointStamped p_map;
                    tf_buffer_->transform(p_in, p_map, "map", tf2::durationFromSec(0.2));
                    pedestrians_map.push_back(p_map);
                } catch (const tf2::TransformException &e) {
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                        "[TF] laser->map transform failed: %s (frame='%s')",
                        e.what(), det.laser_frame_id.c_str());
                }
            }
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                "[timer] TF done: laser_peds=%zu -> map_peds=%zu",
                det.pedestrians_laser.size(), pedestrians_map.size());

            if (!pedestrians_map.empty()) {
                // 发布可视化
                geometry_msgs::msg::PoseArray poses_msg;
                poses_msg.header.frame_id = "map";
                poses_msg.header.stamp = this->now();
                poses_msg.poses.reserve(pedestrians_map.size());

                visualization_msgs::msg::MarkerArray marker_array;
                marker_array.markers.reserve(pedestrians_map.size() + 1);

                // 先清除旧的marker
                visualization_msgs::msg::Marker delete_all;
                delete_all.header.frame_id = "map";
                delete_all.header.stamp = poses_msg.header.stamp;
                delete_all.ns = "pedestrians";
                delete_all.id = 0;
                delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
                marker_array.markers.push_back(delete_all);

                for (size_t i = 0; i < pedestrians_map.size(); ++i) {
                    const auto &p = pedestrians_map[i];
                    geometry_msgs::msg::Pose pose;
                    pose.position = p.point;
                    pose.orientation.w = 1.0;
                    poses_msg.poses.push_back(pose);

                    visualization_msgs::msg::Marker marker;
                    marker.header.frame_id = "map";
                    marker.header.stamp = poses_msg.header.stamp;
                    marker.ns = "pedestrians";
                    marker.id = static_cast<int>(i) + 1;
                    marker.type = visualization_msgs::msg::Marker::SPHERE;
                    marker.action = visualization_msgs::msg::Marker::ADD;
                    marker.pose.position = p.point;
                    marker.pose.orientation.w = 1.0;
                    marker.scale.x = 0.30;
                    marker.scale.y = 0.30;
                    marker.scale.z = 0.30;
                    marker.color.r = 0.0F;
                    marker.color.g = 1.0F;
                    marker.color.b = 0.0F;
                    marker.color.a = 0.9F;
                    marker_array.markers.push_back(marker);
                }

                if (pub_pedestrians_map_) pub_pedestrians_map_->publish(poses_msg);
                if (pub_pedestrians_markers_) pub_pedestrians_markers_->publish(marker_array);

                // 检查行人是否在规划路径上
                const double search_distance = this->get_parameter("global_search_distance").as_double();
                const double threshold = this->get_parameter("pedestrian_distance_threshold").as_double();

                const bool has_global_path = last_global_plan_ && !last_global_plan_->poses.empty();
                const bool has_local_path = last_local_poses_ && !last_local_poses_->poses.empty();

                const bool on_global = has_global_path
                    ? checkPedestrianOnGlobalPath(pedestrians_map, *last_global_plan_, search_distance, threshold)
                    : false;
                const bool on_local = has_local_path
                    ? checkPedestrianOnLocalPath(pedestrians_map, last_local_poses_, threshold)
                    : false;

                trigger_now = (on_global || on_local);
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                    "[timer] on_global=%d on_local=%d trigger=%d",
                    on_global, on_local, trigger_now);
            }
        }
    }

    const rclcpp::Time now_time = this->now();
    if (trigger_now) {
        has_recent_detection_ = true;
        last_pedestrian_detect_time_ = now_time;
    }

    bool is_avoiding = false;
    if (trigger_now) {
        is_avoiding = true;
    } else if (has_recent_detection_) {
        const double dt = (now_time - last_pedestrian_detect_time_).seconds();
        if (dt < avoid_hold_seconds_) {
            is_avoiding = true;
        } else {
            has_recent_detection_ = false;
            is_avoiding = false;
        }
    }

    if (is_avoiding != last_published_avoiding_) {
        auto out = std_msgs::msg::Bool();
        out.data = is_avoiding;
        pub_avoiding_->publish(out);
        //加个日志
        if (is_avoiding) {
            ++warning_event_id_;
            RCLCPP_WARN(
                this->get_logger(),
                "having pedestrians warning, id: %llu",
                static_cast<unsigned long long>(warning_event_id_));
        }
        last_published_avoiding_ = is_avoiding;
    }
}

std::vector<geometry_msgs::msg::PointStamped> BehaviorDetectionNode::fuseAndTrackPedestrians(
    const visualization_msgs::msg::MarkerArray::SharedPtr detections,
    const sensor_msgs::msg::LaserScan::SharedPtr scan)
{
    (void)detections;
    (void)scan;
    return {};
}

std::vector<geometry_msgs::msg::Point> BehaviorDetectionNode::downsamplePath(
    //下采样的函数
    const std::vector<geometry_msgs::msg::Point> &nav_points,
    double min_distance,
    double lookahead_distance)
{
    std::vector<geometry_msgs::msg::Point> out_points;
    if (nav_points.empty()) return out_points;

    if (nav_points.size() >= 2) {
        const double end_dx = nav_points.back().x - nav_points.front().x;
        const double end_dy = nav_points.back().y - nav_points.front().y;
        const double end_dist = std::sqrt(end_dx * end_dx + end_dy * end_dy);
        if (end_dist < lookahead_distance) {
            return nav_points;
        }
    }

    out_points.push_back(nav_points.front());
    if (nav_points.size() == 1) return out_points;

    const double sample_dist = (min_distance > 0.0) ? min_distance : 0.0;
    const double sample_dist_sq = sample_dist * sample_dist;
    const double max_lookahead = (lookahead_distance > 0.0) ? lookahead_distance : local_search_distance_;

    size_t last_keep_idx = 0;
    double accumulated_len = 0.0;

    for (size_t i = 1; i < nav_points.size(); ++i) {
        const double seg_dx = nav_points[i].x - nav_points[i - 1].x;
        const double seg_dy = nav_points[i].y - nav_points[i - 1].y;
        const double seg_len = std::sqrt(seg_dx * seg_dx + seg_dy * seg_dy);

        if (accumulated_len + seg_len > max_lookahead) {
            break;
        }
        accumulated_len += seg_len;

        if (sample_dist <= 0.0) {
            out_points.push_back(nav_points[i]);
            last_keep_idx = i;
            continue;
        }

        const double dx = nav_points[i].x - nav_points[last_keep_idx].x;
        const double dy = nav_points[i].y - nav_points[last_keep_idx].y;
        const double dist_sq = dx * dx + dy * dy;

        if (dist_sq >= sample_dist_sq) {
            out_points.push_back(nav_points[i]);
            last_keep_idx = i;
        }
    }

    return out_points;
}

bool BehaviorDetectionNode::checkPedestrianOnGlobalPath(
    const std::vector<geometry_msgs::msg::PointStamped> &pedestrians,
    const nav_msgs::msg::Path &path,
    double search_distance,
    double threshold)
{
    if (pedestrians.empty() || path.poses.empty()) return false;

    std::vector<geometry_msgs::msg::Point> global_points;
    global_points.reserve(path.poses.size());
    for (const auto &pose_stamped : path.poses) {
        global_points.push_back(pose_stamped.pose.position);
    }

    const auto optimized_points = downsamplePath(global_points, threshold, search_distance);
    if (optimized_points.empty()) return false;

    const double threshold_sq = threshold * threshold;
    double min_dist_global = std::numeric_limits<double>::max();
    for (const auto &ped : pedestrians) {
        for (const auto &pt : optimized_points) {
            const double dx = ped.point.x - pt.x;
            const double dy = ped.point.y - pt.y;
            const double dist_sq = dx * dx + dy * dy;
            if (dist_sq < min_dist_global) min_dist_global = dist_sq;
            if (dist_sq <= threshold_sq) {
                return true;
            }
        }
    }
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
        "[global_path] no match: min_dist=%.2fm threshold=%.2fm path_pts=%zu ped_pts=%zu path_frame='%s'",
        std::sqrt(min_dist_global), threshold, optimized_points.size(), pedestrians.size(),
        path.header.frame_id.c_str());
    if (!pedestrians.empty()) {
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
            "[global_path] ped[0] map=(%.2f,%.2f) path[0]=(%.2f,%.2f)",
            pedestrians[0].point.x, pedestrians[0].point.y,
            optimized_points[0].x, optimized_points[0].y);
    }

    return false;
}

bool BehaviorDetectionNode::checkPedestrianOnLocalPath(
    const std::vector<geometry_msgs::msg::PointStamped> &pedestrians,
    const geometry_msgs::msg::PoseArray::SharedPtr local_poses,
    double threshold)
{
    if (pedestrians.empty() || !local_poses || local_poses->poses.empty()) return false;

    std::vector<geometry_msgs::msg::Point> local_points;
    local_points.reserve(local_poses->poses.size());
    for (const auto &pose : local_poses->poses) {
        local_points.push_back(pose.position);
    }

    const auto optimized_points = downsamplePath(local_points, threshold, local_search_distance_);
    if (optimized_points.empty()) return false;

    const double threshold_sq = threshold * threshold;
    double min_dist_local = std::numeric_limits<double>::max();
    for (const auto &ped : pedestrians) {
        for (const auto &pt : optimized_points) {
            const double dx = ped.point.x - pt.x;
            const double dy = ped.point.y - pt.y;
            const double dist_sq = dx * dx + dy * dy;
            if (dist_sq < min_dist_local) min_dist_local = dist_sq;
            if (dist_sq <= threshold_sq) {
                return true;
            }
        }
    }
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
        "[local_path] no match: min_dist=%.2fm threshold=%.2fm path_pts=%zu ped_pts=%zu path_frame='%s'",
        std::sqrt(min_dist_local), threshold, optimized_points.size(), pedestrians.size(),
        local_poses->header.frame_id.c_str());
    if (!pedestrians.empty()) {
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
            "[local_path] ped[0] map=(%.2f,%.2f) path[0]=(%.2f,%.2f)",
            pedestrians[0].point.x, pedestrians[0].point.y,
            optimized_points[0].x, optimized_points[0].y);
    }

    return false;
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BehaviorDetectionNode>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}