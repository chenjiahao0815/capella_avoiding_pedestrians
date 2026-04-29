// capella_inspection_node.cpp

#include "capella_inspection_node.hpp"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace capella_inspection
{

// ===========================================================================
//  YoloDetector 实现
// ===========================================================================

YoloDetector::YoloDetector(const std::string & model_path, bool use_cuda)
: env_(ORT_LOGGING_LEVEL_WARNING, "YoloDetector")
{
    session_options_.SetIntraOpNumThreads(4);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 如果有 CUDA，添加 CUDA EP；否则回退到 CPU
    if (use_cuda) {
        try {
            OrtCUDAProviderOptions cuda_opts{};
            cuda_opts.device_id = 0;
            session_options_.AppendExecutionProvider_CUDA(cuda_opts);
        } catch (const Ort::Exception & e) {
            // CUDA 不可用时静默回退到 CPU
            (void)e;
        }
    }

    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);

    // 读取输入名称和形状
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_inputs = session_->GetInputCount();
    for (size_t i = 0; i < num_inputs; ++i) {
        auto name = session_->GetInputNameAllocated(i, allocator);
        input_names_.emplace_back(name.get());
    }

    // 尝试从模型元信息中获取输入尺寸（假设 NCHW）
    auto input_shape = session_->GetInputTypeInfo(0)
                           .GetTensorTypeAndShapeInfo()
                           .GetShape();
    if (input_shape.size() == 4) {
        if (input_shape[2] > 0) input_h_ = input_shape[2];
        if (input_shape[3] > 0) input_w_ = input_shape[3];
    }

    // 读取输出名称
    size_t num_outputs = session_->GetOutputCount();
    for (size_t i = 0; i < num_outputs; ++i) {
        auto name = session_->GetOutputNameAllocated(i, allocator);
        output_names_.emplace_back(name.get());
    }

    // 缓存 c_str 指针
    for (auto & n : input_names_)  input_names_cstr_.push_back(n.c_str());
    for (auto & n : output_names_) output_names_cstr_.push_back(n.c_str());
}


std::vector<Detection> YoloDetector::detect(const cv::Mat & bgr_image,
                                            float conf_threshold,
                                            float iou_threshold)
{
    float scale = 1.0f;
    int pad_x = 0, pad_y = 0;

    // 前处理：letterbox + normalize + HWC→CHW
    std::vector<float> input_blob = preprocess(bgr_image, scale, pad_x, pad_y);

    // 构造输入 tensor
    std::array<int64_t, 4> input_shape = {1, 3, input_h_, input_w_};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_blob.data(),
        input_blob.size(),
        input_shape.data(),
        input_shape.size());

    // 推理
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_cstr_.data(),
        &input_tensor,
        1,
        output_names_cstr_.data(),
        output_names_cstr_.size());

    // 后处理
    return postprocess(output_tensors[0],
                       conf_threshold, iou_threshold,
                       scale, pad_x, pad_y,
                       bgr_image.cols, bgr_image.rows);
}


std::vector<float> YoloDetector::preprocess(const cv::Mat & bgr_image,
                                            float & scale,
                                            int & pad_x, int & pad_y)
{
    int orig_w = bgr_image.cols;
    int orig_h = bgr_image.rows;

    // letterbox 缩放比例
    scale = std::min(static_cast<float>(input_w_) / orig_w,
                     static_cast<float>(input_h_) / orig_h);

    int new_w = static_cast<int>(orig_w * scale);
    int new_h = static_cast<int>(orig_h * scale);

    pad_x = (static_cast<int>(input_w_) - new_w) / 2;
    pad_y = (static_cast<int>(input_h_) - new_h) / 2;

    cv::Mat resized;
    cv::resize(bgr_image, resized, cv::Size(new_w, new_h));

    // 创建灰色画布 (114, 114, 114)
    cv::Mat canvas(static_cast<int>(input_h_), static_cast<int>(input_w_),
                   CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(pad_x, pad_y, new_w, new_h)));

    // BGR → RGB
    cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);

    // 归一化到 [0, 1] 并转为 CHW
    canvas.convertTo(canvas, CV_32FC3, 1.0 / 255.0);

    int img_area = static_cast<int>(input_h_ * input_w_);
    std::vector<float> blob(3 * img_area);

    // 拆分通道 → CHW
    std::vector<cv::Mat> channels(3);
    cv::split(canvas, channels);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(blob.data() + c * img_area,
                    channels[c].data,
                    img_area * sizeof(float));
    }

    return blob;
}


std::vector<Detection> YoloDetector::postprocess(const Ort::Value & output_tensor,
                                                  float conf_threshold,
                                                  float iou_threshold,
                                                  float scale,
                                                  int pad_x, int pad_y,
                                                  int orig_w, int orig_h)
{
    // YOLOv8 输出形状: [1, (4 + num_classes), num_boxes]
    auto shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
    int num_features = static_cast<int>(shape[1]);  // 4 + num_classes
    int num_boxes    = static_cast<int>(shape[2]);
    int num_classes  = num_features - 4;

    const float * raw = output_tensor.GetTensorData<float>();

    std::vector<Detection> detections;
    detections.reserve(256);

    for (int b = 0; b < num_boxes; ++b) {
        // 前 4 个特征: cx, cy, w, h（在 640×640 letterbox 空间中）
        float cx = raw[0 * num_boxes + b];
        float cy = raw[1 * num_boxes + b];
        float bw = raw[2 * num_boxes + b];
        float bh = raw[3 * num_boxes + b];

        // 找到最大类别置信度
        int   best_cls  = 0;
        float best_conf = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            float conf = raw[(4 + c) * num_boxes + b];
            if (conf > best_conf) {
                best_conf = conf;
                best_cls  = c;
            }
        }

        if (best_conf < conf_threshold) continue;

        // 从 letterbox 空间映射回原始图像空间
        float real_cx = (cx - pad_x) / scale;
        float real_cy = (cy - pad_y) / scale;
        float real_w  = bw / scale;
        float real_h  = bh / scale;

        // 裁剪到图像边界
        real_cx = std::clamp(real_cx, real_w / 2.0f, orig_w - real_w / 2.0f);
        real_cy = std::clamp(real_cy, real_h / 2.0f, orig_h - real_h / 2.0f);

        Detection det;
        det.class_id   = best_cls;
        det.confidence = best_conf;
        det.x_center   = real_cx;
        det.y_center   = real_cy;
        det.width      = real_w;
        det.height     = real_h;
        detections.push_back(det);
    }

    return nms(detections, iou_threshold);
}


float YoloDetector::compute_iou(const Detection & a, const Detection & b)
{
    float ax1 = a.x_center - a.width  / 2.0f;
    float ay1 = a.y_center - a.height / 2.0f;
    float ax2 = a.x_center + a.width  / 2.0f;
    float ay2 = a.y_center + a.height / 2.0f;

    float bx1 = b.x_center - b.width  / 2.0f;
    float by1 = b.y_center - b.height / 2.0f;
    float bx2 = b.x_center + b.width  / 2.0f;
    float by2 = b.y_center + b.height / 2.0f;

    float inter_x1 = std::max(ax1, bx1);
    float inter_y1 = std::max(ay1, by1);
    float inter_x2 = std::min(ax2, bx2);
    float inter_y2 = std::min(ay2, by2);

    float inter_area = std::max(0.0f, inter_x2 - inter_x1) *
                       std::max(0.0f, inter_y2 - inter_y1);

    float area_a = a.width * a.height;
    float area_b = b.width * b.height;

    return inter_area / (area_a + area_b - inter_area + 1e-6f);
}


std::vector<Detection> YoloDetector::nms(std::vector<Detection> & dets, float iou_threshold)
{
    // 按置信度降序排列
    std::sort(dets.begin(), dets.end(),
              [](const Detection & a, const Detection & b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<Detection> result;
    result.reserve(dets.size());

    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) continue;
            if (dets[i].class_id == dets[j].class_id &&
                compute_iou(dets[i], dets[j]) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}


// ===========================================================================
//  CapellaInspectionNode 实现
// ===========================================================================

// ---------------------------------------------------------------------------
//  UUID 生成
// ---------------------------------------------------------------------------
std::string CapellaInspectionNode::generate_uuid()
{
    static std::mt19937 gen{std::random_device{}()};
    static std::uniform_int_distribution<int> dist(0, 15);

    const char hex[] = "0123456789abcdef";
    // 格式: 8-4-4-4-12
    const int groups[] = {8, 4, 4, 4, 12};

    std::ostringstream oss;
    for (int g = 0; g < 5; ++g) {
        if (g > 0) oss << '-';
        for (int i = 0; i < groups[g]; ++i) {
            oss << hex[dist(gen)];
        }
    }
    return oss.str();
}


// ---------------------------------------------------------------------------
//  构造函数
// ---------------------------------------------------------------------------
CapellaInspectionNode::CapellaInspectionNode(const rclcpp::NodeOptions & options)
: Node("capella_inspection_node", options),
  last_process_time_(this->get_clock()->now())
{
    // ---- 声明并获取参数 ----
    this->declare_parameter<std::vector<std::string>>(
        "camera_topics", {"/rgb_camera_front/image_raw"});
    this->declare_parameter<std::vector<std::string>>(
        "camera_tf_names", {"front_camera_color_frame"});
    this->declare_parameter<std::vector<double>>(
        "horizontal_fov_degs", {84.3});
    this->declare_parameter<int>("max_publish_per_uuid", 2);
    this->declare_parameter<double>("process_interval_sec", 2.0);
    this->declare_parameter<std::string>(
        "model_path",
        "/capella/lib/python3.10/site-packages/capella_inspection_node/2-24.onnx");

    camera_topics_   = this->get_parameter("camera_topics").as_string_array();
    camera_tf_names_ = this->get_parameter("camera_tf_names").as_string_array();
    horizontal_fovs_ = this->get_parameter("horizontal_fov_degs").as_double_array();
    max_publish_per_uuid_ = this->get_parameter("max_publish_per_uuid").as_int();
    process_interval_sec_ = this->get_parameter("process_interval_sec").as_double();
    std::string model_path = this->get_parameter("model_path").as_string();

    // ---- 参数校验 ----
    if (camera_topics_.size() != camera_tf_names_.size() ||
        camera_topics_.size() != horizontal_fovs_.size()) {
        RCLCPP_FATAL(this->get_logger(),
                     "camera_topics, camera_tf_names, horizontal_fov_degs 的长度必须一致!");
        throw std::runtime_error("Parameter array length mismatch");
    }

    // ---- 加载 YOLO 模型 ----
    RCLCPP_INFO(this->get_logger(), "Loading YOLO model from: %s", model_path.c_str());
    yolo_detector_ = std::make_unique<YoloDetector>(model_path, /*use_cuda=*/true);
    RCLCPP_INFO(this->get_logger(), "YOLO model loaded.");

    // ---- TF ----
    tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // ---- QoS ----
    rclcpp::QoS qos_best_effort(1);
    qos_best_effort.best_effort();

    rclcpp::QoS qos_transient_local(1);
    qos_transient_local.transient_local();

    // ---- 创建多相机图像订阅 ----
    for (size_t i = 0; i < camera_topics_.size(); ++i) {
        callback_groups_.push_back(
            this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive));

        rclcpp::SubscriptionOptions sub_opts;
        sub_opts.callback_group = callback_groups_.back();

        // 用 lambda 捕获当前相机的参数，等价于 Python 的 functools.partial
        std::string topic   = camera_topics_[i];
        std::string tf_name = camera_tf_names_[i];
        double fov          = horizontal_fovs_[i];

        auto sub = this->create_subscription<sensor_msgs::msg::Image>(
            topic,
            qos_best_effort,
            [this, topic, tf_name, fov](const sensor_msgs::msg::Image::SharedPtr msg) {
                this->image_callback(msg, topic, tf_name, fov);
            },
            sub_opts);

        image_subs_.push_back(sub);
    }

    // ---- 控制话题订阅 ----
    ctr_sub_ = this->create_subscription<std_msgs::msg::Bool>(
        "/inspections_ctr",
        qos_transient_local,
        std::bind(&CapellaInspectionNode::control_callback, this, std::placeholders::_1));

    // ---- 发布者 ----
    result_pub_ = this->create_publisher<capella_ros_msg::msg::Recognitions>(
        "/inspections_data", 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "/inspection_rays", 10);

    RCLCPP_INFO(this->get_logger(), "Capella Inspection Node initialized !");
}


// ---------------------------------------------------------------------------
//  control_callback
// ---------------------------------------------------------------------------
void CapellaInspectionNode::control_callback(const std_msgs::msg::Bool::SharedPtr msg)
{
    is_active_ = msg->data;
    RCLCPP_INFO(this->get_logger(), "Inspection control set to: %s",
                is_active_ ? "ON" : "OFF");
}


// ---------------------------------------------------------------------------
//  image_callback —— 核心处理逻辑
// ---------------------------------------------------------------------------
void CapellaInspectionNode::image_callback(
    const sensor_msgs::msg::Image::SharedPtr msg,
    const std::string & topic_name,
    const std::string & tf_name,
    double horizontal_fov)
{
    (void)topic_name;  // 当前未使用，保留以备日志输出

    if (!is_active_) {
        // 关闭状态下清空历史
        uuid_publish_count_.clear();
        ray_history_.clear();
        return;
    }

    if (msg->data.empty()) return;

    // ---- 图像转换 ----
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (const cv_bridge::Exception & e) {
        RCLCPP_WARN(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    cv::Mat & color_data = cv_ptr->image;
    int color_width  = color_data.cols;
    int color_height = color_data.rows;

    rclcpp::Time current_time(msg->header.stamp);

    // ---- YOLO 推理 ----
    std::vector<Detection> detections = yolo_detector_->detect(color_data, 0.65f);

    // 过滤：只保留 class_id 为 1 或 2 的（火焰）
    std::vector<Detection> fire_dets;
    for (auto & d : detections) {
        if (d.class_id == 1 || d.class_id == 2) {
            fire_dets.push_back(d);
        }
    }

    if (fire_dets.empty()) return;

    // ---- 时间间隔控制 ----
    if (!first_frame_processed_) {
        first_frame_processed_ = true;
        last_process_time_ = current_time;
    } else {
        double elapsed_ns = static_cast<double>(
            (current_time - last_process_time_).nanoseconds());
        double interval_ns = process_interval_sec_ * 1e9;
        if (elapsed_ns < interval_ns) {
            return;
        }
        last_process_time_ = current_time;
    }

    // ---- TF 查询 ----
    geometry_msgs::msg::TransformStamped transform_base;
    geometry_msgs::msg::TransformStamped transform_camera;
    try {
        transform_base = tf_buffer_->lookupTransform(
            "map", "base_link", current_time, rclcpp::Duration::from_seconds(0.5));
        transform_camera = tf_buffer_->lookupTransform(
            "map", tf_name, current_time, rclcpp::Duration::from_seconds(0.5));
    } catch (const tf2::TransformException & e) {
        RCLCPP_WARN(this->get_logger(), "Failed to get transformation: %s", e.what());
        return;
    }

    double X_R = transform_base.transform.translation.x;
    double Y_R = transform_base.transform.translation.y;

    // ---- 构造发布消息 ----
    capella_ros_msg::msg::Recognitions inspections_msg;
    int last_count = 0;  // 用于最后的日志输出

    for (auto & det : fire_dets) {
        int x1 = det.x1();
        int y1 = det.y1();
        int x2 = det.x2();
        int y2 = det.y2();

        // 计算水平角度（弧度）
        double theta_x_center_rad =
            (0.5 - static_cast<double>(det.x_center) / color_width) * horizontal_fov;
        theta_x_center_rad = theta_x_center_rad * M_PI / 180.0;

        // 在相机坐标系中构造方向 pose
        geometry_msgs::msg::PoseStamped pose_in_camera;
        pose_in_camera.header.stamp    = msg->header.stamp;
        pose_in_camera.header.frame_id = tf_name;
        pose_in_camera.pose.position.x = std::cos(theta_x_center_rad);
        pose_in_camera.pose.position.y = std::sin(theta_x_center_rad);
        pose_in_camera.pose.position.z = 0.0;
        pose_in_camera.pose.orientation.w = 1.0;

        // 变换到 map 坐标系
        geometry_msgs::msg::PoseStamped pose_in_map;
        tf2::doTransform(pose_in_camera, pose_in_map, transform_camera);

        // 当前射线：起点 = base_link 在 map 中的位置，方向 = 指向变换后的点
        Vec2d origin_map = {X_R, Y_R};
        Vec2d target_point = {pose_in_map.pose.position.x,
                              pose_in_map.pose.position.y};

        Vec2d direction_map = {target_point[0] - origin_map[0],
                               target_point[1] - origin_map[1]};
        double norm = std::hypot(direction_map[0], direction_map[1]);
        if (norm > 1e-9) {
            direction_map[0] /= norm;
            direction_map[1] /= norm;
        }

        // ---- 射线可视化 & 同一目标判断 ----
        if (last_ray_origin_.has_value() && last_ray_direction_.has_value()) {
            publish_ray_marker(msg->header.stamp,
                               last_ray_origin_.value(),
                               last_ray_direction_.value(),
                               0, {0.0f, 1.0f, 0.0f});  // 上一帧绿色
            publish_ray_marker(msg->header.stamp,
                               origin_map, direction_map,
                               1, {1.0f, 0.0f, 0.0f});  // 当前帧红色

            bool is_same_object = false;

            for (auto & [prev_origin, prev_direction] : ray_history_) {
                if (rays_have_common_part(prev_direction, prev_origin,
                                          direction_map, origin_map)) {
                    is_same_object = true;
                    break;
                }
            }

            // 用于演示持续发布（和 Python 版一致，强制设为 false）
            is_same_object = false;

            if (is_same_object) {
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                                     "It's the same detection object !");
            } else {
                img_uuid_ = generate_uuid();
            }
        }

        // 添加当前射线到历史
        ray_history_.push_back({origin_map, direction_map});
        if (static_cast<int>(ray_history_.size()) > max_ray_history_) {
            ray_history_.pop_front();
        }

        // ---- 构造图像消息 ----
        cv::Mat rgb_image;
        cv::cvtColor(color_data, rgb_image, cv::COLOR_BGR2RGB);
        sensor_msgs::msg::Image img_msg =
            *cv_bridge::CvImage(msg->header, "rgb8", rgb_image).toImageMsg();

        // ---- UUID 管理 & 发布 ----
        if (img_uuid_.empty()) {
            img_uuid_ = generate_uuid();
        }

        int count = 0;
        auto it = uuid_publish_count_.find(img_uuid_);
        if (it != uuid_publish_count_.end()) {
            count = it->second;
        }

        if (count < max_publish_per_uuid_) {
            capella_ros_msg::msg::SingleRecognition single_msg;
            single_msg.class_id    = det.class_id;
            single_msg.uuid        = img_uuid_;
            single_msg.left_cord_x  = x1;
            single_msg.left_cord_y  = y1;
            single_msg.right_cord_x = x2;
            single_msg.right_cord_y = y2;
            single_msg.image       = img_msg;

            inspections_msg.recognitions.push_back(single_msg);
            uuid_publish_count_[img_uuid_] = count + 1;
            last_count = count + 1;

            // 限制最大保留 UUID 数量
            while (static_cast<int>(uuid_publish_count_.size()) > MAX_UUID_HISTORY) {
                uuid_publish_count_.erase(uuid_publish_count_.begin());
            }
        }

        // 更新射线状态
        last_ray_origin_    = origin_map;
        last_ray_direction_ = direction_map;
    }

    // ---- 发布结果 ----
    if (!inspections_msg.recognitions.empty()) {
        result_pub_->publish(inspections_msg);
        RCLCPP_INFO(this->get_logger(), "Have published %d !", last_count);
    }
}


// ---------------------------------------------------------------------------
//  rays_have_common_part
// ---------------------------------------------------------------------------
bool CapellaInspectionNode::rays_have_common_part(
    const Vec2d & ray1, const Vec2d & origin1,
    const Vec2d & ray2, const Vec2d & origin2,
    double eps) const
{
    // 归一化方向向量
    double norm1 = std::hypot(ray1[0], ray1[1]);
    double norm2 = std::hypot(ray2[0], ray2[1]);
    Vec2d r1 = {ray1[0] / norm1, ray1[1] / norm1};
    Vec2d r2 = {ray2[0] / norm2, ray2[1] / norm2};

    double cross_val = r1[0] * r2[1] - r1[1] * r2[0];

    // ---- 情况一：共线 ----
    if (std::abs(cross_val) < eps) {
        double dist = std::hypot(origin1[0] - origin2[0], origin1[1] - origin2[1]);
        if (dist < eps) {
            // 起点相同但方向不同 → 无重合
            double dot = r1[0] * r2[0] + r1[1] * r2[1];
            if (dot < 1.0 - eps) {
                return false;
            }
        }

        // origin2 - origin1 在 r1 方向上的投影
        double vec_x  = origin2[0] - origin1[0];
        double vec_y  = origin2[1] - origin1[1];
        double proj1  = vec_x * r1[0] + vec_y * r1[1];

        // origin1 - origin2 在 r2 方向上的投影
        double vec2_x = origin1[0] - origin2[0];
        double vec2_y = origin1[1] - origin2[1];
        double proj2  = vec2_x * r2[0] + vec2_y * r2[1];

        return (proj1 >= -eps || proj2 >= -eps);
    }

    // ---- 情况二：不共线，计算交点 ----
    double dx1 = r1[0], dy1 = r1[1];
    double dx2 = r2[0], dy2 = r2[1];
    double x1 = origin1[0], y1_val = origin1[1];
    double x2 = origin2[0], y2_val = origin2[1];

    // 解线性方程组:
    //   t1 * dx1 - t2 * dx2 = x2 - x1
    //   t1 * dy1 - t2 * dy2 = y2 - y1
    double det = dx1 * (-dy2) - (-dx2) * dy1;  // dx1*(-dy2) + dx2*dy1
    if (std::abs(det) < 1e-12) {
        return false;
    }

    double bx = x2 - x1;
    double by = y2_val - y1_val;

    double t1 = (bx * (-dy2) - (-dx2) * by) / det;
    double t2 = (dx1 * by - bx * dy1) / det;

    return (t1 >= -eps && t2 >= -eps);
}


// ---------------------------------------------------------------------------
//  publish_ray_marker
// ---------------------------------------------------------------------------
void CapellaInspectionNode::publish_ray_marker(
    const builtin_interfaces::msg::Time & stamp,
    const Vec2d & origin,
    const Vec2d & direction,
    int marker_id,
    const std::array<float, 3> & color)
{
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp    = stamp;
    marker.ns              = "inspection_rays";
    marker.id              = marker_id;
    marker.type            = visualization_msgs::msg::Marker::ARROW;
    marker.action          = visualization_msgs::msg::Marker::ADD;

    // 起点
    Vec2d end_point = {origin[0] + direction[0] * 3.0,
                       origin[1] + direction[1] * 3.0};

    marker.points.push_back(to_point_msg(origin));
    marker.points.push_back(to_point_msg(end_point));

    marker.scale.x = 0.05;  // 箭杆直径
    marker.scale.y = 0.1;   // 箭头宽度
    marker.scale.z = 0.1;   // 箭头高度

    marker.color.r = color[0];
    marker.color.g = color[1];
    marker.color.b = color[2];
    marker.color.a = 1.0f;

    marker.lifetime.sec     = 2;
    marker.lifetime.nanosec = 0;

    marker_pub_->publish(marker);
}


// ---------------------------------------------------------------------------
//  to_point_msg
// ---------------------------------------------------------------------------
geometry_msgs::msg::Point CapellaInspectionNode::to_point_msg(const Vec2d & v)
{
    geometry_msgs::msg::Point p;
    p.x = v[0];
    p.y = v[1];
    p.z = 0.0;
    return p;
}


}  // namespace capella_inspection


// ===========================================================================
//  main
// ===========================================================================
int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<capella_inspection::CapellaInspectionNode>();

    rclcpp::executors::MultiThreadedExecutor executor(
        rclcpp::ExecutorOptions(), /*number_of_threads=*/4);
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}