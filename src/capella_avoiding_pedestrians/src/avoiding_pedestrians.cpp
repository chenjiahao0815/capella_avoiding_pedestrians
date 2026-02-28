#include "capella_avoiding_pedestrians/avoiding_pedestrians.hpp"

#include <sensor_msgs/image_encodings.hpp>
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
    // 加载YOLO模型
    OpenCvYoloTracker(const std::string &model_path, const rclcpp::Logger &logger)
        : logger_(logger)
    {
        try {
            net_ = cv::dnn::readNet(model_path);
            if (net_.empty()) {
                RCLCPP_ERROR(logger_, "Failed to load YOLO model: %s", model_path.c_str());
                enabled_ = false;
                return;
            }
            enabled_ = true;
            RCLCPP_INFO(logger_, "YOLO model loaded: %s", model_path.c_str());
        } catch (const std::exception &e) {
            RCLCPP_ERROR(logger_, "Exception while loading YOLO model: %s", e.what());
            enabled_ = false;
        }
    }
    // 运行YOLO跟踪
    std::vector<TrackItem> track(const cv::Mat &frame) override
    {   
        //储存结果的向量
        std::vector<TrackItem> tracks;
        if (!enabled_ || frame.empty()) return tracks;
        // 这里是要手动缩放一下输入图像到640x640，保持宽高比，并且进行归一化
        constexpr int input_size = 640;
        cv::Mat blob; // 一个OpenCV DNN 的输入 blob[batch, channels, height, width]
        cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(input_size, input_size), cv::Scalar(), true, false);
        net_.setInput(blob); // 把frame转成blob输入到网络中去

        std::vector<cv::Mat> outputs;  // 网络输出层用mat
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());
        if (outputs.empty()) return tracks;

        const cv::Mat &out = outputs[0];
        if (out.dims != 3) return tracks;

        cv::Mat det;
        if (out.size[1] < out.size[2]) { //矩阵转置
            cv::Mat raw(out.size[1], out.size[2], CV_32F, const_cast<float *>(out.ptr<float>()));
            cv::transpose(raw, det);
        } else {
            det = cv::Mat(out.size[1], out.size[2], CV_32F, const_cast<float *>(out.ptr<float>())).clone();
        }
        // 计算缩放因子
        const float x_scale = static_cast<float>(frame.cols) / static_cast<float>(input_size);
        const float y_scale = static_cast<float>(frame.rows) / static_cast<float>(input_size);

        for (int i = 0; i < det.rows; ++i) { // 每一个检测框
            const float *row = det.ptr<float>(i);
            if (!row) continue;
            // 对检测框取做高得分的类别，第0-3列是bbox坐标，后面是类别得分
            const int class_count = det.cols - 4;
            if (class_count <= 0) continue;

            int best_class = -1;
            float best_score = 0.0F;
            for (int c = 0; c < class_count; ++c) {
                const float score = row[4 + c];
                if (score > best_score) {
                    best_score = score;
                    best_class = c;
                }
            }

            if (best_class < 0 || best_score < 0.05F) continue;
            // 解析边界框坐标
            const float cx = row[0] * x_scale;
            const float cy = row[1] * y_scale;
            const float w = row[2] * x_scale;
            const float h = row[3] * y_scale;

            TrackItem item;
            item.class_id = best_class;
            item.confidence = best_score;
            item.bbox.x = cx - 0.5F * w;
            item.bbox.y = cy - 0.5F * h;
            item.bbox.width = w;
            item.bbox.height = h;
            tracks.push_back(item);
        }

        return tracks;
    }

private:
    rclcpp::Logger logger_;
    cv::dnn::Net net_;
    bool enabled_{false};
};
}  // namespace

BehaviorDetectionNode::BehaviorDetectionNode() : Node("behavior_detection_node")
{
    this->declare_parameter<double>("path_search_distance", 5.0);   // 搜索行人时考虑的路径距离
    this->declare_parameter<double>("pedestrian_distance_threshold", 1.0); // 行人距离阈值
    this->declare_parameter<double>("robot_radius", 0.5); // 机器人半径
    this->declare_parameter<std::string>("global_frame", "map"); // 全局坐标系
    // 仿真模式：跳过相机/YOLO，直接注入一个假行人（正前方2m，激光系）用于逻辑验证
    this->declare_parameter<bool>("use_sim_detection", false);
    this->declare_parameter<std::string>("sim_laser_frame", "front_laser"); // 仿真用激光frame
    //默认的模型的位置
    const std::string default_model_path =
        (std::filesystem::path(__FILE__).parent_path() / "car8.onnx").string();

    this->declare_parameter<std::string>("yolo_model_path", default_model_path);
    //默认的雷达和相机话题
    this->declare_parameter<std::string>("scan_topic_name_front", "/front_scan");
    std::string camera_topic = "/rgb_camera_front/image_raw";

    const std::string scan_topic = this->get_parameter("scan_topic_name_front").as_string();
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
    //qos类型
    auto qos_transient_local = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable();
    //警报话题
    pub_avoiding_ = this->create_publisher<std_msgs::msg::Bool>("is_avoiding_pedestrians", qos_transient_local);
    pub_annotated_image_ = this->create_publisher<sensor_msgs::msg::Image>("/rgb_camera_front/annotated_image", 10);
    pub_pedestrians_map_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/pedestrians/map", 10);
    pub_pedestrians_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/pedestrians/map_markers", 10);
    
    sub_camera_ = this->create_subscription<sensor_msgs::msg::Image>(
        camera_topic,
        rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::Image::SharedPtr msg)
        {
            this->imageCallback(msg);
        }); //相机回调

    sub_laser_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        scan_topic,
        10,
        [this](const sensor_msgs::msg::LaserScan::SharedPtr msg)
        {
            this->laserCallback(msg);
        }); //激光回调

    sub_global_plan_ = this->create_subscription<nav_msgs::msg::Path>(
        "teb_global_plan",
        10,
        [this](const nav_msgs::msg::Path::SharedPtr msg)
        {
            this->globalPlanCallback(msg);
        });    //全局地图回调

    sub_local_poses_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
        "teb_poses",
        //rclcpp::SensorDataQoS(),   // 如果有订阅但是不触发回调就改成这个
        10,
        [this](const geometry_msgs::msg::PoseArray::SharedPtr msg)
        {
            this->localPosesCallback(msg);
        });   //局部回调

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

void BehaviorDetectionNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    if (!msg) return;
    if (msg->data.empty()) return;
    // 把ros转成cv
    cv::Mat frame;
    if (msg->encoding == sensor_msgs::image_encodings::BGR8) {
        frame = cv::Mat(
            static_cast<int>(msg->height),
            static_cast<int>(msg->width),
            CV_8UC3,
            const_cast<unsigned char *>(msg->data.data()),
            static_cast<size_t>(msg->step)).clone();
    } else if (msg->encoding == sensor_msgs::image_encodings::RGB8) {
        cv::Mat rgb(
            static_cast<int>(msg->height),
            static_cast<int>(msg->width),
            CV_8UC3,
            const_cast<unsigned char *>(msg->data.data()),
            static_cast<size_t>(msg->step));
        cv::cvtColor(rgb, frame, cv::COLOR_RGB2BGR);
    } else {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            2000,
            "Unsupported image encoding: %s",
            msg->encoding.c_str());
        return;
    }
    // YOLO检测人体框
    std::vector<TrackItem> tracks;
    if (!runYoloTrack(frame, tracks)) return;

    // 获取最新的激光雷达数据
    sensor_msgs::msg::LaserScan::SharedPtr latest_scan;
    {
        std::lock_guard<std::mutex> lock(laser_queue_mutex_);
        if (!laser_queue_.empty()) {
            latest_scan = laser_queue_.back().second;
        }
    }

    // 更新相机到激光雷达的角度偏差（通过TF，硬件固定值，失败则沿用上次值）
    if (latest_scan && !msg->header.frame_id.empty()) {
        try {
            const auto tf_cam_laser = tf_buffer_->lookupTransform(
                msg->header.frame_id,
                latest_scan->header.frame_id,
                tf2::TimePointZero,
                tf2::durationFromSec(0.05));
            camera_laser_offset_ = -tf2::getYaw(tf_cam_laser.transform.rotation);
        } catch (const tf2::TransformException &) {}
    }

    cv::Mat annotated = frame.clone();

    DetectionResult result;
    result.detected = false;
    result.stamp = msg->header.stamp;
    if (latest_scan) {
        result.laser_frame_id = latest_scan->header.frame_id;
    }

    for (const auto &det : tracks) {
        if (det.class_id != 0) continue;
        if (det.confidence < static_cast<float>(person_conf_threshold_)) continue;

        // 由框中心像素坐标计算水平方向角
        const float cx = det.bbox.x + det.bbox.width * 0.5F;
        const float norm = (cx / static_cast<float>(msg->width)) - 0.5f;
        const float yaw = -norm * static_cast<float>(h_fov_rad_);

        // 将相机角度映射到激光雷达坐标系，查询该方向的测距值
        float distance = std::numeric_limits<float>::quiet_NaN();
        if (latest_scan) {
            const float laser_angle = yaw + static_cast<float>(camera_laser_offset_);
            const int idx = static_cast<int>(
                (laser_angle - latest_scan->angle_min) / latest_scan->angle_increment);
            if (idx >= 0 && idx < static_cast<int>(latest_scan->ranges.size())) {
                const float r = latest_scan->ranges[idx];
                if (r > latest_scan->range_min && r < latest_scan->range_max) {
                    distance = r;
                }
            }
        }

        // 只有激光距离有效才认为检测到行人
        if (!std::isnan(distance)) {
            const float laser_angle = yaw + static_cast<float>(camera_laser_offset_);
            geometry_msgs::msg::Point p;
            p.x = distance * std::cos(laser_angle);
            p.y = distance * std::sin(laser_angle);
            p.z = 0.0;
            result.pedestrians_laser.push_back(p);
            result.detected = true;

            // 打印 base_link 和 map 坐标日志
            if (latest_scan) {
                geometry_msgs::msg::PointStamped p_laser;
                p_laser.header.frame_id = latest_scan->header.frame_id;
                p_laser.header.stamp = msg->header.stamp;
                p_laser.point = p;

                std::ostringstream log_stream;
                log_stream << std::fixed << std::setprecision(2);
                log_stream << "[laser] (" << p.x << ", " << p.y << ")";

                try {
                    geometry_msgs::msg::PointStamped p_base;
                    tf_buffer_->transform(p_laser, p_base, "base_link", tf2::durationFromSec(0.1));
                    log_stream << "  [base_link] (" << p_base.point.x << ", " << p_base.point.y << ")";
                } catch (const tf2::TransformException &e) {
                    log_stream << "  [base_link] TF failed: " << e.what();
                }

                try {
                    geometry_msgs::msg::PointStamped p_map;
                    tf_buffer_->transform(p_laser, p_map, "map", tf2::durationFromSec(0.1));
                    log_stream << "  [map] (" << p_map.point.x << ", " << p_map.point.y << ")";
                } catch (const tf2::TransformException &e) {
                    log_stream << "  [map] TF failed: " << e.what();
                }

                RCLCPP_INFO_THROTTLE(
                    this->get_logger(),
                    *this->get_clock(),
                    500,
                    "pedestrian %s  dist=%.1fm",
                    log_stream.str().c_str(),
                    distance);
            }
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

    if (result.detected && pub_annotated_image_) {
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
    RCLCPP_INFO(this->get_logger(), "globalPlanCallback: received path with %zu poses at %d.%09d",
            msg->poses.size(), msg->header.stamp.sec, msg->header.stamp.nanosec);    
    if (!msg) return;
    last_global_plan_ = msg;

    static bool once = true;
    if (once) {
        RCLCPP_INFO(this->get_logger(), "get global plan");
        once = false;
    }
}

void BehaviorDetectionNode::localPosesCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
{
    if (!msg) return;
    last_local_poses_ = msg;

    static bool once = true;
    if (once) {
        RCLCPP_INFO(this->get_logger(), "get local poses plan");
        once = false;
    }
}

void BehaviorDetectionNode::timerCallback()
{
    bool trigger_now = false;  // 默认没检测到人

    // 仿真模式：直接注入假行人，绕过相机/YOLO
    if (this->get_parameter("use_sim_detection").as_bool()) {
        const std::string sim_frame = this->get_parameter("sim_laser_frame").as_string();
        DetectionResult sim_det;
        sim_det.detected = true;
        sim_det.stamp = rclcpp::Time(0, 0, this->get_clock()->get_clock_type());  // 使用最新可用TF
        sim_det.laser_frame_id = sim_frame;
        // 假行人：激光坐标系正前方 2.0m
        geometry_msgs::msg::Point sim_p;
        sim_p.x = 2.0;
        sim_p.y = 0.0;
        sim_p.z = 0.0;
        sim_det.pedestrians_laser.push_back(sim_p);
        {
            std::lock_guard<std::mutex> lock(detection_mutex_);
            detection_result_ = sim_det;
        }
        RCLCPP_INFO_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "[SIM] injected pedestrian at laser (2.0, 0.0)");
    }

    if (last_global_plan_ || last_local_poses_) {
        DetectionResult det;
        {
            std::lock_guard<std::mutex> lock(detection_mutex_);
            det = detection_result_;
        }
        // imageCallback中已完成激光测距，直接使用激光坐标系下的行人坐标
        if (det.detected && !det.pedestrians_laser.empty() && !det.laser_frame_id.empty()) {

            // 将行人从激光坐标系转换到map坐标系
            std::vector<geometry_msgs::msg::PointStamped> pedestrians_map;
            pedestrians_map.reserve(det.pedestrians_laser.size());
            for (const auto &pt_laser : det.pedestrians_laser) {
                geometry_msgs::msg::PointStamped p_in;
                p_in.header.frame_id = det.laser_frame_id;
                p_in.header.stamp = det.stamp;
                p_in.point = pt_laser;
                try {
                    geometry_msgs::msg::PointStamped p_map;
                    tf_buffer_->transform(p_in, p_map, "map", tf2::durationFromSec(0.2));
                    pedestrians_map.push_back(p_map);
                } catch (const tf2::TransformException &e) {
                    RCLCPP_WARN_THROTTLE(
                        this->get_logger(), *this->get_clock(), 2000,
                        "TF transform %s->map failed: %s", det.laser_frame_id.c_str(), e.what());
                }
            }

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
                const double search_distance = this->get_parameter("path_search_distance").as_double();
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
        if (end_dist < path_lookahead_distance_) {
            return nav_points;
        }
    }

    out_points.push_back(nav_points.front());
    if (nav_points.size() == 1) return out_points;

    const double sample_dist = (min_distance > 0.0) ? min_distance : 0.0;
    const double sample_dist_sq = sample_dist * sample_dist;
    const double max_lookahead = (lookahead_distance > 0.0) ? lookahead_distance : path_lookahead_distance_;

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
    for (const auto &ped : pedestrians) {
        for (const auto &pt : optimized_points) {
            const double dx = ped.point.x - pt.x;
            const double dy = ped.point.y - pt.y;
            const double dist_sq = dx * dx + dy * dy;
            if (dist_sq <= threshold_sq) {
                return true;
            }
        }
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

    const auto optimized_points = downsamplePath(local_points, threshold, path_lookahead_distance_);
    if (optimized_points.empty()) return false;

    const double threshold_sq = threshold * threshold;
    for (const auto &ped : pedestrians) {
        for (const auto &pt : optimized_points) {
            const double dx = ped.point.x - pt.x;
            const double dy = ped.point.y - pt.y;
            const double dist_sq = dx * dx + dy * dy;
            if (dist_sq <= threshold_sq) {
                return true;
            }
        }
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