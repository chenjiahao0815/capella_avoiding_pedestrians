#include "capella_avoiding_pedestrians/avoiding_pedestrians.hpp"

#include <sensor_msgs/image_encodings.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/utils.h>

#include <chrono>
#include <cmath>
#include <opencv2/imgproc.hpp>

using namespace std::chrono_literals;

BehaviorDetectionNode::BehaviorDetectionNode() : Node("behavior_detection_node")
{
    this->declare_parameter<double>("path_search_distance", 5.0);
    this->declare_parameter<double>("pedestrian_distance_threshold", 1.0);
    this->declare_parameter<double>("robot_radius", 0.5);
    this->declare_parameter<std::string>("global_frame", "map");
    this->declare_parameter<std::string>("scan_topic_name_front", "/front_scan");

    std::string camera_topic = "/rgb_camera_front/image_raw"
    const std::string scan_topic = this->get_parameter("scan_topic_name_front").as_string();

    auto qos_transient_local = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable();
    pub_avoiding_ = this->create_publisher<std_msgs::msg::Bool>("is_avoiding_pedestrians", qos_transient_local);

    sub_camera_ = this->create_subscription<sensor_msgs::msg::Image>(
        camera_topic,
        rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::Image::SharedPtr msg)
        {
            this->imageCallback(msg);
        });

    sub_laser_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        scan_topic,
        10,
        [this](const sensor_msgs::msg::LaserScan::SharedPtr msg)
        {
            this->laserCallback(msg);
        });

    sub_global_plan_ = this->create_subscription<nav_msgs::msg::Path>(
        "teb_global_plan",
        10,
        [this](const nav_msgs::msg::Path::SharedPtr msg)
        {
            this->globalPlanCallback(msg);
        });

    sub_local_poses_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
        "teb_poses",
        10,
        [this](const geometry_msgs::msg::PoseArray::SharedPtr msg)
        {
            this->localPosesCallback(msg);
        });

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    timer_ = this->create_wall_timer(
        100ms,
        [this]()
        {
            this->timerCallback();
        });

    RCLCPP_INFO(this->get_logger(), "Behavior Detection Node initialized.");
}

void BehaviorDetectionNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    if (!msg) return;
    if (msg->data.empty()) return;

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

    const auto tracks = yolo_->track(frame);

    DetectionResult result;
    result.detected = false;
    result.stamp = msg->header.stamp;
    result.frame_id = msg->header.frame_id;

    for (size_t i = 0; i < tracks.size(); ++i) {
        const auto &det = tracks[i];
        if (det.class_id != 0) continue;
        if (det.confidence < static_cast<float>(person_conf_threshold_)) continue;

        const float cx = det.bbox.x + det.bbox.width * 0.5f;
        const float norm = (cx / static_cast<float>(msg->width)) - 0.5f;
        const float yaw = -norm * static_cast<float>(h_fov_rad_);

        result.detected = true;
        result.person_angles.push_back(yaw);
    }

    {
        std::lock_guard<std::mutex> lock(detection_mutex_);
        detection_result_ = std::move(result);
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
    last_global_plan_ = msg;
}

void BehaviorDetectionNode::localPosesCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
{
    last_local_poses_ = msg;
}

void BehaviorDetectionNode::timerCallback()
{
    bool trigger_now = false;

    if (last_global_plan_ && last_local_poses_) {
        DetectionResult det;
        {
            std::lock_guard<std::mutex> lock(detection_mutex_);
            det = detection_result_;
        }

        if (det.detected && !det.person_angles.empty()) {
            sensor_msgs::msg::LaserScan::SharedPtr best_scan;
            double best_dt = 1e9;
            const double img_ts = static_cast<double>(det.stamp.sec)
                                + static_cast<double>(det.stamp.nanosec) * 1e-9;

            {
                std::lock_guard<std::mutex> lock(laser_queue_mutex_);
                for (const auto &item : laser_queue_) {
                    const double dt = std::abs(item.first - img_ts);
                    if (dt < best_dt) {
                        best_dt = dt;
                        best_scan = item.second;
                    }
                }
            }

            if (best_scan) {
                try {
                    const auto tf_cam_laser = tf_buffer_->lookupTransform(
                        det.frame_id,
                        best_scan->header.frame_id,
                        tf2::TimePointZero,
                        tf2::durationFromSec(0.1));
                    const double yaw_laser_to_cam = tf2::getYaw(tf_cam_laser.transform.rotation);
                    camera_laser_offset_ = -yaw_laser_to_cam;
                } catch (const tf2::TransformException &) {
                }

                std::vector<geometry_msgs::msg::PointStamped> pedestrians_laser;
                pedestrians_laser.reserve(det.person_angles.size());

                for (size_t i = 0; i < det.person_angles.size(); ++i) {
                    const float cam_angle = det.person_angles[i];
                    const float laser_angle = cam_angle + camera_laser_offset_;
                    const int index = static_cast<int>((laser_angle - best_scan->angle_min) / best_scan->angle_increment);

                    if (index < 0 || index >= static_cast<int>(best_scan->ranges.size())) continue;

                    const float r = best_scan->ranges[index];
                    if (r <= best_scan->range_min || r >= best_scan->range_max) continue;

                    geometry_msgs::msg::PointStamped p;
                    p.header = best_scan->header;
                    p.point.x = r * std::cos(laser_angle);
                    p.point.y = r * std::sin(laser_angle);
                    p.point.z = 0.0;
                    pedestrians_laser.push_back(p);
                }

                if (!pedestrians_laser.empty()) {
                    std::vector<geometry_msgs::msg::PointStamped> pedestrians_map;
                    pedestrians_map.reserve(pedestrians_laser.size());

                    for (const auto &p : pedestrians_laser) {
                        try {
                            geometry_msgs::msg::PointStamped p_map;
                            tf_buffer_->transform(p, p_map, "map", tf2::durationFromSec(0.2));
                            pedestrians_map.push_back(p_map);
                        } catch (const tf2::TransformException &) {
                        }
                    }

                    if (!pedestrians_map.empty()) {
                        const double search_distance = this->get_parameter("path_search_distance").as_double();
                        const double threshold = this->get_parameter("pedestrian_distance_threshold").as_double();

                        const bool on_global = checkPedestrianOnGlobalPath(
                            pedestrians_map,
                            *last_global_plan_,
                            search_distance,
                            threshold);

                        const bool on_local = checkPedestrianOnLocalPath(
                            pedestrians_map,
                            last_local_poses_,
                            threshold);

                        trigger_now = (on_global || on_local);
                    }
                }
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
