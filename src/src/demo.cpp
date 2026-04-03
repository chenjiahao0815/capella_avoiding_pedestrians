#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cmath>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <algorithm>
#include <limits>
#include <numeric>

using namespace std::chrono_literals;

// -------------------- YOLO 检测器 --------------------
struct Detection {
    int class_id;
    float confidence;
    cv::Rect bbox;
    float cx, cy;
};

class YoloDetector {
public:
    YoloDetector(const std::string &model_path, const rclcpp::Logger &logger, bool use_gpu)
        : logger_(logger), enabled_(false), use_gpu_(use_gpu) {
        try {
            net_ = cv::dnn::readNet(model_path);
            if (net_.empty()) {
                RCLCPP_ERROR(logger_, "Failed to load YOLO model: %s", model_path.c_str());
                return;
            }

            if (use_gpu_) {
                try {
                    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                    net_.enableWinograd(false);
                    RCLCPP_INFO(logger_, "YOLO model loaded with CUDA backend");
                } catch (const std::exception &e) {
                    RCLCPP_WARN(logger_, "CUDA backend failed, falling back to CPU: %s", e.what());
                    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                }
            } else {
                net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                RCLCPP_INFO(logger_, "YOLO model loaded with CPU backend");
            }

            enabled_ = true;
            RCLCPP_INFO(logger_, "YOLO model loaded: %s", model_path.c_str());
        } catch (const std::exception &e) {
            RCLCPP_ERROR(logger_, "Exception while loading YOLO model: %s", e.what());
        }
    }

    std::vector<Detection> detect(const cv::Mat &frame) {
        std::vector<Detection> detections;
        if (!enabled_ || frame.empty()) return detections;

        constexpr int input_size = 640;
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0/255.0,
                               cv::Size(input_size, input_size),
                               cv::Scalar(), true, false);
        net_.setInput(blob);

        std::vector<cv::Mat> outputs;
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());
        if (outputs.empty()) return detections;

        const cv::Mat &out = outputs[0];
        if (out.dims != 3) return detections;

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
        if (class_count <= 0) return detections;

        const float x_scale = static_cast<float>(frame.cols) / static_cast<float>(input_size);
        const float y_scale = static_cast<float>(frame.rows) / static_cast<float>(input_size);

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;
        std::vector<std::pair<float, float>> centers;

        for (int i = 0; i < det.rows; ++i) {
            const float *row = det.ptr<float>(i);
            if (!row) continue;

            int best_class = -1;
            float best_score = 0.0F;
            for (int c = 0; c < class_count; ++c) {
                float score = row[4 + c];
                if (score > best_score) {
                    best_score = score;
                    best_class = c;
                }
            }

            if (best_class != 0 || best_score < 0.15F) continue;

            const float cx = row[0] * x_scale;
            const float cy = row[1] * y_scale;
            const float w  = row[2] * x_scale;
            const float h  = row[3] * y_scale;

            int x1 = static_cast<int>(cx - 0.5F * w);
            int y1 = static_cast<int>(cy - 0.5F * h);
            int bw = std::max(static_cast<int>(w), 30);
            int bh = std::max(static_cast<int>(h), 30);
            
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            if (x1 + bw > frame.cols) bw = frame.cols - x1;
            if (y1 + bh > frame.rows) bh = frame.rows - y1;

            boxes.emplace_back(x1, y1, bw, bh);
            confidences.push_back(best_score);
            class_ids.push_back(best_class);
            centers.emplace_back(cx, cy);
        }

        std::vector<int> indices;
        if (!boxes.empty()) {
            cv::dnn::NMSBoxes(boxes, confidences, 0.15F, 0.65F, indices);
        }

        detections.reserve(indices.size());
        for (int idx : indices) {
            Detection d;
            d.class_id = class_ids[idx];
            d.confidence = confidences[idx];
            d.bbox = boxes[idx];
            d.cx = centers[idx].first;
            d.cy = centers[idx].second;
            detections.push_back(d);
        }

        return detections;
    }

private:
    rclcpp::Logger logger_;
    cv::dnn::Net net_;
    bool enabled_;
    bool use_gpu_;
};

// -------------------- 节点类 --------------------
class CrowdStatisticsNode : public rclcpp::Node {
public:
    CrowdStatisticsNode() : Node("crowd_statistics_node") {
        // 参数声明
        this->declare_parameter<std::string>("model_path", "/capella/lib/python3.10/site-packages/crowd_statistics_ws/yolov8l.onnx");
        this->declare_parameter<double>("distance_threshold", 20.0);
        this->declare_parameter<double>("confidence_threshold", 0.4);
        this->declare_parameter<std::string>("base_frame", "base_link");
        this->declare_parameter<std::string>("map_frame", "map");
        this->declare_parameter<bool>("publish_map_frame", true);
        this->declare_parameter<bool>("use_gpu", true);
        this->declare_parameter<int>("frame_skip", 1);
        this->declare_parameter<double>("h_fov_rad", 1.0472);
        this->declare_parameter<bool>("debug_mode", false);
        this->declare_parameter<double>("default_person_width_rad", 0.20);
        this->declare_parameter<double>("density_radius", 2.0);
        this->declare_parameter<bool>("publish_visualization", true);
        this->declare_parameter<double>("angle_tolerance", 0.10);

        // 相机配置
        this->declare_parameter<std::string>("camera_topic_front", "/rgb_camera_front/compressed");
        this->declare_parameter<std::string>("camera_topic_left", "/rgb_camera_left/compressed");
        this->declare_parameter<std::string>("camera_topic_right", "/rgb_camera_right/compressed");
        this->declare_parameter<std::string>("camera_topic_back", "/rgb_camera_back/compressed");
        this->declare_parameter<std::string>("camera_frame_front", "front_camera_color_frame");
        this->declare_parameter<std::string>("camera_frame_left", "left_camera_color_frame");
        this->declare_parameter<std::string>("camera_frame_right", "right_camera_color_frame");
        this->declare_parameter<std::string>("camera_frame_back", "back_camera_color_frame");

        // 激光雷达话题
        this->declare_parameter<std::string>("front_scan_topic", "/front_scan");
        this->declare_parameter<std::string>("back_scan_topic", "/back_scan");

        // 获取参数
        std::string model_path = this->get_parameter("model_path").as_string();
        distance_threshold_ = this->get_parameter("distance_threshold").as_double();
        confidence_threshold_ = this->get_parameter("confidence_threshold").as_double();
        base_frame_ = this->get_parameter("base_frame").as_string();
        map_frame_ = this->get_parameter("map_frame").as_string();
        publish_map_frame_ = this->get_parameter("publish_map_frame").as_bool();
        bool use_gpu = this->get_parameter("use_gpu").as_bool();
        frame_skip_ = this->get_parameter("frame_skip").as_int();
        h_fov_rad_ = this->get_parameter("h_fov_rad").as_double();
        debug_mode_ = this->get_parameter("debug_mode").as_bool();
        default_person_width_rad_ = this->get_parameter("default_person_width_rad").as_double();
        density_radius_ = this->get_parameter("density_radius").as_double();
        publish_viz_ = this->get_parameter("publish_visualization").as_bool();
        angle_tolerance_ = this->get_parameter("angle_tolerance").as_double();

        // 相机配置
        camera_names_ = {"front", "left", "right", "back"};
        std::vector<std::string> topics = {
            this->get_parameter("camera_topic_front").as_string(),
            this->get_parameter("camera_topic_left").as_string(),
            this->get_parameter("camera_topic_right").as_string(),
            this->get_parameter("camera_topic_back").as_string()
        };
        std::vector<std::string> frames = {
            this->get_parameter("camera_frame_front").as_string(),
            this->get_parameter("camera_frame_left").as_string(),
            this->get_parameter("camera_frame_right").as_string(),
            this->get_parameter("camera_frame_back").as_string()
        };

        // 初始化 YOLO
        yolo_ = std::make_unique<YoloDetector>(model_path, this->get_logger(), use_gpu);

        // TF
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // 订阅相机
        cameras_.resize(4);
        for (size_t i = 0; i < topics.size(); ++i) {
            cameras_[i].topic = topics[i];
            cameras_[i].frame_id = frames[i];
            cameras_[i].mutex = std::make_shared<std::mutex>();
            cameras_[i].camera_name = camera_names_[i];

            cameras_[i].sub = this->create_subscription<sensor_msgs::msg::CompressedImage>(
                cameras_[i].topic,
                rclcpp::SensorDataQoS(),
                [this, i](const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
                    this->imageCallback(msg, i);
                });
            RCLCPP_INFO(this->get_logger(), "Subscribed to %s [%s] (frame=%s)",
                        camera_names_[i].c_str(), topics[i].c_str(), frames[i].c_str());
        }

        // 订阅激光雷达
        front_scan_topic_ = this->get_parameter("front_scan_topic").as_string();
        back_scan_topic_ = this->get_parameter("back_scan_topic").as_string();
        sub_front_scan_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            front_scan_topic_, rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::LaserScan::SharedPtr msg) { this->laserCallback(msg, 0); });
        sub_back_scan_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            back_scan_topic_, rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::LaserScan::SharedPtr msg) { this->laserCallback(msg, 1); });

        // 发布器
        pub_pedestrians_ = this->create_publisher<geometry_msgs::msg::PointStamped>("detected_pedestrians", 10);
        pub_pedestrians_map_ = this->create_publisher<geometry_msgs::msg::PointStamped>("detected_pedestrians_map", 10);
        pub_person_poses_ = this->create_publisher<geometry_msgs::msg::PoseArray>("person_poses", 10);
        pub_stats_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("crowd_statistics", 10);
        pub_viz_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("crowd_visualization", 10);

        // 定时器 (500ms周期)
        timer_ = this->create_wall_timer(500ms, [this]() { this->timerCallback(); });

        RCLCPP_INFO(this->get_logger(), "CrowdStatisticsNode initialized [Non-overlapping FOV mode]");
    }

private:
    // 结构体定义
    struct LaserCamPoint {
        float xc, yc;
        float angle;
        float dist;
        float orig_dist;
        float orig_angle;
    };

    struct Person3D {
        geometry_msgs::msg::Point position_base;
        geometry_msgs::msg::Point position_map;
        double distance;
        double angle;
        size_t camera_id;
        double confidence;
        bool has_map_position{false};
    };

    struct CameraInfo {
        std::string topic;
        std::string frame_id;
        std::string camera_name;
        rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr sub;
        std::shared_ptr<std::mutex> mutex;
        std::vector<Person3D> latest_persons;
        rclcpp::Time last_update{rclcpp::Time(0, 0, RCL_ROS_TIME)};
        size_t frame_count{0};  // 统计帧计数
    };

    struct LaserCache {
        rclcpp::Time stamp{rclcpp::Time(0, 0, RCL_ROS_TIME)};
        sensor_msgs::msg::LaserScan::SharedPtr scan;
    };

    // 成员变量
    std::unique_ptr<YoloDetector> yolo_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    std::vector<std::string> camera_names_;
    std::vector<CameraInfo> cameras_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_front_scan_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_back_scan_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr pub_pedestrians_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr pub_pedestrians_map_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_person_poses_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_stats_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_viz_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::string front_scan_topic_, back_scan_topic_;
    std::string base_frame_;
    std::string map_frame_;
    double distance_threshold_;
    double confidence_threshold_;
    double angle_tolerance_;
    int frame_skip_;
    double h_fov_rad_;
    double default_person_width_rad_;
    double density_radius_;
    bool debug_mode_;
    bool publish_viz_;
    bool publish_map_frame_;
    int frame_counter_{0};

    std::deque<LaserCache> front_scan_queue_, back_scan_queue_;
    std::mutex front_mutex_, back_mutex_;
    static constexpr size_t MAX_QUEUE_SIZE = 50;

    rclcpp::Time now() {
        return this->get_clock()->now();
    }

    sensor_msgs::msg::LaserScan::SharedPtr getClosestLaserScan(const rclcpp::Time &img_stamp, int type) {
        std::deque<LaserCache> *queue = (type == 0) ? &front_scan_queue_ : &back_scan_queue_;
        std::mutex *mtx = (type == 0) ? &front_mutex_ : &back_mutex_;
        std::lock_guard<std::mutex> lock(*mtx);
        if (queue->empty()) return nullptr;

        auto best = queue->begin();
        double best_diff = std::abs((best->stamp - img_stamp).seconds());
        for (auto it = queue->begin(); it != queue->end(); ++it) {
            double diff = std::abs((it->stamp - img_stamp).seconds());
            if (diff < best_diff) {
                best_diff = diff;
                best = it;
            }
        }
        if (best_diff > 0.3) return nullptr;
        return best->scan;
    }

    std::vector<LaserCamPoint> transformLaserToCamera(const sensor_msgs::msg::LaserScan::SharedPtr &scan,
                                                      const std::string &camera_frame,
                                                      const rclcpp::Time &img_stamp) {
        std::vector<LaserCamPoint> points;
        if (!scan) return points;

        geometry_msgs::msg::TransformStamped tf_cam_laser;
        try {
            tf_cam_laser = tf_buffer_->lookupTransform(
                camera_frame, 
                scan->header.frame_id,
                img_stamp,
                tf2::durationFromSec(0.1)
            );
        } catch (const tf2::TransformException &e) {
            try {
                tf_cam_laser = tf_buffer_->lookupTransform(
                    camera_frame, 
                    scan->header.frame_id,
                    tf2::TimePointZero,
                    tf2::durationFromSec(0.05)
                );
            } catch (const tf2::TransformException &e2) {
                return points;
            }
        }

        const auto &q = tf_cam_laser.transform.rotation;
        const auto &t = tf_cam_laser.transform.translation;
        float qx = static_cast<float>(q.x), qy = static_cast<float>(q.y);
        float qz = static_cast<float>(q.z), qw = static_cast<float>(q.w);
        float R00 = 1.0f - 2.0f*(qy*qy + qz*qz);
        float R01 = 2.0f*(qx*qy - qz*qw);
        float R10 = 2.0f*(qx*qy + qz*qw);
        float R11 = 1.0f - 2.0f*(qx*qx + qz*qz);
        float tx = static_cast<float>(t.x);
        float ty = static_cast<float>(t.y);

        const int n = static_cast<int>(scan->ranges.size());
        points.reserve(n);
        
        for (int i = 0; i < n; ++i) {
            float r = scan->ranges[i];
            if (!std::isfinite(r) || r < scan->range_min || r > scan->range_max) continue;
            
            float theta = scan->angle_min + i * scan->angle_increment;
            float xl = r * std::cos(theta);
            float yl = r * std::sin(theta);
            float xc = R00*xl + R01*yl + tx;
            float yc = R10*xl + R11*yl + ty;
            
            float dist = std::hypot(xc, yc);
            if (dist > distance_threshold_ || dist < 0.05f) continue;
            
            float angle = std::atan2(yc, xc);
            points.push_back({xc, yc, angle, dist, r, theta});
        }

        return points;
    }

    // 修正：添加负号，修正图像坐标到相机角度的映射
    void computeBboxAngles(const Detection &det, int img_width,
                           float &left_angle, float &right_angle) {
        const float img_w = static_cast<float>(img_width);
        
        float cx_norm = (det.cx / img_w) - 0.5f;
        // 关键修复：添加负号，使图像左侧对应相机左侧（正角度）
        float center_angle = -cx_norm * h_fov_rad_;
        
        float half_width_rad;
        if (det.bbox.width > 20 && det.bbox.width < img_width) {
            float width_norm = static_cast<float>(det.bbox.width) / img_w;
            half_width_rad = (width_norm * h_fov_rad_) / 2.0f;
            half_width_rad = std::max<float>(half_width_rad, static_cast<float>(default_person_width_rad_ / 2.0));
        } else {
            half_width_rad = static_cast<float>(default_person_width_rad_ / 2.0);
        }
        
        left_angle = center_angle - half_width_rad;
        right_angle = center_angle + half_width_rad;
        
        float max_angle = static_cast<float>(h_fov_rad_ / 2.0);
        left_angle = std::max<float>(left_angle, -max_angle);
        right_angle = std::min<float>(right_angle, max_angle);
        
        if (left_angle > right_angle) {
            std::swap(left_angle, right_angle);
        }
    }

    bool findNearestInAngleRange(const std::vector<LaserCamPoint> &points,
                                 float left_angle, float right_angle,
                                 LaserCamPoint &best) {
        float min_dist = std::numeric_limits<float>::max();
        bool found = false;
        
        float search_left = left_angle - static_cast<float>(angle_tolerance_);
        float search_right = right_angle + static_cast<float>(angle_tolerance_);
        
        for (const auto &p : points) {
            float angle = p.angle;
            if (angle >= search_left && angle <= search_right) {
                if (p.dist < min_dist) {
                    min_dist = p.dist;
                    best = p;
                    found = true;
                }
            }
        }
        return found;
    }

    bool transformToBase(const std::string &laser_frame, 
                         const geometry_msgs::msg::Point &pt_laser,
                         const rclcpp::Time &stamp,
                         geometry_msgs::msg::PointStamped &pt_base) {
        geometry_msgs::msg::PointStamped pt_in;
        pt_in.header.frame_id = laser_frame;
        pt_in.header.stamp = stamp;
        pt_in.point = pt_laser;
        
        try {
            tf_buffer_->transform(pt_in, pt_base, base_frame_);
            return true;
        } catch (const tf2::TransformException &e) {
            try {
                pt_in.header.stamp = rclcpp::Time(0, 0, RCL_ROS_TIME);
                tf_buffer_->transform(pt_in, pt_base, base_frame_);
                return true;
            } catch (const tf2::TransformException &e2) {
                return false;
            }
        }
    }

    bool transformToMap(const geometry_msgs::msg::Point &pt_base,
                        const rclcpp::Time &stamp,
                        geometry_msgs::msg::PointStamped &pt_map) {
        if (!publish_map_frame_) return false;
        
        geometry_msgs::msg::PointStamped pt_in;
        pt_in.header.frame_id = base_frame_;
        pt_in.header.stamp = stamp;
        pt_in.point = pt_base;
        
        try {
            tf_buffer_->transform(pt_in, pt_map, map_frame_);
            return true;
        } catch (const tf2::TransformException &e) {
            return false;
        }
    }

    void imageCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg, size_t cam_idx) {
        if (!yolo_ || !msg || msg->data.empty()) return;

        if (++frame_counter_ % (frame_skip_ + 1) != 0) return;

        cv::Mat frame = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
        if (frame.empty()) return;

        std::vector<Detection> detections = yolo_->detect(frame);
        
        rclcpp::Time img_stamp(msg->header.stamp, RCL_ROS_TIME);
        auto front_scan = getClosestLaserScan(img_stamp, 0);
        auto back_scan = getClosestLaserScan(img_stamp, 1);
        
        // 即使无检测也更新last_update，表示相机活跃
        {
            std::lock_guard<std::mutex> lock(*cameras_[cam_idx].mutex);
            cameras_[cam_idx].last_update = now();
            cameras_[cam_idx].frame_count++;
        }

        if (detections.empty()) {
            std::lock_guard<std::mutex> lock(*cameras_[cam_idx].mutex);
            cameras_[cam_idx].latest_persons.clear();
            return;
        }

        if (!front_scan && !back_scan) return;

        std::vector<LaserCamPoint> all_points_cam;
        std::string laser_frame;
        if (front_scan) {
            auto pts = transformLaserToCamera(front_scan, cameras_[cam_idx].frame_id, img_stamp);
            all_points_cam.insert(all_points_cam.end(), pts.begin(), pts.end());
            laser_frame = front_scan->header.frame_id;
        }
        if (back_scan) {
            auto pts = transformLaserToCamera(back_scan, cameras_[cam_idx].frame_id, img_stamp);
            all_points_cam.insert(all_points_cam.end(), pts.begin(), pts.end());
            if (laser_frame.empty()) laser_frame = back_scan->header.frame_id;
        }
        
        if (all_points_cam.empty()) return;

        std::vector<Person3D> new_persons;
        
        for (const auto &det : detections) {
            if (det.class_id != 0) continue;  // 只检测person
            if (det.confidence < confidence_threshold_) continue;

            float left_angle, right_angle;
            computeBboxAngles(det, frame.cols, left_angle, right_angle);

            if (debug_mode_) {
                RCLCPP_INFO(this->get_logger(), 
                    "[%s] cx=%.1f (%.1f%%), angles: [%.3f, %.3f] rad", 
                    cameras_[cam_idx].camera_name.c_str(),
                    det.cx, (det.cx/frame.cols)*100, left_angle, right_angle);
            }

            LaserCamPoint best;
            if (findNearestInAngleRange(all_points_cam, left_angle, right_angle, best)) {
                geometry_msgs::msg::Point pt_laser;
                pt_laser.x = best.orig_dist * std::cos(best.orig_angle);
                pt_laser.y = best.orig_dist * std::sin(best.orig_angle);
                pt_laser.z = 0.0;

                geometry_msgs::msg::PointStamped pt_base;
                
                if (transformToBase(laser_frame, pt_laser, img_stamp, pt_base)) {
                    double dist = std::hypot(pt_base.point.x, pt_base.point.y);
                    
                    if (dist <= distance_threshold_) {
                        Person3D person;
                        person.position_base = pt_base.point;
                        person.distance = dist;
                        person.angle = std::atan2(pt_base.point.y, pt_base.point.x);
                        person.camera_id = cam_idx;
                        person.confidence = det.confidence;
                        
                        geometry_msgs::msg::PointStamped pt_map;
                        if (transformToMap(pt_base.point, img_stamp, pt_map)) {
                            person.position_map = pt_map.point;
                            person.has_map_position = true;
                            pt_map.header.stamp = msg->header.stamp;
                            pub_pedestrians_map_->publish(pt_map);
                        }
                        
                        new_persons.push_back(person);
                        
                        pt_base.header.stamp = msg->header.stamp;
                        pub_pedestrians_->publish(pt_base);
                    }
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(*cameras_[cam_idx].mutex);
            cameras_[cam_idx].latest_persons = std::move(new_persons);
        }
    }

    void laserCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg, int type) {
        if (!msg) return;
        LaserCache cache{rclcpp::Time(msg->header.stamp, RCL_ROS_TIME), msg};
        if (type == 0) {
            std::lock_guard<std::mutex> lock(front_mutex_);
            front_scan_queue_.push_back(cache);
            if (front_scan_queue_.size() > MAX_QUEUE_SIZE) front_scan_queue_.pop_front();
        } else {
            std::lock_guard<std::mutex> lock(back_mutex_);
            back_scan_queue_.push_back(cache);
            if (back_scan_queue_.size() > MAX_QUEUE_SIZE) back_scan_queue_.pop_front();
        }
    }

    // 无重叠区域：简单汇总，无需去重
    std::vector<Person3D> aggregateAllPersons(std::vector<size_t> &camera_counts) {
        std::vector<Person3D> all_persons;
        camera_counts.assign(4, 0);
        
        for (size_t i = 0; i < cameras_.size(); ++i) {
            std::lock_guard<std::mutex> cam_lock(*cameras_[i].mutex);
            
            if (cameras_[i].last_update.nanoseconds() > 0) {
                double diff = (now() - cameras_[i].last_update).seconds();
                if (diff <= 2.0 && !cameras_[i].latest_persons.empty()) {
                    camera_counts[i] = cameras_[i].latest_persons.size();
                    all_persons.insert(all_persons.end(), 
                                    cameras_[i].latest_persons.begin(), 
                                    cameras_[i].latest_persons.end());
                }
            }
        }
        
        return all_persons;
    }

    double computeLocalDensity(const Person3D &person, const std::vector<Person3D> &all_persons) {
        double density = 0.0;
        for (const auto &other : all_persons) {
            double dx = person.position_base.x - other.position_base.x;
            double dy = person.position_base.y - other.position_base.y;
            double dist = std::hypot(dx, dy);
            if (dist < density_radius_ && dist > 0.01) {
                density += std::exp(-(dist * dist) / (2.0 * (density_radius_ / 3.0) * (density_radius_ / 3.0)));
            }
        }
        double area = M_PI * density_radius_ * density_radius_;
        return density / area;
    }

    double computeCrowdPressure(const std::vector<Person3D> &persons) {
        if (persons.size() < 2) return 0.0;
        
        double total_pressure = 0.0;
        for (const auto &p : persons) {
            double local_density = computeLocalDensity(p, persons);
            double dist_from_center = std::hypot(p.position_base.x, p.position_base.y);
            double pressure = local_density * (1.0 + 0.1 * dist_from_center);
            total_pressure += pressure;
        }
        
        return total_pressure / persons.size();
    }

    void publishVisualization(const std::vector<Person3D> &persons, const std::vector<size_t> &camera_counts) {
        if (!publish_viz_) return;
        
        visualization_msgs::msg::MarkerArray markers;
        int marker_id = 0;
        rclcpp::Time stamp = now();
        
        // 清空旧marker
        visualization_msgs::msg::MarkerArray clear_markers;
        visualization_msgs::msg::Marker clear_marker;
        clear_marker.header.frame_id = base_frame_;
        clear_marker.header.stamp = stamp;
        clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        clear_markers.markers.push_back(clear_marker);
        pub_viz_->publish(clear_markers);
        
        // 检测范围圆柱体
        visualization_msgs::msg::Marker range_marker;
        range_marker.header.frame_id = base_frame_;
        range_marker.header.stamp = stamp;
        range_marker.ns = "crowd_range";
        range_marker.id = marker_id++;
        range_marker.type = visualization_msgs::msg::Marker::CYLINDER;
        range_marker.action = visualization_msgs::msg::Marker::ADD;
        range_marker.pose.position.x = 0;
        range_marker.pose.position.y = 0;
        range_marker.pose.position.z = -0.5;
        range_marker.pose.orientation.w = 1.0;
        range_marker.scale.x = distance_threshold_ * 2;
        range_marker.scale.y = distance_threshold_ * 2;
        range_marker.scale.z = 1.0;
        range_marker.color.r = 0.0f;
        range_marker.color.g = 0.5f;
        range_marker.color.b = 0.5f;
        range_marker.color.a = 0.1f;
        markers.markers.push_back(range_marker);
        
        // 各摄像头人数文本标签（在机器人上方显示）
        const std::string cam_names[4] = {"F", "L", "R", "B"};
        const float colors[4][3] = {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 0.0f}};
        
        for (int i = 0; i < 4; ++i) {
            visualization_msgs::msg::Marker count_marker;
            count_marker.header.frame_id = base_frame_;
            count_marker.header.stamp = stamp;
            count_marker.ns = "camera_counts";
            count_marker.id = marker_id++;
            count_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            count_marker.action = visualization_msgs::msg::Marker::ADD;
            count_marker.pose.position.x = 0;
            count_marker.pose.position.y = 0;
            count_marker.pose.position.z = 2.0 + i * 0.3;  // 垂直排列
            count_marker.scale.z = 0.3f;
            count_marker.color.r = colors[i][0];
            count_marker.color.g = colors[i][1];
            count_marker.color.b = colors[i][2];
            count_marker.color.a = 1.0f;
            count_marker.text = cam_names[i] + std::string(": ") + std::to_string(camera_counts[i]);
            markers.markers.push_back(count_marker);
        }
        
        // 每个人员
        for (size_t i = 0; i < persons.size(); ++i) {
            const auto &p = persons[i];
            
            visualization_msgs::msg::Marker person_marker;
            person_marker.header.frame_id = base_frame_;
            person_marker.header.stamp = stamp;
            person_marker.ns = "persons_base";
            person_marker.id = marker_id++;
            person_marker.type = visualization_msgs::msg::Marker::SPHERE;
            person_marker.action = visualization_msgs::msg::Marker::ADD;
            person_marker.pose.position = p.position_base;
            person_marker.pose.position.z = 0.5;
            person_marker.pose.orientation.w = 1.0;
            person_marker.scale.x = 0.4;
            person_marker.scale.y = 0.4;
            person_marker.scale.z = 0.4;
            
            person_marker.color.r = colors[p.camera_id % 4][0];
            person_marker.color.g = colors[p.camera_id % 4][1];
            person_marker.color.b = colors[p.camera_id % 4][2];
            person_marker.color.a = 0.8f;
            markers.markers.push_back(person_marker);
            
            // 距离标签
            visualization_msgs::msg::Marker text_marker;
            text_marker.header.frame_id = base_frame_;
            text_marker.header.stamp = stamp;
            text_marker.ns = "person_labels";
            text_marker.id = marker_id++;
            text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            text_marker.action = visualization_msgs::msg::Marker::ADD;
            text_marker.pose.position = p.position_base;
            text_marker.pose.position.z = 1.0;
            text_marker.scale.z = 0.25f;
            text_marker.color.r = 1.0f;
            text_marker.color.g = 1.0f;
            text_marker.color.b = 1.0f;
            text_marker.color.a = 1.0f;
            char buf[128];
            snprintf(buf, sizeof(buf), "P%zu: %.1fm [%s]", 
                    i, p.distance, camera_names_[p.camera_id].c_str());
            text_marker.text = buf;
            markers.markers.push_back(text_marker);
        }
        
        pub_viz_->publish(markers);
        
        // 发布PoseArray
        geometry_msgs::msg::PoseArray pose_array;
        pose_array.header.frame_id = base_frame_;
        pose_array.header.stamp = stamp;
        for (const auto &p : persons) {
            geometry_msgs::msg::Pose pose;
            pose.position = p.position_base;
            pose.orientation.w = 1.0;
            pose_array.poses.push_back(pose);
        }
        pub_person_poses_->publish(pose_array);
    }

    void timerCallback() {
        // 分别获取各摄像头人数和全部人员
        std::vector<size_t> camera_counts(4, 0);
        std::vector<Person3D> all_persons = aggregateAllPersons(camera_counts);
        
        size_t total_people = all_persons.size();
        double area = M_PI * distance_threshold_ * distance_threshold_;
        double global_density = (area > 0) ? static_cast<double>(total_people) / area : 0.0;
        
        // 计算平均局部密度
        double avg_local_density = 0.0;
        if (!all_persons.empty()) {
            for (const auto &p : all_persons) {
                avg_local_density += computeLocalDensity(p, all_persons);
            }
            avg_local_density /= all_persons.size();
        }
        
        double pressure = computeCrowdPressure(all_persons);
        
        // 构建统计消息: [总数, 全局密度, 平均局部密度, 压力, 前, 左, 右, 后]
        std_msgs::msg::Float64MultiArray stats_msg;
        stats_msg.data.resize(8);
        stats_msg.data[0] = static_cast<double>(total_people);
        stats_msg.data[1] = global_density;
        stats_msg.data[2] = avg_local_density;
        stats_msg.data[3] = pressure;
        stats_msg.data[4] = static_cast<double>(camera_counts[0]);
        stats_msg.data[5] = static_cast<double>(camera_counts[1]);
        stats_msg.data[6] = static_cast<double>(camera_counts[2]);
        stats_msg.data[7] = static_cast<double>(camera_counts[3]);
        
        pub_stats_->publish(stats_msg);
        
        publishVisualization(all_persons, camera_counts);
        
        // 日志输出各摄像头人数
        RCLCPP_INFO(this->get_logger(), 
            "Total: %zu | Front: %zu | Left: %zu | Right: %zu | Back: %zu | Density: %.3f | Pressure: %.3f",
            total_people, camera_counts[0], camera_counts[1], camera_counts[2], camera_counts[3],
            global_density, pressure);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CrowdStatisticsNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}