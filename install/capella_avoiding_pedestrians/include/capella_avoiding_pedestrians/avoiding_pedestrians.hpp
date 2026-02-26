#pragma once

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <nav_msgs/msg/path.hpp>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/bool.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker_array.hpp>

#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

struct TrackBBox
{
	float x{0.0F};
	float y{0.0F};
	float width{0.0F};
	float height{0.0F};
};

struct TrackItem
{
	int class_id{0};
	float confidence{0.0F};
	TrackBBox bbox;
};

class YoloTracker
{
public:
	virtual ~YoloTracker() = default;
	virtual std::vector<TrackItem> track(const cv::Mat &frame) = 0;
};

class BehaviorDetectionNode : public rclcpp::Node
{
public:
	BehaviorDetectionNode();
private:
	struct DetectionResult
	{
		bool detected{false};
		builtin_interfaces::msg::Time stamp;
		std::string frame_id;
		std::vector<float> person_angles;
	};

	void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
	bool runYoloTrack(const cv::Mat &frame, std::vector<TrackItem> &tracks);
	void laserCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
	void globalPlanCallback(const nav_msgs::msg::Path::SharedPtr msg);
	void localPosesCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg);
	void timerCallback();
	//降采样的函数，减少路径点用的
    std::vector<geometry_msgs::msg::Point> downsamplePath(
		const std::vector<geometry_msgs::msg::Point> &nav_points,
		double min_distance,
		double lookahead_distance);
	//融合相机和激光数据用滤波跟踪行人
    std::vector<geometry_msgs::msg::PointStamped> fuseAndTrackPedestrians(
		const visualization_msgs::msg::MarkerArray::SharedPtr detections,
		const sensor_msgs::msg::LaserScan::SharedPtr scan);

	bool checkPedestrianOnGlobalPath(
		const std::vector<geometry_msgs::msg::PointStamped> &pedestrians,
		const nav_msgs::msg::Path &path,
		double search_distance,
		double threshold);
	bool checkPedestrianOnLocalPath(
		const std::vector<geometry_msgs::msg::PointStamped> &pedestrians,
		const geometry_msgs::msg::PoseArray::SharedPtr local_poses,
		double threshold);

	rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_camera_;
	rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_laser_;
	rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_global_plan_;
	rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr sub_local_poses_;
    // 是否在避让行人
	rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_avoiding_;

	std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
	std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

	rclcpp::TimerBase::SharedPtr timer_;

	sensor_msgs::msg::Image::SharedPtr last_rgb_; //最近RGB图像
	sensor_msgs::msg::LaserScan::SharedPtr last_laser_; //最近激光数据
	nav_msgs::msg::Path::SharedPtr last_global_plan_; //最近全局路径
	geometry_msgs::msg::PoseArray::SharedPtr last_local_poses_; //最近局部路径
	nav_msgs::msg::Path downsampled_global_plan_; //降采样后的全局路径
    //共享数据
	std::mutex detection_mutex_;
	DetectionResult detection_result_;
    // 激光雷达
	std::mutex laser_queue_mutex_;
	std::deque<std::pair<double, sensor_msgs::msg::LaserScan::SharedPtr>> laser_queue_;
	size_t laser_queue_max_size_{30};

	double person_conf_threshold_{0.5}; //行人检测置信度阈值
	double camera_laser_offset_{0.0};  //默认的相机和激光雷达的角度差
	double h_fov_rad_{1.0472};  //相机水平视场角，默认60度
	double path_lookahead_distance_{5.0}; //保持多远的前方路径点用于判断
	double avoid_hold_seconds_{3.0}; //避让行人的保持时间
	bool has_recent_detection_{false}; //是否有最近的行人检测
	bool last_published_avoiding_{false}; //上一次发布的避让状态
	rclcpp::Time last_pedestrian_detect_time_; //最近一次行人检测时间

	std::shared_ptr<YoloTracker> yolo_;
};
