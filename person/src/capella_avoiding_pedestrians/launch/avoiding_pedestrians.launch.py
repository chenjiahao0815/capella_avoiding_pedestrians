import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    pkg_share = get_package_share_directory("capella_avoiding_pedestrians")

    # YOLO 模型路径
    yolo_model_path = os.path.join(pkg_share, "car8.onnx")

    return LaunchDescription([
        Node(
            package="capella_avoiding_pedestrians",
            executable="behavior_detection_node",
            name="behavior_detection_node",
            output="screen",
            parameters=[{
                # -------------------------------------------------------
                # 路径相关
                # -------------------------------------------------------
                # 全局路径搜索距离
                "global_search_distance": 5.0,

                # 局部路径搜索距离
                "local_search_distance": 5.0,

                # 行人与路径的距离小于这个值就要触发预警
                "pedestrian_distance_threshold": 1.0,

                # 如果检测到人以后就要持续发布，直到超过这个时间没有检测到人了才停止发布
                "avoid_hold_seconds": 3.0,

                # 行人检测置信度阈值
                "person_conf_threshold": 0.3,

                # -------------------------------------------------------
                # 坐标系 / TF
                # 用于激光-相机 TF 查询
                "rgb_camera_frame": "front_camera_color_frame",

                # 激光雷达话题
                "scan_topic_name_front": "/front_scan",

                # 相机图像话题
                "camera_topic": "/rgb_camera_front/compressed",

                # -------------------------------------------------------
                # YOLO 模型
                # -------------------------------------------------------
                # 用了一个pytho变量，不是字符串，在文件顶部已经定义了
                "yolo_model_path": yolo_model_path,
            }],
        )
    ])
