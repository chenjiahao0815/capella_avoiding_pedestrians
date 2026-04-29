import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_pose_stamped

from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, PoseStamped

import cv2
import os
import uuid
import torch
import math
import numpy as np
from ultralytics import YOLO
from functools import partial
from cv_bridge import CvBridge

from capella_ros_msg.msg import Recognitions, SingleRecognition



class CapellaInspectionNode(Node):
    def __init__(self):
        super().__init__('capella_inspection_node')
        self.device="cuda:0" if torch.cuda.is_available() else 'cpu'
        self.model_fire = YOLO(os.path.join(os.path.dirname(__file__), 'fire_2025-2-11_best.pt')).to(self.device)
        
        # self.declare_parameter('camera_topics', ['/camera2/color/image_raw', '/camera3/color/image_raw'])
        # self.declare_parameter('camera_tf_names', ['camera2_color_frame', 'camera3_color_frame'])
        # self.declare_parameter('horizontal_fov_degs', [84.3, 67.8])
        self.declare_parameter('camera_topics', ['/rgb_camera_front/image_raw'])
        self.declare_parameter('camera_tf_names', ['front_camera_color_frame'])
        self.declare_parameter('horizontal_fov_degs', [84.3])
        self.declare_parameter('max_publish_per_uuid', 2)
        self.declare_parameter('process_interval_sec', 2.0)

        self.camera_topics = self.get_parameter('camera_topics').get_parameter_value().string_array_value
        self.camera_tf_names = self.get_parameter('camera_tf_names').get_parameter_value().string_array_value
        self.horizontal_fovs = self.get_parameter('horizontal_fov_degs').get_parameter_value().double_array_value
        self.max_publish_per_uuid = self.get_parameter('max_publish_per_uuid').get_parameter_value().integer_value
        self.process_interval_sec = self.get_parameter('process_interval_sec').get_parameter_value().double_value
        
        self.bridge = CvBridge()
        self.is_active = False
        
        qos_1 = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        qos_2 = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)    
            
        # 创建多个图像订阅者
        self.image_subs = []
        self.callback_groups = []  # 存储所有回调组

        for i, topic in enumerate(self.camera_topics):
            # 创建回调组并保存到成员变量中
            self.callback_groups.append(MutuallyExclusiveCallbackGroup())
            
            # 创建订阅器时，从成员变量列表中获取回调组，并绑定对应参数：topic, tf_name, fov
            sub = self.create_subscription(
                Image,
                topic,
                partial(self.image_callback, topic_name=topic, tf_name=self.camera_tf_names[i], horizontal_fov=self.horizontal_fovs[i]),
                qos_1,
                callback_group=self.callback_groups[i] 
            )
            self.image_subs.append(sub)

        self.ctr_sub = self.create_subscription(Bool, '/inspections_ctr', self.control_callback, qos_2)
        
        self.result_pub = self.create_publisher(Recognitions, '/inspections_data', 10)
        
        self.marker_pub = self.create_publisher(Marker, '/inspection_rays', 10)

            
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.get_logger().info("Capella Inspection Node initialized !")

        self.last_ray_origin = None  # 上一帧射线起点（在 map 中）
        self.last_ray_direction = None  # 上一帧射线方向（在 map 中）
        self.last_intersection = None  # 上一次计算出的交点
        self.img_uuid = None
        self.uuid_publish_count = {}  # 记录每个 UUID 的发布次数
        self.first_frame_processed = False
        self.last_process_time = self.get_clock().now()
        self.ray_history = []  # 储存最近的射线（最多10条），每条是 (origin, direction)
        self.max_ray_history = 7



    def control_callback(self, msg):
        self.is_active = msg.data
        self.get_logger().info(f"Inspection control set to: {'ON' if self.is_active else 'OFF'}")


    def image_callback(self, msg, topic_name, tf_name, horizontal_fov):
        if self.is_active:
            if len(msg.data) > 0:
                current_time = rclpy.time.Time.from_msg(msg.header.stamp)
                color_width = msg.width
                color_height = msg.height
                color_data = np.array(msg.data).reshape((color_height, color_width, 3))
                for i, topic in enumerate(self.camera_topics):
                    if topic.startswith('/rgb'):
                        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
                
                reuslt_fires = self.model_fire.predict(color_data, conf=0.65, show=False, verbose=False, save=False)
                
                if reuslt_fires[0].boxes.cls is not None:
                    if not self.first_frame_processed:
                        self.first_frame_processed = True
                        self.last_process_time = current_time
                    else:
                        interval_ns = self.process_interval_sec * 1e9
                        if (current_time - self.last_process_time).nanoseconds < interval_ns:
                            return
                        self.last_process_time = current_time
                    
                    inspections_msg = Recognitions()
                    
                    boxes = reuslt_fires[0].boxes.xywh.tolist()
                    
                    try:
                        transform = self.tf_buffer.lookup_transform('map', 'base_link', current_time, rclpy.duration.Duration(seconds=0.5))
                        transform1 = self.tf_buffer.lookup_transform('map', tf_name, current_time, rclpy.duration.Duration(seconds=0.5))
                    except Exception as e:
                        self.get_logger().warn(f"Failed to get transformation: {str(e)}")
                        return None
                    
                    X_R, Y_R = transform.transform.translation.x, transform.transform.translation.y
                    
                    for i in range(len(boxes)):
                        x_center, y_center, w, h = boxes[i]
                        x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
                        x2, y2 = int(x_center + w / 2), int(y_center + h / 2)
                    
                        theta_x_center_rad = np.deg2rad((0.5 - x_center / color_width) * horizontal_fov)
                        
                        pose_in_camera = PoseStamped()
                        pose_in_camera.header.stamp = msg.header.stamp
                        pose_in_camera.pose.position.x = np.cos(theta_x_center_rad)
                        pose_in_camera.pose.position.y = np.sin(theta_x_center_rad)
                        
                        pose_in_map = do_transform_pose_stamped(pose_in_camera, transform1)

                        # 当前射线在 map 中的起点和方向（单位向量）
                        origin_map = np.array([X_R, Y_R])
                        target_point = np.array([pose_in_map.pose.position.x, pose_in_map.pose.position.y])
                        # 射线方向：从 base_link 位置指向目标点
                        direction_map = target_point - origin_map
                        direction_map /= np.linalg.norm(direction_map)  # 归一化

                        # 如果上一帧存在数据，判断是否同一个物体
                        if self.last_ray_origin is not None and self.last_ray_direction is not None:
                            # 覆盖方式显示射线：使用固定 ID
                            self.publish_ray_marker(msg.header.stamp, self.last_ray_origin, self.last_ray_direction, marker_id=0, color=[0.0, 1.0, 0.0])  # 上一帧绿色
                            self.publish_ray_marker(msg.header.stamp, origin_map, direction_map, marker_id=1, color=[1.0, 0.0, 0.0])  # 当前帧红色
                        
                            is_same_object = False

                            # 与历史射线一一比较，只要有一条不同，就视为新目标
                            for prev_origin, prev_direction in self.ray_history:
                                if  self.rays_have_common_part(
                                    ray1=prev_direction,
                                    origin1=prev_origin,
                                    ray2=direction_map,
                                    origin2=origin_map
                                ):
                                    is_same_object = True
                                    break
                            # 用于演示持续发布
                            is_same_object = False
                            if is_same_object:
                                self.get_logger().info("It's the same detection object !", throttle_duration_sec=3.0)
                            else:
                                self.img_uuid = str(uuid.uuid4())

                        # 添加当前射线到历史列表
                        self.ray_history.append((origin_map.copy(), direction_map.copy()))
                        if len(self.ray_history) > self.max_ray_history:
                            self.ray_history.pop(0)

                        # result = reuslt_fires[0]
                        # plotted_img = result.plot()  # RGB 格式的 numpy array

                        # cv2.rectangle(color_data, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # info_text = "fire"
                        # cv2.putText(color_data, info_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
                        img_msg = self.bridge.cv2_to_imgmsg(color_data, encoding='rgb8')
                        img_msg.header = msg.header 
                        
                        if self.img_uuid is None:
                            self.img_uuid = str(uuid.uuid4())
                        
                        # 检查是否超出最大发布次数
                        count = self.uuid_publish_count.get(self.img_uuid, 0)
                        if count < self.max_publish_per_uuid:
                            single_msg = SingleRecognition()
                            single_msg.class_id = 0
                            single_msg.uuid = self.img_uuid
                            single_msg.left_cord_x = x1
                            single_msg.left_cord_y = y1
                            single_msg.right_cord_x = x2
                            single_msg.right_cord_y = y2
                            single_msg.image = img_msg

                            inspections_msg.recognitions.append(single_msg)
                            
                            self.uuid_publish_count[self.img_uuid] = count + 1

                            # 限制最大保留 UUID 数量为 10
                            MAX_UUID_HISTORY = 10
                            if len(self.uuid_publish_count) > MAX_UUID_HISTORY:
                                oldest_uuid = next(iter(self.uuid_publish_count))
                                del self.uuid_publish_count[oldest_uuid]


                        # 更新当前射线信息供下一帧使用
                        self.last_ray_origin = origin_map
                        self.last_ray_direction = direction_map
                        
                    if len(inspections_msg.recognitions) > 0:
                        self.result_pub.publish(inspections_msg)
                        self.get_logger().info(f"Have published {count+1} !")

        else:
            self.uuid_publish_count.clear()
            self.ray_history = []


    # def compute_ray_intersection(self, ray1, origin1, ray2, origin2):
    #     """
    #     计算两条二维射线在正方向上的的交点(以参数方程形式表示)
    #     射线1: origin1 + t1 * ray1
    #     射线2: origin2 + t2 * ray2
    #     返回: 交点 np.array([x, y]) 或 None(平行/无解)
    #     """
    #     dx1, dy1 = ray1
    #     dx2, dy2 = ray2
    #     x1, y1 = origin1
    #     x2, y2 = origin2

    #     A = np.array([
    #         [dx1, -dx2],
    #         [dy1, -dy2]
    #     ])
    #     b = np.array([x2 - x1, y2 - y1])
        
    #     # 判断是否近似平行（行列式接近0）
    #     det = dx1 * dy2 - dx2 * dy1
    #     if abs(det) < 1e-4:
    #         return None  # 射线近似平行，认为无交点
            
    #     try:
    #         t = np.linalg.solve(A, b)
    #         t1, t2 = t

    #         # 只接受正方向的交点
    #         if t1 >= 0 and t2 >= 0:
    #             intersection = origin1 + t1 * ray1
    #             return intersection
    #         else:
    #             return None
    #     except np.linalg.LinAlgError:
    #         # 平行或奇异矩阵
    #         return None


    def rays_have_common_part(self, ray1, origin1, ray2, origin2, eps=1e-1):
        """
        判断两条二维射线是否有公共部分（相交 或 共线重合）
        不包含起点相同方向不同的情况
        """
        ray1 = ray1 / np.linalg.norm(ray1)
        ray2 = ray2 / np.linalg.norm(ray2)

        # ---------- 情况一：共线 ----------
        if abs(np.cross(ray1, ray2)) < eps:
            # 起点相同但方向不同，认为无重合
            if np.linalg.norm(origin1 - origin2) < eps:
                if np.dot(ray1, ray2) < 1 - eps:
                    return False

            vec = origin2 - origin1
            proj1 = np.dot(vec, ray1)
            vec2 = origin1 - origin2
            proj2 = np.dot(vec2, ray2)

            if proj1 >= -eps or proj2 >= -eps:
                return True
            else:
                return False

        # ---------- 情况二：不共线，尝试计算交点 ----------
        # else:
        dx1, dy1 = ray1
        dx2, dy2 = ray2
        x1, y1 = origin1
        x2, y2 = origin2

        A = np.array([
            [dx1, -dx2],
            [dy1, -dy2]
        ])
        b = np.array([x2 - x1, y2 - y1])

        t = np.linalg.solve(A, b)
        t1, t2 = t
        if t1 >= -eps and t2 >= -eps:
            return True
        else:
            return False




    def publish_ray_marker(self, stamp, origin, direction, marker_id, color):
        if origin is None or direction is None:
            return
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = stamp
        marker.ns = 'inspection_rays'
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        start = origin
        end = origin + direction * 3.0

        marker.points = [self.to_point_msg(start), self.to_point_msg(end)]
        marker.scale.x = 0.05
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0
        marker.lifetime = Duration(sec=2)

        self.marker_pub.publish(marker)

    def to_point_msg(self, np_array):
        p = Point()
        p.x, p.y, p.z = float(np_array[0]), float(np_array[1]), 0.0
        return p



def main(args=None):
    rclpy.init(args=args)
    node = CapellaInspectionNode()
    multi_executor = MultiThreadedExecutor(num_threads=4)
    multi_executor.add_node(node)
    multi_executor.spin()
    node.destroy_node()
    multi_executor.shutdown()


if __name__ == '__main__':
    main()