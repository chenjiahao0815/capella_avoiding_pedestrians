#!/usr/bin/env python3
"""
完整模拟脚本：图片 + 雷达 + TF + 路径
把这个脚本和 1.png 放在同一个目录下运行
"""
import sys
import os
import math
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage, LaserScan
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, TransformStamped
from tf2_ros import StaticTransformBroadcaster


class FullSimulator(Node):
    def __init__(self):
        super().__init__('full_simulator')

        # ==================== 1. 加载图片 ====================
        script_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = None
        for name in ['1.jpg', '1.png', '1.jpeg']:
            p = os.path.join(script_dir, name)
            if os.path.exists(p):
                img_path = p
                break

        if img_path is None:
            self.get_logger().error(f'在 {script_dir} 下找不到 1.jpg 或 1.png')
            sys.exit(1)

        img = cv2.imread(img_path)
        if img is None:
            self.get_logger().error(f'图片读取失败: {img_path}')
            sys.exit(1)

        self.get_logger().info(f'已加载图片: {img_path} ({img.shape[1]}x{img.shape[0]})')

        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            self.get_logger().error('JPEG 编码失败')
            sys.exit(1)
        self.jpeg_data = buf.tobytes()

        # ==================== 2. QoS 配置 ====================
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ==================== 3. 发布者 ====================

        # ① 图片
        self.pub_img = self.create_publisher(
            CompressedImage, '/rgb_camera_front/compressed', sensor_qos)

        # ② 雷达
        self.pub_scan = self.create_publisher(
            LaserScan, '/front_scan', sensor_qos)

        # ④ 全局路径 (teb_global_plan)
        self.pub_global_plan = self.create_publisher(
            Path, 'teb_global_plan', 10)

        # ⑤ 局部路径 (teb_poses)
        self.pub_local_poses = self.create_publisher(
            PoseArray, 'teb_poses', 10)

        # ③ TF 静态变换
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_tf()

        # ==================== 4. 定时器 ====================
        self.count = 0
        # 图片 15 FPS
        self.timer_img = self.create_timer(1.0 / 15.0, self.publish_image)
        # 雷达 10 Hz
        self.timer_scan = self.create_timer(0.1, self.publish_scan)
        # 路径 2 Hz（不需要太频繁）
        self.timer_path = self.create_timer(0.5, self.publish_paths)

        self.get_logger().info('='*60)
        self.get_logger().info('完整模拟已启动：')
        self.get_logger().info('  [1] 图片  → /rgb_camera_front/compressed (15Hz)')
        self.get_logger().info('  [2] 雷达  → /front_scan (10Hz, 正前方1.2m有障碍)')
        self.get_logger().info('  [3] TF    → laser_front ↔ front_camera_color_frame ↔ map')
        self.get_logger().info('  [4] 全局路径 → teb_global_plan (经过正前方1.2m处)')
        self.get_logger().info('  [5] 局部路径 → teb_poses (经过正前方1.2m处)')
        self.get_logger().info('='*60)
        self.get_logger().info('预期触发流程：')
        self.get_logger().info('  YOLO检测到人 → 雷达测距1.2m → TF转到map坐标(1.2,0)')
        self.get_logger().info('  → 路径经过(1.2,0)附近 → 距离<1.0m阈值 → 发布报警')
        self.get_logger().info('='*60)

    def publish_static_tf(self):
        """
        发布静态TF，让 laser_front / front_camera_color_frame / map 之间可以互相变换。
        
        你的程序需要两个TF：
        1. laser_front → front_camera_color_frame （imageCallback中用）
        2. laser_front → map （timerCallback中用）
        
        这里简单起见，全部设为同一个位置（单位变换），
        即假设相机、雷达、map原点重合。
        """
        transforms = []

        # map → laser_front（单位变换，雷达在map原点）
        t1 = TransformStamped()
        t1.header.stamp = self.get_clock().now().to_msg()
        t1.header.frame_id = 'map'
        t1.child_frame_id = 'laser_front'
        t1.transform.translation.x = 0.0
        t1.transform.translation.y = 0.0
        t1.transform.translation.z = 0.0
        t1.transform.rotation.x = 0.0
        t1.transform.rotation.y = 0.0
        t1.transform.rotation.z = 0.0
        t1.transform.rotation.w = 1.0
        transforms.append(t1)

        # laser_front → front_camera_color_frame（单位变换，相机和雷达重合）
        t2 = TransformStamped()
        t2.header.stamp = self.get_clock().now().to_msg()
        t2.header.frame_id = 'laser_front'
        t2.child_frame_id = 'front_camera_color_frame'
        t2.transform.translation.x = 0.0
        t2.transform.translation.y = 0.0
        t2.transform.translation.z = 0.0
        t2.transform.rotation.x = 0.0
        t2.transform.rotation.y = 0.0
        t2.transform.rotation.z = 0.0
        t2.transform.rotation.w = 1.0
        transforms.append(t2)

        self.tf_broadcaster.sendTransform(transforms)
        self.get_logger().info('[TF] 已发布静态变换: map → laser_front → front_camera_color_frame')

    def publish_image(self):
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'front_camera_color_frame'
        msg.format = 'jpeg'
        msg.data = self.jpeg_data
        self.pub_img.publish(msg)

        self.count += 1
        if self.count % 30 == 0:
            self.get_logger().info(f'[图片] 已发布 {self.count} 帧')

    def publish_scan(self):
        """
        模拟激光雷达：正前方1.2m处有障碍物（模拟一个人）。
        
        你的代码中激光点会被TF变换到相机坐标系，然后按角度匹配检测框。
        相机坐标系约定：X前、Y左。
        因为TF是单位变换，所以激光坐标系和相机坐标系重合。
        """
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'laser_front'

        # 模拟 270 度激光雷达
        msg.angle_min = -math.radians(135.0)
        msg.angle_max = math.radians(135.0)
        msg.angle_increment = math.radians(0.5)
        msg.time_increment = 0.0
        msg.scan_time = 0.1
        msg.range_min = 0.05
        msg.range_max = 12.0

        num_points = int((msg.angle_max - msg.angle_min) / msg.angle_increment) + 1

        # 默认 5.0m（远处无障碍）
        ranges = [5.0] * num_points

        # 正前方（角度=0）附近放一个 1.2m 的障碍物
        # 覆盖 ±15 度，模拟一个人站在前方的宽度
        obstacle_distance = 1.2
        center_idx = num_points // 2
        spread = int(15.0 / 0.5)  # ±15度

        for i in range(center_idx - spread, center_idx + spread + 1):
            if 0 <= i < num_points:
                ranges[i] = obstacle_distance

        msg.ranges = ranges
        msg.intensities = [100.0] * num_points

        self.pub_scan.publish(msg)

    def publish_paths(self):
        """
        发布全局路径和局部路径，路径经过正前方1.2m处。
        
        你的代码判断逻辑：
        - pedestrian_distance_threshold = 1.0m
        - 行人在map坐标约 (1.2, 0)
        - 路径点也经过 (1.2, 0) 附近
        - 行人到路径点距离 < 1.0m → 触发报警
        
        所以路径点直接穿过 (1.2, 0) 就行了。
        """
        now = self.get_clock().now().to_msg()

        # ---- 全局路径 (nav_msgs/Path) ----
        global_path = Path()
        global_path.header.stamp = now
        global_path.header.frame_id = 'map'

        # 从机器人位置 (0,0) 到前方 (3,0)，路径经过 (1.2, 0)
        for x in [i * 0.2 for i in range(16)]:  # 0.0, 0.2, 0.4, ..., 3.0
            ps = PoseStamped()
            ps.header.stamp = now
            ps.header.frame_id = 'map'
            ps.pose.position.x = x
            ps.pose.position.y = 0.0
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            global_path.poses.append(ps)

        self.pub_global_plan.publish(global_path)

        # ---- 局部路径 (geometry_msgs/PoseArray) ----
        local_poses = PoseArray()
        local_poses.header.stamp = now
        local_poses.header.frame_id = 'map'

        # 局部路径也经过 (1.2, 0)
        for x in [i * 0.1 for i in range(21)]:  # 0.0, 0.1, ..., 2.0
            pose = Pose()
            pose.position.x = x
            pose.position.y = 0.0
            pose.position.z = 0.0
            pose.orientation.w = 1.0
            local_poses.poses.append(pose)

        self.pub_local_poses.publish(local_poses)


def main():
    rclpy.init()
    node = FullSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('停止')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()