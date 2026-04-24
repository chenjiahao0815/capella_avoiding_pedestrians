#!/usr/bin/env python3
"""

"""

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import tf2_ros


class FakeGlobalPlanPublisher(Node):
    def __init__(self):
        super().__init__('fake_global_plan_publisher')

        # 参数：点的数量和间距
        self.declare_parameter('num_points', 5)
        self.declare_parameter('point_spacing', 1.0)       # 米
        self.declare_parameter('publish_rate', 2.0)         # Hz
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('map_frame', 'map')

        self.num_points = self.get_parameter('num_points').value
        self.point_spacing = self.get_parameter('point_spacing').value
        rate = self.get_parameter('publish_rate').value
        self.base_frame = self.get_parameter('base_frame').value
        self.map_frame = self.get_parameter('map_frame').value

        # 发布到和避障节点订阅的同一个话题
        self.pub = self.create_publisher(Path, 'teb_global_plan', 10)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 定时发布
        period = 1.0 / rate
        self.timer = self.create_timer(period, self.timer_callback)

        self.get_logger().info(
            f'FakeGlobalPlanPublisher started: {self.num_points} points, '
            f'{self.point_spacing}m spacing, {rate}Hz'
        )

    def timer_callback(self):
        # 查询机器人在 map 下的位姿
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}', throttle_duration_sec=2.0)
            return

        # 提取位置和朝向
        tx = tf.transform.translation.x
        ty = tf.transform.translation.y
        q = tf.transform.rotation
        yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)

        # 构造 Path：从机器人前方 1m 开始，每隔 1m 放一个点
        path = Path()
        path.header.frame_id = self.map_frame
        path.header.stamp = self.get_clock().now().to_msg()

        for i in range(1, self.num_points + 1):
            dist = self.point_spacing * i
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = tx + dist * math.cos(yaw)
            pose.pose.position.y = ty + dist * math.sin(yaw)
            pose.pose.position.z = 0.0
            pose.pose.orientation = q
            path.poses.append(pose)

        self.pub.publish(path)
        self.get_logger().info(
            f'Published path: robot=({tx:.2f},{ty:.2f}) yaw={math.degrees(yaw):.1f}deg '
            f'points=[{self.point_spacing}m..{self.point_spacing * self.num_points}m]',
            throttle_duration_sec=5.0
        )

    @staticmethod
    def quaternion_to_yaw(x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)


def main():
    rclpy.init()
    node = FakeGlobalPlanPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()