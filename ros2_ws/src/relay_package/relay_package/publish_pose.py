import rclpy
from rclpy.node import Node

from tf2_ros import Buffer, PoseStamped, TransformListener
from tf2_ros import TransformException


class CameraPoseNode(Node):

    def __init__(self):
        super().__init__("camera_pose_node")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.publisher = self.create_publisher(PoseStamped, "camera_pose", 10)

        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):

        try:
            transform = self.tf_buffer.lookup_transform(
                "odom", "camera_link", rclpy.time.Time()  # target frame  # source frame
            )

            t = transform.transform.translation
            q = transform.transform.rotation
            
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            # Correct Way 1: Use .to_msg()
            # msg.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = "odom"
            pose.pose.position.x = t.x
            pose.pose.position.y = t.y
            pose.pose.position.z = t.z
            pose.pose.orientation.x = q.x
            pose.pose.orientation.y = q.y
            pose.pose.orientation.z = q.z
            pose.pose.orientation.w = q.w
            self.publisher.publish(pose)

        except TransformException as ex:
            self.get_logger().warn(str(ex))


def main():
    rclpy.init()

    node = CameraPoseNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
