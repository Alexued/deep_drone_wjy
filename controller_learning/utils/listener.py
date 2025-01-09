#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud

# 标志位，确保只打印一次数据
printed = False

def feature_callback(msg):
    """
    回调函数，处理接收到的 /feature_tracker/feature 消息。
    """
    global printed

    # 如果已经打印过数据，直接返回
    if printed:
        return

    # 打印消息的基本信息
    print("Received PointCloud message:")
    print(f"Header: {msg.header}")
    print(f"Number of points: {len(msg.points)}")
    print(f"Number of channels: {len(msg.channels)}")

    # 打印每个通道的名称和值
    for i, channel in enumerate(msg.channels):
        print(f"\nChannel {i}:")
        print(f"  Name: {channel.name}")
        print(f"  Values: {channel.values}")

    # 打印每个点的坐标
    print("\nPoints:")
    for j, point in enumerate(msg.points):
        print(f"  Point {j}: x={point.x}, y={point.y}, z={point.z}")

    # 设置标志位为 True，表示已经打印过数据
    printed = True

def listener():
    """
    初始化 ROS 节点并订阅 /feature_tracker/feature 话题。
    """
    # 初始化节点
    rospy.init_node('feature_listener', anonymous=True)

    # 订阅话题
    rospy.Subscriber("/vins_estimator/left_features", PointCloud, feature_callback)

    # 保持节点运行
    print("Listening to /feature_tracker/feature...")
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass