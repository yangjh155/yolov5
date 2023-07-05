#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from detect import MyDetect

def main():
    # 创建一个CvBridge对象
    bridge = CvBridge()

    # 定义回调函数，用于处理接收到的图像消息
    def image_callback(msg):
        # 将ROS图像消息转换为OpenCV图像格式
        try:
            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        # 目标识别
        MyDetect(source=cv_image)
        
        # 显示图像
        cv2.imshow("Camera Feed", cv_image)
        cv2.waitKey(1)

    # 初始化ROS节点
    rospy.init_node("camera_subscriber")

    # 创建一个订阅者对象，订阅相机话题
    image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, image_callback)

    # 进入ROS循环
    rospy.spin()
    
if __name__ == "__main__":
    main()