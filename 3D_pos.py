import pyrealsense2 as rs
import numpy as np
import cv2
from realsense_test import realsense_alien_clip
from edge_contour import color_edge
from MyDetect import MyDetect

if __name__ == "__main__":
    
    pipeline = rs.pipeline()  #定义流程pipeline
    config = rs.config()   #定义配置config

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)
        
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  #配置depth流
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)   #配置color流
    profile = pipeline.start(config)  #流程开始
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align_to = rs.stream.color  #与color流对齐
    align = rs.align(align_to)
    
    Pos = np.zeros((4, 3))  # 4个角点的坐标
    
    try:
        while True:         

            frames = pipeline.wait_for_frames()  #等待获取图像帧
            bg_removed, depth_image, aligned_depth_frame = realsense_alien_clip(frames, depth_scale, clip_max_dist = float('inf'), clip_min_dist = 0)
             # fake color of depth
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            
            yolo_result, pos_list = MyDetect(bg_removed)

            if yolo_result is not None:
                # 左-绿
                cv2.line(bg_removed, (pos_list[0], pos_list[2]), (pos_list[0], pos_list[3]), (0, 255, 0), 10)
                # 右-蓝
                cv2.line(bg_removed, (pos_list[1], pos_list[2]), (pos_list[1], pos_list[3]), (255, 0, 0), 10)
                # 上-红
                cv2.line(bg_removed, (pos_list[0], pos_list[2]), (pos_list[1], pos_list[2]), (0, 0, 255), 10)
                # 下-白
                cv2.line(bg_removed, (pos_list[0], pos_list[3]), (pos_list[1], pos_list[3]), (255, 255, 255), 10)
                
                # 根据裁切后的值进行角点检测，返回四边形的四个角点的相对位置（相对裁切图片而言）[height, width]
                corner_pos_list = color_edge(yolo_result) # left_top, right_top, left_bottom, right_bottom
                # 对得到的边界进行坐标转换
                # 集合起来坐标变换
                if corner_pos_list is None:
                    continue
                for corner_pos in corner_pos_list:
                    corner_pos[0] += pos_list[2]
                    corner_pos[1] += pos_list[0]
                # print(corner_pos_list)
                # 画出角点
                for i, corner_pos in enumerate(corner_pos_list):                    
                    x = corner_pos[1]
                    y = corner_pos[0]
                    dis = aligned_depth_frame.get_distance(x, y)  #（x, y)点的真实深度值
                    #（x, y)点在相机坐标系下的真实值，为一个三维向量。其中camera_coordinate[2]仍为dis，camera_coordinate[0]和camera_coordinate[1]为相机坐标系下的xy真实距离。
                    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], dis)  
                    # 限制小数3位
                    for i in range(len(camera_coordinate)):
                        camera_coordinate[i] = round(camera_coordinate[i], 3)
                    cv2.circle(bg_removed, (x, y), 5, (0, 0, 255), -1)
                    cv2.putText(bg_removed, str(camera_coordinate), (x-70, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
            # combine clipped_image and depth
            # images = np.hstack(( bg_removed, depth_colormap))
            
            # cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            # cv2.imshow('Align Example', images)
            cv2.imshow('RGB image',bg_removed)  #显示彩色图像

            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                pipeline.stop()
                break
        cv2.destroyAllWindows()
    finally:
        pipeline.stop()
