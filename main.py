import pyrealsense2 as rs
import numpy as np
import cv2
from realsense_test import realsense_alien_clip
from MyDetect import MyDetect
from Plane_ransac import Plane_RANSAC


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
    
    Plane = None
    
    try:
        while True:         
            frames = pipeline.wait_for_frames()  #等待获取图像帧
            bg_removed, depth_image, aligned_depth_frame = realsense_alien_clip(frames, depth_scale, clip_max_dist = float('inf'), clip_min_dist = 0)
             # fake color of depth
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            # yolo detect large square
            yolo_result, pos_list, Note_pos_list = MyDetect(bg_removed)
            if yolo_result is None:
                print("Antenna not detected")
                continue
            print("Antenna detected")
            # get the distance of every point of the whole picture
            points = np.zeros((yolo_result.shape[0], yolo_result.shape[1], 3))
            for y in range(yolo_result.shape[0]):
                for x in range(yolo_result.shape[1]):
                    # pos_3d[x, y] = rs.rs2_deproject_pixel_to_point(depth_intrin, [x + pos_list[2], y+ pos_list[0]], depth_image[x + pos_list[2], y+ pos_list[0]])
                    x_ = x + pos_list[0]
                    y_ = y + pos_list[2]
                    dis_ = aligned_depth_frame.get_distance(x_, y_)
                    points[y, x] = rs.rs2_deproject_pixel_to_point(depth_intrin, [x_, y_], dis_)
            # pos_3d *= depth_scale
            
            if Note_pos_list is None:
                print("Note not detected")
                continue
            else:
                print("Note detected")
            # 可信内点坐标
            inliers_points = points[Note_pos_list[2]:Note_pos_list[3], Note_pos_list[0]:Note_pos_list[1]]
            
            # 求平面坐标，输入pos_3d
            Res, mask= Plane_RANSAC(inliers_points)
            if Plane is None:
                Plane = np.asarray([Res]).reshape(1,8)
            else:
                Plane = np.append(Plane, np.reshape(Res, (1,8)), axis=0)
            print(f"{Plane.shape[0]} planes is saved")

            # 显示图像
            cv2.imshow('mask',mask)  #显示彩色图像
            key = cv2.waitKey(1)
            
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                pipeline.stop()
                np.savetxt('Plane0_2.txt', Plane, fmt='%.6f')
                break
        cv2.destroyAllWindows()
    finally:
        pipeline.stop()
            





