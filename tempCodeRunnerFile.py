# Get frameset of color and depth
            # frames = pipeline.wait_for_frames()
            # # frames.get_depth_frame() is a 640x360 depth image

            # bg_removed, depth_image = realsense_alien_clip(frames, depth_scale, clip_max_dist = float('inf'), clip_min_dist = 0)
            
            # # fake color of depth
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            # # 数据放入yolo得到裁切后的值
            # yolo_result, pos_list = MyDetect(bg_removed)


            # if yolo_result is not None:
            #     # 左-绿
            #     cv2.line(bg_removed, (pos_list[0], pos_list[2]), (pos_list[0], pos_list[3]), (0, 255, 0), 10)
            #     # 右-红
            #     cv2.line(bg_removed, (pos_list[1], pos_list[2]), (pos_list[1], pos_list[3]), (0, 0, 255), 10)
            #     # 上-蓝
            #     cv2.line(bg_removed, (pos_list[0], pos_list[2]), (pos_list[1], pos_list[2]), (255, 0, 0), 10)
            #     # 下-白
            #     cv2.line(bg_removed, (pos_list[0], pos_list[3]), (pos_list[1], pos_list[3]), (255, 255, 255), 10)
                
            #     # 根据裁切后的值进行角点检测，返回四边形的四个角点的相对位置（相对裁切图片而言）[height, width]
            #     corner_pos_list = color_edge(yolo_result) # left_top, right_top, left_bottom, right_bottom
            #     # 对得到的边界进行坐标转换
            #     # 集合起来坐标变换
            #     if corner_pos_list is None:
            #         continue
            #     for corner_pos in corner_pos_list:
            #         corner_pos[0] += pos_list[2]
            #         corner_pos[1] += pos_list[0]
            #     print(corner_pos_list)
            #     # 画出角点
            #     for corner_pos in corner_pos_list:
            #         cv2.circle(bg_removed, (corner_pos[1], corner_pos[0]), 5, (0, 0, 255), -1)
                
            # # combine clipped_image and depth
            # images = np.hstack(( bg_removed, depth_colormap))
            
            # cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            # cv2.imshow('Align Example', images)