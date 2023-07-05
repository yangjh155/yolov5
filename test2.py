# import cv2
# import socket
# import struct
# import numpy as np
# from detect import MyDetect


# # 创建一个OpenCV视频显示窗口
# cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

# # 创建一个TCP套接字
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# # 绑定到本地IP地址和端口号
# sock.bind(("0.0.0.0", 8888))

# # 监听传入的连接请求
# sock.listen(1)

# # 等待传入的连接请求
# conn, addr = sock.accept()

# # 接收视频帧数据并显示
# while True:
#     # 接收图像大小
#     size_data = conn.recv(4)
#     size = struct.unpack("i", size_data)[0]
    
#     # 接收图像数据
#     jpeg_data = b""
#     while len(jpeg_data) < size:
#         data = conn.recv(size - len(jpeg_data))
#         if not data:
#             break
#         jpeg_data += data
    
#     # 将JPEG数据解码为OpenCV图像格式
#     frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
    
#     cv2.imwrite('test.jpg', frame)
#     MyDetect(source='test.jpg')
    
    # 显示图像
    # cv2.imshow("Video", frame)
    # cv2.waitKey(500)

