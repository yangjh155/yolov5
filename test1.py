# import cv2
# import socket
# import struct

# # 创建一个OpenCV视频捕获对象
# cap = cv2.VideoCapture(0)

# # 创建一个TCP套接字
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# # 连接到接收端IP地址和端口号
# sock.connect(("127.0.0.1", 8888))

# # 发送视频帧数据
# while True:
#     # 从摄像头读取一帧图像
#     ret, frame = cap.read()
    
#     # 将图像转换为JPEG格式
#     _, jpeg_frame = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    
#     # 将图像大小打包为4字节的二进制数据
#     size = struct.pack("i", len(jpeg_frame))
    
#     # 发送图像大小和图像数据
#     sock.sendall(size)
#     sock.sendall(jpeg_frame.tobytes())
#     cv2.waitKey(1000)
import cv2

# 定义本机摄像头
cap = cv2.VideoCapture(0)

# 定义编码器和输出格式
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# 定义RTSP地址
rtsp_url = r"rtsp://example.com/stream"

# 打开RTSP流
rtsp_out = cv2.VideoWriter(rtsp_url, fourcc, 20.0, (640, 480))

# 读取并输出视频帧
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        out.write(frame)
        rtsp_out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# 清理资源
cap.release()
out.release()
rtsp_out.release()
cv2.destroyAllWindows()

