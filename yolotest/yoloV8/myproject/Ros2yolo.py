import cv2
import time
import numpy as np
import math
import argparse
from ultralytics import YOLO
import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import rosbag2_py
from rosidl_runtime_py.utilities import get_message

# 加载 YOLOv8 模型
model = YOLO("/home/nvidia/huangjie/YM_VLN/yolotest/yoloV8/models/yolov8x.pt")


# 从ROS2 bag读取数据的类
class ROS2BagReader:
    def __init__(self, bag_path):
        self.bridge = CvBridge()
        self.bag_path = bag_path

        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions('', '')
        self.reader = rosbag2_py.SequentialReader()
        self.reader.open(storage_options, converter_options)

        self.color_topic = '/camera/color/image_raw'
        self.depth_topic = '/camera/depth/image_rect_raw'
        self.camera_info_topic = '/camera/color/camera_info'

        self.camera_info = None
        self.depth_scale = 0.001  # 默认深度比例
        
        # 获取 topic 类型映射
        self.topic_types = {}
        topic_types_list = self.reader.get_all_topics_and_types()
        for topic_metadata in topic_types_list:
            self.topic_types[topic_metadata.name] = topic_metadata.type
        
        print("Available topics in bag file:")
        for topic, msg_type in self.topic_types.items():
            print(f"  {topic}: {msg_type}")

    def get_next_frame(self):
        """从bag文件读取下一帧"""
        color_image = None
        depth_image = None

        while self.reader.has_next():
            topic, data, timestamp = self.reader.read_next()
            
            # 根据 topic 获取消息类型
            if topic not in self.topic_types:
                continue
                
            msg_type = get_message(self.topic_types[topic])
            msg = deserialize_message(data, msg_type)

            if topic == self.color_topic:
                color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            elif topic == self.depth_topic:
                depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            elif topic == self.camera_info_topic and self.camera_info is None:
                self.camera_info = msg

            if color_image is not None and depth_image is not None:
                return color_image, depth_image, self.camera_info

        return None, None, None

    def get_camera_intrinsics(self):
        """从camera_info获取相机内参"""
        if self.camera_info:
            class Intrinsics:
                def __init__(self, info):
                    self.fx = info.k[0]
                    self.fy = info.k[4]
                    self.ppx = info.k[2]
                    self.ppy = info.k[5]
                    self.width = info.width
                    self.height = info.height
            return Intrinsics(self.camera_info)
        return None


def get_3d_coordinate_from_depth(depth_pixel, depth_image, intrinsics, depth_scale=0.001):
    """从深度图计算3D坐标"""
    x, y = depth_pixel
    if 0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]:
        depth_value = depth_image[y, x] * depth_scale  # 转换为米

        # 使用相机内参计算3D坐标
        z = depth_value
        x_3d = (x - intrinsics.ppx) * z / intrinsics.fx
        y_3d = (y - intrinsics.ppy) * z / intrinsics.fy

        return depth_value, [x_3d, y_3d, z]
    return 0, [0, 0, 0]


# 初始化 FPS 计算
fps = 0
frame_count = 0
start_time = time.time()

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='YOLOv8 with ROS2 bag')
parser.add_argument('--bag', type=str, required=True, help='Path to ROS2 bag file')
args = parser.parse_args()

# 初始化bag读取器
print(f"Using ROS2 bag: {args.bag}")
rclpy.init()
bag_reader = ROS2BagReader(args.bag)

try:
    while True:
        # 从bag文件读取
        color_image, depth_image, camera_info = bag_reader.get_next_frame()
        if color_image is None:
            print("End of bag file")
            break

        intr = bag_reader.get_camera_intrinsics()
        if intr is None:
            continue

        if not depth_image.any() or not color_image.any():
            continue

        # 获取当前时间
        time1 = time.time()

        # 将图像转为numpy数组
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
            depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap))

        # 使用 YOLOv8 进行目标检测
        results = model.predict(color_image, conf=0.5)
        annotated_frame = results[0].plot()
        detected_boxes = results[0].boxes.xyxy  # 获取边界框坐标

        for i, box in enumerate(detected_boxes):
            x1, y1, x2, y2 = map(int, box)  # 获取边界框坐标

            # 计算步长
            xrange = max(1, math.ceil(abs((x1 - x2) / 30)))
            yrange = max(1, math.ceil(abs((y1 - y2) / 30)))

            point_cloud_data = []

            # 获取范围内点的三维坐标
            for x_position in range(x1, x2, xrange):
                for y_position in range(y1, y2, yrange):
                    depth_pixel = [x_position, y_position]
                    dis, camera_coordinate = get_3d_coordinate_from_depth(
                        depth_pixel, depth_image, intr)
                    point_cloud_data.append(f"{camera_coordinate} ")

            # 一次性写入所有数据
            # with open("point_cloud_data.txt", "a") as file:
            #     file.write(f"\nTime: {time.time()}\n")
            #     file.write(" ".join(point_cloud_data))

            # 显示中心点坐标
            ux = int((x1 + x2) / 2)
            uy = int((y1 + y2) / 2)
            dis, camera_coordinate = get_3d_coordinate_from_depth(
                [ux, uy], depth_image, intr)

            formatted_camera_coordinate = f"({camera_coordinate[0]:.2f}, {camera_coordinate[1]:.2f}, {camera_coordinate[2]:.2f})"

            cv2.circle(annotated_frame, (ux, uy), 4, (255, 255, 255), 5)  # 标出中心点
            cv2.putText(annotated_frame, formatted_camera_coordinate, (ux + 20, uy + 10), 0, 1,
                        [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)  # 标出坐标

        # 计算 FPS
        frame_count += 1
        time2 = time.time()
        fps = int(1 / (time2 - time1))
        
        # 显示 FPS
        cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

        # 显示结果
        cv2.imshow('YOLOv8 ROS2 Bag', annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # 按空格键暂停/继续
        elif key == ord(' '):
            cv2.waitKey(0)

finally:
    rclpy.shutdown()
    cv2.destroyAllWindows()