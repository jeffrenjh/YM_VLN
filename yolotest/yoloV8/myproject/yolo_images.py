import cv2
import time
import numpy as np
import math
import os
import argparse
from pathlib import Path
from ultralytics import YOLO
 
# 加载 YOLOv8 模型
model = YOLO("/home/nvidia/huangjie/YM_VLN/yolotest/yoloV8/models/yolov8x.pt")

def process_image_pair(color_path, depth_path, output_path=None):
    """
    处理一对 RGB 和 Depth 图像
    """
    # 读取图像
    color_image = cv2.imread(color_path)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    if color_image is None or depth_image is None:
        print(f"[ERROR] 无法读取图像: {color_path} 或 {depth_path}")
        return None
    
    # 使用 YOLOv8 进行目标检测
    results = model.predict(color_image, conf=0.5)
    annotated_frame = results[0].plot()
    detected_boxes = results[0].boxes.xyxy  # 获取边界框坐标
    
    point_cloud_data_all = []
    
    for i, box in enumerate(detected_boxes):
        x1, y1, x2, y2 = map(int, box)  # 获取边界框坐标
 
        # 计算步长
        xrange = max(1, math.ceil(abs((x1 - x2) / 30)))
        yrange = max(1, math.ceil(abs((y1 - y2) / 30)))
 
        point_cloud_data = []
 
        # 获取范围内点的深度坐标
        for x_position in range(x1, x2, xrange):
            for y_position in range(y1, y2, yrange):
                if 0 <= x_position < depth_image.shape[1] and 0 <= y_position < depth_image.shape[0]:
                    depth_value = depth_image[y_position, x_position]
                    point_cloud_data.append(f"({x_position}, {y_position}, {depth_value}) ")
 
        point_cloud_data_all.extend(point_cloud_data)
 
        # 显示中心点坐标
        ux = int((x1 + x2) / 2)
        uy = int((y1 + y2) / 2)
        
        if 0 <= ux < depth_image.shape[1] and 0 <= uy < depth_image.shape[0]:
            center_depth = depth_image[uy, ux]
            formatted_coordinate = f"({ux}, {uy}, D:{center_depth})"
 
            cv2.circle(annotated_frame, (ux, uy), 4, (255, 255, 255), 5)  # 标出中心点
            cv2.putText(annotated_frame, formatted_coordinate, (ux + 20, uy + 10), 0, 0.6,
                        [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)  # 标出坐标
    
    # 保存点云数据
    if output_path and point_cloud_data_all:
        os.makedirs(output_path, exist_ok=True)
        base_name = Path(color_path).stem
        point_cloud_file = os.path.join(output_path, f"{base_name}_pointcloud.txt")
        with open(point_cloud_file, "w") as file:
            file.write(f"Image: {color_path}\n")
            file.write(f"Time: {time.time()}\n")
            file.write(" ".join(point_cloud_data_all))
        print(f"[INFO] 点云数据已保存: {point_cloud_file}")
    
    return annotated_frame

def process_dataset(input_path, output_path=None, show_preview=True):
    """
    批量处理数据集中的图像
    """
    # 查找所有 color 图像
    color_files = sorted(Path(input_path).glob("*_color.jpg"))
    
    if not color_files:
        print(f"[ERROR] 在 {input_path} 中未找到图像文件")
        return
    
    print(f"[INFO] 找到 {len(color_files)} 张图像")
    
    for color_path in color_files:
        # 构建对应的 depth 图像路径
        depth_path = str(color_path).replace("_color.jpg", "_depth.png")
        
        if not os.path.exists(depth_path):
            print(f"[WARN] 未找到对应的深度图: {depth_path}")
            continue
        
        print(f"[INFO] 处理: {color_path.name}")
        
        # 处理图像对
        annotated_frame = process_image_pair(str(color_path), depth_path, output_path)
        
        if annotated_frame is not None:
            # 保存结果
            if output_path:
                result_path = os.path.join(output_path, f"{color_path.stem}_result.jpg")
                cv2.imwrite(result_path, annotated_frame)
                print(f"[INFO] 结果已保存: {result_path}")
            
            # 显示预览
            if show_preview:
                cv2.imshow('YOLOv8 Detection', annotated_frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    print("[INFO] 用户退出")
                    break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='从本地图像进行 YOLOv8 检测')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入图像文件夹路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='结果保存路径（可选）')
    parser.add_argument('--no-preview', action='store_true',
                        help='不显示预览窗口')
    args = parser.parse_args()
    
    # 处理数据集
    process_dataset(
        input_path=args.input,
        output_path=args.output,
        show_preview=not args.no_preview
    )
    
    print("[INFO] 处理完成")