"""
测试脚本：使用 data 文件夹中的 RGBD 图像测试 YOLO 检测和坐标转换
"""

import cv2
import numpy as np
import json
import os
from ultralytics import YOLO
from calibration import CameraCalibration


def load_rgbd_images(color_path, depth_path):
    """加载 RGB 和深度图像"""
    color_image = cv2.imread(color_path)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # 读取深度图（16位）
    return color_image, depth_image


def pixel_to_camera_coordinate(u, v, depth_mm, camera_intrinsics):
    """
    将像素坐标和深度值转换为相机坐标系下的3D坐标
    
    参数:
        u, v: 像素坐标
        depth_mm: 深度值（毫米）
        camera_intrinsics: 相机内参字典 {'fx', 'fy', 'cx', 'cy'}
    
    返回:
        相机坐标系下的 [x, y, z] 坐标（单位：米）
    """
    fx = camera_intrinsics['fx']
    fy = camera_intrinsics['fy']
    cx = camera_intrinsics['cx']
    cy = camera_intrinsics['cy']
    
    # 深度值转换为米
    depth_m = depth_mm / 1000.0
    
    # 像素坐标转相机坐标（RealSense 坐标系）
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m
    
    return np.array([x, y, z])


def get_default_camera_intrinsics():
    """
    获取 RealSense D435 相机的默认内参（848x480 分辨率）
    如果有实际标定的内参，应该从文件中读取
    """
    return {
        'fx': 615.0,  # 焦距 x
        'fy': 615.0,  # 焦距 y
        'cx': 424.0,  # 主点 x
        'cy': 240.0,  # 主点 y
        'width': 848,
        'height': 480
    }


def main():
    # 设置路径
    project_dir = os.path.dirname(os.path.abspath(__file__))
    #data_dir = os.path.join(project_dir, 'data')
    data_dir = '/home/nvidia/huangjie/YM_VLN/test/YM_VLN/yolotest/yoloV8/myproject/data/0106task1'
    color_image_path = os.path.join(data_dir, 'color_000001.png')
    depth_image_path = os.path.join(data_dir, 'depth_000001.png')
    
    # 检查文件是否存在
    if not os.path.exists(color_image_path):
        print(f"错误：找不到彩色图像文件 {color_image_path}")
        return
    if not os.path.exists(depth_image_path):
        print(f"错误：找不到深度图像文件 {depth_image_path}")
        return
    
    # 加载图像
    print("正在加载 RGBD 图像...")
    color_image, depth_image = load_rgbd_images(color_image_path, depth_image_path)
    print(f"彩色图像尺寸: {color_image.shape}")
    print(f"深度图像尺寸: {depth_image.shape}")
    
    # 获取相机内参（实际使用时应该从标定文件读取）
    camera_intrinsics = get_default_camera_intrinsics()
    print(f"\n相机内参: fx={camera_intrinsics['fx']}, fy={camera_intrinsics['fy']}, "
          f"cx={camera_intrinsics['cx']}, cy={camera_intrinsics['cy']}")
    
    # 初始化相机标定
    # 假设相机安装在小车前方 30cm，高度 50cm，向下俯仰 30 度
    print("\n初始化相机标定...")
    calibration = CameraCalibration(
        x=0.3,      # 相机在小车前方 30cm
        y=0.0,      # 相机在小车中心线上
        z=0.5,      # 相机高度 50cm
        yaw=0.0,    # 没有偏航
        pitch=np.radians(-30),  # 向下俯仰 30 度
        roll=0.0    # 没有横滚
    )
    print(calibration)
    
    # 加载 YOLO 模型
    print("\n正在加载 YOLO 模型...")
    # 尝试多个可能的模型路径
    #model_paths = "/home/nvidia/huangjie/YM_VLN/yolotest/yoloV8/models/yolov8x.pt"
    
    model = YOLO("/home/nvidia/huangjie/YM_VLN/test/YM_VLN/yolotest/yoloV8/models/yolov8l.pt")
    
    # 进行目标检测
    print("\n正在进行目标检测...")
    results = model.predict(color_image, conf=0.5)
    annotated_frame = results[0].plot()
    detected_boxes = results[0].boxes.xyxy
    class_ids = results[0].boxes.cls
    confidences = results[0].boxes.conf
    
    print(f"检测到 {len(detected_boxes)} 个目标")
    
    # 处理每个检测到的目标
    for i, (box, class_id, conf) in enumerate(zip(detected_boxes, class_ids, confidences)):
        x1, y1, x2, y2 = map(int, box)
        class_name = model.names[int(class_id)]
        
        # 计算目标中心点
        ux = int((x1 + x2) / 2)
        uy = int((y1 + y2) / 2)
        
        # 获取中心点的深度值
        if 0 <= ux < depth_image.shape[1] and 0 <= uy < depth_image.shape[0]:
            depth_value = depth_image[uy, ux]
            
            # 将像素坐标转换为相机坐标系
            camera_coord = pixel_to_camera_coordinate(ux, uy, depth_value, camera_intrinsics)
            
            # 将相机坐标系转换为小车底盘坐标系
            chassis_coord = calibration.camera_to_chassis(camera_coord)
            
            print(f"\n目标 {i+1}: {class_name} (置信度: {conf:.2f})")
            print(f"  边界框: ({x1}, {y1}) -> ({x2}, {y2})")
            print(f"  中心点像素坐标: ({ux}, {uy})")
            print(f"  深度值: {depth_value} mm ({depth_value/1000:.3f} m)")
            print(f"  相机坐标系: X={camera_coord[0]:.3f}m, Y={camera_coord[1]:.3f}m, Z={camera_coord[2]:.3f}m")
            print(f"  底盘坐标系: X={chassis_coord[0]:.3f}m, Y={chassis_coord[1]:.3f}m, Z={chassis_coord[2]:.3f}m")
            
            # 在图像上标注坐标信息
            cv2.circle(annotated_frame, (ux, uy), 4, (255, 255, 255), 5)
            
            # 标注相机坐标系坐标
            camera_text = f"Cam: ({camera_coord[0]:.2f}, {camera_coord[1]:.2f}, {camera_coord[2]:.2f})m"
            cv2.putText(annotated_frame, camera_text, (ux + 20, uy + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 255], 
                       thickness=1, lineType=cv2.LINE_AA)
            
            # 标注底盘坐标系坐标
            chassis_text = f"Chassis: ({chassis_coord[0]:.2f}, {chassis_coord[1]:.2f}, {chassis_coord[2]:.2f})m"
            cv2.putText(annotated_frame, chassis_text, (ux + 20, uy + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 
                       thickness=1, lineType=cv2.LINE_AA)
    
    # 显示结果
    print("\n正在显示结果图像...")
    
    # 创建深度图可视化
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03), 
        cv2.COLORMAP_JET
    )
    
    # 显示图像
    cv2.imshow('YOLO Detection with Coordinates', annotated_frame)
    cv2.imshow('Depth Image', depth_colormap)
    
    # 保存结果
    output_path = os.path.join(project_dir, 'detection_result.jpg')
    cv2.imwrite(output_path, annotated_frame)
    print(f"\n结果已保存到: {output_path}")
    
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
