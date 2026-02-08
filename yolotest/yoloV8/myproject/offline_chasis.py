"""
测试脚本：使用 data 文件夹中的 RGBD 图像测试 YOLO 检测和坐标转换
功能增强：添加全局坐标系转换
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


def chassis_to_global(chassis_coord, base_to_odom_trans=None, base_to_odom_yaw=None, 
                      odom_to_map_trans=None, odom_to_map_yaw=None):
    """
    将底盘坐标系的坐标转换到全局坐标系（map）
    
    变换链: base_link -> odom -> map
    
    参数:
        chassis_coord: numpy数组 [x, y, z] 底盘坐标系下的坐标
        base_to_odom_trans: base_link到odom的平移向量 [x, y, z]，默认 [-5.270, -1.117, 0]
        base_to_odom_yaw: base_link到odom的偏航角（弧度），默认 2.986
        odom_to_map_trans: odom到map的平移向量 [x, y, z]，默认 [0.540, 1.102, 0]
        odom_to_map_yaw: odom到map的偏航角（弧度），默认 -0.059
    
    返回:
        全局坐标系下的坐标 [x, y, z]
    """
    # 使用默认TF参数（如果未提供）
    if base_to_odom_trans is None:
        base_to_odom_trans = np.array([-5.270, -1.117, 0.0])
    if base_to_odom_yaw is None:
        base_to_odom_yaw = 2.986
    if odom_to_map_trans is None:
        odom_to_map_trans = np.array([0.540, 1.102, 0.0])
    if odom_to_map_yaw is None:
        odom_to_map_yaw = -0.059
    
    # 第一步: base_link -> odom
    cos_yaw1 = np.cos(base_to_odom_yaw)
    sin_yaw1 = np.sin(base_to_odom_yaw)
    x_odom = cos_yaw1 * chassis_coord[0] - sin_yaw1 * chassis_coord[1] + base_to_odom_trans[0]
    y_odom = sin_yaw1 * chassis_coord[0] + cos_yaw1 * chassis_coord[1] + base_to_odom_trans[1]
    z_odom = chassis_coord[2] + base_to_odom_trans[2]
    
    # 第二步: odom -> map
    cos_yaw2 = np.cos(odom_to_map_yaw)
    sin_yaw2 = np.sin(odom_to_map_yaw)
    x_map = cos_yaw2 * x_odom - sin_yaw2 * y_odom + odom_to_map_trans[0]
    y_map = sin_yaw2 * x_odom + cos_yaw2 * y_odom + odom_to_map_trans[1]
    z_map = z_odom + odom_to_map_trans[2]
    
    return np.array([x_map, y_map, z_map])


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
    
    # 存储检测结果
    detection_results = []
    
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
            
            # 将底盘坐标系转换为全局坐标系
            global_coord = chassis_to_global(chassis_coord)
            
            print(f"\n目标 {i+1}: {class_name} (置信度: {conf:.2f})")
            print(f"  边界框: ({x1}, {y1}) -> ({x2}, {y2})")
            print(f"  中心点像素坐标: ({ux}, {uy})")
            print(f"  深度值: {depth_value} mm ({depth_value/1000:.3f} m)")
            print(f"  相机坐标系: X={camera_coord[0]:.3f}m, Y={camera_coord[1]:.3f}m, Z={camera_coord[2]:.3f}m")
            print(f"  底盘坐标系: X={chassis_coord[0]:.3f}m, Y={chassis_coord[1]:.3f}m, Z={chassis_coord[2]:.3f}m")
            print(f"  全局坐标系: X={global_coord[0]:.3f}m, Y={global_coord[1]:.3f}m, Z={global_coord[2]:.3f}m")
            
            # 保存检测结果
            detection_result = {
                'id': i + 1,
                'class': class_name,
                'confidence': float(conf),
                'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                'pixel_coords': {'u': ux, 'v': uy},
                'depth_mm': int(depth_value),
                'camera_coords': {
                    'x': float(camera_coord[0]),
                    'y': float(camera_coord[1]),
                    'z': float(camera_coord[2])
                },
                'chassis_coords': {
                    'x': float(chassis_coord[0]),
                    'y': float(chassis_coord[1]),
                    'z': float(chassis_coord[2])
                },
                'global_coords': {
                    'x': float(global_coord[0]),
                    'y': float(global_coord[1]),
                    'z': float(global_coord[2])
                }
            }
            detection_results.append(detection_result)
            
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
            
            # 标注全局坐标系坐标
            global_text = f"Global: ({global_coord[0]:.2f}, {global_coord[1]:.2f}, {global_coord[2]:.2f})m"
            cv2.putText(annotated_frame, global_text, (ux + 20, uy + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 255], 
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
    
    # 保存检测结果到JSON文件
    json_output_path = os.path.join(project_dir, 'detection_result.json')
    results_data = {
        'source_images': {
            'color': color_image_path,
            'depth': depth_image_path
        },
        'camera_intrinsics': camera_intrinsics,
        'calibration_params': {
            'x': calibration.x,
            'y': calibration.y,
            'z': calibration.z,
            'yaw': calibration.yaw,
            'pitch': calibration.pitch,
            'roll': calibration.roll
        },
        'detections': detection_results
    }
    
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)
    print(f"检测结果已保存到: {json_output_path}")
    
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
