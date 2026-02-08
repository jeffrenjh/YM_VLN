"""
测试脚本：使用 data 文件夹中的 RGBD 图像测试 YOLO 检测和坐标转换
功能增强：添加全局坐标系转换，支持批量处理和指定目标筛选
"""

import argparse
import csv
import cv2
import json
import numpy as np
import os
import re
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


def find_rgbd_pairs(data_dir):
    """查找文件夹中匹配的 RGB 和深度图像对"""
    color_pattern = re.compile(r"^color_(\d+)\.png$")
    depth_pattern = re.compile(r"^depth_(\d+)\.png$")

    color_map = {}
    depth_map = {}

    for name in os.listdir(data_dir):
        color_match = color_pattern.match(name)
        depth_match = depth_pattern.match(name)
        if color_match:
            color_map[color_match.group(1)] = os.path.join(data_dir, name)
        elif depth_match:
            depth_map[depth_match.group(1)] = os.path.join(data_dir, name)

    shared_ids = sorted(color_map.keys() & depth_map.keys(), key=lambda x: int(x))
    return [(frame_id, color_map[frame_id], depth_map[frame_id]) for frame_id in shared_ids]


def parse_target_classes(targets_text):
    """解析目标类别字符串"""
    if not targets_text:
        return None
    targets = [item.strip() for item in targets_text.split(",") if item.strip()]
    return set(targets) if targets else None


def main():
    parser = argparse.ArgumentParser(description="批量处理 RGBD 图像并进行 YOLO 检测和坐标转换")
    parser.add_argument(
        "--data-dir",
        default="/home/nvidia/huangjie/YM_VLN/test/YM_VLN/yolotest/yoloV8/myproject/data/0106task1",
        help="包含 color_*.png 和 depth_*.png 的文件夹路径",
    )
    parser.add_argument(
        "--model-path",
        default="/home/nvidia/huangjie/YM_VLN/test/YM_VLN/yolotest/yoloV8/models/yolov8l.pt",
        help="YOLO 模型路径",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument(
        "--targets",
        default="",
        help="指定需要检测的物体名称（逗号分隔），为空则检测所有",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/nvidia/huangjie/YM_VLN/test/YM_VLN/yolotest/yoloV8/myproject/output",
        help="输出目录 (默认为脚本所在目录)",
    )
    parser.add_argument(
        "--save-annotated",
        action="store_true",
        help="保存每一帧的标注结果图像",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="实时显示检测结果",
    )

    args = parser.parse_args()

    # 设置路径
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir
    output_dir = args.output_dir or project_dir
    os.makedirs(output_dir, exist_ok=True)

    # 检查数据目录
    if not os.path.isdir(data_dir):
        print(f"错误：找不到数据文件夹 {data_dir}")
        return

    rgbd_pairs = find_rgbd_pairs(data_dir)
    if not rgbd_pairs:
        print(f"错误：未找到匹配的 RGBD 图像对（color_*.png / depth_*.png），目录: {data_dir}")
        return
    
    print(f"找到 {len(rgbd_pairs)} 对图像")

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

    target_classes = parse_target_classes(args.targets)
    if target_classes:
        print(f"\n仅记录这些目标类别: {sorted(target_classes)}")
    else:
        print("\n记录所有检测到的目标类别")

    # 加载 YOLO 模型
    print(f"\n正在加载 YOLO 模型: {args.model_path} ...")
    if not os.path.exists(args.model_path):
        print(f"错误：找不到模型文件 {args.model_path}")
        return
        
    model = YOLO(args.model_path)

    detection_results = []

    for frame_index, (frame_id, color_image_path, depth_image_path) in enumerate(rgbd_pairs, start=1):
        print(f"\n处理帧 {frame_index}/{len(rgbd_pairs)}: {frame_id}")

        color_image, depth_image = load_rgbd_images(color_image_path, depth_image_path)
        if color_image is None or depth_image is None:
            print(f"  警告：无法读取图像，跳过 {frame_id}")
            continue

        results = model.predict(color_image, conf=args.conf, verbose=False)
        annotated_frame = results[0].plot()
        detected_boxes = results[0].boxes.xyxy
        class_ids = results[0].boxes.cls
        confidences = results[0].boxes.conf

        print(f"  检测到 {len(detected_boxes)} 个目标")

        for i, (box, class_id, conf) in enumerate(zip(detected_boxes, class_ids, confidences), start=1):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(class_id)]

            # 筛选指定物体
            if target_classes and class_name not in target_classes:
                continue

            # 计算目标中心点
            ux = int((x1 + x2) / 2)
            uy = int((y1 + y2) / 2)

            # 获取中心点的深度值
            if 0 <= ux < depth_image.shape[1] and 0 <= uy < depth_image.shape[0]:
                depth_value = depth_image[uy, ux]
                
                # 如果深度无效（0），尝试搜索邻域
                if depth_value == 0:
                     # 简单的 3x3 邻域搜索非零最小值
                     roi = depth_image[max(0, uy-1):min(uy+2, depth_image.shape[0]), 
                                       max(0, ux-1):min(ux+2, depth_image.shape[1])]
                     valid_depths = roi[roi > 0]
                     if len(valid_depths) > 0:
                         depth_value = np.median(valid_depths)

                if depth_value > 0:
                    # 将像素坐标转换为相机坐标系
                    camera_coord = pixel_to_camera_coordinate(ux, uy, depth_value, camera_intrinsics)

                    # 将相机坐标系转换为小车底盘坐标系
                    chassis_coord = calibration.camera_to_chassis(camera_coord)

                    # 将底盘坐标系转换为全局坐标系
                    global_coord = chassis_to_global(chassis_coord)

                    print(f"  目标 {i}: {class_name} (置信度: {conf:.2f})")

                    detection_result = {
                        'frame_id': frame_id,
                        'color_image': color_image_path,
                        'depth_image': depth_image_path,
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

                    if args.save_annotated or args.show:
                        # 绘制中心点
                        cv2.circle(annotated_frame, (ux, uy), 4, (255, 255, 255), 5)
                        # 绘制坐标文本
                        global_text = (
                            f"Global: ({global_coord[0]:.2f}, {global_coord[1]:.2f}, {global_coord[2]:.2f})m"
                        )
                        cv2.putText(
                            annotated_frame,
                            global_text,
                            (ux + 20, uy + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            [255, 0, 255],
                            thickness=1,
                            lineType=cv2.LINE_AA,
                        )

        if args.save_annotated:
            output_path = os.path.join(output_dir, f"annotated_{frame_id}.jpg")
            cv2.imwrite(output_path, annotated_frame)

        if args.show:
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            cv2.imshow('YOLO Detection with Coordinates', annotated_frame)
            cv2.imshow('Depth Image', depth_colormap)
            if cv2.waitKey(1) & 0xFF == 27: # Esc key
                print("  已手动终止显示")
                break

    if args.show:
        cv2.destroyAllWindows()

    # 保存检测结果到JSON文件（扁平列表，便于数据处理） - 名称更通用
    json_output_path = os.path.join(output_dir, 'detection_results.json')
    results_data = {
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
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"\\n检测结果 JSON 已保存到: {json_output_path}")

    # 额外导出 CSV，便于表格处理
    csv_output_path = os.path.join(output_dir, 'detection_results.csv')
    csv_fields = [
        'frame_id', 'class', 'confidence',
        'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
        'pixel_u', 'pixel_v', 'depth_mm',
        'camera_x', 'camera_y', 'camera_z',
        'chassis_x', 'chassis_y', 'chassis_z',
        'global_x', 'global_y', 'global_z',
        'color_image', 'depth_image'
    ]
    
    if detection_results:
        with open(csv_output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for item in detection_results:
                writer.writerow({
                    'frame_id': item['frame_id'],
                    'class': item['class'],
                    'confidence': item['confidence'],
                    'bbox_x1': item['bbox']['x1'],
                    'bbox_y1': item['bbox']['y1'],
                    'bbox_x2': item['bbox']['x2'],
                    'bbox_y2': item['bbox']['y2'],
                    'pixel_u': item['pixel_coords']['u'],
                    'pixel_v': item['pixel_coords']['v'],
                    'depth_mm': item['depth_mm'],
                    'camera_x': item['camera_coords']['x'],
                    'camera_y': item['camera_coords']['y'],
                    'camera_z': item['camera_coords']['z'],
                    'chassis_x': item['chassis_coords']['x'],
                    'chassis_y': item['chassis_coords']['y'],
                    'chassis_z': item['chassis_coords']['z'],
                    'global_x': item['global_coords']['x'],
                    'global_y': item['global_coords']['y'],
                    'global_z': item['global_coords']['z'],
                    'color_image': item['color_image'],
                    'depth_image': item['depth_image'],
                })
        print(f"检测结果 CSV 已保存到: {csv_output_path}")
    else:
        print("\\n未检测到任何目标，未生成 CSV 文件。")

if __name__ == "__main__":
    main()
