"""
实时 YOLO 检测与坐标标定系统
功能：
1. 实时从 RealSense 相机获取 RGBD 数据
2. 使用 YOLO 进行目标检测
3. 估计目标的 XYZ 坐标（相机坐标系）
4. 将坐标转换到小车底盘坐标系（base_link）
5. 将坐标转换到全局坐标系（map）
6. 可选保存检测结果（图像和坐标数据）
"""

import cv2
import pyrealsense2 as rs
import numpy as np
import time
import json
import os
import argparse
from datetime import datetime
from ultralytics import YOLO
from calibration import CameraCalibration


def chassis_to_global(chassis_coord):
    """
    将底盘坐标系的坐标转换到全局坐标系（map）
    
    变换链: base_link -> odom -> map
    - odom <- base_link: 平移 (-5.270, -1.117, 0), yaw 2.986 rad
    - map <- odom: 平移 (0.540, 1.102, 0), yaw -0.059 rad
    
    参数:
        chassis_coord: numpy数组 [x, y, z] 底盘坐标系下的坐标
    
    返回:
        全局坐标系下的坐标 [x, y, z]
    """
    # TF参数: base_link -> odom
    base_to_odom_trans = np.array([-5.270, -1.117, 0.0])
    base_to_odom_yaw = 2.986
    
    # TF参数: odom -> map
    odom_to_map_trans = np.array([0.540, 1.102, 0.0])
    odom_to_map_yaw = -0.059
    
    # 第一步: base_link -> odom
    # 旋转
    cos_yaw1 = np.cos(base_to_odom_yaw)
    sin_yaw1 = np.sin(base_to_odom_yaw)
    x_odom = cos_yaw1 * chassis_coord[0] - sin_yaw1 * chassis_coord[1] + base_to_odom_trans[0]
    y_odom = sin_yaw1 * chassis_coord[0] + cos_yaw1 * chassis_coord[1] + base_to_odom_trans[1]
    z_odom = chassis_coord[2] + base_to_odom_trans[2]
    
    # 第二步: odom -> map
    # 旋转
    cos_yaw2 = np.cos(odom_to_map_yaw)
    sin_yaw2 = np.sin(odom_to_map_yaw)
    x_map = cos_yaw2 * x_odom - sin_yaw2 * y_odom + odom_to_map_trans[0]
    y_map = sin_yaw2 * x_odom + cos_yaw2 * y_odom + odom_to_map_trans[1]
    z_map = z_odom + odom_to_map_trans[2]
    
    return np.array([x_map, y_map, z_map])


class YOLOCalibrationSystem:
    """实时 YOLO 检测与坐标标定系统"""
    
    def __init__(self, model_path, camera_params=None, calibration_params=None, 
                 save_results=False, output_dir='results'):
        """
        初始化系统
        
        参数:
            model_path: YOLO 模型路径
            camera_params: 相机内参字典 {'fx', 'fy', 'cx', 'cy'}
            calibration_params: 标定参数字典 {'x', 'y', 'z', 'yaw', 'pitch', 'roll'}
            save_results: 是否保存检测结果
            output_dir: 结果保存目录
        """
        # 加载 YOLO 模型
        print(f"正在加载 YOLO 模型: {model_path}")
        #self.model = YOLO(model_path)
        self.model = YOLO("/home/nvidia/huangjie/YM_VLN/test/YM_VLN/yolotest/yoloV8/models/yolov8l.pt")
 
        
        # 设置相机内参（默认值为 RealSense D435 848x480）
        if camera_params is None:
            self.camera_intrinsics = {
                'fx': 615.0,
                'fy': 615.0,
                'cx': 424.0,
                'cy': 240.0,
                'width': 848,
                'height': 480
            }
        else:
            self.camera_intrinsics = camera_params
        
        # 初始化相机标定
        if calibration_params is None:
            # 默认标定参数：相机在小车前方30cm，高度50cm，向下俯仰30度
            calibration_params = {
                'x': 0.3,
                'y': 0.0,
                'z': 0.5,
                'yaw': 0.0,
                'pitch': np.radians(-30),
                'roll': 0.0
            }
        
        self.calibration = CameraCalibration(**calibration_params)
        print("相机标定参数:")
        print(self.calibration)
        
        # 配置 RealSense 相机
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        
        # 启动相机流
        print("正在启动 RealSense 相机...")
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        
        # 保存结果设置
        self.save_results = save_results
        self.output_dir = output_dir
        if self.save_results:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"结果将保存到: {self.output_dir}")
        
        # FPS 计算
        self.fps = 0
        self.frame_count = 0
        
    def get_aligned_images(self):
        """获取对齐的彩色图像和深度图像"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        # 获取相机内参
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        
        # 转换为 numpy 数组
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        return color_image, depth_image, aligned_depth_frame, depth_intrin
    
    def pixel_to_camera_coordinate(self, u, v, depth_mm):
        """
        将像素坐标和深度值转换为相机坐标系下的3D坐标
        
        参数:
            u, v: 像素坐标
            depth_mm: 深度值（毫米）
        
        返回:
            相机坐标系下的 [x, y, z] 坐标（单位：米）
        """
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']
        
        # 深度值转换为米
        depth_m = depth_mm / 1000.0
        
        # 像素坐标转相机坐标
        x = (u - cx) * depth_m / fx
        y = (v - cy) * depth_m / fy
        z = depth_m
        
        return np.array([x, y, z])
    
    def get_3d_camera_coordinate_rs(self, depth_pixel, aligned_depth_frame, depth_intrin):
        """
        使用 RealSense SDK 获取3D坐标（备用方法）
        
        参数:
            depth_pixel: [x, y] 像素坐标
            aligned_depth_frame: 对齐的深度帧
            depth_intrin: 深度相机内参
        
        返回:
            距离和相机坐标
        """
        x, y = depth_pixel
        dis = aligned_depth_frame.get_distance(x, y)  # 获取该像素点对应的深度（米）
        camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
        return dis, camera_coordinate
    
    def save_detection_results(self, annotated_frame, detections_data):
        """
        保存检测结果
        
        参数:
            annotated_frame: 标注后的图像
            detections_data: 检测数据列表
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        # 保存图像
        image_path = os.path.join(self.output_dir, f"detection_{timestamp}.jpg")
        cv2.imwrite(image_path, annotated_frame)
        
        # 保存坐标数据
        json_path = os.path.join(self.output_dir, f"detection_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(detections_data, f, indent=2)
        
        return image_path, json_path
    
    def run(self):
        """运行实时检测系统"""
        print("\n=== 开始实时检测 ===")
        print("按 'q' 退出")
        print("按 's' 保存当前帧检测结果（仅在启用保存功能时）")
        
        try:
            while True:
                start_time = time.time()
                
                # 获取对齐的图像
                color_image, depth_image, aligned_depth_frame, depth_intrin = self.get_aligned_images()
                
                if color_image is None or depth_image is None:
                    continue
                
                # 使用 YOLO 进行目标检测
                results = self.model.predict(color_image, conf=0.5, verbose=False)
                annotated_frame = results[0].plot()
                detected_boxes = results[0].boxes.xyxy
                class_ids = results[0].boxes.cls if len(results[0].boxes) > 0 else []
                confidences = results[0].boxes.conf if len(results[0].boxes) > 0 else []
                
                # 存储当前帧的检测数据
                detections_data = {
                    'timestamp': datetime.now().isoformat(),
                    'num_detections': len(detected_boxes),
                    'detections': []
                }
                
                # 处理每个检测到的目标
                for i, (box, class_id, conf) in enumerate(zip(detected_boxes, class_ids, confidences)):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = self.model.names[int(class_id)]
                    
                    # 计算目标中心点
                    ux = int((x1 + x2) / 2)
                    uy = int((y1 + y2) / 2)
                    
                    # 获取中心点的深度值
                    if 0 <= ux < depth_image.shape[1] and 0 <= uy < depth_image.shape[0]:
                        depth_value = depth_image[uy, ux]
                        
                        if depth_value > 0:  # 确保深度值有效
                            # 将像素坐标转换为相机坐标系
                            camera_coord = self.pixel_to_camera_coordinate(ux, uy, depth_value)
                            
                            # 将相机坐标系转换为小车底盘坐标系
                            chassis_coord = self.calibration.camera_to_chassis(camera_coord)
                            
                            # 将底盘坐标系转换为全局坐标系（map）
                            global_coord = chassis_to_global(chassis_coord)
                            
                            # 存储检测数据
                            detection_info = {
                                'id': i + 1,
                                'class': class_name,
                                'confidence': float(conf),
                                'bbox': [x1, y1, x2, y2],
                                'center_pixel': [ux, uy],
                                'depth_mm': int(depth_value),
                                'camera_coord': {
                                    'x': float(camera_coord[0]),
                                    'y': float(camera_coord[1]),
                                    'z': float(camera_coord[2])
                                },
                                'chassis_coord': {
                                    'x': float(chassis_coord[0]),
                                    'y': float(chassis_coord[1]),
                                    'z': float(chassis_coord[2])
                                },
                                'global_coord': {
                                    'x': float(global_coord[0]),
                                    'y': float(global_coord[1]),
                                    'z': float(global_coord[2])
                                }
                            }
                            detections_data['detections'].append(detection_info)
                            
                            # 在图像上标注
                            cv2.circle(annotated_frame, (ux, uy), 4, (255, 255, 255), 5)
                            
                            # 标注相机坐标系坐标（青色）
                            camera_text = f"Cam: ({camera_coord[0]:.2f}, {camera_coord[1]:.2f}, {camera_coord[2]:.2f})m"
                            cv2.putText(annotated_frame, camera_text, (ux + 20, uy), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255, 255, 0], 
                                       thickness=1, lineType=cv2.LINE_AA)
                            
                            # 标注底盘坐标系坐标（绿色）
                            chassis_text = f"Base: ({chassis_coord[0]:.2f}, {chassis_coord[1]:.2f}, {chassis_coord[2]:.2f})m"
                            cv2.putText(annotated_frame, chassis_text, (ux + 20, uy + 20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, [0, 255, 0], 
                                       thickness=1, lineType=cv2.LINE_AA)
                            
                            # 标注全局坐标系坐标（粉红色）
                            global_text = f"Map: ({global_coord[0]:.2f}, {global_coord[1]:.2f}, {global_coord[2]:.2f})m"
                            cv2.putText(annotated_frame, global_text, (ux + 20, uy + 40), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255, 0, 255], 
                                       thickness=1, lineType=cv2.LINE_AA)
                            
                            # 打印到控制台
                            print(f"\n目标 {i+1}: {class_name} (置信度: {conf:.2f})")
                            print(f"  底盘坐标系: X={chassis_coord[0]:.3f}m, Y={chassis_coord[1]:.3f}m, Z={chassis_coord[2]:.3f}m")
                            print(f"  全局坐标系: X={global_coord[0]:.3f}m, Y={global_coord[1]:.3f}m, Z={global_coord[2]:.3f}m")
                
                # 计算并显示 FPS
                end_time = time.time()
                self.fps = 1.0 / (end_time - start_time)
                cv2.putText(annotated_frame, f'FPS: {self.fps:.2f}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # 显示结果
                cv2.imshow('YOLO Real-time Detection with Calibration', annotated_frame)
                
                # 键盘控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n退出程序...")
                    break
                elif key == ord('s') and self.save_results:
                    # 保存当前帧
                    image_path, json_path = self.save_detection_results(annotated_frame, detections_data)
                    print(f"\n已保存结果:")
                    print(f"  图像: {image_path}")
                    print(f"  数据: {json_path}")
                
                self.frame_count += 1
        
        finally:
            # 清理资源
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("\n相机已停止")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='实时 YOLO 检测与坐标标定系统')
    
    # 模型参数
    parser.add_argument('--model', type=str, 
                       default='/home/nvidia/huangjie/YM_VLN/yolotest/yoloV8/models/yolov8x.pt',
                       help='YOLO 模型路径')
    
    # 相机标定参数
    parser.add_argument('--cam_x', type=float, default=0.3, 
                       help='相机在底盘坐标系的 X 位置（米）')
    parser.add_argument('--cam_y', type=float, default=0.0, 
                       help='相机在底盘坐标系的 Y 位置（米）')
    parser.add_argument('--cam_z', type=float, default=0.5, 
                       help='相机在底盘坐标系的 Z 位置（米）')
    parser.add_argument('--cam_yaw', type=float, default=0.0, 
                       help='相机偏航角（度）')
    parser.add_argument('--cam_pitch', type=float, default=-30.0, 
                       help='相机俯仰角（度）')
    parser.add_argument('--cam_roll', type=float, default=0.0, 
                       help='相机横滚角（度）')
    
    # 相机内参
    parser.add_argument('--fx', type=float, default=615.0, help='焦距 fx')
    parser.add_argument('--fy', type=float, default=615.0, help='焦距 fy')
    parser.add_argument('--cx', type=float, default=424.0, help='主点 cx')
    parser.add_argument('--cy', type=float, default=240.0, help='主点 cy')
    
    # 保存设置
    parser.add_argument('--save', action='store_true', 
                       help='启用保存检测结果功能')
    parser.add_argument('--output_dir', type=str, default='results', 
                       help='结果保存目录')
    
    args = parser.parse_args()
    
    # 准备相机内参
    camera_params = {
        'fx': args.fx,
        'fy': args.fy,
        'cx': args.cx,
        'cy': args.cy,
        'width': 848,
        'height': 480
    }
    
    # 准备标定参数
    calibration_params = {
        'x': args.cam_x,
        'y': args.cam_y,
        'z': args.cam_z,
        'yaw': np.radians(args.cam_yaw),
        'pitch': np.radians(args.cam_pitch),
        'roll': np.radians(args.cam_roll)
    }
    
    # 打印配置信息
    print("=== 系统配置 ===")
    print(f"YOLO 模型: {args.model}")
    print(f"相机位置: X={args.cam_x}m, Y={args.cam_y}m, Z={args.cam_z}m")
    print(f"相机姿态: Yaw={args.cam_yaw}°, Pitch={args.cam_pitch}°, Roll={args.cam_roll}°")
    print(f"相机内参: fx={args.fx}, fy={args.fy}, cx={args.cx}, cy={args.cy}")
    print(f"保存结果: {'是' if args.save else '否'}")
    if args.save:
        print(f"保存目录: {args.output_dir}")
    print()
    
    # 初始化并运行系统
    try:
        system = YOLOCalibrationSystem(
            model_path=args.model,
            camera_params=camera_params,
            calibration_params=calibration_params,
            save_results=args.save,
            output_dir=args.output_dir
        )
        system.run()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
