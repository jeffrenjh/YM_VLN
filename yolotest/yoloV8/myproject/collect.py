import os
import cv2
import time
import numpy as np
import pyrealsense2 as rs
import argparse

def initialize_single_camera():
    """
    初始化单个 RealSense 相机
    """
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 配置 RGB 和 Depth 流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    pipeline.start(config)
    print("[INFO] RealSense 相机已初始化")
    return pipeline

def save_camera_data(color_img, depth_img, output_path, index):
    """
    保存相机的 RGB 和 Depth 数据
    """
    os.makedirs(output_path, exist_ok=True)
    
    # 保存 RGB 图像
    color_path = os.path.join(output_path, f"{index:06d}_color.jpg")
    cv2.imwrite(color_path, color_img)
    print(f"[INFO] RGB 图像已保存: {color_path}")
    
    # 保存 Depth 图像
    depth_path = os.path.join(output_path, f"{index:06d}_depth.png")
    cv2.imwrite(depth_path, depth_img)
    print(f"[INFO] Depth 图像已保存: {depth_path}")

def collect_and_save(pipeline, output_path, frame_index):
    """
    采集并保存一帧相机数据
    """
    try:
        frames = pipeline.wait_for_frames(timeout_ms=1000)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame and depth_frame:
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # 显示预览
            cv2.imshow('RGB', color_image)
            cv2.imshow('Depth', depth_image)
            
            save_camera_data(color_image, depth_image, output_path, frame_index)
            return True
        else:
            print(f"[WARN] 相机数据获取失败")
            return False
    except Exception as e:
        print(f"[ERROR] 相机采集出错: {e}")
        return False

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='RealSense 相机数据采集')
    parser.add_argument('--output', '-o', type=str, required=True, 
                        help='数据保存路径，例如: ./datasets/my_data')
    args = parser.parse_args()
    
    # 使用命令行指定的路径
    output_path = args.output
    
    # 自动创建文件夹（如果不存在）
    os.makedirs(output_path, exist_ok=True)
    print(f"[INFO] 数据将保存到: {output_path}")

    # 初始化 RealSense 相机
    pipeline = initialize_single_camera()
    time.sleep(2)

    frame_index = 0
    print("[INFO] 开始采集数据，按 'q' 退出")
    
    try:
        while True:
            success = collect_and_save(pipeline, output_path, frame_index)
            if success:
                frame_index += 1
                print(f"[INFO] 已保存第 {frame_index} 帧")
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] 用户退出")
                break
            
            time.sleep(0.1)  # 控制采集频率
            
    except KeyboardInterrupt:
        print("[INFO] 采集中断")
    finally:
        # 清理资源
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] 采集结束")


