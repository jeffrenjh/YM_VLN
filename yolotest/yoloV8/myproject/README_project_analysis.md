# YOLOv8 RealSense 物体检测与定位项目说明文档

## 项目概述
本项目旨在利用 Intel RealSense D435i 深度相机和 YOLOv8 目标检测模型，实现实时的环境物体检测，并将检测到的物体坐标从**像素坐标系**转换到**相机坐标系**，进而转换到**机器人底盘坐标系**，最终映射到**全局地图坐标系**。

项目包含相机标定、数据采集、离线测试、实时检测以及 IMU 姿态解算等功能模块。

## 目录结构分析 (myproject)

该文件夹 (`/home/nvidia/huangjie/YM_VLN/test/YM_VLN/yolotest/yoloV8/myproject`) 下的主要 Python 脚本功能说明如下：

| 文件名 | 功能描述 | 依赖 |
| :--- | :--- | :--- |
| **`calibration.py`** | **[核心] 相机标定类**。<br>定义了 `CameraCalibration` 类，用于管理相机相对于机器人底盘的安装位置参数 (x, y, z, yaw, pitch, roll)，并提供坐标变换矩阵计算和点变换功能。 | `numpy` |
| **`yolo_calibration.py --> online_global.py`** | **[核心] 实时检测与定位系统**。<br>集成了 RealSense 相机流、YOLOv8 检测、坐标变换（相机 -> 底盘 -> 全局 Map）。包含硬编码的 TF 变换参数（Base_link -> Odom -> Map）。 | `calibration.py`, `ultralytics`, `pyrealsense2`, `opencv` |
| **`testyolo.py --> online_camera.py`** | **实时 YOLO 检测测试脚本**。<br>使用 RealSense 运行 YOLOv8，在图像上显示检测框和物体中心点的相机系 3D 坐标。功能比 `yolo_calibration.py` 简单，不包含到底盘坐标系的转换。 | `ultralytics`, `pyrealsense2`, `opencv` |
| **`collect2.py --> data_collector.py`** | **数据采集脚本**。<br>从 RealSense 相机采集 RGB 图像和深度图像，并按序保存到指定文件夹。用于制作离线数据集。 | `pyrealsense2`, `opencv` |
| **`test_calibration_with_data.py --> offline_global.py`** | **离线标定测试脚本**。<br>加载本地保存的 RGBD 图像对（由 collect2.py 采集），运行 YOLO 检测并应用 `calibration.py` 中的转换逻辑，验证坐标转换的正确性。 | `calibration.py`, `ultralytics`, `opencv` |
| **`yolo_images.py --> offline_camera.py`** | **批量图片处理脚本**。<br>批量处理文件夹中的所有图像对，运行 YOLO 检测并显示/保存带有点云坐标的检测结果。 | `ultralytics`, `opencv` |
| **`imutest.py`** | **IMU 姿态解算模块**。<br>读取 RealSense 的加速度计和陀螺仪数据，使用互补滤波算法解算相机的实时姿态（四元数和欧拉角）。 | `pyrealsense2`, `numpy` |
| **`imu2euler.py`** | **IMU 欧拉角计算演示**。<br>一个简单的脚本，演示如何从 IMU 数据直接计算欧拉角（主要用于测试算法原理）。 | `pyrealsense2` |

## 详细功能说明

### 1. 坐标系定义
在 `calibration.py` 中定义了主要的坐标系转换逻辑：
- **相机坐标系 (RealSense)**:
    - X: 右
    - Y: 下
    - Z: 前 (深度方向)
- **底盘坐标系 (Base_link)**:
    - X: 前
    - Y: 左
    - Z: 上

### 2. 核心模块：实时检测 (`yolo_calibration.py`)
这是最完整的执行脚本，其工作流程如下：
1. **初始化**: 加载 YOLO 模型，设置相机内参，设置相机安装位置（标定参数）。
2. **获取数据**: 从 RealSense 获取对齐的 RGB 和 深度帧。
3. **推理**: 运行 YOLOv8 模型检测物体。
4. **深度提取**: 获取检测框中心点的深度值。
5. **坐标链变换**:
    - `Pixel (u, v)` + `Depth` -> **Camera Frame (Xc, Yc, Zc)** (利用内参)
    - **Camera Frame** -> **Chassis Frame (Xb, Yb, Zb)** (利用 `calibration.py` 中的外参矩阵)
    - **Chassis Frame** -> **Global Map Frame (Xw, Yw, Zw)** (利用硬编码的 TF 变换参数)
6. **可视化**: 在图像上绘制 BBox 和三套坐标系的坐标值。

### 3. 硬编码参数注意
在 `yolo_calibration.py` 中，函数 `chassis_to_global` 包含硬编码的变换参数，实际部署时可能需要修改：
```python
# TF参数: base_link -> odom
base_to_odom_trans = np.array([-5.270, -1.117, 0.0])
base_to_odom_yaw = 2.986

# TF参数: odom -> map
odom_to_map_trans = np.array([0.540, 1.102, 0.0])
odom_to_map_yaw = -0.059
```

## 使用示例

### 运行实时检测与标定
```bash
# 运行主程序，使用默认参数
python yolo_calibration.py

# 指定模型路径和相机安装高度（例如相机高 0.6米，下倾 45度）
python yolo_calibration.py --model ../models/yolov8l.pt --cam_z 0.6 --cam_pitch -45
```

### 采集数据
```bash
# 采集数据并保存到 my_data 文件夹
python collect2.py --save_dir ./data/my_data
```

### 测试 IMU
```bash
python imutest.py
```
