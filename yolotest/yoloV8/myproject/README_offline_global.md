# offline_chasis.py - 离线RGBD图像检测与全局坐标转换

## 功能概述

此脚本基于 `offline_chasis.py` 进行了增强，实现了以下功能：

1. **离线RGBD图像检测**：使用 `data` 文件夹中的彩色和深度图像
2. **YOLO目标检测**：自动检测图像中的目标物体
3. **多坐标系转换**：
   - 像素坐标 → 相机坐标系
   - 相机坐标系 → 底盘坐标系
   - **底盘坐标系 → 全局坐标系（map）** ⭐ 新增功能
4. **结果保存**：
   - 标注图像：包含所有坐标系信息的可视化结果
   - JSON文件：详细的检测结果和坐标数据

## 坐标系转换链

```
Pixel Coords (像素坐标)
    ↓
Camera Coords (相机坐标系)
    ↓
Chassis Coords (底盘坐标系 / base_link)
    ↓
Odom Coords (里程计坐标系)
    ↓
Global Coords (全局坐标系 / map) ⭐
```

## 使用方法

### 1. 准备数据

确保 `data` 文件夹中有以下文件：
- `color_000001.png` - 彩色图像
- `depth_000001.png` - 深度图像（16位PNG）

### 2. 运行脚本

```bash
python offline_chasis.py
```

### 3. 查看结果

脚本会生成两个输出文件：

#### a. 图像结果 (`detection_result.jpg`)
- 标注了所有检测到的目标
- 显示每个目标的三个坐标系信息：
  - **黄色文本**：相机坐标系 (Cam)
  - **绿色文本**：底盘坐标系 (Chassis)
  - **洋红色文本**：全局坐标系 (Global) ⭐

#### b. JSON结果 (`detection_result.json`)
包含完整的检测数据：

```json
{
    "source_images": {
        "color": "...",
        "depth": "..."
    },
    "camera_intrinsics": {
        "fx": 615.0,
        "fy": 615.0,
        "cx": 424.0,
        "cy": 240.0
    },
    "calibration_params": {
        "x": 0.3,
        "y": 0.0,
        "z": 0.5,
        "yaw": 0.0,
        "pitch": -0.524,
        "roll": 0.0
    },
    "detections": [
        {
            "id": 1,
            "class": "person",
            "confidence": 0.85,
            "bbox": {...},
            "pixel_coords": {...},
            "depth_mm": 2150,
            "camera_coords": {...},
            "chassis_coords": {...},
            "global_coords": {
                "x": -4.523,
                "y": -2.134,
                "z": 0.315
            }
        }
    ]
}
```

## 核心参数配置

### 相机内参（Camera Intrinsics）
```python
camera_intrinsics = {
    'fx': 615.0,  # 焦距 x
    'fy': 615.0,  # 焦距 y
    'cx': 424.0,  # 主点 x
    'cy': 240.0,  # 主点 y
}
```
⚠️ **注意**：这些是 RealSense D435 的默认参数，实际使用时应该从标定文件读取。

### 相机标定参数（Camera Calibration）
```python
calibration = CameraCalibration(
    x=0.3,      # 相机在小车前方 30cm
    y=0.0,      # 相机在小车中心线上
    z=0.5,      # 相机高度 50cm
    yaw=0.0,    # 没有偏航
    pitch=np.radians(-30),  # 向下俯仰 30°
    roll=0.0    # 没有横滚
)
```
⚠️ **注意**：这些参数需要根据实际的相机安装位置进行调整。

### 全局坐标转换参数（Global Transform）
```python
# base_link -> odom
base_to_odom_trans = [-5.270, -1.117, 0.0]  # 平移 (米)
base_to_odom_yaw = 2.986                     # 偏航角 (弧度)

# odom -> map
odom_to_map_trans = [0.540, 1.102, 0.0]     # 平移 (米)
odom_to_map_yaw = -0.059                     # 偏航角 (弧度)
```
⚠️ **注意**：这些参数应该从ROS的TF树中获取，代码中的是示例值。

### 修改全局坐标转换参数

如果需要使用不同的TF参数，可以在调用 `chassis_to_global` 时传入：

```python
global_coord = chassis_to_global(
    chassis_coord,
    base_to_odom_trans=np.array([x, y, z]),
    base_to_odom_yaw=yaw_rad,
    odom_to_map_trans=np.array([x, y, z]),
    odom_to_map_yaw=yaw_rad
)
```

## YOLO模型

脚本会按照以下顺序查找YOLO模型：
1. `../models/yolov8l.pt`（当前项目相对路径）
2. `../../../models/yolov8l.pt`（上层目录）
3. 硬编码的Linux路径
4. 自动下载 `yolov8l.pt`

如果需要使用其他模型，请修改 `model_paths` 列表。

## 依赖项

```bash
pip install opencv-python numpy ultralytics
```

## 输出示例

```
正在加载 RGBD 图像...
彩色图像尺寸: (480, 848, 3)
深度图像尺寸: (480, 848)

相机内参: fx=615.0, fy=615.0, cx=424.0, cy=240.0

初始化相机标定...
============================================================
坐标系说明:
  相机坐标系 (RealSense): X-右, Y-下, Z-前
  底盘坐标系 (Robot):    X-前, Y-左, Z-上
============================================================
...

正在加载 YOLO 模型...
找到模型: ../models/yolov8l.pt

正在进行目标检测...
检测到 3 个目标

目标 1: person (置信度: 0.85)
  边界框: (123, 45) -> (267, 389)
  中心点像素坐标: (195, 217)
  深度值: 2150 mm (2.150 m)
  相机坐标系: X=-0.234m, Y=-0.123m, Z=2.150m
  底盘坐标系: X=2.385m, Y=0.234m, Z=0.315m
  全局坐标系: X=-4.523m, Y=-2.134m, Z=0.315m ⭐
...

结果已保存到: D:\...\detection_result.jpg
检测结果已保存到: D:\...\detection_result.json
```

## 与其他脚本的关系

- **offline_camera.py**: 只做离线图像的YOLO检测，不做坐标转换
- **offline_chasis.py**: 原版本，只转换到底盘坐标系
- **offline_chasis.py (增强版)**: **当前版本，支持全局坐标系转换** ⭐
- **online_global.py**: 实时相机检测，支持全局坐标系（但需要实时ROS TF）

## 注意事项

1. **深度值有效性**：如果深度图中某个位置的深度值为0或异常，该点的3D坐标会不准确
2. **坐标系对齐**：确保相机标定参数正确，否则坐标转换会有误差
3. **TF参数时效性**：全局坐标转换的TF参数会随着机器人移动而变化，离线模式使用的是固定值
4. **模型性能**：yolov8l.pt 是较大的模型，检测精度高但速度较慢，可以根据需求选择其他版本

## 扩展功能建议

1. **批量处理**：修改脚本以处理多张图像
2. **动态TF参数**：从ROS bag文件中提取对应时刻的TF参数
3. **可视化改进**：添加3D可视化，显示物体在全局坐标系中的位置
4. **结果分析**：添加统计和分析功能，比如物体分布热图

## 参考文档

- [README_calibration.md](README_calibration.md) - 相机标定说明
- [README_global_coordinates.md](README_global_coordinates.md) - 全局坐标系转换说明
- [calibration.py](calibration.py) - 相机标定实现
