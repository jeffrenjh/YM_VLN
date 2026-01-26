# 相机标定使用说明

## 坐标系定义

### 1. 相机坐标系（RealSense 标准）
与 `testyolo.py` 中使用的 RealSense SDK 坐标系完全一致：
- **X 轴**：指向右侧（从相机视角看）
- **Y 轴**：指向下方
- **Z 轴**：指向前方（相机光轴方向，深度方向）
- **原点**：相机光学中心

### 2. 小车底盘坐标系（机器人标准）
- **X 轴**：指向小车前方
- **Y 轴**：指向小车左侧
- **Z 轴**：指向上方
- **原点**：小车底盘中心

## 文件说明

### calibration.py
相机到小车底盘坐标系转换的核心模块，包含：
- `CameraCalibration` 类：处理坐标系转换
- `get_camera_to_chassis_transform()` 便捷函数

### testyolo.py
使用 RealSense 相机进行实时 YOLO 检测，计算物体在相机坐标系下的位置。

### test_calibration_with_data.py
测试脚本，使用 `data/` 文件夹中的 RGBD 图像进行离线测试。

## 使用方法

### 方法 1: 基本使用

```python
from calibration import CameraCalibration
import numpy as np

# 创建标定对象
# 假设相机安装在小车前方 30cm，高度 50cm，向下俯仰 30 度
calibration = CameraCalibration(
    x=0.3,      # 相机在小车前方 30cm
    y=0.0,      # 相机在小车中心线上
    z=0.5,      # 相机高度 50cm
    yaw=0.0,    # 没有偏航
    pitch=np.radians(-30),  # 向下俯仰 30 度
    roll=0.0    # 没有横滚
)

# 转换单个点（相机坐标系 -> 底盘坐标系）
camera_point = [1.0, 0.0, 2.0]  # 相机坐标系下的点
chassis_point = calibration.camera_to_chassis(camera_point)
print(f"相机坐标: {camera_point}")
print(f"底盘坐标: {chassis_point}")

# 批量转换多个点
camera_points = np.array([
    [1.0, 0.0, 2.0],
    [1.0, 0.5, 2.0],
    [1.0, -0.5, 2.0]
])
chassis_points = calibration.camera_to_chassis(camera_points)
```

### 方法 2: 整合到 testyolo.py

在 `testyolo.py` 中添加以下代码：

```python
from calibration import CameraCalibration
import numpy as np

# 在脚本开头初始化标定对象
calibration = CameraCalibration(
    x=0.3,      # 根据实际安装位置调整
    y=0.0,
    z=0.5,
    pitch=np.radians(-30)
)

# 在检测循环中，获取物体坐标后进行转换
for i, box in enumerate(detected_boxes):
    # ... 原有代码 ...
    
    # 获取相机坐标系下的坐标
    dis, camera_coordinate = get_3d_camera_coordinate(
        [ux, uy], aligned_depth_frame, depth_intrin
    )
    
    # 转换到底盘坐标系
    chassis_coordinate = calibration.camera_to_chassis(camera_coordinate)
    
    print(f"相机坐标: {camera_coordinate}")
    print(f"底盘坐标: {chassis_coordinate}")
```

### 方法 3: 使用测试脚本

```bash
# 使用 data 文件夹中的测试数据
python test_calibration_with_data.py
```

## 标定参数说明

### 位置参数 (x, y, z)
相机光学中心相对于小车底盘中心的位置偏移（单位：米）：
- **x**: 正值表示相机在小车前方，负值在后方
- **y**: 正值表示相机在小车左侧，负值在右侧
- **z**: 正值表示相机在小车上方，负值在下方

### 姿态参数 (yaw, pitch, roll)
相机相对于小车底盘的姿态角（单位：弧度）：
- **yaw**: 绕 Z 轴旋转（偏航角），正值逆时针
- **pitch**: 绕 Y 轴旋转（俯仰角），正值抬头，负值低头
- **roll**: 绕 X 轴旋转（横滚角），正值向右倾斜

### 常见安装配置示例

#### 1. 相机水平向前安装
```python
CameraCalibration(x=0.3, y=0.0, z=0.5, yaw=0.0, pitch=0.0, roll=0.0)
```

#### 2. 相机向下俯仰 30 度
```python
CameraCalibration(x=0.3, y=0.0, z=0.5, yaw=0.0, pitch=np.radians(-30), roll=0.0)
```

#### 3. 相机向下俯仰 45 度
```python
CameraCalibration(x=0.3, y=0.0, z=0.5, yaw=0.0, pitch=np.radians(-45), roll=0.0)
```

#### 4. 相机安装在左侧，向内旋转 90 度
```python
CameraCalibration(x=0.0, y=0.2, z=0.5, yaw=np.radians(-90), pitch=0.0, roll=0.0)
```

## 验证坐标转换

### 手动验证示例

假设相机坐标系下检测到物体位于 `(0, 0, 2)` 米（正前方 2 米）：

1. **无旋转，相机在小车前方 0.3m，高度 0.5m**
   ```python
   calibration = CameraCalibration(x=0.3, y=0.0, z=0.5, yaw=0, pitch=0, roll=0)
   chassis_point = calibration.camera_to_chassis([0, 0, 2])
   # 结果: [2.3, 0.0, 0.5] (物体在底盘前方 2.3m，高度 0.5m)
   ```

2. **向下俯仰 30 度**
   ```python
   calibration = CameraCalibration(x=0.3, y=0.0, z=0.5, pitch=np.radians(-30))
   chassis_point = calibration.camera_to_chassis([0, 0, 2])
   # 由于俯仰角，Z 轴会受到影响
   ```

## 注意事项

1. **角度单位**：所有角度参数使用弧度制，使用 `np.radians()` 转换度数
2. **坐标系一致性**：确保相机坐标系与 RealSense SDK 一致（X-右，Y-下，Z-前）
3. **测量精度**：准确测量相机相对于底盘中心的位置和姿态角
4. **深度精度**：RealSense 深度测量在近距离（0.3-3m）较准确
5. **标定验证**：使用已知位置的物体验证转换结果的准确性

## 测试数据

`data/` 文件夹包含：
- `color_000001.png`: RGB 彩色图像
- `depth_000001.png`: 深度图像（16 位）

可用于离线测试和调试标定参数。

## 故障排查

### 问题 1: 转换后的坐标明显错误
- 检查标定参数是否正确
- 确认姿态角的符号（俯仰向下应为负值）
- 验证相机内参是否正确

### 问题 2: 深度值为 0
- 检查深度图像是否正确加载
- 确认物体在相机深度测量范围内（0.3-10m）
- 某些材质（透明、反光）可能导致深度测量失败

### 问题 3: 检测不到目标
- 降低 YOLO 检测阈值（conf 参数）
- 确认模型文件路径正确
- 检查图像质量和光照条件
