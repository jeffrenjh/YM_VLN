# YOLO实时检测 - 全局坐标系转换

## 功能说明

`yolo_calibration.py` 现在支持将检测到的物体坐标从底盘坐标系转换到全局坐标系（map）。

### 坐标系转换链

```
相机坐标系 (camera)
    ↓
底盘坐标系 (base_link)
    ↓
全局坐标系 (map)
```

### TF变换参数

已内置以下TF变换参数：

- **odom ← base_link**: 平移 (-5.270, -1.117, 0), yaw 2.986 rad
- **map ← odom**: 平移 (0.540, 1.102, 0), yaw -0.059 rad

## 使用方法

运行程序（使用默认参数）：

```bash
python yolo_calibration.py
```

启用结果保存：

```bash
python yolo_calibration.py --save --output_dir results
```

## 输出说明

### 屏幕显示

检测到的物体会在图像上显示三种坐标标注：

- **青色（Cam）** - 相机坐标系 [x, y, z]
- **绿色（Base）** - 底盘坐标系 [x, y, z]  
- **粉红色（Map）** - 全局坐标系 [x, y, z]

### 控制台输出

```
目标 1: person (置信度: 0.95)
  底盘坐标系: X=1.234m, Y=0.567m, Z=0.123m
  全局坐标系: X=-3.456m, Y=0.789m, Z=0.123m
```

### 保存的JSON文件

```json
{
  "timestamp": "2026-01-29T15:46:00.000000",
  "num_detections": 1,
  "detections": [
    {
      "id": 1,
      "class": "person",
      "confidence": 0.95,
      "camera_coord": {"x": 0.5, "y": 0.2, "z": 1.5},
      "chassis_coord": {"x": 1.234, "y": 0.567, "z": 0.123},
      "global_coord": {"x": -3.456, "y": 0.789, "z": 0.123}
    }
  ]
}
```

## 修改TF参数

如需修改TF变换参数，编辑 `yolo_calibration.py` 文件中的 `chassis_to_global()` 函数：

```python
def chassis_to_global(chassis_coord):
    # 修改这里的参数
    base_to_odom_trans = np.array([-5.270, -1.117, 0.0])
    base_to_odom_yaw = 2.986
    
    odom_to_map_trans = np.array([0.540, 1.102, 0.0])
    odom_to_map_yaw = -0.059
    ...
```

## 控制键

- **q** - 退出程序
- **s** - 保存当前帧（需要启用 `--save`）
