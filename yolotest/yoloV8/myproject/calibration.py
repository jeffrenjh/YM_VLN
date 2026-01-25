# 用于实现相机-小车中心标定
import numpy as np


class CameraCalibration:
    """
    相机到小车底盘坐标系的标定类
    
    用于将相机坐标系下的物体坐标转换为小车底盘坐标系下的坐标
    """
    
    def __init__(self, x=0.0, y=0.0, z=0.0, yaw=0.0, pitch=0.0, roll=0.0):
        """
        初始化相机标定参数
        
        参数:
            x: 相机相对小车底盘中心的x坐标 (单位: 米)
            y: 相机相对小车底盘中心的y坐标 (单位: 米)
            z: 相机相对小车底盘中心的z坐标 (单位: 米)
            yaw: 相机绕z轴的旋转角度 (单位: 弧度)
            pitch: 相机绕y轴的旋转角度 (单位: 弧度)
            roll: 相机绕x轴的旋转角度 (单位: 弧度)
        """
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        
        # 计算变换矩阵
        self.transform_matrix = self._compute_transform_matrix()
    
    def _compute_transform_matrix(self):
        """
        计算从相机坐标系到小车底盘坐标系的4x4变换矩阵
        
        变换顺序: 先旋转(Roll-Pitch-Yaw)，再平移
        
        返回:
            4x4的齐次变换矩阵
        """
        # 计算旋转矩阵 (使用ZYX欧拉角顺序)
        # Roll (绕X轴旋转)
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(self.roll), -np.sin(self.roll)],
            [0, np.sin(self.roll), np.cos(self.roll)]
        ])
        
        # Pitch (绕Y轴旋转)
        R_y = np.array([
            [np.cos(self.pitch), 0, np.sin(self.pitch)],
            [0, 1, 0],
            [-np.sin(self.pitch), 0, np.cos(self.pitch)]
        ])
        
        # Yaw (绕Z轴旋转)
        R_z = np.array([
            [np.cos(self.yaw), -np.sin(self.yaw), 0],
            [np.sin(self.yaw), np.cos(self.yaw), 0],
            [0, 0, 1]
        ])
        
        # 组合旋转矩阵: R = Rz * Ry * Rx
        R = R_z @ R_y @ R_x
        
        # 构建4x4齐次变换矩阵
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = [self.x, self.y, self.z]
        
        return transform
    
    def camera_to_chassis(self, point_camera):
        """
        将相机坐标系下的点转换到小车底盘坐标系
        
        参数:
            point_camera: 相机坐标系下的点坐标
                         可以是:
                         - 3元素列表/数组 [x, y, z]
                         - Nx3的numpy数组 (批量转换)
        
        返回:
            小车底盘坐标系下的点坐标 (与输入格式相同)
        """
        point_camera = np.array(point_camera)
        
        # 判断是单个点还是多个点
        if point_camera.ndim == 1:
            # 单个点: [x, y, z]
            point_homogeneous = np.append(point_camera, 1)  # 转为齐次坐标 [x, y, z, 1]
            point_chassis = self.transform_matrix @ point_homogeneous
            return point_chassis[:3]  # 返回 [x, y, z]
        else:
            # 多个点: Nx3
            ones = np.ones((point_camera.shape[0], 1))
            points_homogeneous = np.hstack([point_camera, ones])  # Nx4
            points_chassis = (self.transform_matrix @ points_homogeneous.T).T
            return points_chassis[:, :3]  # 返回 Nx3
    
    def get_transform_matrix(self):
        """
        获取变换矩阵
        
        返回:
            4x4的从相机坐标系到小车底盘坐标系的齐次变换矩阵
        """
        return self.transform_matrix.copy()
    
    def set_calibration_params(self, x=None, y=None, z=None, yaw=None, pitch=None, roll=None):
        """
        更新标定参数
        
        参数:
            x, y, z: 相机位置 (单位: 米)
            yaw, pitch, roll: 相机姿态角 (单位: 弧度)
        """
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if z is not None:
            self.z = z
        if yaw is not None:
            self.yaw = yaw
        if pitch is not None:
            self.pitch = pitch
        if roll is not None:
            self.roll = roll
        
        # 重新计算变换矩阵
        self.transform_matrix = self._compute_transform_matrix()
    
    def __str__(self):
        """打印标定信息"""
        return (f"相机标定参数:\n"
                f"  位置 (x, y, z): ({self.x:.3f}, {self.y:.3f}, {self.z:.3f}) m\n"
                f"  姿态 (yaw, pitch, roll): ({np.degrees(self.yaw):.2f}°, "
                f"{np.degrees(self.pitch):.2f}°, {np.degrees(self.roll):.2f}°)\n"
                f"变换矩阵:\n{self.transform_matrix}")


# 便捷函数
def get_camera_to_chassis_transform(x, y, z, yaw, pitch, roll):
    """
    快速获取相机到小车底盘的变换矩阵
    
    参数:
        x, y, z: 相机相对小车底盘中心的位置 (单位: 米)
        yaw, pitch, roll: 相机的姿态角 (单位: 弧度)
    
    返回:
        4x4的齐次变换矩阵
    """
    calibration = CameraCalibration(x, y, z, yaw, pitch, roll)
    return calibration.get_transform_matrix()


# 使用示例
if __name__ == "__main__":
    # 示例1: 创建标定对象
    # 假设相机位于小车前方0.3米，高度0.5米，向下俯仰30度
    calibration = CameraCalibration(
        x=0.3,      # 相机在小车前方30cm
        y=0.0,      # 相机在小车中心线上
        z=0.5,      # 相机高度50cm
        yaw=0.0,    # 没有偏航
        pitch=np.radians(-30),  # 向下俯仰30度
        roll=0.0    # 没有横滚
    )
    
    print(calibration)
    print("\n" + "="*50 + "\n")
    
    # 示例2: 转换单个点
    # 假设相机看到一个物体在相机坐标系下的坐标为 (1.0, 0.0, 2.0)
    point_in_camera = [1.0, 0.0, 2.0]
    point_in_chassis = calibration.camera_to_chassis(point_in_camera)
    
    print(f"相机坐标系下的点: {point_in_camera}")
    print(f"小车底盘坐标系下的点: {point_in_chassis}")
    print("\n" + "="*50 + "\n")
    
    # 示例3: 批量转换多个点
    points_in_camera = np.array([
        [1.0, 0.0, 2.0],
        [1.0, 0.5, 2.0],
        [1.0, -0.5, 2.0]
    ])
    points_in_chassis = calibration.camera_to_chassis(points_in_camera)
    
    print("批量转换多个点:")
    print("相机坐标系下的点:")
    print(points_in_camera)
    print("\n小车底盘坐标系下的点:")
    print(points_in_chassis)
    print("\n" + "="*50 + "\n")
    
    # 示例4: 使用便捷函数
    transform_matrix = get_camera_to_chassis_transform(
        x=0.3, y=0.0, z=0.5,
        yaw=0.0, pitch=np.radians(-30), roll=0.0
    )
    print("使用便捷函数获取变换矩阵:")
    print(transform_matrix)
