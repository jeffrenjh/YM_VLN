# ai写的测量方法
# 读取 RealSense 的加速度计和陀螺仪数据，使用互补滤波算法解算相机的实时姿态（四元数和欧拉角）
import pyrealsense2 as rs
import numpy as np
import threading
import time
from collections import deque
 
class RealSenseIMUPoseEstimator:
    def __init__(self):
        # 初始化相机和数据流
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 仅配置IMU流，不配置彩色和深度流
        self.config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
        self.config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
        
        # 姿态解算参数
        self.imu_data_lock = threading.Lock()
        self.accel_data = deque(maxlen=10)
        self.gyro_data = deque(maxlen=10)
        
        # 姿态四元数，初始化为[w, x, y, z]
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        
        # 时间戳
        self.last_timestamp = None
        
        # 线程控制
        self.is_running = False
        
        # 互补滤波参数
        self.filter_beta = 0.05  # 互补滤波器参数，越小表示更信任陀螺仪
    
    def start(self):
        # 启动相机
        profile = self.pipeline.start(self.config)
        print("IMU数据采集已启动")
        
        # 启动IMU数据收集和解算线程
        self.is_running = True
        self.imu_thread = threading.Thread(target=self._process_imu_data)
        self.imu_thread.daemon = True
        self.imu_thread.start()
        
        try:
            # 主线程等待用户中断
            print("按Ctrl+C停止程序")
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n程序已停止")
        finally:
            # 停止程序时释放资源
            self.is_running = False
            self.imu_thread.join()
            self.pipeline.stop()
    
    def _process_imu_data(self):
        # 处理IMU数据的线程函数
        while self.is_running:
            try:
                # 等待获取IMU帧
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                
                # 处理IMU数据
                accel_frame = frames.first_or_default(rs.stream.accel)
                gyro_frame = frames.first_or_default(rs.stream.gyro)
                
                if accel_frame and gyro_frame:
                    accel = accel_frame.as_motion_frame().get_motion_data()
                    gyro = gyro_frame.as_motion_frame().get_motion_data()
                    timestamp = accel_frame.get_timestamp() / 1000.0  # 转换为秒
                    
                    with self.imu_data_lock:
                        self.accel_data.append((accel.x, accel.y, accel.z))
                        self.gyro_data.append((gyro.x, gyro.y, gyro.z))
                        
                        # 姿态解算
                        self._update_pose(accel, gyro, timestamp)
                        
                        # 打印姿态数据
                        self._print_pose()
            except Exception as e:
                if self.is_running:
                    print(f"IMU数据收集错误: {e}")
    
 
 
    def _update_pose(self, accel, gyro, timestamp):
        # 更新姿态四元数
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            return
            
        # 计算时间间隔(秒)
        dt = timestamp - self.last_timestamp
        self.last_timestamp = timestamp
        
        # 转换为numpy数组
        accel = np.array([accel.x, accel.y, accel.z])
        gyro = np.array([gyro.x, gyro.y, gyro.z])
        
        # 归一化加速度计数据
        accel_norm = np.linalg.norm(accel)
        if accel_norm > 0:
            accel = accel / accel_norm
        
        # 基于陀螺仪更新四元数
        q_dot = self._quaternion_multiply(self.quaternion, [0, gyro[0], gyro[1], gyro[2]]) * 0.5
        self.quaternion = self.quaternion + q_dot * dt
        
        # 归一化四元数
        self.quaternion = self.quaternion / np.linalg.norm(self.quaternion)
        
        # 基于加速度计估计重力方向
        # 并使用互补滤波融合两种估计
        if len(self.accel_data) > 5:  # 确保有足够的数据
            # 加速度计估计的重力方向
            accel_gravity = self._accel_to_gravity(accel)
            
            # 四元数估计的重力方向
            q_gravity = self._quaternion_multiply(
                self._quaternion_multiply(self.quaternion, [0, 0, 0, 1]),
                self._quaternion_conjugate(self.quaternion)
            )[1:]  # 取向量部分
            
            # 计算误差并应用互补滤波
            error = np.cross(q_gravity, accel_gravity)
            
            # 仅更新四元数的向量部分(x,y,z)
            self.quaternion[1:] = self.quaternion[1:] + error * self.filter_beta * dt
            
            # 重新归一化四元数
            self.quaternion = self.quaternion / np.linalg.norm(self.quaternion)
 
 
    
    def _accel_to_gravity(self, accel):
        # 从加速度计数据估计重力方向
        # 加速度计在静止时测量的是重力加速度
        return accel
    
    def _quaternion_multiply(self, q1, q2):
        # 四元数乘法
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([w, x, y, z])
    
    def _quaternion_conjugate(self, q):
        # 四元数共轭
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    def _quaternion_to_euler(self, q):
        # 四元数转欧拉角(roll, pitch, yaw)
        w, x, y, z = q
        
        # 滚转(绕x轴)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # 俯仰(绕y轴)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # 万向锁
        else:
            pitch = np.arcsin(sinp)
        
        # 偏航(绕z轴)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.degrees([roll, pitch, yaw])  # 转换为角度
    
    def _print_pose(self):
        # 打印姿态数据
        euler_angles = self._quaternion_to_euler(self.quaternion)
        roll, pitch, yaw = euler_angles
        
        # 格式化输出
        print("四元素：", self.quaternion)
        print(f"姿态欧拉角: 滚转={roll:.2f}°, 俯仰={pitch:.2f}°, 偏航={yaw:.2f}°")
        # print(f"姿态: 滚转={roll:.2f}°, 俯仰={pitch:.2f}°, 偏航={yaw:.2f}°", end='\r')
 
if __name__ == "__main__":
    estimator = RealSenseIMUPoseEstimator()
    estimator.start()    