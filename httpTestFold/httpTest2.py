#!/usr/bin/env python3
import requests
import time
import math

class SimpleBunnyController:
    def __init__(self, robot_ip):
        port = 10001  # 默认端口

        self.base_url = f"http://{robot_ip}:{port}"
        
    def get_position(self):
        """获取当前位置"""
        try:
            response = requests.get(f"{self.base_url}/bunny/robot/get_localization_pose")
            print(f"获取位置: {response.url}")
            if response.status_code != 200:
                print(f"获取位置失败: {response.status_code}")
                return None
            data = response.json()
            if data.get('code') == 0:
                return data.get('data', {})
        except:
            pass
        return None
    
    def move_to(self, target_x, target_y):
        """移动到目标位置"""
        print(f"移动到: ({target_x}, {target_y})")
        
        # 获取当前位置
        current = self.get_position()
        if not current:
            print("无法获取当前位置")
            return False
        
        # 计算相对移动距离
        dx = target_x - current.get('x', 0)
        dy = target_y - current.get('y', 0)
        
        # 发送相对移动命令
        try:
            params = {"x": dx, "y": dy, "th": 0}
            response = requests.get(f"{self.base_url}/bunny/robot/relative_move", params=params)
            result = response.json()

            print(f"返回: {msg.url}")
            if response.status_code != 200:
                print(f"失败: {response.status_code}")
            
            if result.get('code') == 0:
                print("移动命令发送成功")
                time.sleep(5)  # 等待移动完成
                return True
            else:
                print(f"移动失败: {result.get('msg', '')}")
                return False
        except Exception as e:
            print(f"移动异常: {e}")
            return False

# 使用示例
if __name__ == '__main__':
    # 替换为您的机器人IP地址
    robot = SimpleBunnyController("192.168.80.100")
    
    # 开启定位
    msg = requests.get(f"{robot.base_url}/bunny/robot/start_localization")
    
    # 等待定位开启
    print("等待定位开启...")

    print(f"获取位置: {msg.url}")
    if msg.status_code != 200:
        print(f"失败: {msg.status_code}")

    time.sleep(2)
    
    # 移动到目标位置
    target_positions = [
        (0.1, 0.1),
        (0.1, 0.1),
        (0.0, 0.0)
    ]
    
    for x, y in target_positions:
        robot.move_to(x, y)
        time.sleep(3)  # 在每个点停留3秒
    
    print("移动任务完成!")


#!/usr/bin/env python3
# import requests
# import time
# import math

# class SimpleBunnyController:
#     def __init__(self, robot_ip, port=45769):
#         """
#         初始化控制器
        
#         Args:
#             robot_ip (str): 机器人IP地址
#             port (int): 端口号，默认为80
#         """
#         self.robot_ip = robot_ip
#         self.port = port
#         self.base_url = f"http://{robot_ip}:{port}"
        
#     def get_position(self):
#         """获取当前位置"""
#         try:
#             response = requests.get(f"{self.base_url}/bunny/robot/get_localization_pose")
#             data = response.json()
#             if data.get('code') == 0:
#                 return data.get('data', {})
#         except Exception as e:
#             print(f"获取位置异常: {e}")
#         return None
    
#     def move_to(self, target_x, target_y):
#         """移动到目标位置"""
#         print(f"移动到: ({target_x}, {target_y})")
        
#         # 获取当前位置
#         current = self.get_position()
#         if not current:
#             print("无法获取当前位置")
#             return False
        
#         # 计算相对移动距离
#         dx = target_x - current.get('x', 0)
#         dy = target_y - current.get('y', 0)
        
#         # 发送相对移动命令
#         try:
#             params = {"x": dx, "y": dy, "th": 0}
#             response = requests.get(f"{self.base_url}/bunny/robot/relative_move", params=params)
#             result = response.json()
            
#             if result.get('code') == 0:
#                 print("移动命令发送成功")
#                 time.sleep(5)  # 等待移动完成
#                 return True
#             else:
#                 print(f"移动失败: {result.get('msg', '')}")
#                 return False
#         except Exception as e:
#             print(f"移动异常: {e}")
#             return False
    
#     def start_localization(self):
#         """开启定位"""
#         try:
#             response = requests.get(f"{self.base_url}/bunny/robot/start_localization")
#             result = response.json()
#             if result.get('code') == 0:
#                 print("定位开启成功")
#                 return True
#             else:
#                 print(f"定位开启失败: {result.get('msg', '')}")
#                 return False
#         except Exception as e:
#             print(f"开启定位异常: {e}")
#             return False

# # 使用示例
# if __name__ == '__main__':
#     # 方式1: 使用默认端口80
#     robot = SimpleBunnyController("192.168.80.100")
    
#     # 方式2: 指定端口
#     # robot = SimpleBunnyController("192.168.80.100", 8080)
    
#     # 方式3: 使用自定义端口
#     # robot = SimpleBunnyController("192.168.80.100", port=9000)
    
#     print(f"连接到机器人: {robot.base_url}")
    
#     # 开启定位
#     if robot.start_localization():
#         time.sleep(2)
        
#         # 移动到目标位置
#         target_positions = [
#             (0.1, 0.1),
#             (0.1, 0.1),
#             (0.0, 0.0)
#         ]
        
#         for x, y in target_positions:
#             if robot.move_to(x, y):
#                 time.sleep(3)  # 在每个点停留3秒
#             else:
#                 print(f"移动到 ({x}, {y}) 失败，跳过")
        
#         print("移动任务完成!")
#     else:
#         print("定位开启失败，无法执行移动任务")
