import requests
import json
import time

class BunnyRobot:
    def __init__(self, robot_ip):
        """
        初始化机器人控制类
        :param robot_ip: 机器人的IP地址
        """
        self.base_url = f"http://{robot_ip}:10001"
        
    def set_cmd_vel(self, vx=0.0, vy=0.0, vth=0.0):
        """
        设置机器人速度
        :param vx: x方向线速度 (m/s)
        :param vy: y方向线速度 (m/s) - 文档显示无效果
        :param vth: z轴角速度 (rad/s)
        """
        url = f"{self.base_url}/bunny/robot/set_cmd_vel"
        params = {
            "vx": vx,
            "vy": vy,
            "vth": vth
        }
        
        try:
            response = requests.get(url, params=params)
            result = response.json()
            if result.get("code") == 0:
                print(f"速度设置成功: vx={vx}, vth={vth}")
                return True
            else:
                print(f"速度设置失败: {result.get('msg', '未知错误')}")
                return False
        except Exception as e:
            print(f"请求失败: {e}")
            return False
    
    def move_forward(self, speed=0.3, duration=1.0):
        """
        前进
        :param speed: 前进速度 (m/s)
        :param duration: 持续时间 (秒)
        """
        print(f"前进 - 速度: {speed} m/s, 持续时间: {duration}秒")
        self.set_cmd_vel(vx=speed, vth=0.0)
        time.sleep(duration)
        self.stop()
    
    def move_backward(self, speed=0.3, duration=1.0):
        """
        后退
        :param speed: 后退速度 (m/s)
        :param duration: 持续时间 (秒)
        """
        print(f"后退 - 速度: {speed} m/s, 持续时间: {duration}秒")
        self.set_cmd_vel(vx=-speed, vth=0.0)
        time.sleep(duration)
        self.stop()
    
    def turn_left(self, angular_speed=0.5, duration=1.0):
        """
        左转
        :param angular_speed: 角速度 (rad/s)
        :param duration: 持续时间 (秒)
        """
        print(f"左转 - 角速度: {angular_speed} rad/s, 持续时间: {duration}秒")
        self.set_cmd_vel(vx=0.0, vth=angular_speed)
        time.sleep(duration)
        self.stop()
    
    def turn_right(self, angular_speed=0.5, duration=1.0):
        """
        右转
        :param angular_speed: 角速度 (rad/s)
        :param duration: 持续时间 (秒)
        """
        print(f"右转 - 角速度: {angular_speed} rad/s, 持续时间: {duration}秒")
        self.set_cmd_vel(vx=0.0, vth=-angular_speed)
        time.sleep(duration)
        self.stop()
    
    def stop(self):
        """
        停止移动
        """
        print("停止移动")
        self.set_cmd_vel(vx=0.0, vth=0.0)
    
    def get_robot_speed(self):
        """
        获取当前机器人速度
        """
        # url = f"{self.base_url}/bunny/robot/speed"
        # try:
        #     response = requests.get(url)
        #     result = response.json()
        #     if result.get("code") == 0:
        #         data = result.get("data", {})
        #         vel_x = data.get("vel_x", 0)
        #         vel_theta = data.get("vel_theta", 0)
        #         print(f"当前速度 - 线速度: {vel_x} m/s, 角速度: {vel_theta} rad/s")
        #         return vel_x, vel_theta
        #     else:
        #         print(f"获取速度失败: {result.get('msg', '未知错误')}")
        #         return None, None
        # except Exception as e:
        #     print(f"获取速度请求失败: {e}")
        #     return None, None
        """
        获取当前机器人速度
        """
        url = f"{self.base_url}/bunny/robot/speed"
        result = self._make_request(url)
        
        if result and result.get("code") == 0:
            data = result.get("data", {})
            vel_x = data.get("vel_x", 0)
            vel_theta = data.get("vel_theta", 0)
            print(f"✓ 当前速度 - 线速度: {vel_x} m/s, 角速度: {vel_theta} rad/s")
            return vel_x, vel_theta
        else:
            error_msg = result.get('msg', '未知错误') if result else '请求失败'
            print(f"✗ 获取速度失败: {error_msg}")
            return None, None
    
    def get_robot_status(self):
        """
        获取机器人状态
        """
        url = f"{self.base_url}/bunny/robot/robot_status"
        print(f"请求机器人状态: {url}")
        try:
            response = requests.get(url)
            result = response.json()
            if result.get("code") == 0:
                data = result.get("data", {})
                robot_state = data.get("robot_state", 0)
                error_code = data.get("error_code", 0)
                message = data.get("message", "")
                
                # 状态码对应表
                status_map = {
                    100: "空闲状态",
                    110: "初始化定位",
                    120: "导航功能中",
                    121: "正在导航",
                    122: "停障中",
                    123: "导航失败",
                    124: "自动充电中",
                    125: "导航成功",
                    126: "自动充电成功",
                    127: "自动充电失败",
                    130: "建图功能中",
                    200: "发生错误",
                    201: "充电器直连充电",
                    202: "急停按下"
                }
                
                status_desc = status_map.get(robot_state, f"未知状态({robot_state})")
                print(f"机器人状态: {status_desc}")
                
                if error_code != 0:
                    print(f"错误码: {error_code}, 错误信息: {message}")
                
                return robot_state, error_code, message
            else:
                print(f"获取状态失败: {result.get('msg', '未知错误')}")
                return None, None, None
        except Exception as e:
            print(f"获取状态请求失败: {e}")
            return None, None, None

def main():
    """
    主函数 - 演示机器人移动控制
    """
    # 请替换为您的机器人IP地址
    robot_ip = "192.168.80.100"  # 示例IP，请根据实际情况修改
    
    # 创建机器人控制对象
    robot = BunnyRobot(robot_ip)
    
    print("=== Bunny机器人移动控制演示 ===")
    
    # 检查机器人状态
    print("\n1. 检查机器人状态:")
    robot.get_robot_status()
    
    # 获取当前速度
    print("\n2. 获取当前速度:")
    robot.get_robot_speed()
    
    print("\n3. 开始移动演示:")
    
    # 前进
    print("\n--- 前进测试 ---")
    robot.move_forward(speed=0.2, duration=0.5)
    time.sleep(1)
    
    # 后退
    print("\n--- 后退测试 ---")
    robot.move_backward(speed=0.2, duration=0.5)
    time.sleep(1)
    
    # 左转
    print("\n--- 左转测试 ---")
    robot.turn_left(angular_speed=0.3, duration=1.0)
    time.sleep(1)
    
    # 右转
    print("\n--- 右转测试 ---")
    robot.turn_right(angular_speed=0.3, duration=1.0)
    time.sleep(1)
    
    # 最终停止
    print("\n--- 最终停止 ---")
    robot.stop()
    
    print("\n移动演示完成!")

# 交互式控制函数
def interactive_control():
    """
    交互式控制机器人
    """
    # 请替换为您的机器人IP地址
    robot_ip = input("请输入机器人IP地址 (例如: 192.168.1.100): ").strip()
    if not robot_ip:
        robot_ip = "192.168.1.100"  # 默认IP
    
    robot = BunnyRobot(robot_ip)
    
    print("\n=== 交互式机器人控制 ===")
    print("命令说明:")
    print("w - 前进")
    print("s - 后退") 
    print("a - 左转")
    print("d - 右转")
    print("x - 停止")
    print("q - 退出")
    print("status - 查看状态")
    print("speed - 查看速度")
    
    while True:
        command = input("\n请输入命令: ").strip().lower()
        
        if command == 'w':
            robot.move_forward(speed=0.3, duration=1.0)
        elif command == 's':
            robot.move_backward(speed=0.3, duration=1.0)
        elif command == 'a':
            robot.turn_left(angular_speed=0.5, duration=1.0)
        elif command == 'd':
            robot.turn_right(angular_speed=0.5, duration=1.0)
        elif command == 'x':
            robot.stop()
        elif command == 'status':
            robot.get_robot_status()
        elif command == 'speed':
            robot.get_robot_speed()
        elif command == 'q':
            robot.stop()
            print("退出控制程序")
            break
        else:
            print("无效命令，请重新输入")

if __name__ == "__main__":
    # 选择运行模式
    mode = input("选择运行模式 (1-演示模式, 2-交互模式): ").strip()
    
    if mode == "1":
        main()
    elif mode == "2":
        interactive_control()
    else:
        print("无效选择，运行演示模式")
        main()
