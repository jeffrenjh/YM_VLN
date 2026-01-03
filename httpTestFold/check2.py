#!/usr/bin/env python3
import requests
import json
import time

class DeepChassisAnalyzer:
    def __init__(self, ip="192.168.0.117", port=10001):
        self.base_url = f"http://{ip}:{port}/bunny/robot"
        
    def get_all_status(self):
        """获取所有可能的状态信息"""
        endpoints = [
            "/chassis",
            "/status", 
            "/robot_status",
            "/system_status",
            "/mode",
            "/control_mode",
            "/safety_status"
        ]
        
        results = {}
        for endpoint in endpoints:
            try:
                url = self.base_url + endpoint
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    results[endpoint] = response.json()
                    print(f"✓ {endpoint}: 获取成功")
                else:
                    print(f"✗ {endpoint}: HTTP {response.status_code}")
            except Exception as e:
                print(f"✗ {endpoint}: {e}")
        
        return results
    
    def test_all_movement_apis(self):
        """测试所有可能的移动API"""
        movement_apis = [
            ("/set_cmd_vel", {"vx": 0.3, "vy": 0.0, "vth": 0.0}),
            ("/cmd_vel", {"vx": 0.3, "vy": 0.0, "vth": 0.0}),
            ("/move", {"linear": 0.3, "angular": 0.0}),
            ("/velocity", {"x": 0.3, "y": 0.0, "theta": 0.0}),
            ("/control", {"cmd": "move", "vx": 0.3, "vy": 0.0, "vth": 0.0}),
        ]
        
        print("\n=== 测试所有移动API ===")
        for endpoint, payload in movement_apis:
            try:
                url = self.base_url + endpoint
                response = requests.post(url, json=payload, timeout=3)
                print(f"{endpoint}: HTTP {response.status_code}")
                if response.status_code == 200:
                    result = response.json()
                    print(f"  响应: {result}")
                time.sleep(1)
            except Exception as e:
                print(f"{endpoint}: 异常 - {e}")
    
    def check_safety_locks(self):
        """检查安全锁定状态"""
        print("\n=== 检查安全状态 ===")
        
        # 获取底盘状态
        try:
            response = requests.get(f"{self.base_url}/chassis", timeout=3)
            if response.status_code == 200:
                data = response.json().get('data', {})
                
                safety_checks = [
                    ("急停状态", "hard_estop", 0),
                    ("软急停", "soft_estop", 0), 
                    ("电机伺服", "motor_servo", 1),
                    ("底盘使能", "chassis_enable", 1),
                    ("运动使能", "motion_enable", 1),
                    ("控制模式", "control_mode", None),
                    ("系统状态", "system_state", None),
                    ("错误码", "error_code", 0),
                    ("警告码", "warning_code", 0),
                ]
                
                for name, key, expected in safety_checks:
                    value = data.get(key, "未知")
                    if expected is not None:
                        status = "✓" if value == expected else "✗"
                        print(f"  {status} {name}: {value} (期望: {expected})")
                    else:
                        print(f"  ℹ {name}: {value}")
                
                # 检查其他可能的锁定字段
                lock_fields = ['locked', 'disabled', 'blocked', 'paused', 'inhibited']
                for field in lock_fields:
                    if field in data:
                        print(f"  ⚠ {field}: {data[field]}")
                        
        except Exception as e:
            print(f"获取安全状态失败: {e}")
    
    def test_enable_commands(self):
        """尝试各种使能命令"""
        print("\n=== 尝试使能命令 ===")
        
        enable_commands = [
            ("/enable", {}),
            ("/enable_chassis", {}),
            ("/enable_motion", {}),
            ("/set_enable", {"enable": True}),
            ("/motor_enable", {"enable": True}),
            ("/servo_enable", {"enable": True}),
            ("/unlock", {}),
            ("/resume", {}),
            ("/start", {}),
        ]
        
        for endpoint, payload in enable_commands:
            try:
                url = self.base_url + endpoint
                if payload:
                    response = requests.post(url, json=payload, timeout=3)
                else:
                    response = requests.post(url, timeout=3)
                
                print(f"{endpoint}: HTTP {response.status_code}")
                if response.status_code == 200:
                    result = response.json()
                    print(f"  响应: {result}")
                    
            except Exception as e:
                print(f"{endpoint}: 异常 - {e}")
    
    def monitor_real_movement(self):
        """监控实际运动状态"""
        print("\n=== 监控实际运动 ===")
        
        # 记录初始位置
        initial_status = self.get_chassis_status()
        if not initial_status:
            print("无法获取初始状态")
            return
            
        print(f"初始状态:")
        print(f"  位置: x={initial_status.get('pos_x', 0):.3f}, y={initial_status.get('pos_y', 0):.3f}")
        print(f"  角度: {initial_status.get('pos_theta', 0):.3f}")
        print(f"  速度: vx={initial_status.get('vel_x', 0):.3f}, vth={initial_status.get('vel_theta', 0):.3f}")
        
        # 发送移动指令
        print(f"\n发送移动指令...")
        move_success = self.send_move_command(0.3, 0.0, 0.0)
        if not move_success:
            print("移动指令发送失败")
            return
            
        # 监控5秒
        print(f"监控5秒...")
        for i in range(10):
            time.sleep(0.5)
            current_status = self.get_chassis_status()
            if current_status:
                pos_x = current_status.get('pos_x', 0)
                pos_y = current_status.get('pos_y', 0)
                vel_x = current_status.get('vel_x', 0)
                vel_th = current_status.get('vel_theta', 0)
                
                # 计算位置变化
                dx = pos_x - initial_status.get('pos_x', 0)
                dy = pos_y - initial_status.get('pos_y', 0)
                distance = (dx**2 + dy**2)**0.5
                
                print(f"  {i*0.5:.1f}s: 位移={distance:.4f}m, 速度=({vel_x:.3f}, {vel_th:.3f})")
                
                if distance > 0.01:  # 移动了1cm以上
                    print(f"  ✓ 检测到实际移动!")
                    break
        else:
            print(f"  ✗ 未检测到实际移动")
        
        # 停止
        self.send_move_command(0, 0, 0)
    
    def get_chassis_status(self):
        """获取底盘状态"""
        try:
            response = requests.get(f"{self.base_url}/chassis", timeout=3)
            if response.status_code == 200:
                return response.json().get('data', {})
        except:
            pass
        return None
    
    def send_move_command(self, vx, vy, vth):
        """发送移动指令"""
        try:
            payload = {"vx": vx, "vy": vy, "vth": vth}
            response = requests.post(f"{self.base_url}/set_cmd_vel", json=payload, timeout=3)
            return response.status_code == 200 and response.json().get('code') == 0
        except:
            return False
    
    def full_diagnostic(self):
        """完整诊断"""
        print("=== 底盘深度诊断 ===\n")
        
        # 1. 获取所有状态
        print("1. 获取所有状态信息...")
        all_status = self.get_all_status()
        
        # 2. 检查安全锁定
        self.check_safety_locks()
        
        # 3. 尝试使能命令
        self.test_enable_commands()
        
        # 4. 测试所有移动API
        self.test_all_movement_apis()
        
        # 5. 监控实际运动
        self.monitor_real_movement()
        
        # 6. 输出完整状态用于分析
        print(f"\n=== 完整状态信息 ===")
        for endpoint, data in all_status.items():
            print(f"\n{endpoint}:")
            print(json.dumps(data, indent=2, ensure_ascii=False))

def main():
    analyzer = DeepChassisAnalyzer()
    
    try:
        analyzer.full_diagnostic()
    except KeyboardInterrupt:
        print("\n\n诊断被中断")
    except Exception as e:
        print(f"\n诊断出现异常: {e}")

if __name__ == "__main__":
    main()
