import requests
import json
import time

class BunnyRobotDiagnostic:
    def __init__(self, robot_ip):
        """
        初始化机器人诊断类
        :param robot_ip: 机器人的IP地址
        """
        self.base_url = f"http://{robot_ip}"
        self.timeout = 5
        
    def _make_request(self, url, params=None, method='GET'):
        """统一的请求处理方法"""
        try:
            print(f"🔗 发送请求: {url}")
            if params:
                print(f"📝 请求参数: {params}")
            
            if method.upper() == 'GET':
                response = requests.get(url, params=params, timeout=self.timeout)
            elif method.upper() == 'POST':
                response = requests.post(url, json=params, timeout=self.timeout)
            
            print(f"📊 HTTP状态码: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    print(f"✅ 响应内容: {json.dumps(result, indent=2, ensure_ascii=False)}")
                    return result
                except json.JSONDecodeError:
                    print(f"❌ JSON解析失败，原始响应: {response.text}")
                    return None
            else:
                print(f"❌ HTTP错误: {response.status_code}, 响应: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ 请求异常: {e}")
            return None
    
    def get_full_robot_status(self):
        """获取完整的机器人状态信息"""
        print("\n" + "="*50)
        print("🔍 完整机器人状态诊断")
        print("="*50)
        
        # 1. 获取机器人状态
        print("\n1️⃣ 机器人状态:")
        url = f"{self.base_url}/bunny/robot/robot_status"
        status_result = self._make_request(url)
        
        if status_result and status_result.get("code") == 0:
            data = status_result.get("data", {})
            robot_state = data.get("robot_state", 0)
            error_code = data.get("error_code", 0)
            message = data.get("message", "")
            
            status_map = {
                100: "空闲状态 ✅",
                110: "初始化定位 🔄",
                120: "导航功能中 🧭",
                121: "正在导航 🚀",
                122: "停障中 ⚠️",
                123: "导航失败 ❌",
                124: "自动充电中 🔋",
                125: "导航成功 ✅",
                126: "自动充电成功 ✅",
                127: "自动充电失败 ❌",
                130: "建图功能中 🗺️",
                200: "发生错误 ❌",
                201: "充电器直连充电 🔌",
                202: "急停按下 🛑"
            }
            
            status_desc = status_map.get(robot_state, f"未知状态({robot_state})")
            print(f"   状态码: {robot_state}")
            print(f"   状态描述: {status_desc}")
            print(f"   错误码: {error_code}")
            print(f"   消息: {message}")
            
            # 判断是否可以移动
            movable_states = [100, 110, 120]  # 空闲、初始化定位、导航功能中
            if robot_state in movable_states:
                print("   ✅ 状态允许移动")
            else:
                print("   ❌ 当前状态不允许移动")
                return False
        else:
            print("   ❌ 无法获取机器人状态")
            return False
        
        # 2. 获取底盘状态
        print("\n2️⃣ 底盘状态:")
        url = f"{self.base_url}/bunny/robot/chassis"
        chassis_result = self._make_request(url)
        
        if chassis_result and chassis_result.get("code") == 0:
            data = chassis_result.get("data", {})
            battery_soc = data.get("battery_soc", 0)
            error_code = data.get("error_code", 0)
            hard_estop = data.get("hard_estop", 0)
            motor_servo = data.get("motor_servo", 0)
            vel_x = data.get("vel_x", 0)
            vel_theta = data.get("vel_theta", 0)
            
            print(f"   电池电量: {battery_soc}%")
            print(f"   错误码: {error_code}")
            print(f"   急停状态: {hard_estop} {'❌ 急停激活' if hard_estop else '✅ 正常'}")
            print(f"   电机伺服: {motor_servo} {'✅ 已启用' if motor_servo else '❌ 未启用'}")
            print(f"   当前线速度: {vel_x} m/s")
            print(f"   当前角速度: {vel_theta} rad/s")
            
            # 检查底盘是否可以移动
            if hard_estop:
                print("   ❌ 急停被激活，无法移动")
                return False
            if error_code != 0:
                print("   ⚠️ 底盘有错误码")
            if battery_soc < 10:
                print("   ⚠️ 电池电量过低")
        else:
            print("   ❌ 无法获取底盘状态")
        
        # 3. 获取当前速度
        # print("\n3️⃣ 当前速度:")
        # url = f"{self.base_url}/bunny/robot/speed"
        # speed_result = self._make_request(url)
        
        # if speed_result and speed_result.get("code") == 0:
        #     data = speed_result.get("data", {})
        #     vel_x = data.get("vel_x", 0)
        #     vel_theta = data.get("vel_theta", 0)
        #     print(f"   线速度: {vel_x} m/s")
        #     print(f"   角速度: {vel_theta} rad/s")
        
        # 4. 获取定位状态
        # print("\n4️⃣ 定位状态:")
        # url = f"{self.base_url}/bunny/robot/get_localization_pose"
        # loc_result = self._make_request(url)
        
        # if loc_result and loc_result.get("code") == 0:
        #     data = loc_result.get("data", {})
        #     x = data.get("x", 0)
        #     y = data.get("y", 0)
        #     theta = data.get("theta", 0)
        #     print(f"   位置: x={x:.3f}, y={y:.3f}, theta={theta:.3f}")
        #     print("   ✅ 定位正常")
        # else:
        #     print("   ❌ 定位未启用或异常")
        #     print("   💡 提示: 可能需要先启用定位")
        
        return True
    
    def start_localization(self):
        """启动定位"""
        print("\n🧭 启动定位...")
        url = f"{self.base_url}/bunny/robot/start_localization"
        result = self._make_request(url)
        
        if result and result.get("code") == 0:
            print("✅ 定位启动成功")
            time.sleep(2)  # 等待定位启动
            return True
        else:
            print("❌ 定位启动失败")
            return False
    
    def test_movement_step_by_step(self):
        """逐步测试移动功能"""
        print("\n" + "="*50)
        print("🎮 逐步测试移动功能")
        print("="*50)
        
        # 1. 先检查状态
        if not self.get_full_robot_status():
            print("\n❌ 机器人状态不允许移动，尝试启动定位...")
            if not self.start_localization():
                print("❌ 无法启动定位，请检查机器人状态")
                return
            
            # 重新检查状态
            time.sleep(3)
            if not self.get_full_robot_status():
                print("❌ 启动定位后状态仍不正常")
                return
        
        # 2. 测试设置速度命令
        print("\n🧪 测试速度设置...")
        
        test_cases = [
            {"name": "微小前进", "vx": 0.3, "vth": 0.0, "duration": 1},
            {"name": "停止", "vx": 0.0, "vth": 0.0, "duration": 1},
            {"name": "微小后退", "vx": -0.3, "vth": 0.0, "duration": 1},
            {"name": "停止", "vx": 0.0, "vth": 0.0, "duration": 1},
            {"name": "微小左转", "vx": 0.0, "vth": 0.4, "duration": 1},
            {"name": "停止", "vx": 0.0, "vth": 0.0, "duration": 1},
            {"name": "微小右转", "vx": 0.0, "vth": -0.4, "duration": 1},
            {"name": "最终停止", "vx": 0.0, "vth": 0.0, "duration": 1},
        ]
        
        for i, test in enumerate(test_cases):
            print(f"\n--- 测试 {i+1}: {test['name']} ---")
            
            # 发送速度命令
            url = f"{self.base_url}/bunny/robot/set_cmd_vel"
            params = {
                "vx": test["vx"],
                "vy": 0.0,
                "vth": test["vth"]
            }
            
            result = self._make_request(url, params)
            
            if result and result.get("code") == 0:
                print(f"✅ 速度命令发送成功")
                print(f"⏱️ 等待 {test['duration']} 秒...")
                
                # 等待指定时间
                time.sleep(test['duration'])
                
                # 检查当前速度
                print("📊 检查当前速度:")
                speed_url = f"{self.base_url}/bunny/robot/speed"
                speed_result = self._make_request(speed_url)
                
                if speed_result and speed_result.get("code") == 0:
                    data = speed_result.get("data", {})
                    actual_vx = data.get("vel_x", 0)
                    actual_vth = data.get("vel_theta", 0)
                    
                    print(f"   期望速度: vx={test['vx']}, vth={test['vth']}")
                    print(f"   实际速度: vx={actual_vx}, vth={actual_vth}")
                    
                    # 检查速度是否匹配（允许一定误差）
                    vx_match = abs(actual_vx - test['vx']) < 0.05
                    vth_match = abs(actual_vth - test['vth']) < 0.05
                    
                    if vx_match and vth_match:
                        print("   ✅ 速度匹配，机器人应该在移动")
                    else:
                        print("   ⚠️ 速度不匹配，可能有问题")
                
            else:
                print(f"❌ 速度命令发送失败")
            
            # 询问用户是否继续
            if i < len(test_cases) - 1:
                user_input = input("按回车继续下一个测试，或输入 'q' 退出: ").strip()
                if user_input.lower() == 'q':
                    break
    
    def manual_speed_control(self):
        """手动速度控制"""
        print("\n" + "="*50)
        print("🎮 手动速度控制模式")
        print("="*50)
        print("命令格式: vx,vth (例如: 0.2,0 表示前进)")
        print("快捷命令:")
        print("  w - 前进 (0.2,0)")
        print("  s - 后退 (-0.2,0)")
        print("  a - 左转 (0,0.3)")
        print("  d - 右转 (0,-0.3)")
        print("  x - 停止 (0,0)")
        print("  q - 退出")
        
        while True:
            try:
                cmd = input("\n🎮 输入命令: ").strip().lower()
                
                if cmd == 'q':
                    break
                elif cmd == 'w':
                    vx, vth = 0.2, 0.0
                elif cmd == 's':
                    vx, vth = -0.2, 0.0
                elif cmd == 'a':
                    vx, vth = 0.0, 0.3
                elif cmd == 'd':
                    vx, vth = 0.0, -0.3
                elif cmd == 'x':
                    vx, vth = 0.0, 0.0
                elif ',' in cmd:
                    try:
                        parts = cmd.split(',')
                        vx = float(parts[0])
                        vth = float(parts[1])
                    except:
                        print("❌ 格式错误，请使用 vx,vth 格式")
                        continue
                else:
                    print("❌ 无效命令")
                    continue
                
                # 发送速度命令
                url = f"{self.base_url}/bunny/robot/set_cmd_vel"
                params = {"vx": vx, "vy": 0.0, "vth": vth}
                
                result = self._make_request(url, params)
                
                if result and result.get("code") == 0:
                    print(f"✅ 速度设置成功: vx={vx}, vth={vth}")
                else:
                    print(f"❌ 速度设置失败")
                
            except KeyboardInterrupt:
                print("\n🛑 停止机器人...")
                url = f"{self.base_url}/bunny/robot/set_cmd_vel"
                params = {"vx": 0.0, "vy": 0.0, "vth": 0.0}
                self._make_request(url, params)
                break
            except Exception as e:
                print(f"❌ 错误: {e}")

def main():
    print("🤖 Bunny机器人诊断和控制工具")
    
    robot_ip = input("请输入机器人IP地址: ").strip()
    robot_ip = robot_ip + ":10001" 
    if not robot_ip:
        print("❌ 请输入有效的IP地址")
        return
    
    diagnostic = BunnyRobotDiagnostic(robot_ip)
    
    while True:
        print("\n" + "="*50)
        print("选择操作:")
        print("1 - 完整状态诊断")
        print("2 - 逐步测试移动")
        print("3 - 手动速度控制")
        print("4 - 启动定位")
        print("q - 退出")
        
        choice = input("请选择 (1-4/q): ").strip()
        
        if choice == '1':
            diagnostic.get_full_robot_status()
        elif choice == '2':
            diagnostic.test_movement_step_by_step()
        elif choice == '3':
            diagnostic.manual_speed_control()
        elif choice == '4':
            diagnostic.start_localization()
        elif choice.lower() == 'q':
            break
        else:
            print("❌ 无效选择")

if __name__ == "__main__":
    main()
