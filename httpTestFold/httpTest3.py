import requests
import json

class RobotStatusClient:
    def __init__(self, base_url):
        """
        初始化机器人状态客户端
        :param base_url: 机器人API的基础URL，例如 'http://192.168.1.100:8080'
        """
        self.base_url = base_url.rstrip('/')
        
        # 状态码映射
        self.status_codes = {
            100: "IDLE - 空闲状态",
            110: "INIT - 初始化定位",
            120: "NAVIGATION - 机器人处于导航功能中",
            121: "NAVIGATING - 机器人正在导航",
            122: "NAV_STOP_OBS - 机器人停障中",
            123: "NAV_ERROR - 导航失败",
            124: "AUTO_CHARGING - 自动充电中",
            125: "NAV_FINISHED - 导航成功",
            126: "AUTO_CHARGING_FINISHED - 自动充电成功",
            127: "AUTO_CHARGING_FAILED - 自动充电失败",
            130: "MAPPING - 机器人处于建图功能中",
            200: "ERROR - 发生错误",
            201: "MANUAL_CHARGING - 充电器直连充电",
            202: "HARDWARE - 急停按下"
        }
    
    def get_robot_status(self):
        """
        获取机器人状态
        :return: 包含状态信息的字典
        """
        try:
            url = f"{self.base_url}/bunny/robot/robot_status"
            print(f"请求机器人状态: {url}")
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self.parse_status_response(data)
            else:
                return {
                    "success": False,
                    "error": f"HTTP错误: {response.status_code}",
                    "raw_response": response.text
                }
                
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"请求异常: {str(e)}"
            }
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"JSON解析错误: {str(e)}"
            }
    
    def parse_status_response(self, response_data):
        """
        解析状态响应数据
        :param response_data: API返回的原始数据
        :return: 解析后的状态信息
        """
        try:
            code = response_data.get("code", -1)
            data = response_data.get("data", {})
            
            if code == 0:  # 成功
                robot_state = data.get("robot_state", -1)
                error_code = data.get("error_code", -1)
                message = data.get("message", "")
                
                status_description = self.status_codes.get(robot_state, f"未知状态码: {robot_state}")
                
                return {
                    "success": True,
                    "robot_state": robot_state,
                    "status_description": status_description,
                    "error_code": error_code,
                    "message": message,
                    "type": response_data.get("type", ""),
                    "raw_response": response_data
                }
            else:
                return {
                    "success": False,
                    "error": f"API返回错误码: {code}",
                    "message": response_data.get("msg", ""),
                    "raw_response": response_data
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"数据解析错误: {str(e)}",
                "raw_response": response_data
            }
    
    def is_robot_idle(self):
        """检查机器人是否处于空闲状态"""
        status = self.get_robot_status()
        return status.get("success", False) and status.get("robot_state") == 100
    
    def is_robot_navigating(self):
        """检查机器人是否正在导航"""
        status = self.get_robot_status()
        robot_state = status.get("robot_state", -1)
        return status.get("success", False) and robot_state in [120, 121]
    
    def is_robot_charging(self):
        """检查机器人是否正在充电"""
        status = self.get_robot_status()
        robot_state = status.get("robot_state", -1)
        return status.get("success", False) and robot_state in [124, 201]
    
    def is_robot_error(self):
        """检查机器人是否处于错误状态"""
        status = self.get_robot_status()
        robot_state = status.get("robot_state", -1)
        return status.get("success", False) and robot_state in [200, 123, 127, 202]

# 使用示例
if __name__ == "__main__":
    # 创建客户端实例（请替换为实际的机器人IP地址和端口）
    client = RobotStatusClient("http://192.168.80.100:10001")
    
    # 获取机器人状态
    status = client.get_robot_status()
    
    if status["success"]:
        print(f"机器人状态: {status['status_description']}")
        print(f"状态码: {status['robot_state']}")
        print(f"错误码: {status['error_code']}")
        print(f"消息: {status['message']}")
        
        # 检查特定状态
        if client.is_robot_idle():
            print("机器人当前空闲")
        elif client.is_robot_navigating():
            print("机器人正在导航中")
        elif client.is_robot_charging():
            print("机器人正在充电")
        elif client.is_robot_error():
            print("机器人处于错误状态")
    else:
        print(f"获取状态失败: {status['error']}")
