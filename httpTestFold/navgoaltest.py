import requests
import json

# 底盘基础配置
BASE_IP = "192.168.80.100"
NAV_GOAL_URL = f"http://{BASE_IP}:10001/bunny/nav/nav_goal"

def nav_to_coordinate(x: float, y: float, theta: float, obstacle_avoid: int = 1):
    """
    导航到指定坐标点
    :param x: 目标点x坐标
    :param y: 目标点y坐标
    :param theta: 目标点theta坐标
    :param obstacle_avoid: 避障策略 0-停障 1-绕障
    :return: 响应结果字典
    """
    # 验证避障参数
    if obstacle_avoid not in (0, 1):
        raise ValueError("obstacle_avoid只能是0(停障)或1(绕障)")
    
    # 构造请求参数
    params = {
        "x": x,
        "y": y,
        "theta": theta,
        "obstacle_avoid": obstacle_avoid
    }
    
    try:
        # 发送GET请求（注意：GET请求参数会自动拼接到URL后）
        response = requests.get(
            url=NAV_GOAL_URL,
            params=params,
            timeout=10  # 设置10秒超时
        )
        
        # 检查响应状态码
        response.raise_for_status()
        
        # 解析JSON响应
        result = response.json()
        print(f"导航请求响应成功：")
        print(f"状态码: {result.get('code')}")
        print(f"描述信息: {result.get('msg')}")
        print(f"指令类型: {result.get('type')}")
        
        return result
        
    except requests.exceptions.Timeout:
        print("错误：请求超时，请检查网络连接或底盘IP是否正确")
        return {"code": 1, "msg": "请求超时", "type": "nav_goal"}
    except requests.exceptions.ConnectionError:
        print(f"错误：无法连接到底盘IP {BASE_IP}，请检查网络或IP配置")
        return {"code": 1, "msg": "连接失败", "type": "nav_goal"}
    except requests.exceptions.HTTPError as e:
        print(f"错误：HTTP请求失败 {e}")
        return {"code": 1, "msg": f"HTTP错误: {e}", "type": "nav_goal"}
    except json.JSONDecodeError:
        print("错误：响应数据不是有效的JSON格式")
        return {"code": 1, "msg": "响应格式错误", "type": "nav_goal"}
    except Exception as e:
        print(f"未知错误：{e}")
        return {"code": 1, "msg": f"未知错误: {e}", "type": "nav_goal"}

def nav_to_mark_point(mark_id: int, obstacle_avoid: int = 1):
    """
    导航到标记点（扩展功能）
    :param mark_id: 地图上的点号
    :param obstacle_avoid: 避障策略 0-停障 1-绕障
    :return: 响应结果字典
    """
    # 构造标记点请求参数
    params = {
        "id": mark_id,
        "obstacle_avoid": obstacle_avoid
    }
    
    try:
        response = requests.get(
            url=NAV_GOAL_URL,
            params=params,
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        print(f"标记点导航响应：")
        print(f"状态码: {result.get('code')}")
        print(f"描述信息: {result.get('msg')}")
        return result
    except Exception as e:
        print(f"标记点导航失败：{e}")
        return {"code": 1, "msg": str(e), "type": "nav_goal"}

def main():
    """主函数：交互式输入坐标并发送导航请求"""
    print("===== 底盘导航控制程序 =====")
    print("请输入导航目标坐标信息：")
    
    # 交互式输入坐标
    try:
        x = float(input("目标点x坐标："))
        y = float(input("目标点y坐标："))
        theta = float(input("目标点theta坐标："))
        
        # 选择避障策略
        while True:
            obstacle_input = input("避障策略（0-停障，1-绕障）：")
            if obstacle_input in ("0", "1"):
                obstacle_avoid = int(obstacle_input)
                break
            print("输入错误，请输入0或1！")
        
        # 发送导航请求
        print("\n正在发送导航请求...")
        nav_to_coordinate(x, y, theta, obstacle_avoid)
        
    except ValueError:
        print("输入错误：坐标必须是数字格式！")
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行出错：{e}")

if __name__ == "__main__":
    # 方式1：交互式运行
    main()
    
    # 方式2：直接调用（示例）
    # nav_to_coordinate(x=1.5, y=2.8, theta=0.0, obstacle_avoid=1)
    
    # 方式3：导航到标记点（示例）
    # nav_to_mark_point(mark_id=5, obstacle_avoid=1)