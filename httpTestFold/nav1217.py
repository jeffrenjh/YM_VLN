import requests
import json
import os

# 底盘基础配置
BASE_IP = "192.168.0.5"
NAV_GOAL_URL = f"http://{BASE_IP}:10001/bunny/nav/nav_goal"

# JSON文件路径
POINT_JSON_PATH = "point.json"

def load_points_from_json(file_path: str) -> dict:
    """
    从JSON文件加载导航点信息
    :param file_path: JSON文件路径
    :return: 包含所有导航点的字典
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            points_dict = {}
            for point in data.get('points', []):
                points_dict[point['id']] = {
                    'x': point['x'],
                    'y': point['y'],
                    'theta': point['theta']
                }
            return points_dict
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON文件格式错误: {e}")
    except KeyError as e:
        raise ValueError(f"JSON文件缺少必要字段: {e}")

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

def display_available_points(points_dict: dict):
    """
    显示可用的导航点
    :param points_dict: 导航点字典
    """
    print("\n===== 可用导航点列表 =====")
    print(f"{'点编号':<10} {'X坐标':<15} {'Y坐标':<15} {'角度':<15}")
    print("-" * 60)
    for point_id, coords in points_dict.items():
        print(f"{point_id:<10} {coords['x']:<15.3f} {coords['y']:<15.3f} {coords['theta']:<15.3f}")
    print()

def select_and_navigate(points_dict: dict):
    """
    让用户选择导航点并执行导航
    :param points_dict: 导航点字典
    """
    display_available_points(points_dict)
    
    # 获取用户选择的点
    while True:
        selected_id = input("请选择要导航到的点编号（输入'q'退出）: ").strip()
        
        if selected_id.lower() == 'q':
            print("退出导航程序")
            return
            
        if selected_id in points_dict:
            break
        else:
            print(f"无效的点编号 '{selected_id}'，请重新输入！")
    
    # 选择避障策略
    while True:
        obstacle_input = input("避障策略（0-停障，1-绕障）：")
        if obstacle_input in ("0", "1"):
            obstacle_avoid = int(obstacle_input)
            break
        print("输入错误，请输入0或1！")
    
    # 获取选定点的坐标信息
    point_info = points_dict[selected_id]
    print(f"\n正在导航到点 {selected_id} (x={point_info['x']}, y={point_info['y']}, theta={point_info['theta']})...")
    
    # 执行导航
    nav_to_coordinate(point_info['x'], point_info['y'], point_info['theta'], obstacle_avoid)

def main():
    """主函数：从JSON文件加载导航点，并允许用户选择导航点"""
    print("===== 底盘导航控制程序 =====")
    
    try:
        # 加载导航点数据
        points_dict = load_points_from_json(POINT_JSON_PATH)
        
        if not points_dict:
            print("警告：未找到任何导航点数据")
            return
            
        # 进入导航选择循环
        while True:
            select_and_navigate(points_dict)
            
            # 询问是否继续导航
            continue_nav = input("\n是否继续导航到其他点？(y/n): ").strip().lower()
            if continue_nav != 'y' and continue_nav != 'yes':
                print("导航程序结束")
                break
                
    except FileNotFoundError as e:
        print(f"错误：{e}")
    except ValueError as e:
        print(f"错误：{e}")
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行出错：{e}")

if __name__ == "__main__":
    main()