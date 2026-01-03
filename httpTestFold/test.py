import requests

# 假设机器人的IP地址是 192.168.1.100
robot_ip = "192.168.80.100" 
base_url = f"http://{robot_ip}:43013"

def move_to_goal(x, y, theta, avoid_obstacles=True):
    """
    使用HTTP API命令机器人移动到指定目标点。

    :param x: 目标点 x 坐标 (米)
    :param y: 目标点 y 坐标 (米)
    :param theta: 目标姿态 (弧度)
    :param avoid_obstacles: 是否避障 (1 for true, 0 for false)
    """
    endpoint = "/bunny/robot/relative_move"
    
    # 根据文档，参数是放在URL里的
    # obstacle_avoid: 0表示停障, 1表示避障
    params = {
        "x": x,
        "y": y,
        "theta": theta
    }
    
    try:
        # 发送GET请求
        response = requests.get(f"{base_url}{endpoint}", params=params, timeout=5)
        response.raise_for_status()  # 如果请求失败 (如 404, 500), 会抛出异常

        # 解析返回的JSON数据
        result = response.json()
        print(f"请求发送成功: {result}")

        if result.get("code") == 0:
            print("机器人已接收导航指令，开始移动...")
        else:
            print(f"机器人返回错误: {result.get('msg')}")

    except requests.exceptions.RequestException as e:
        print(f"HTTP请求失败: {e}")

# --- 主程序 ---
if __name__ == "__main__":
    # 示例：让机器人移动到地图坐标 (2.5, 1.0)，最终朝向 0 弧度（通常是正前方）
    target_x = 0.1
    target_y = 0.1
    target_theta = 0.0
    
    print(f"准备发送导航指令到 ({target_x}, {target_y}, {target_theta})")
    move_to_goal(target_x, target_y, target_theta)