import requests

# 底盘IP和关闭导航的接口URL
BASE_IP = "192.168.0.5"
CLOSE_NAV_URL = f"http://{BASE_IP}:10001/bunny/nav/close_navigation"

def close_navigation():
    """发送关闭导航的HTTP GET请求"""
    try:
        # 发送GET请求，超时10秒
        response = requests.get(CLOSE_NAV_URL, timeout=10)
        # 解析响应JSON
        result = response.json()
        
        print(f"关闭导航响应结果：")
        print(f"状态码: {result.get('code')}")
        print(f"描述信息: {result.get('msg')}")
        print(f"指令类型: {result.get('type')}")
        
        return result
        
    except requests.exceptions.Timeout:
        print("错误：请求超时，检查网络或底盘IP")
        return {"code": 1, "msg": "请求超时", "type": "close_navigation"}
    except requests.exceptions.ConnectionError:
        print(f"错误：无法连接到底盘 {BASE_IP}")
        return {"code": 1, "msg": "连接失败", "type": "close_navigation"}
    except Exception as e:
        print(f"错误：{e}")
        return {"code": 1, "msg": str(e), "type": "close_navigation"}

# 直接调用关闭导航函数
if __name__ == "__main__":
    close_navigation()