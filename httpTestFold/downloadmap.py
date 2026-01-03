import requests
import json
import os

# 底盘基础配置
BASE_IP = "192.168.0.5"
DOWNLOAD_MAP_URL = f"http://{BASE_IP}:10001/bunny/map/download_map"

def download_map(map_id: str, save_path: str = "./") -> bool:
    """
    下载指定map_id的地图文件
    :param map_id: 地图ID（如"2300000"）
    :param save_path: 地图文件保存路径（默认当前目录）
    :return: 下载成功返回True，失败返回False
    """
    # 验证map_id参数
    if not isinstance(map_id, str) or not map_id.strip():
        print("错误：map_id必须是非空字符串！")
        return False
    
    # 构造请求参数
    params = {"map_id": 250805200640}
    
    try:
        # 发送GET请求，流式下载（适配大文件）
        response = requests.get(
            url=DOWNLOAD_MAP_URL,
            params=params,
            timeout=30,  # 下载超时设为30秒（可根据实际调整）
            stream=True  # 流式处理，避免大文件占用过多内存
        )
        response.raise_for_status()  # 检查HTTP状态码（非200则抛出异常）

        # 判断响应类型：成功返回二进制文件，失败返回JSON
        try:
            # 尝试解析为JSON（失败场景）
            error_info = response.json()
            print(f"下载失败：")
            print(f"错误码: {error_info.get('code')}")
            print(f"失败原因: {error_info.get('msg')}")
            print(f"指令类型: {error_info.get('type')}")
            return False
        except json.JSONDecodeError:
            # 解析JSON失败，说明是二进制文件（成功场景）
            # 拼接保存路径
            file_path = os.path.join(save_path, "map.zip")
            # 写入二进制文件
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):  # 分块写入
                    if chunk:
                        f.write(chunk)
            print(f"地图下载成功！文件保存至：{os.path.abspath(file_path)}")
            return True

    except requests.exceptions.Timeout:
        print("错误：请求超时，请检查网络连接或底盘IP是否正确")
        return False
    except requests.exceptions.ConnectionError:
        print(f"错误：无法连接到底盘IP {BASE_IP}，请检查网络或IP配置")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"错误：HTTP请求失败 {e}")
        return False
    except IOError as e:
        print(f"错误：文件保存失败 {e}，请检查保存路径权限")
        return False
    except Exception as e:
        print(f"未知错误：{e}")
        return False

if __name__ == "__main__":
    # 示例：下载map_id为"2300000"的地图，保存到当前目录
    download_map(map_id="2300000", save_path="./")