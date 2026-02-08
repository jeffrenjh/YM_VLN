import requests
import json
import argparse
import sys

def query_llm(target_object_name):
    # 1. 读取本地物体坐标JSON文件
    coord_file_path = "/home/nvidia/huangjie/YM_VLN/test/YM_VLN/yolotest/yoloV8/myproject/output/detection_results_clustered.json"
    try:
        with open(coord_file_path, "r", encoding="utf-8") as f:
            coord_data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 {coord_file_path}")
        return

    # 2. 构造 Prompt
    # 增加多语言支持提示，并明确告知模型环境数据为英文class
    prompt = f"""
你是一个空间坐标助手。请根据[环境数据]回答用户的查询。
注意：[环境数据]中的 class 字段是英文。如果用户使用中文查询，请根据语义匹配对应的英文类别（例如："笔记本"-> "laptop", "电视"-> "tv"）。

[环境数据]
{json.dumps(coord_data, ensure_ascii=False, indent=2)}

[用户查询]
请找到与 "{target_object_name}" 匹配的物体，输出它的 id 和 xyz 坐标。
如果没有找到匹配的物体，请回答“未找到”。

[回答]
"""
    # print("-" * 20 + " 发送给 LLM 的 Prompt " + "-" * 20)
    # print(prompt.strip())
    # print("-" * 60)

    # 3. 调用本地LLM服务
    # 使用 localhost，如果需要远程请修改 IP
    url = "http://192.168.0.189:8080/v1/completions"
    headers = {"Content-Type": "application/json"}
    
    data = {
        "model": "/home/ym/data/huangjie/dataset/Qwen3-8B",
        "prompt": prompt,
        "max_tokens": 128,
        "temperature": 0.1,
        "stop": ["[用户查询]", "[环境数据]", "User:", "###"],
        "echo": False
    }

    # 发送请求并处理结果
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        generated_text = result["choices"][0]["text"].strip()
        
        print("\n" + "=" * 20 + " LLM 返回结果 " + "=" * 20)
        print(generated_text)
        print("=" * 56)
        
    except Exception as e:
        print(f"调用失败：{e}")
        if 'response' in locals():
            print(f"响应详情：{response.text}")

if __name__ == "__main__":
    # 如果命令行有参数，则使用参数模式
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="使用 LLM 查询物体坐标")
        parser.add_argument("--target", type=str, required=True, help="想要查找的物体名称 (例如: laptop, tv, 电视)")
        args = parser.parse_args()
        query_llm(args.target)
    else:
        # 否则进入交互模式
        print(">>> 已进入交互模式 (输入 'quit', 'exit' 或 'q' 退出) <<<")
        while True:
            try:
                user_input = input("\n请输入想要查的物体名称 (支持中英文): ").strip()
                if user_input.lower() in ['q', 'exit', 'quit']:
                    print("退出程序。")
                    break
                if not user_input:
                    continue
                
                query_llm(user_input)
            except KeyboardInterrupt:
                print("\n退出程序。")
                break