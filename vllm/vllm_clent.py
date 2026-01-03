#注意：请先安装 openai
#pip install openai
from openai import OpenAI

#设置 OpenAI API 密钥和 API 基础地址
openai_api_key = "EMPTY"  # 请替换为您的 API 密钥
openai_api_base = "http://localhost:8080/v1"  # 本地服务地址

client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

models = client.models.list()
model = models.data[0].id
prompt = "描述一下北京的秋天"

#Completion API 调用
completion= client.completions.create(model=model, prompt=prompt)
res = completion.choices[0].text.strip()
print(f"Prompt: {prompt}\nResponse: {res}")