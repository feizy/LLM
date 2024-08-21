import json

with open("E:\data\dolma\c4-0000.json", 'r', encoding='utf-8') as file:
    data = json.load(file)

# 假设 data 是一个列表，展示前 5 个元素
if isinstance(data, list):
    print("前5个元素:", data[:5])

# 如果 data 是一个字典，展示前5个键值对
elif isinstance(data, dict):
    for key in list(data.keys())[:5]:
        print(f"{key}: {data[key]}")