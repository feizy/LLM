import numpy as np
from transformers import AutoTokenizer
import os

# 加载原始tokenizer
original_tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-0724-hf")

# 加载目标tokenizer
target_tokenizer = AutoTokenizer.from_pretrained("Qwen/CodeQwen1.5-7B-Chat")

# 设置数据路径
data_path = "/mnt/zzb_Term_4/TeamOne/data/eval_data/"

def process_file(file_path):
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint32)
    
    # 使用原始tokenizer解码
    try:
        decoded_text = original_tokenizer.decode(data, skip_special_tokens=True)
        print(f"解码后的文本长度: {len(decoded_text)}")
        print(f"解码后的文本片段: {decoded_text[:100]}...")
    except Exception as e:
        print(f"解码错误: {e}")
        return data, None, None
    
    # 使用目标tokenizer重新编码
    try:
        re_encoded = target_tokenizer.encode(decoded_text, add_special_tokens=False)
        if len(re_encoded) == 0:
            print("警告：重新编码后的列表为空")
    except Exception as e:
        print(f"编码错误: {e}")
        return data, decoded_text, None
    
    return data, decoded_text, re_encoded

# 遍历所有子文件夹和文件
output_path = "/mnt/zzb_Term_4/TeamOne/data/eval_data_v1/"
for root, dirs, files in os.walk(data_path):
    for filename in files:
        if filename.endswith('.npy'):
            file_path = os.path.join(root, filename)
            original_data, decoded_text, re_encoded = process_file(file_path)
            
            if decoded_text is None or re_encoded is None:
                print(f"跳过文件 {file_path} 由于处理错误")
                continue
            # 创建对应的输出目录
            relative_path = os.path.relpath(root, data_path)
            output_dir = os.path.join(output_path, relative_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # 将重新编码的数据写入新文件
            output_file = os.path.join(output_dir, filename)
            np.array(re_encoded, dtype=np.uint32).tofile(output_file)
            
            print(f"处理文件: {file_path}")
            print(f"原始数据形状: {original_data.shape}")
            print(f"解码后的文本片段: {decoded_text[:100]}...")
            print(f"重新编码后的形状: {len(re_encoded)}")
            print(f"重新编码后的片段: {re_encoded[:20]}")
            print(f"已保存到: {output_file}")
            print("-" * 50)


