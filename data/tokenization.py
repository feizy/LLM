import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
import os

# 第一步：下载数据集
dataset = load_dataset("jed351/Chinese-Common-Crawl-Filtered", split='train')
print(dataset.column_names)
print(dataset.features)
# 第二步：加载预训练的tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/CodeQwen1.5-7B-Chat")


# 第三步：定义token化函数并进行处理
def tokenize_function(examples, tokenizer, max_seq_len):
    # Token化并处理批次数据
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_seq_len)


# 指定最大序列长度
max_seq_len = 2048

# 第四步：设置输出目录
output_dir = Path("/mnt/zzb_Term_4/TeamOne/data/QW_ccchinese_tokens")
output_dir.mkdir(parents=True, exist_ok=True)

# 定义每个npy文件的最大大小（以字节为单位），例如2GB
max_file_size = 2 * 1024 * 1024 * 1024  # 2GB

# 初始化变量
current_file_index = 0
current_data = []


def save_to_npy(data, file_index):
    """将数据保存为npy文件"""
    npy_path = output_dir / f"input_ids_{file_index}.npy"
    np.save(npy_path, np.array(data, dtype=np.uint16))
    print(f"Saved {npy_path} with {len(data)} items")


# 处理数据集并分块保存
for example in dataset:
    # 使用tokenizer对每个样本进行token化
    tokenized_output = tokenize_function(example, tokenizer, max_seq_len)

    # 将tokenized_output中的input_ids添加到当前数据块
    current_data.append(tokenized_output['input_ids'])

    # 检查当前数据块的大小
    current_data_size = len(current_data) * max_seq_len * np.dtype(np.uint16).itemsize

    if current_data_size >= max_file_size:
        # 保存当前数据块到npy文件
        save_to_npy(current_data, current_file_index)

        # 更新文件索引和重置当前数据块
        current_file_index += 1
        current_data = []

# 检查最后一个数据块是否有剩余
if current_data:
    save_to_npy(current_data, current_file_index)

print("Tokenization and saving completed!")