import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from functools import partial

# 第一步：加载数据集
dataset = load_dataset("jed351/Chinese-Common-Crawl-Filtered", split='train')

# 第二步：加载Qwen/CodeQwen1.5-7B-Chat的tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/CodeQwen1.5-7B-Chat")


# 第三步：定义token化函数
def tokenize_function(examples, tokenizer, max_seq_len):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_seq_len)


# 设置最大序列长度
max_seq_len = 2048

# 第四步：设置输出目录
output_dir = Path("/mnt_llm/zzb_Term_4/TeamOne/data/QW_ccchinese_tokens")
output_dir.mkdir(parents=True, exist_ok=True)

# 定义每个npy文件的最大大小（以字节为单位），例如2GB
max_file_size = 3 * 1024 * 1024 * 1024  # 2GB

# 初始化变量
current_file_index = 0
current_data = []


def save_to_npy(data, file_index):
    """将数据保存为npy文件"""
    npy_path = output_dir / f"input_ids_{file_index}.npy"
    np.save(npy_path, np.array(data, dtype=np.uint16))
    print(f"Saved {npy_path} with {len(data)} items")


# 使用tqdm显示进度条
def tokenize_with_progress(dataset, tokenizer, max_seq_len):
    # 获取数据集长度
    dataset_length = len(dataset)

    # 创建一个进度条
    with tqdm(total=dataset_length, desc="Tokenizing data") as pbar:
        def update_progress(batch):
            nonlocal pbar
            pbar.update(len(batch['text']))
            return tokenize_function(batch, tokenizer, max_seq_len)

        return dataset.map(update_progress, batched=True, remove_columns=['text'], num_proc=1)


# 执行tokenization并分块保存
tokenized_dataset = tokenize_with_progress(dataset, tokenizer, max_seq_len)

for example in tokenized_dataset:
    # 将tokenized_output中的input_ids添加到当前数据块
    current_data.append(example['input_ids'])

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