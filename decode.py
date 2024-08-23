from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
with open("/mnt/zzb_Term_4/TeamOne/data/QW_wiki_tokens/part-01-00000.npy", 'rb') as f:
    data = np.fromfile(f, dtype=np.uint32)
print(data.shape)
print(tokenizer.decode(data[:1000]))