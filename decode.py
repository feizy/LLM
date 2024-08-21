from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
import numpy as np
with open("/mnt/zzb_Term_4/feizy/LLM/data/tokens/",'rb' ) as f :
	data=np.fromfile(f , dtype=np.uint16)
print("shappppp",data.shape)
print("decoddd",tokenizer.decode(data[:100]))