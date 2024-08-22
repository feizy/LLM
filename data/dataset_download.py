from datasets import load_dataset
from datasets.config import HF_DATASETS_CACHE

print("Current cache directory:", HF_DATASETS_CACHE)
ds = load_dataset("jed351/Chinese-Common-Crawl-Filtered"):