import requests
from pathlib import Path

# 定义数据集下载链接
datasets = {
    "v3-small-c4_en-validation": [
        "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/c4_en/val/part-0-00000.npy"],
    "v3-small-dolma_books-validation": [
        "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/dolma_books/val/part-0-00000.npy"],
    "v3-small-dolma_common-crawl-validation": [
        "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/dolma_common-crawl/val/part-0-00000.npy"],
    "v3-small-dolma_pes2o-validation": [
        "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/dolma_pes2o/val/part-0-00000.npy"],
    "v3-small-dolma_reddit-validation": [
        "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/dolma_reddit/val/part-0-00000.npy"],
    "v3-small-dolma_stack-validation": [
        "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/dolma_stack/val/part-0-00000.npy"],
    "v3-small-dolma_wiki-validation": [
        "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/dolma_wiki/val/part-0-00000.npy"],
    "v3-small-ice-validation": [
        "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/ice/val/part-0-00000.npy"],
    "v3-small-m2d2_s2orc-validation": [
        "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/m2d2_s2orc/val/part-0-00000.npy"],
    "v3-small-pile-validation": [
        "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/pile/val/part-0-00000.npy"],
    "v3-small-wikitext_103-validation": [
        "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/wikitext_103/val/part-0-00000.npy"],
    "v2-small-4chan-validation": ["https://olmo-data.org/eval-data/perplexity/v2_small_gptneox20b/4chan/val.npy"],
    "v2-small-c4_100_domains-validation": [
        "https://olmo-data.org/eval-data/perplexity/v2_small_gptneox20b/c4_100_domains/val.npy"],
    "v2-small-c4_en-validation": ["https://olmo-data.org/eval-data/perplexity/v2_small_gptneox20b/c4_en/val.npy"],
    "v2-small-gab-validation": ["https://olmo-data.org/eval-data/perplexity/v2_small_gptneox20b/gab/val.npy"],
    "v2-small-ice-validation": ["https://olmo-data.org/eval-data/perplexity/v2_small_gptneox20b/ice/val.npy"],
    "v2-small-m2d2_s2orc-validation": [
        "https://olmo-data.org/eval-data/perplexity/v2_small_gptneox20b/m2d2_s2orc/val.npy"],
    "v2-small-m2d2_wiki-validation": [
        "https://olmo-data.org/eval-data/perplexity/v2_small_gptneox20b/m2d2_wiki/val.npy"],
    "v2-small-manosphere-validation": [
        "https://olmo-data.org/eval-data/perplexity/v2_small_gptneox20b/manosphere/val.npy"],
    "v2-small-mc4_en-validation": ["https://olmo-data.org/eval-data/perplexity/v2_small_gptneox20b/mc4_en/val.npy"],
    "v2-small-pile-validation": ["https://olmo-data.org/eval-data/perplexity/v2_small_gptneox20b/pile/val.npy"],
    "v2-small-ptb-validation": ["https://olmo-data.org/eval-data/perplexity/v2_small_gptneox20b/ptb/val.npy"],
    "v2-small-twitterAEE-validation": [
        "https://olmo-data.org/eval-data/perplexity/v2_small_gptneox20b/twitterAEE/val.npy"],
    "v2-small-wikitext_103-validation": [
        "https://olmo-data.org/eval-data/perplexity/v2_small_gptneox20b/wikitext_103/val.npy"],
}

# 定义保存数据的路径
save_path = Path("/mnt/zzb_Term_4/TeamOne/data/eval_data")
save_path.mkdir(parents=True, exist_ok=True)

# 下载数据集并保存
for dataset_name, urls in datasets.items():
    dataset_dir = save_path / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for url in urls:
        file_name = url.split("/")[-1]
        file_path = dataset_dir / file_name
        print(f"Downloading {file_name} to {file_path}...")

        # 下载文件
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # 保存文件
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

print("All datasets downloaded successfully.")