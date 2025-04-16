import json
from datasets import load_dataset

# all_datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
#                     "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
#                     "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

# for dataset in all_datasets:
#     data = load_dataset('THUDM/LongBench', dataset, split='test')
#     data.save_to_disk(f"benchmark/data/longbench/{dataset}")

from datasets import load_dataset

ds = load_dataset("THUDM/LongBench-v2")
print(ds)
train_data = ds["train"]

with open("/home/xujie/TaoTie/benchmark/data/longbench-v2/data.jsonl", "w", encoding="utf-8") as f:
    for data in train_data:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")