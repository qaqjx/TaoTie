import json
import os
import torch
from utils_exp import combine_contexts, get_model_and_tokenizer, get_pred, load_data_jsonl, load_dataset, build_qa_prompt, compute_f1, normalize_context

dataset = "loong_process"
config_path = "/home/xujie/TaoTie/"
result_path_prefix = "/home/xujie/TaoTie/cb-bench/loong/"
special_tokens = "[[TAOTIE]]"

inf_llm_config_path = config_path + "config/mistral-inf-llm.yaml"
result_path = result_path_prefix  + "1_4_result-taotie.json"
dataset_path = config_path + "benchmark/data/loong/" + dataset + ".jsonl"

def clean_json(data):
    """
    清理 JSON 数据中的字符串：
    1. 去掉所有字符串中的 '#' 符号。
    2. 去掉字符串首尾的多余空格。
    
    参数:
        data (dict | list): 输入的 JSON 数据（可以是字典或列表）。
    
    返回:
        dict | list: 清理后的 JSON 数据。
    """
    def clean_string(s):
        """清理单个字符串：去掉 '#' 和首尾空格"""
        return s.replace("#", "").strip() if isinstance(s, str) else s

    if isinstance(data, dict):  # 如果是字典，递归处理每个值
        return {key: clean_json(value) for key, value in data.items()}
    elif isinstance(data, list):  # 如果是列表，递归处理每个元素
        return [clean_json(item) for item in data]
    else:  # 如果是字符串或其他类型，直接清理或返回
        return clean_string(data)
    
def divide_prompt(prompt):
    """
    Divide the prompt into two parts: the first part is the prefix, and the second part is the rest of the prompt.
    """
    if special_tokens in prompt:
        prefix_prompt = prompt.split(special_tokens)[0]
        question = prompt.split(special_tokens)[-1]
        doc_chunk_ids = prompt.split(special_tokens)[1:-1]
        doc_chunk_ids = [doc_chunk_id for doc_chunk_id in doc_chunk_ids if doc_chunk_id != ""]
        # print("doc_chunk_ids:", doc_chunk_ids)
        return prefix_prompt, doc_chunk_ids , question
    else:
        return "", [prompt]

if __name__ == '__main__':
    from omegaconf import OmegaConf
    args = OmegaConf.load(inf_llm_config_path)  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # define your model

    if not hasattr(args.model, "tokenizer_path"):
      args.model.tokenizer_path = args.model.path
    model, tokenizer = get_model_and_tokenizer(args.model,device)

    eval_dataset = load_data_jsonl(dataset_path)

    results = []
    f1_scores = []
    len = len(eval_dataset) / 4
    doc_str = dict()

    for i , ex in enumerate(eval_dataset):
        if i > 500:
            break
        
        prefix_prompt, doc_chunk_ids, q_prompt = divide_prompt(ex["prompt"])

        for prompt in doc_chunk_ids:
            # print("length:", len(prompt))
            prompt = "[INST]" + normalize_context(prompt) + "[/INST]"
            if prompt not in doc_str:
                doc_str[prompt] = 1
                get_pred(model, tokenizer, prompt,
                      args.max_len,1, verbose=True)
            else:
                continue

        answer = clean_json(ex["answer"])
        answer = json.dumps(answer)

        doc_chunk_ids = prefix_prompt + combine_contexts(doc_chunk_ids) + q_prompt 
        user_promt = "[INST]" + doc_chunk_ids + "[/INST]"
    
        output,ttft = get_pred(model, tokenizer, user_promt,
                      args.max_len,100, args.chunk_size , verbose=True)
        # print("Predsss:", output)
        f1 = compute_f1(output[0].split(")")[0], answer, tokenizer)
        # store the result
        result = {
            "id": ex["id"],
            "pred": output,
            "TTFT": ttft,
            "answer": answer,
            "f1 socre": f1,
        }
        results.append(result)
        f1_scores.append(f1)
        with open(result_path, "a") as f:
            for result in results:
                json.dump(result, f)
                f.write("\n")

    # print("F1 scores:", f1_scores)
    print("Average F1 score:", sum(f1_scores) / len(f1_scores))
    result_dir = os.path.dirname(result_path)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)



## [##TAOTIE##] who are u ？