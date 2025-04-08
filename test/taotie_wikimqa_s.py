import json
import argparse
import os
import time
import torch
from utils_exp import combine_contexts, get_model_and_tokenizer, get_pred, load_dataset, normalize_context, build_qa_prompt, compute_f1


prefix_prompt = "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n"
query_prompt = f"\n\nAnswer the question based on the given passages. Answer the question within 5 words. Do NOT repeat the question or output any other words. Question: "

dataset = "wikimqa_s"
config_path = "/home/xujie/TaoTie/"
result_path_prefix = "/home/xujie/TaoTie/cb-bench/"

if __name__ == '__main__':
    inf_llm_config_path = config_path + "config/mistral-inf-llm.yaml"
    result_path = result_path_prefix + dataset + "/taotie-result.json"
    from omegaconf import OmegaConf
    args = OmegaConf.load(inf_llm_config_path)  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # define your model

    if not hasattr(args.model, "tokenizer_path"):
      args.model.tokenizer_path = args.model.path
    model, tokenizer = get_model_and_tokenizer(args.model)

    dataset_path = config_path + "benchmark/data/inputs/" + dataset + ".json"
    
    eval_dataset = load_dataset(dataset_path)
    results = []
    f1_scores = []
    for ex in eval_dataset:
        answers = ex["answers"]
        doc_prompts, q_prompt = build_qa_prompt(ex, query_prompt)
        doc_chunk_ids = [(doc)[1:] for doc in doc_prompts]
        q_ids = (q_prompt)[1:]

        for prompt in doc_chunk_ids:
            prompt = "[INST]" + normalize_context(prompt) + "[/INST]"
            get_pred(model, tokenizer, prompt,
                      args.max_len,1, verbose=True)
        
        # print("--------------------")
        doc_chunk_ids = prefix_prompt +  combine_contexts(doc_chunk_ids) + q_ids  
        user_promt = "[INST]" + doc_chunk_ids + "[/INST]"
    
        output,ttft = get_pred(model, tokenizer, user_promt,
                      args.max_len,100, verbose=True)
        # print("Predsss:", output)
        f1 = max([compute_f1(output[0], answer[0], tokenizer) for answer in answers])
        # store the result
        result = {
            "pred": output,
            "TTFT": ttft,
            "answers": answers,
            "f1 socre": f1,
        }
        results.append(result)
        f1_scores.append(f1)
        # break

    # print("F1 scores:", f1_scores)
    print("Average F1 score:", sum(f1_scores) / len(f1_scores))
    result_dir = os.path.dirname(result_path)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    with open(result_path, "w") as f:
        for result in results:
            json.dump(result, f)
            f.write("\n")

## [##TAOTIE##] who are u ？