import json
import os
import torch
from utils_exp import get_model_and_tokenizer, get_pred, load_dataset, build_qa_prompt, compute_f1, load_result


dataset = "wikimqa_s"
config_path = "/home/xujie/TaoTie/"
result_path_prefix = "/home/xujie/TaoTie/cb-bench/"

if __name__ == '__main__':
    inf_llm_config_path = config_path + "config/mistral-inf-llm.yaml"
    infllm_result_path = result_path_prefix + dataset + "/result.json"
    taotie_result_path = result_path_prefix + dataset + "/taotie-result.json"

    from omegaconf import OmegaConf
    args = OmegaConf.load(inf_llm_config_path)  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # define your model
    if not hasattr(args.model, "tokenizer_path"):
      args.model.tokenizer_path = args.model.path
    model, tokenizer = get_model_and_tokenizer(args.model)

    dataset_path = config_path + "benchmark/data/inputs/" + dataset + ".json"
    
    eval_dataset = load_dataset(dataset_path)
    # load the results from the inf-llm
    infllm_results = load_result(infllm_result_path)
    # load the results from the taotie
    taotie_results = load_result(taotie_result_path)

    inf_f1_scores = []
    taotie_f1_scores = []
    inf_ttft = []
    taotie_ttft = []
    for i in range(len(eval_dataset)):
        infllm_result = infllm_results[i]
        taotie_result = taotie_results[i]

        answers = eval_dataset[i]["answers"]
        infllm_pred = infllm_result["pred"]
        taotie_pred = taotie_result["pred"]

        infllm_f1 = max([compute_f1(infllm_pred[0], answer[0], tokenizer) for answer in answers])
        taotie_f1 = max([compute_f1(taotie_pred[0], answer[0], tokenizer) for answer in answers])
        # store the result
        inf_f1_scores.append(infllm_f1)
        taotie_f1_scores.append(taotie_f1)

        inf_ttft.append(infllm_result["TTFT"])
        taotie_ttft.append(taotie_result["TTFT"])

    print("InfLLM Average F1 score:", sum(inf_f1_scores) / len(inf_f1_scores))
    print("TaoTie Average F1 score:", sum(taotie_f1_scores) / len(taotie_f1_scores))
    print("InfLLM Average TTFT:", sum(inf_ttft) / len(inf_ttft))
    print("TaoTie Average TTFT:", sum(taotie_ttft) / len(taotie_ttft))

