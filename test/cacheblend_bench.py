import json
import argparse
import os
import torch
from benchmark.pred import serialize_and_hash
from inf_llm.utils import patch_hf, GreedySearch, patch_model_center, find_special_tokens
from transformers import AutoModelForCausalLM, AutoTokenizer
from inf_llm.utils.patch import SPECIAL_TOKENS
from utils_exp import load_dataset, normalize_question, build_qa_prompt, compute_f1



def get_model_and_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(config.path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda")
    model = patch_hf(model, config.type, **config)
    return model, tokenizer

def get_pred(
    model, tokenizer, prompt, max_length,
    max_gen,gen_chunk_size = None, truncation: str = None, 
    rank: int = None, world_size: int = None,
    verbose: bool = False
):
    preds = []
    extra_end_token_ids = []

    if world_size is not None:
        prompt = prompt[rank::world_size]
    model.is_blend = 0
    searcher = GreedySearch(model, tokenizer)
    
    # add the special token.
    tokenizer.add_special_tokens({"additional_special_tokens": [SPECIAL_TOKENS]})
    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    spec_id = tokenizer(SPECIAL_TOKENS).input_ids[-1]
        
    indices = [idx for idx,input_id in enumerate(tokenized_prompt) if input_id == spec_id]
    hash_str = []
    if(len(indices) == 1):
        hash_str.append(serialize_and_hash(tokenized_prompt[4 : -5]))
    else :
        for idx in range(0,len(indices),2):
            hash_str.append(serialize_and_hash(tokenized_prompt[indices[idx] + 1 : indices[idx + 1] ]))
    model.hash_str = hash_str

    spec_count = (tokenized_prompt == spec_id).sum().item()
    if spec_count == 0:
        model.is_blend = 0
    elif spec_count == 1:
        model.is_blend = 2
    else:
        model.is_blend = 1

    indices = [x - i for i, x in enumerate(indices) ]

    model.cacheblend_indices = indices
    tokenized_prompt = tokenized_prompt[tokenized_prompt != spec_id]  
        
    output = searcher.generate(
        input_ids = tokenized_prompt,
        max_length=max_gen,
        chunk_size=gen_chunk_size,
        extra_end_token_ids=extra_end_token_ids
    )

    TTFT = output[-1]
    output = output[:-1]
    searcher.clear()
    # print("Length: ", len(tokenized_prompt))
    # print("Question:", prompt)
    # print("Pred:", output)
    # print("")

    return output

prefix_prompt = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
query_prompt = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"


if __name__ == '__main__':
    inf_llm_config_path = "/home/xujie/TaoTie/config/mistral-inf-llm.yaml"
    dataset = "wikimqa_s"
    result_path = "/home/xujie/TaoTie/cb-bench/" + dataset + "/result.json"
    from omegaconf import OmegaConf
    args = OmegaConf.load(inf_llm_config_path)  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # define your model

    if not hasattr(args.model, "tokenizer_path"):
      args.model.tokenizer_path = args.model.path
    model, tokenizer = get_model_and_tokenizer(args.model)

    dataset_path = "/home/xujie/TaoTie/benchmark/data/inputs/" + dataset + ".json"
    
    eval_dataset = load_dataset("/home/xujie/TaoTie/benchmark/data/inputs/wikimqa_s.json")
    results = []
    f1_scores = []
    for ex in eval_dataset:
        answers = ex["answers"]
        doc_prompts, q_prompt = build_qa_prompt(ex, query_prompt)
        doc_chunk_ids = [(doc)[1:] for doc in doc_prompts]
        q_ids = (q_prompt)[1:]
    
        doc_chunk_ids = [prefix_prompt] + doc_chunk_ids + [q_ids]    
        user_promt = "[INST]" + "".join(doc_chunk_ids) + "[/INST]"
    
        output = get_pred(model, tokenizer, user_promt,
                      args.max_len,100, verbose=True)
        # print("Predsss:", output)
        f1 = max([compute_f1(output[0], answer[0], tokenizer) for answer in answers])
        # store the result
        result = {
            "pred": output,
            "answers": answers,
            "f1 socre": f1,
        }
        results.append(result)
        f1_scores.append(f1)

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