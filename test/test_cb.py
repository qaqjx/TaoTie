import json
import argparse
import torch
from benchmark.pred import serialize_and_hash
from inf_llm.utils import patch_hf, GreedySearch, patch_model_center, find_special_tokens
from transformers import AutoModelForCausalLM, AutoTokenizer
from inf_llm.utils.patch import SPECIAL_TOKENS



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
            hash_str.append(serialize_and_hash(tokenized_prompt[indices[idx] + 1 :indices[idx + 1]]))
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
    print("Length: ", len(tokenized_prompt))
    print("Question:", prompt)
    print("Pred:", output)
    print("")


    return preds



if __name__ == '__main__':
    inf_llm_config_path = "/home/xujie/TaoTie/config/mistral-inf-llm.yaml"
    from omegaconf import OmegaConf
    args = OmegaConf.load(inf_llm_config_path)  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # define your model

    if not hasattr(args.model, "tokenizer_path"):
      args.model.tokenizer_path = args.model.path
    model, tokenizer = get_model_and_tokenizer(args.model)

    while True:
      input_text = input("input: ")

      if(input_text == "exit"):
        break
      
      prompt = "<s>" + input_text + "</s>"

      preds = get_pred(model, tokenizer, prompt,
                      args.max_len,100, verbose=True)
