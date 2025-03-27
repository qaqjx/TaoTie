import os
from datasets import load_from_disk
import torch
import json
from tqdm import tqdm
import argparse
from omegaconf import OmegaConf
from inf_llm.utils import patch_hf, GreedySearch, patch_model_center
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--model_center", action="store_true", default=False)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args, extra_args = parser.parse_known_args()
    conf = OmegaConf.load(args.config_path)
    cli_conf = OmegaConf.from_cli(extra_args)
    conf = OmegaConf.merge(conf, cli_conf)
    conf.model.model_center = args.model_center
    conf.rank = args.rank
    conf.world_size = args.world_size
    conf.verbose = args.verbose
    if not hasattr(conf.model, "tokenizer_path"):
        conf.model.tokenizer_path = conf.model.path
    if not hasattr(conf, "truncation"):
        conf.truncation = None
    return conf

def get_model_and_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    if config.model_center:
        import bmtrain as bmt
        bmt.init_distributed(seed=233)
        from model_center.model import Llama, LlamaConfig
        model_config = LlamaConfig.from_pretrained(config.path)
        model_config.dtype = torch.bfloat16
        model = Llama(model_config)
        bmt.load(model, os.path.join(config.path, "pytorch_model.pt"), strict=False)
        model = patch_model_center(model, config.type, **config)
    else:
        model = AutoModelForCausalLM.from_pretrained(config.path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda")
        model = patch_hf(model, config.type, **config)
    return model, tokenizer

def get_pred(
    searcher:GreedySearch, tokenizer, max_length,
    max_gen, prompt, model_name, 
    gen_chunk_size=None, truncation=None, 
    rank=None, world_size=None,
    verbose=False
):
  
    cur = 0
    extra_end_token_ids = []
    if model_name == "llama-3-inst":
        extra_end_token_ids.append(tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0])
    if model_name == "qwen":
        extra_end_token_ids.append(tokenizer.encode("<|im_end|>", add_special_tokens=False)[0])

    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
    
    if truncation is None:
        if len(tokenized_prompt) > max_length - max_gen:
            if verbose:
                print(f"Length {len(tokenized_prompt)}. Skipped.")
    else:
        if truncation == "suffix":
            length = len(tokenized_prompt)
            if length > max_length - max_gen:
                if verbose:
                    print("over length")
                init_token_num = 128
                prompt = tokenizer.decode(tokenized_prompt[:init_token_num].tolist() + tokenized_prompt[- (max_length - max_gen - init_token_num):].tolist())
                tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids[0]
        else:
            raise NotImplementedError
    output = searcher.generate(
        input_ids=tokenized_prompt,
        max_length=max_gen,
        chunk_size=gen_chunk_size,
        extra_end_token_ids=extra_end_token_ids
    )
    
    print(output)

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = get_model_and_tokenizer(args.model)
    
    max_gen = 512
    
    searcher = GreedySearch(model, tokenizer)
    
    prompt = "Pink Floyd is a band that"
    preds = get_pred(
        searcher, tokenizer, 
        args.max_len, max_gen, 
        prompt, "llama-3", 
        args.chunk_size, args.truncation,
        args.rank, args.world_size,
        args.verbose
    )
