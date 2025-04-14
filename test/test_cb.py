import json
import argparse
import torch
from benchmark.pred import serialize_and_hash
from inf_llm.utils import patch_hf, GreedySearch, patch_model_center, find_special_tokens
from transformers import AutoModelForCausalLM, AutoTokenizer
from inf_llm.utils.patch import SPECIAL_TOKENS
from utils_exp import combine_contexts, normalize_context

from utils_exp import combine_contexts, get_model_and_tokenizer, get_pred, load_dataset, normalize_context, build_qa_prompt, compute_f1




if __name__ == '__main__':
    inf_llm_config_path = "/home/xujie/TaoTie/config/mistral-inf-llm.yaml"
    from omegaconf import OmegaConf
    args = OmegaConf.load(inf_llm_config_path)  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # define your model

    if not hasattr(args.model, "tokenizer_path"):
      args.model.tokenizer_path = args.model.path
    model, tokenizer = get_model_and_tokenizer(args.model,device)

    while True:
      input_text = input("input: ")

      if(input_text == "exit"):
        break
    #   prompt = "[INST]" + normalize_context(input_text) + "[/INST]"

      prompt = "[INST]" + input_text + "[/INST]"

      preds = get_pred(model, tokenizer, prompt,
                      args.max_len,100, verbose=True)
      
      print("preds: ", preds)
      

## [##TAOTIE##] who are u？