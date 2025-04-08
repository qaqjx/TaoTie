import json
import collections
import string
import re
from rouge_score import rouge_scorer
from inf_llm.utils.patch import SPECIAL_TOKENS
from benchmark.pred import serialize_and_hash
from inf_llm.utils import patch_hf, GreedySearch, patch_model_center, find_special_tokens
from transformers import AutoModelForCausalLM, AutoTokenizer
from inf_llm.utils.patch import SPECIAL_TOKENS
import torch

def load_result(result_path):
    loaded_results = []
    with open(result_path, "r") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                result = json.loads(line)
                loaded_results.append(result)
    return loaded_results

def load_dataset(dataset_path):
    print("Loading dataset:", dataset_path)
    with open(dataset_path) as f:
        return json.load(f)

def normalize_question(question):
    if not question.endswith("?"):
        question = question + "?"

    return question[0].lower() + question[1:]

def parse_generation(s):
    s = s.lstrip('\n').split('\n')[0]
    if s.startswith("Yes") or s.startswith("yes"):
        s = "Yes"
    elif (s.split()[0]).startswith("No") or (s.split()[0]).startswith("no"):
        s = "No"
    return s

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def build_qa_prompt(example, query_prompt):

    q = normalize_question(example["question"])
    doc_prompts = [f"{ctx['title']}\n\n{ctx['text']}\n\n" for ctx in example["ctxs"]]
    #ex_prompt = f"{docs_text}\n\nBased on these texts, answer the question:\nQ: {q}\nA:"
    #q_prompt = f"\n\nAnswer the question based on the given passages. Answer the question within 5 words. Do NOT repeat the question or output any other words. Question: {q}\nAnswer:"
    q_prompt = f"{query_prompt}{q}\nAnswer:"
    return doc_prompts, q_prompt

def build_fewshot_prompt(example):
    q = "\n\n"+example["question"]
    doc_prompts = [f"{ctx['text']}" for ctx in example["ctxs"]]
    q_prompt = f"{q}"
    return doc_prompts, q_prompt

def compute_f1(a_pred, a_gold, tokenizer):
    a_pred = parse_generation(a_pred)
    gold_toks = tokenizer.encode(normalize_answer(a_gold))[1:]
    pred_toks = tokenizer.encode(normalize_answer(a_pred))[1:]
    #gold_toks = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=[UserMessage(content=normalize_answer(a_gold))])).tokens[4:-4]
    #pred_toks = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=[UserMessage(content=normalize_answer(a_pred))])).tokens[4:-4]
    #pdb.set_trace()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_rl(pred, gold):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rougeL = scorer.score(gold, pred)['rougeL'].fmeasure
    return rougeL

def normalize_context(s):
    s += SPECIAL_TOKENS
    return s

def combine_contexts(contexts):
    combined = ""
    for context in contexts:
        combined +=  SPECIAL_TOKENS + context + SPECIAL_TOKENS
    return combined

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

    return output,TTFT
