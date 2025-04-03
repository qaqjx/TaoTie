import torch
import time


class GreedySearch:
    def __init__(self, model, tokenizer):
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.past_kv = None

    def clear(self):
        self.past_kv = None

    def _process_texts(self, input_text):
        model_inputs = {}
        input_ids = self.tokenizer.encode(input_text)

        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = [1] * len(model_inputs["input_ids"])

        for key in model_inputs:
            model_inputs[key] = torch.tensor(model_inputs[key]).int().unsqueeze(0).cuda()

        return model_inputs

    def generate(self, text=None, input_ids=None, **kwargs):
        if input_ids is None:
            model_inputs = self._process_texts(text)
            input_ids = model_inputs['input_ids']

        with torch.inference_mode():
            result = self._decode(input_ids, **kwargs)
        return result

    def _decode(self, input_ids, max_length=100, extra_end_token_ids=[], chunk_size: int = 4096, output=False):
        if input_ids.dim() == 1:
            input_ids = input_ids[None, :]
        input_ids = input_ids.cuda()
        attention_mask = torch.ones_like(input_ids)
        assert input_ids.size(0) == 1
        length = input_ids.size(1)
        end_token_ids = extra_end_token_ids + [self.tokenizer.eos_token_id]
        logits = None
        past_key_values = self.past_kv
        if output:
            output_text = ""

        self.model.model.is_blend = self.model.is_blend
        # self.model.model.hash_str = self.model.hash_str

        start = time.time()  
        for i in range(max_length + 1):
            if i == 0:
                # prefill phase
                if chunk_size is None:
                    chunk_size = input_ids.size(1)
                
                # segmentation
                for st in range(0, input_ids.size(1) - 1, chunk_size):
                    ed = min(input_ids.size(1) - 1, st + chunk_size)
                    cb_indices = []
                    st_hash_idx = 10000000
                    ed_hash_idx = 0
                    if self.model.model.is_blend == 1:
                        for i in range(len(self.model.hash_str)):
                            if (self.model.cacheblend_indices[2 * i] > ed or self.model.cacheblend_indices[2 * i + 1] < st) is False:
                                st_hash_idx = min(st_hash_idx,i)
                                ed_hash_idx = max(ed_hash_idx,i)
                                # find the slotting idx
                                slot_st = max(self.model.cacheblend_indices[2 * i], st)
                                slot_ed = min(self.model.cacheblend_indices[2 * i + 1], ed)
                                cb_indices.append([slot_st - st, slot_ed - st, slot_st - self.model.cacheblend_indices[2 * i] , slot_ed - self.model.cacheblend_indices[2 * i]])
                    
                        self.model.model.cacheblend_indices = cb_indices
                        self.model.model.hash_str = self.model.hash_str[st_hash_idx : ed_hash_idx + 1]
                    else:
                        self.model.model.cacheblend_indices = []
                        self.model.model.hash_str = self.model.hash_str
                    out = self.model(
                        input_ids = input_ids[:, st: ed],
                        attention_mask = attention_mask[:, :ed],
                        use_cache = True,
                        return_dict = True,
                        past_key_values = past_key_values
                    )
                    logits, past_key_values = out.logits, out.past_key_values
                self.model.model.is_blend = 0
                self.model.model.hash_str = []

                # decode phase
                out = self.model(
                    input_ids = input_ids[:, -1:],
                    attention_mask = attention_mask,
                    use_cache = True,
                    return_dict = True,
                    past_key_values = past_key_values
                )
                logits, past_key_values = out.logits, out.past_key_values
                end = time.time()  
                eplised = end - start
            else:
                out = self.model(
                    input_ids = input_ids[:, -1:],
                    attention_mask = attention_mask,
                    past_key_values = past_key_values,
                    use_cache = True,
                    return_dict = True
                )
                logits, past_key_values = out.logits, out.past_key_values
            
            logits = logits[:, -1, :]
            word = logits.argmax(dim=-1)
            if word.item() in end_token_ids or i == max_length:
                break

            input_ids = torch.cat((input_ids, word.view(1, 1)), dim=-1)
            attention_mask = torch.cat(
                (attention_mask, torch.ones((attention_mask.size(0), 1), dtype=torch.int, device=attention_mask.device)),
                dim=-1
            )
            if output:
                tmp = self.tokenizer.decode(input_ids.squeeze(0)[length:])
                if len(tmp) > len(output_text):
                    import sys               
                    sys.stdout.write(tmp[len(output_text):])
                    sys.stdout.flush()
                    output_text = tmp

        self.past_kv = past_key_values

        if output:
            sys.stdout.write("\n")
            sys.stdout.flush()

        return [self.tokenizer.decode(input_ids.squeeze(0)[length:]),eplised]
