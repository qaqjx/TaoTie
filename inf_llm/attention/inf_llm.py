import time
import torch
from typing import Optional
from .context_manager import ContextManager
import nvtx

def inf_llm_forward(
    n_local, n_init, topk, 
    block_size, max_cached_block,
    exc_block_size, fattn,
    repr_topk: int = 1,
    cache_strategy="lru",
    score_decay=None,
    chunk_topk_calc=None,
    async_global_stream=True,
    pin_memory=False,
    faiss=False,
    perhead=False,
    *args, **kwargs
):

    def forward(self, query : torch.Tensor,
                    key_value : torch.Tensor,
                    position_bias : Optional[torch.Tensor],
                    use_cache: bool,
                    past_key_value,
                    project_q, project_k, project_v, attention_out, 
                    dim_head, num_heads, num_heads_kv,
                    is_blend , cacheblend_indices,layer_idx,
                    hash_str , recompute_idx
    ):


        if cacheblend_indices == []:
            is_blend = 0

        batch_size = query.size(0)
        len_q = query.size(1)
        len_k = key_value.size(1)

        assert use_cache

        if past_key_value is None:
            past_key_value = ContextManager(
                position_bias, n_init,
                n_local, block_size,
                max_cached_block, topk,
                exc_block_size,
                score_decay, fattn, repr_topk,
                cache_strategy,
                chunk_topk_calc,
                async_global_stream,
                pin_memory,
                faiss,
                perhead
            )

        token_num = query.shape[1]

        # prefetch the kv cache to the CPU
        if is_blend == 1 :
            past_key_value.flush_loacl_to_block()
            if layer_idx >= 1:
                past_key_value.get_previous_kv(hash_str, layer_idx)
        
        h_q = project_q(query)             # (batch, len_q, num_heads * dim_head)
        h_k = project_k(key_value)         # (batch, len_k, num_heads * dim_head)
        h_v = project_v(key_value)         # (batch, len_k, num_heads * dim_head)

        len_k = key_value.shape[1]

        h_q = h_q.view(batch_size, len_q, num_heads, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads, len_q, dim_head)
        h_k = h_k.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads_kv, len_k, dim_head)
        h_v = h_v.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads_kv, len_k, dim_head)

        local_q, local_k, local_v = h_q, h_k, h_v
        global_q, global_k, global_v = h_q, h_k, h_v

        if is_blend == 1 and layer_idx:
            if layer_idx > 1:
                o = past_key_value.do_blend(h_q, h_k, h_v, recompute_idx, hash_str, cacheblend_indices, layer_idx)      
            elif layer_idx == 1:
                o, recompute_idx = past_key_value.selective_recompute_block(
                    local_q, local_k, local_v ,
                    global_q, global_k, global_v,
                    hash_str, cacheblend_indices, layer_idx
                )
        else:
            # how to store the kv cache
            # store the kv cache of precompute pharse

            if is_blend == 2:
                block_st = past_key_value.flush_loacl_to_block()
            # add the kv to the cache
            o = past_key_value.append(
                local_q, local_k, local_v,
                global_q, global_k, global_v,
            )

            if is_blend == 2:
                past_key_value.flush_loacl_to_block()

                if hash_str != []:
                    # store the kv cache to SSD
                    # print("store the kv cache to SSD")
                    past_key_value.offload_ssd(hash_str, layer_idx , block_st)
                # print("store the kv cache not implemented")

        # get the attention output
        o = o.view(batch_size, num_heads, len(recompute_idx), dim_head).permute(0, 2, 1, 3)
        o = o.reshape(batch_size, len(recompute_idx), dim_head * num_heads)
        o = attention_out(o)

        if use_cache:
            return o, past_key_value, recompute_idx
        else:
            return o


    return forward
