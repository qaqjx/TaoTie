import torch
from typing import Optional
from .context_manager import ContextManager


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

        # divide the token for blend
        if is_blend == 1 and  layer_idx != 0:
            kv_c = []
            kv_c.append(key_value[ :, :cacheblend_indices[0][0],:])
            for idx in range(1 , len(cacheblend_indices) - 1):
                if cacheblend_indices[idx][0] != cacheblend_indices[idx - 1][1]:
                    kv_c.append(key_value[ :, cacheblend_indices[idx - 1][1]:cacheblend_indices[idx][0],:])
            kv_c.append(key_value[ :, cacheblend_indices[-1][1]:,:])               
            key_value = torch.cat(kv_c,dim = 1)

        h_q = project_q(query)             # (batch, len_q, num_heads * dim_head)
        h_k = project_k(key_value)         # (batch, len_k, num_heads * dim_head)
        h_v = project_v(key_value)         # (batch, len_k, num_heads * dim_head)

        len_k = key_value.shape[1]

        h_q = h_q.view(batch_size, len_q, num_heads, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads, len_q, dim_head)
        h_k = h_k.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads_kv, len_k, dim_head)
        h_v = h_v.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads_kv, len_k, dim_head)

        #  check the kv whether reuse 
        # 1. if the kv is not reused, we can directly use the kv
        # 2. if the kv is reused, we need to recover the deviation
        if is_blend == 1 and layer_idx >= 1: 
            # retrieval the kv cache
            h_k, h_v = past_key_value.blend(hash_str, cacheblend_indices, h_k, h_v, layer_idx)
            # Convert recompute_idx from list to tensor
            recompute_idx_tensor = torch.tensor(recompute_idx, dtype=torch.long, device = h_k.device)
            # recover the kv cache
            recover_kv = torch.index_select(query, dim = 1, index = recompute_idx_tensor)
            recover_k = project_k(recover_kv)         # (batch, len_k, num_heads * dim_head)
            recover_v = project_v(recover_kv)         # (batch, len_k, num_heads * dim_head)

            recover_k = recover_k.view(batch_size, token_num, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()
            recover_v = recover_v.view(batch_size, token_num, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()
            # 2. if the kv is reused, we need to recover the deviation
            if layer_idx == 1:
                # choose the kv for the blend
                deviation = torch.abs(h_k - recover_k).sum(dim = 1)
                _, topk_deviation = deviation.topk( k = int(token_num * 0.15), dim = 1)
                recompute_idx = topk_deviation
            else: 
                for i,idx in enumerate(recompute_idx):
                    h_k[:,:,idx,:] = recover_k[:,:,i,:]
                    h_v[:,:,idx,:] = recover_v[:,:,i,:]
                

        local_q, local_k, local_v = h_q, h_k, h_v
        global_q, global_k, global_v = h_q, h_k, h_v

        # add the kv to the cache
        o = past_key_value.append(
            local_q, local_k, local_v,
            global_q, global_k, global_v,
        )

        # store the kv cache of precompute pharse
        if is_blend == 2:
            past_key_value.offload_ssd(hash_str,layer_idx)

        # get the attention output
        o = o.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3)
        o = o.reshape(batch_size, len_q, dim_head * num_heads)
        o = attention_out(o)

        if use_cache:
            return o, past_key_value
        else:
            return o


    return forward
