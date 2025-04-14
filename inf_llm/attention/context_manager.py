import time
import numpy as np
import torch
from typing import Optional, Tuple
from copy import deepcopy
from .dot_production_attention import get_multi_stage_dot_production_attention

class CudaCache:
    def __init__(self, num_units, unit_size, dtype):
        self.num_units = num_units
        self.unit_size = unit_size
        self.dtype = dtype
        self.data = torch.empty(
            (num_units, unit_size),
            device = "cuda",
            dtype=dtype
        )
        self.idle_set = set(list(range(num_units)))

    def alloc(self):
        assert len(self.idle_set) > 0

        idx = self.idle_set.pop()
        return self.data[idx], idx

    def delete(self, idx):
        assert idx not in self.idle_set
        self.idle_set.add(idx)

class CPUCache:
    def __init__(self):
        self.kv = dict()
        self.memory_units = dict()
        self.all_time = 0

    def find(self , filename):
        if filename in self.kv:
            return self.kv[filename]
    
        return None
        
    def insert(self, filename, data):
        self.kv[filename] = data
    
    def get_kv(self, filename, slot_st = 0, slot_ed = -1, offset = 4):
        # start = time.time()
        slot_st = slot_st + offset
        slot_ed = slot_ed + offset
        
        if self.find(filename) is None:
            k,v = self.load_ssd(filename)
            self.insert(filename, [k,v])
        #  end = time.time()
        # self.all_time += end - start
        # print("load time: ", end - start , "all time: ", self.all_time)
        return self.kv[filename][0][:,:,slot_st:slot_ed,:], self.kv[filename][1][:,:,slot_st:slot_ed,:]
    
    def cache_memory_units(self, filename):
        if filename not in self.memory_units:
            self.memory_units[filename] = self.load_to_context_manager(filename)
        
        return self.memory_units[filename]
        
    def get_memory_units(self, filename):
        if filename not in self.memory_units:
            self.memory_units[filename] = self.load_to_context_manager(filename)
        
        return self.memory_units[filename]

    def get_memory_k(self,filename, indices):
        if filename not in self.memory_units:
            self.memory_units[filename] = self.load_to_context_manager(filename)
        all_k , _ , _ = self.memory_units[filename]

        all_k_tensor = torch.cat(all_k, dim=-2)
        
        return all_k_tensor[:, indices[-2] : indices[-1], :]


    def load_to_context_manager(self , file_name):
        with open(file_name, "rb") as f:        
            # load the repr token
            repr_token_shape = np.frombuffer(f.read(8), dtype=np.int32)
            repr_token_data = np.frombuffer(f.read(repr_token_shape.prod() * 2), dtype=np.int16).reshape(repr_token_shape)
            repr_token_data = torch.tensor(repr_token_data).view(torch.bfloat16)
            # Read the number of units and blocks
            batch_size, num_blocks = np.frombuffer(f.read(8), dtype=np.int32)
            # Iterate through each unit and block to deserialize and load their data
            all_k = []
            all_v = []
            for u in range(batch_size):
                for _ in range(num_blocks):
                    # Deserialize the shape and data of each tensor
                    kv_0_shape = np.frombuffer(f.read(12), dtype=np.int32)
                    kv_0_data = np.frombuffer(f.read(kv_0_shape.prod() * 2), dtype=np.int16).reshape(kv_0_shape)
                    kv_1_shape = np.frombuffer(f.read(12), dtype=np.int32)
                    kv_1_data = np.frombuffer(f.read(kv_1_shape.prod() * 2), dtype=np.int16).reshape(kv_1_shape)
                    
                    k_tensor = torch.tensor(kv_0_data)
                    v_tensor = torch.tensor(kv_1_data)

                    all_k.append(k_tensor.view(torch.bfloat16))
                    all_v.append(v_tensor.view(torch.bfloat16))

        return [all_k , all_v, repr_token_data]
    
    def load_ssd(self , file_name):
        with open(file_name, "rb") as f:        
            # Read the number of units and blocks
            num_units, num_blocks, remainder_num = np.frombuffer(f.read(12), dtype=np.int32)
            # Iterate through each unit and block to deserialize and load their data
            all_k = []
            all_v = []
            all_repr_token = []

            for u in range(num_units):
                for _ in range(num_blocks):
                    # Deserialize the shape and data of each tensor
                    kv_0_shape = np.frombuffer(f.read(16), dtype=np.int32)
                    kv_0_data = np.frombuffer(f.read(kv_0_shape.prod() * 2), dtype=torch.int16).reshape(kv_0_shape)
                    kv_1_shape = np.frombuffer(f.read(16), dtype=np.int32)
                    kv_1_data = np.frombuffer(f.read(kv_1_shape.prod() * 2), dtype=torch.int16).reshape(kv_1_shape)
                    repr_token_shape = np.frombuffer(f.read(4), dtype=np.int32)
                    repr_token_data = np.frombuffer(f.read(repr_token_shape.prod() * 2), dtype=torch.int16).reshape(repr_token_shape)
                    
                    k_tensor = torch.tensor(kv_0_data)
                    v_tensor = torch.tensor(kv_1_data)
                    repr_token = torch.tensor(repr_token_data)

                    all_k.append(k_tensor)
                    all_v.append(v_tensor)
                    all_repr_token.append(repr_token)
            
                    # Deserialize the remainder tensors (key and value)
            key_shape = np.frombuffer(f.read(4 * 4), dtype=np.int32)
            key_data = np.frombuffer(f.read(key_shape.prod() * 2), dtype=np.int16).reshape(key_shape)
            value_shape = np.frombuffer(f.read(4 * 4), dtype=np.int32)
            value_data = np.frombuffer(f.read(value_shape.prod() * 2), dtype=np.int16).reshape(value_shape)

            # Convert remainder tensors to PyTorch tensors
            key_tensor = torch.tensor(key_data)
            value_tensor = torch.tensor(value_data)

            # Concatenate all key and value tensors along the appropriate dimension
            concatenated_k = torch.cat(all_k + [key_tensor], dim=-2) # Concatenate along the first dimension
            concatenated_v = torch.cat(all_v + [value_tensor], dim=-2)  # Concatenate along the first dimension
            concatenated_repr = torch.cat(all_repr_token, dim=-2)
        return concatenated_k.view(torch.bfloat16), concatenated_v.view(torch.bfloat16)

class MemoryUnit:
    def __init__(
        self, 
        kv: Tuple[torch.Tensor, torch.Tensor], 
        cache: CudaCache, 
        load_to_cache: bool = False, 
        pin_memory: bool = False,
    ):
        self.cache = cache

        if kv[0].is_cuda:
            cpu_data = tuple(_t.contiguous().to("cpu", non_blocking=True) for _t in kv)
        else:
            cpu_data = tuple(_t.contiguous() for _t in kv)

        if pin_memory:
            # malloc the memory 
            cpu_data = tuple(_t.pin_memory() for _t in cpu_data)

        if load_to_cache:
            gpu_data, gpu_data_id = cache.alloc()
            gpu_data = gpu_data.view((2,) + kv[0].shape)
            gpu_data[0].copy_(kv[0], non_blocking=True)
            gpu_data[1].copy_(kv[1], non_blocking=True)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
        else:
            gpu_data, gpu_data_id = None, None
            event = None

        self.cpu_data = cpu_data
        self.gpu_data = gpu_data
        self.gpu_data_id = gpu_data_id
        self.event = event

    def load(self, target: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> bool:
        if self.gpu_data is not None:
            if target is not None:
                target[0].copy_(self.gpu_data[0], non_blocking=True)
                target[1].copy_(self.gpu_data[1], non_blocking=True)
                target_event = torch.cuda.Event()
                target_event.record(torch.cuda.current_stream())
            else:
                target_event = None


            return False, target_event



        gpu_data, gpu_data_id = self.cache.alloc()
        
        if self.cpu_data[0].numel() * 2 != gpu_data.numel():
            self.cache.delete(gpu_data_id)
            target[0][:,:self.cpu_data[0].size(1),:].copy_(self.cpu_data[0], non_blocking=True)
            target[1][:,:self.cpu_data[0].size(1),:].copy_(self.cpu_data[1], non_blocking=True)
            target_event = torch.cuda.Event()
            target_event.record(torch.cuda.current_stream())
            return False, None

        gpu_data = gpu_data.view((2,) + self.cpu_data[0].shape)
        if target is not None:
            target[0].copy_(self.cpu_data[0], non_blocking=True)
            target[1].copy_(self.cpu_data[1], non_blocking=True)
            target_event = torch.cuda.Event()
            target_event.record(torch.cuda.current_stream())
            gpu_data[0].copy_(target[0], non_blocking=True)
            gpu_data[1].copy_(target[1], non_blocking=True)

        else:
            gpu_data[0].copy_(self.cpu_data[0], non_blocking=True)
            gpu_data[1].copy_(self.cpu_data[1], non_blocking=True)

        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self.event = event
        self.gpu_data = gpu_data
        self.gpu_data_id = gpu_data_id

        return True, target_event

    def get(self):
        assert self.gpu_data is not None
        self.event.wait()
        return self.gpu_data

    def offload(self):
        # assert self.gpu_data is not None
        if self.gpu_data is not None:
            self.event.wait()
            self.gpu_data = None
            self.cache.delete(self.gpu_data_id)
            self.gpu_data_id = None


class VectorTensor:
    def __init__(
        self, 
        hidden_size,
        element_dtype
    ):
        init_cached_size = 16
        self.data = torch.empty(
            (init_cached_size, hidden_size),
            dtype=element_dtype,
            device='cuda'
        )
        self.length = 0
        self.cache_size = init_cached_size
        self.hidden_size = hidden_size

    def append_cache(self):
        new_cache_size = self.cache_size * 2
        data_shape = self.data.shape
        new_data = torch.empty(
            (new_cache_size,) + data_shape[1:],
            device='cuda',
            dtype=self.data.dtype
        )
        new_data[:self.cache_size,...].copy_(self.data)
        self.data = new_data
        self.cache_size = new_cache_size

    def append(self, tensor: torch.Tensor):
        assert tensor.dtype == self.data.dtype
        assert tensor.size(1) == self.hidden_size
        assert tensor.is_contiguous()

        append_l = tensor.size(0)

        while self.length + append_l > self.cache_size:
            self.append_cache()

        self.data[self.length: self.length+append_l, ...].copy_(tensor)

        self.length += append_l

    def update_back(self, tensor: torch.Tensor):
        self.data[self.length - 1: self.length, ...].copy_(tensor)

    def get_data(self):
        return self.data[:self.length, ...]

    def get_topk(self, tensor: torch.Tensor, topk): # inner product
        assert tensor.dim() == 1 and tensor.size(0) == self.hidden_size
        logits = torch.matmul(self.data[:self.length], tensor[:, None]).squeeze(dim=-1)
        assert logits.dim() == 1 and logits.size(0) == self.length
        return logits.topk(topk, dim=0).indices.cpu().tolist()

    def __len__(self):
        return self.length


class Faiss:
    def __init__(self, hidden_size, element_dtype):
        import faiss
        # We use the CPU index here because the GPU index requires a long initialization time
        self.index = faiss.IndexFlatIP(hidden_size)
        self.hidden_size = hidden_size

    def append(self, tensor: torch.Tensor):
        assert tensor.dim() == 2 and tensor.size(1) == self.hidden_size
        self.index.add(tensor.cpu().float().numpy().astype("float32"))

    def get_data(self):
        raise ValueError

    def get_topk(self, tensor: torch.Tensor, topk):
        assert tensor.dim() == 1 and tensor.size(0) == self.hidden_size
        xq = tensor[None, :].cpu().float().numpy().astype("float32")
        topk_index = self.index.search(xq, topk)[1][0].tolist()
        return topk_index

    def __len__(self):
        return self.index.ntotal


GLOBAL_STREAM = None

class ContextManager:
    def __init__(self, 
                 position_embedding,
                 n_init, n_local, 
                 block_size, max_cached_block, topk, exc_block_size, 
                 score_decay: Optional[float] = None, fattn: bool = False,
                 repr_topk: int = 1,
                 cache_strategy = "lru",
                 chunk_topk_calc: Optional[int] = None,
                 async_global_stream: bool = False,
                 pin_memory: bool = False,
                 faiss: bool = False,
                 perhead: bool = False
    ):
        self.length = 0
        self.position_embedding = position_embedding
        self.n_init = n_init
        self.n_local = n_local
        self.block_size = block_size
        self.max_cached_block = max_cached_block
        self.exc_block_size = exc_block_size # default 512
        self.score_decay = score_decay
        assert exc_block_size <= n_local # no global token in input
        self.topk = topk
        self.Attn, _ = get_multi_stage_dot_production_attention(fattn)
        self.fattn = fattn
        self.initialized = False
        self.repr_topk = repr_topk
        self.cache_strategy = cache_strategy
        self.load_count = 0
        self.chunk_topk_calc = chunk_topk_calc
        self.async_global_stream = async_global_stream
        self.pin_memory = pin_memory
        self.faiss = faiss
        self.perhead = perhead
        self.cpucache = CPUCache()

        global GLOBAL_STREAM
        if self.async_global_stream and GLOBAL_STREAM is None:
            GLOBAL_STREAM = torch.cuda.Stream()
            

        assert cache_strategy in ["lru", "lru-s"]

        if cache_strategy == "lru-s":
            self.calc_block_score = True
        else:
            self.calc_block_score = False

        
    def remove_lru_blocks(self, u, num_remove: Optional[int] = None, ignore_blocks = None):
        if num_remove is None:
            num_remove = len(self.cached_blocks[u]) - self.max_cached_block

        if num_remove <= 0:
            return

        lst = list(self.cached_blocks[u].items())
        lst.sort(key=lambda x: x[1])

        removed = 0
        for i in range(len(lst)):
            idx = lst[i][0]
            if ignore_blocks is None or (idx not in ignore_blocks):
                self.global_blocks[u][idx].offload()
                self.cached_blocks[u].pop(idx)
                removed += 1

            if removed >= num_remove:
                return

    def get_block_k(self, k, score):
        assert isinstance(score, torch.Tensor)
        assert k.dim() >= 2
        k = self.from_group_kv(k)

        
        assert k.shape[:-1] == score.shape
        # assert k.shape[-2] == self.block_size
        
        repr_topk = min(self.repr_topk, score.size(-1))
        score_topk = score.topk(repr_topk, dim=-1).indices
        assert score_topk.shape == (self.batch_size, self.num_heads, repr_topk)
        ret = torch.gather(k, -2, score_topk[:, :, :, None].expand(self.batch_size, self.num_heads, repr_topk, self.dim_head))
        return ret

    def from_group_kv(self, tensor):
        assert tensor.dim() == 4 
        assert tensor.size(1) == self.num_heads_kv
        if self.num_heads == self.num_heads_kv:
            return tensor
        _, _, length, dim_head = tensor.shape
        num_group = self.num_heads // self.num_heads_kv
        tensor = tensor.view((self.batch_size, self.unit_size_kv, 1, length, dim_head))
        tensor = tensor.expand((self.batch_size, self.unit_size_kv, num_group, length, dim_head)).reshape((self.batch_size, self.num_heads, length, dim_head))
        return tensor
            
    def init(
        self, 
        local_q, local_k, local_v,
        global_q, global_k, global_v
    ):
        assert local_q.dim() == 4
        batch_size, num_heads, len_q, dim_head = local_q.shape
        num_heads_kv = local_k.size(1)

        for _t in [local_q, local_k, local_v, global_q, global_k, global_v]:
            assert _t.size(0) == batch_size
            assert (_t.size(1) == num_heads or _t.size(1) == num_heads_kv)
            assert _t.size(2) == len_q
            assert _t.size(3) == dim_head
            assert _t.is_cuda


        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.dim_head = dim_head
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.unit_size_kv = num_heads_kv

        self.global_blocks = [[] for _ in range(self.batch_size)] # [[memory_unit]]
        self.cached_blocks = [{} for _ in range(self.batch_size)] # [[block_id: block_score]
        self.num_global_block = 0

        if self.faiss:
            self.block_k = [Faiss(
                dim_head * self.num_heads, global_k.dtype
            ) for _ in range(self.batch_size)]
        else:
            self.block_k = [VectorTensor(
                dim_head * self.num_heads, global_k.dtype
            ) for _ in range(self.batch_size)]

        self.local_k = torch.empty((self.batch_size, self.unit_size_kv, 0, dim_head), dtype=local_k.dtype, device=local_k.device)
        self.local_v = torch.empty((self.batch_size, self.unit_size_kv, 0, dim_head), dtype=local_v.dtype, device=local_v.device)

        self.global_remainder = (
            torch.empty((self.batch_size, self.unit_size_kv, 0, dim_head), dtype=global_k.dtype, device=global_k.device),
            torch.empty((self.batch_size, self.unit_size_kv, 0, dim_head), dtype=global_v.dtype, device=global_v.device),
        )

        self.global_remainder_local_score = torch.empty((self.batch_size, self.num_heads, 0), dtype=global_k.dtype, device=global_k.device)


        self.init_k = torch.empty((self.batch_size, self.unit_size_kv, 0, dim_head), dtype=global_k.dtype, device=global_k.device)
        self.init_v = torch.empty((self.batch_size, self.unit_size_kv, 0, dim_head), dtype=global_k.dtype, device=global_k.device)
        self.init_exc = False
        self.dtype = local_q.dtype
        self.position_embedding._update_cos_sin_tables_len(
            self.n_local + self.exc_block_size + 1, local_k.device, local_k.dim()
        )

        buffer_len = self.topk * self.block_size + self.exc_block_size + self.block_size + self.n_init
        self.global_buffer = torch.zeros(
                (2, self.batch_size, self.unit_size_kv, buffer_len , dim_head),
                dtype = global_k.dtype, device=global_k.device
            )
        self.global_buffer_block_id_list = [[-1] * self.topk for _ in range(self.batch_size)]
        self.global_buffer_init_st = 0
        self.global_buffer_init_ed = 0
        self.cuda_cache = CudaCache(
            self.max_cached_block * self.batch_size,
            self.unit_size_kv * self.block_size * dim_head * 2,
            local_k.dtype
        )

        self.initialized = True

    def calc_block_topk(
        self, global_h_q
    ):
    # calc current block topk
        if not self._use_chunk_topk:
            if self.num_global_block <= self.topk:
                return [list(range(len(self.global_blocks[0]))) for _ in range(self.batch_size)]

            global_h_q = global_h_q.mean(dim=2, keepdim=False)
            assert global_h_q.shape == (self.batch_size, self.num_heads, self.dim_head)
            global_h_q = global_h_q.reshape(self.batch_size, self.dim_head * self.num_heads)
            ret = []
            for u in range(self.batch_size):
                ret.append(self.block_k[u].get_topk(global_h_q[u], self.topk))

        else:
            return self._cached_topk[self._topk_cur]

        return ret

    def get_global_hidden_and_mask(
        self, len_q, block_topk
    ):
        assert len(block_topk) == self.batch_size
        global_block_map = [[] for _ in range(self.batch_size)]
        global_remainder_len = max(self._global_remainder_ed - self._global_remainder_st + len_q - self.n_local, 0)
        init_len = self.init_k.size(-2)
        sliding_window = None

        global_h_k = self.global_buffer[0]
        global_h_v = self.global_buffer[1]

        block_num = len(block_topk[0])
        for u in range(self.batch_size):
            assert len(block_topk[u]) == block_num

            block_topk[u].sort()
            global_block_map[u] = deepcopy(self.global_buffer_block_id_list[u])
            for b_idx in block_topk[u]:
                if b_idx in global_block_map[u]:
                    continue

                st = -1
                ed = -1
                for j in range(self.topk):
                    if global_block_map[u][j] == -1 or global_block_map[u][j] not in block_topk[u]:
                        st = j * self.block_size
                        ed = st + self.block_size
                        global_block_map[u][j] = b_idx
                        break

                
                assert b_idx in self.cached_blocks[u]
                # load the block to the GPU
                self.global_blocks[u][b_idx].load((global_h_k[u, :, st:ed, :], global_h_v[u, :, st:ed, :]))

        # copy the init token to the global buffer
        init_st = block_num * self.block_size
        init_ed = init_st + init_len

        # if the init location no change , need't to copy
        if self.global_buffer_init_st != init_st or self.global_buffer_init_ed != init_ed:
            global_h_k[:, :, init_st: init_ed, :].copy_(self.init_k, non_blocking=True)
            global_h_v[:, :, init_st: init_ed, :].copy_(self.init_v, non_blocking=True)

        ed = init_ed

        rmd_st = init_ed
        rmd_ed = rmd_st + global_remainder_len
        ed = rmd_ed

        global_h_k[:, :, rmd_st: rmd_ed, :].copy_(self.global_remainder[0][:, :, self._global_remainder_st:self._global_remainder_st+global_remainder_len, :], non_blocking=True)
        global_h_v[:, :, rmd_st: rmd_ed, :].copy_(self.global_remainder[1][:, :, self._global_remainder_st:self._global_remainder_st+global_remainder_len, :], non_blocking=True)


        sliding_window = (self.global_remainder[0].size(-2) + rmd_st, self.n_local)

        self.global_buffer_block_id_list = deepcopy(global_block_map)
        self.global_buffer_init_st = init_st
        self.global_buffer_init_ed = init_ed

        for u in range(self.batch_size):
            assert max(global_block_map[u][block_num:] + [-1]) == -1
            assert min(global_block_map[u][:block_num] + [0]) > -1
            global_block_map[u] = list(global_block_map[u][:block_num])


        global_h_k = global_h_k[:, :, :ed, :]
        global_h_v = global_h_v[:, :, :ed, :]
        return global_h_k, global_h_v, sliding_window, global_block_map, block_num

    def update_block_score(
        self, global_score: torch.FloatTensor, global_block_map, global_block_num
    ):
        if global_score is not None:
            global_score = global_score[:, :, :global_block_num * self.block_size]
            assert global_score.shape == (self.batch_size, self.num_heads, global_block_num * self.block_size)
            global_score = global_score.view(self.batch_size, self.num_heads, global_block_num, self.block_size)
            global_score = global_score.sum(dim=-1).sum(dim=1)
            assert global_score.shape == (self.batch_size, global_block_num)
            global_score = global_score.to(device='cpu', non_blocking=False) # (num_units, global_block_num)
            for u in range(self.batch_size):
                for k, v in self.cached_blocks[u].items():
                    self.cached_blocks[u][k] = v * self.score_decay
                score = global_score[u].tolist()
                assert len(score) >= len(global_block_map[u])
                for s, i in zip(score, global_block_map[u]):
                    self.cached_blocks[u][i] += s
    
    def _append(
        self,
        local_q, local_k, local_v, global_q
    ):
        # get local_h_q, local_h_k, local_h_v
        # rotary the q v
        # TODO achieve the correct rotary position
        local_h_q, local_h_k = self.position_embedding(local_q, local_k)
        local_h_v = local_v

        # calc local result first to overlap host-device communication
        attn = self.Attn(local_h_q.shape, local_h_q.dtype, local_h_q.device)
        attn.append(
            local_h_q, local_h_k, local_h_v, 
            get_score=True, sliding_window=self.n_local
        )

        # calc topk global repr k and load cache
        with torch.cuda.stream(GLOBAL_STREAM):
            block_topk = self.calc_block_topk(global_q)
            
            # calc evict block number
            for u in range(self.batch_size):
                num_remove = len(self.cached_blocks[u]) - self.max_cached_block
                for bidx in block_topk[u]:
                    if bidx not in self.cached_blocks[u]:
                        num_remove += 1

                # update cache
                self.remove_lru_blocks(u, num_remove, block_topk[u])

            # update the lru time
            if self.cache_strategy == "lru":
                self.load_count += 1
                for u in range(self.batch_size):
                    for bidx in block_topk[u]:
                        self.cached_blocks[u][bidx] = self.load_count

            elif self.cache_strategy == "lru-s":
                for u in range(self.batch_size):
                    for bidx in block_topk[u]:
                        self.cached_blocks[u][bidx] = 0
            else:
                raise ValueError

            # get global_h_k, global_h_v, global_mask
            #    Beacuse exc_block_size <= n_local, no global_k, global_v used in global part
            global_h_q = global_q
            global_h_k, global_h_v, global_sliding_window, global_block_map, global_block_num = self.get_global_hidden_and_mask(local_h_q.size(-2), block_topk)

        if self.async_global_stream:
            torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)

        # calc global result
        attn.append(
            global_h_q, global_h_k, global_h_v, 
            end=True, get_score=self.calc_block_score, 
            sliding_window=global_sliding_window,
            complement_sliding_window=True
        )

        o, score_list = attn.get_result()
        loc_score = score_list[0]
        glb_score = score_list[1]

        if self.async_global_stream:
            GLOBAL_STREAM.wait_stream(torch.cuda.current_stream())

        # update global score
        with torch.cuda.stream(GLOBAL_STREAM):
            self.update_block_score(glb_score, global_block_map, global_block_num)


        return o.view((self.batch_size, self.num_heads, -1, self.dim_head)), loc_score

    def get_batched_topk(self, global_q):
        length = global_q.shape[2]
        exc_num = (length + self.exc_block_size - 1) // self.exc_block_size
        exc_block_num = length // self.exc_block_size
        ret = []
        if self.num_global_block <= self.topk:
            for _ in range(exc_num):
                ret.append(
                    [list(range(len(self.global_blocks[0]))) for _ in range(self.batch_size)]
                )
            return ret
        
        global_h_q = global_q
        assert global_h_q.dim() == 4
        assert global_h_q.shape[:2] == (self.batch_size, self.num_heads)
        assert global_h_q.shape[3] == self.dim_head

        block_k = torch.cat([self.block_k[u].get_data()[None, :, :] for u in range(self.batch_size)], dim=0)
        assert block_k.shape == (self.batch_size, self.num_global_block, self.dim_head * self.num_heads)
        block_k = block_k.reshape(self.batch_size, self.num_global_block, self.num_heads, self.dim_head).permute(0, 2, 1, 3).contiguous()


        if exc_block_num > 0:
            tmp_global_h_q = global_h_q[:, :, :exc_block_num * self.exc_block_size, :].reshape(
                self.batch_size, self.num_heads, exc_block_num, self.exc_block_size, self.dim_head
            ).mean(dim=-2)
            assert tmp_global_h_q.shape == (self.batch_size, self.num_heads, exc_block_num, self.dim_head)
            block_score = torch.matmul(
                tmp_global_h_q, block_k.transpose(-1, -2)
            ).mean(dim=1) # (num_units, exc_block_num, num_global_block)
            assert block_score.shape == (self.batch_size, exc_block_num, self.num_global_block)

            indices = block_score.topk(self.topk, dim=-1).indices.cpu()
            for b in range(exc_block_num):
                tmp = []
                for u in range(self.batch_size):
                    tmp.append(indices[u, b].tolist())
                    assert len(tmp[-1]) == self.topk
                
                ret.append(tmp)

        if exc_block_num != exc_num: 
            tmp_global_h_q = global_h_q[:, :, exc_block_num * self.exc_block_size:, :].reshape(
                self.batch_size, self.num_heads, length - exc_block_num * self.exc_block_size, self.dim_head
            ).mean(dim=-2, keepdim=True)
            assert tmp_global_h_q.shape == (self.batch_size, self.num_heads, 1, self.dim_head)
            block_score = torch.matmul(
                tmp_global_h_q, block_k.transpose(-1, -2)
            )
            assert block_score.shape == (self.batch_size, self.num_heads, 1, self.num_global_block)
            block_score = block_score.squeeze(dim=2).mean(dim=1)
            assert block_score.shape == (self.batch_size, self.num_global_block)
            indices = block_score.topk(self.topk, dim=-1).indices.cpu()
            tmp = []
            for u in range(self.batch_size):
                tmp.append(indices[u].tolist())
                assert len(tmp[-1]) == self.topk

            ret.append(tmp)

         
        return ret

    def offload_ssd(self , hash_str ,layer_idx , offset = 0):
        # store the global block to ssd by binary format
        file_name =  "kvcache/global_blocks_data" + str(hash_str) + "layer_"+ str(layer_idx) + ".bin"
       
        with open(file_name, "wb") as f:
            # store the repr token
            repr_token_num = self.block_k[0].length
            assert repr_token_num == len(self.global_blocks[0])

            repr_token = self.block_k[0].get_data()[offset:]
            repr_shape = np.array(repr_token.shape, dtype=np.int32)
            f.write(repr_shape.tobytes())  # Write shape
            f.write(repr_token.cpu().view(torch.int16).numpy().tobytes())  # Write raw data

            # store the global block
            f.write(np.array([self.batch_size, repr_token.size(0)], dtype=np.int32).tobytes())
            # Iterate through each unit and block to serialize and store their data
            for u in range(self.batch_size):
                for i in range(offset  , repr_token_num):
                    memory_unit = self.global_blocks[u][i]
                    # Extract tensors from the MemoryUnit
                    kv_0 = memory_unit.cpu_data[0].cpu().view(torch.int16).numpy()  # First tensor (move to CPU if on GPU)
                    kv_1 = memory_unit.cpu_data[1].cpu().view(torch.int16).numpy()  # Second tensor (move to CPU if on GPU)
                    
                    # Serialize the shape and data of each tensor
                    for tensor in [kv_0, kv_1]:
                        shape = np.array(tensor.shape, dtype=np.int32)
                        f.write(shape.tobytes())  # Write shape
                        f.write(tensor.tobytes())  # Write raw data     

    def blend(self, hash_str, indices, partial_k , partial_v , layer_idx):
        # Load the global block data from the SSD
        for i, hash_s in enumerate(hash_str):
            idx = indices[i]
            filename =  "kvcache/global_blocks_data" + str(hash_s) + "layer_"+ str(layer_idx) + ".bin"
            # append to the current ctm 
            start = time.time()
            k, v = self.cpucache.get_kv(filename,idx[-2],idx[-1] )

            k = k.to(partial_k.device)
            v = v.to(partial_v.device)
            end = time.time()
            # print("load time: ", end - start )
            # Insert the concatenated tensors into the correct positions in partial_k and partial_v
            partial_k = torch.cat([partial_k[:,:,:idx[0],:], k , partial_k[: , : , idx[0]: , :]] ,dim = 2)  # Insert as a single-element batch
            partial_v = torch.cat([partial_v[:,:,:idx[0],:], v , partial_v[: , : , idx[0]: , :]] ,dim = 2)  # Insert as a single-element batch   
        return partial_k, partial_v

    def get_previous_kv(self, hash_str, layer_idx):
        # Load the global block data from the SSD
        filename =  "kvcache/global_blocks_data" + str(hash_str) + "layer_"+ str(layer_idx) + ".bin"
        # append to the current ctm 
        self.cpucache.cache_memory_units(filename)

    def append_global(
        self, exc_length, kv_length, local_score
    ):

        global_remainder_ed = self._global_remainder_ed + exc_length
        global_remainder_st = self._global_remainder_st

        global_remainder_len = global_remainder_ed - global_remainder_st

        assert local_score.shape[:3] == (self.batch_size, self.num_heads, kv_length)
        local_score = local_score[:, :, -exc_length-self.n_local:]

        # compute the block score
        self.global_remainder_local_score[:, :, global_remainder_ed-local_score.size(-1):global_remainder_ed].add_(local_score)
        
        if not self.init_exc and global_remainder_len > self.n_local:
            global_k = self.global_remainder[0]
            global_v = self.global_remainder[1]

            append_init_len = min(
                self.n_init - self.init_k.size(-2),
                global_remainder_len - self.n_local
            )
            self.init_k = torch.cat(
                (self.init_k, global_k[:, :, global_remainder_st:global_remainder_st + append_init_len, :]), dim=-2
            )
            self.init_v = torch.cat(
                (self.init_v, global_v[:, :, global_remainder_st:global_remainder_st + append_init_len, :]), dim=-2
            )
            global_remainder_st += append_init_len
            global_remainder_len -= append_init_len

            if self.init_k.size(-2) == self.n_init:
                self.init_exc = True

        while global_remainder_len - self.block_size >= self.n_local:
            global_remainder_len -= self.block_size
            for u in range(self.batch_size):
                self.global_blocks[u].append((
                    MemoryUnit(
                        (
                            self.global_remainder[0][u, :, global_remainder_st:global_remainder_st + self.block_size, :],
                            self.global_remainder[1][u, :, global_remainder_st:global_remainder_st + self.block_size, :]
                        ),
                        self.cuda_cache,
                        False,
                        self.pin_memory
                    )
                ))

            # global_block_k = self.get_block_k(
            #     self.global_remainder[0][:, :, global_remainder_st:global_remainder_st + self.block_size, :],
            #     self.global_remainder_local_score[:, :, global_remainder_st:global_remainder_st + self.block_size]
            # )

            global_block_k = self.global_remainder[0][:, :, global_remainder_st:global_remainder_st + self.block_size, :]
            global_block_k = self.from_group_kv(global_block_k)
            # repr_token 
            # assert global_block_k.shape == (self.batch_size, self.num_heads, self.repr_topk, self.dim_head)
            global_block_k = global_block_k.mean(dim=-2, keepdim=False)
            global_block_k = global_block_k.reshape(self.batch_size, self.num_heads * self.dim_head)
            global_block_k = global_block_k[:, None, :]

            self.num_global_block += 1
            for u in range(self.batch_size):
                self.block_k[u].append(global_block_k[u])
            global_remainder_st += self.block_size

        self._global_remainder_ed = global_remainder_ed
        self._global_remainder_st = global_remainder_st

    def flush_loacl_to_block(self):
        assert self.num_global_block == self.block_k[0].length
        
        if self.async_global_stream:
            GLOBAL_STREAM.wait_stream(torch.cuda.current_stream())

        if self.global_remainder[0].size(-2) > 0:
            global_remainder_len = self._global_remainder_ed - self._global_remainder_st
            assert global_remainder_len == self.global_remainder[0].size(-2)
            assert global_remainder_len == self.global_remainder_local_score.size(-1)

            while global_remainder_len > 0:
                unit_size = min(self.block_size, global_remainder_len)

                global_remainder_len -= unit_size
                for u in range(self.batch_size):
                    self.global_blocks[u].append((
                        MemoryUnit(
                            (
                                self.global_remainder[0][u, :, self._global_remainder_st:self._global_remainder_st + unit_size, :],
                                self.global_remainder[1][u, :, self._global_remainder_st:self._global_remainder_st + unit_size, :]
                            ),
                            self.cuda_cache,
                            False,
                            self.pin_memory
                        )
                    ))

                    with torch.cuda.stream(GLOBAL_STREAM):
                        global_block_k = self.global_remainder[0][:, :, self._global_remainder_st:self._global_remainder_st + unit_size, :]
                        global_block_k = self.from_group_kv(global_block_k)

                        # global_block_k = self.get_block_k(
                        #     self.global_remainder[0][:, :, self._global_remainder_st:self._global_remainder_st + unit_size, :],
                        #     self.global_remainder_local_score[:, :, self._global_remainder_st:self._global_remainder_st + unit_size]
                        # )
                        # repr_token
                        # assert global_block_k.shape == (self.batch_size, self.num_heads, self.repr_topk, self.dim_head)
                        global_block_k = global_block_k.mean(dim=-2, keepdim=False)
                        global_block_k = global_block_k.reshape(self.batch_size, self.num_heads * self.dim_head)
                        global_block_k = global_block_k[:, None, :]
                    
                if self.async_global_stream:
                    torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)
                for u in range(self.batch_size):
                    self.block_k[u].append(global_block_k[u])

                self.num_global_block += 1
                assert self.num_global_block == self.block_k[0].length
                self._global_remainder_st += unit_size

            # update the global remainder and clear the local cache
            with torch.cuda.stream(GLOBAL_STREAM):

                # update the local kv 
                self.local_k = self.local_k[:, :, :-0, :]
                self.local_v = self.local_v[:, :, :-0, :]

                self.global_remainder = (
                    self.global_remainder[0][:, :, :-0, :],
                    self.global_remainder[1][:, :, :-0, :]
                )
                self.global_remainder_local_score = self.global_remainder_local_score[:, :, :-0]
                self._global_remainder_ed = 0
                self._global_remainder_st = 0
            
            assert self._global_remainder_ed == 0
        
        assert self.num_global_block == self.block_k[0].length
        self.init_exc = True
        return self.block_k[0].length
      
    def do_blend(self, rc_q, rc_k, rc_v, rc_idx, hash_str, indices, layer_idx):
        # 1. flush the local cache to global cache
        self.flush_loacl_to_block()
        
        # 2. retrieve the kvcache from ssd
        idx = indices
        filename =  "kvcache/global_blocks_data" + str(hash_str) + "layer_"+ str(layer_idx) + ".bin"
        # append to the current ctm 
        doc_k, doc_v, doc_repr = self.cpucache.get_memory_units(filename)
        device = rc_k.device

        doc_repr = doc_repr.to(device)

        idx_st = idx[0][-2] // self.block_size
        idx_ed = idx[0][-1] // self.block_size + (idx[0][-1] % self.block_size != 0 )

        append_unit = []
        append_unit_repr = []
        for i in range(idx_st, idx_ed):
            append_unit.append(
                MemoryUnit(
                    (
                        doc_k[i],
                        doc_v[i]
                    ),
                    self.cuda_cache,
                    False,
                    self.pin_memory
                )
            )
            append_unit_repr.append(doc_repr[i:i+1,:])

        current_repr_token_num = self.block_k[0].length

        append_idx = 0
        o_list = []
        for i in range(0 , len(rc_idx) , self.block_size):
            block_idx = rc_idx[i] // self.block_size
            block_ed = min(len(rc_idx), i + self.block_size)

            while append_idx < block_idx:
                self.global_blocks[0].append(append_unit[append_idx])
                self.block_k[0].append(append_unit_repr[append_idx])
                append_idx += 1
                self.num_global_block += 1

            # recompute current block
            local_q = rc_q[:, :, i:block_ed, :]
            local_k = rc_k[:, :, i:block_ed, :]
            local_v = rc_v[:, :, i:block_ed, :]
            global_q = rc_q[:, :, i:block_ed, :]
            global_k = rc_k[:, :, i:block_ed, :]
            global_v = rc_v[:, :, i:block_ed, :]
            
            # store the recompute kv cache to global cache
            with torch.cuda.stream(GLOBAL_STREAM):
                global_q = self.position_embedding.apply_rotary_pos_emb_one_angle(
                    global_q, self.n_local
                )

            block_o = self.append(
                local_q, local_k, local_v,
                global_q, global_k, global_v,
            )
            o_list.append(block_o)

            self.flush_loacl_to_block()
            
            # todo update the repr token
            # self.block_k[0].update_back(append_unit_repr[block_idx])
            append_idx += 1

        while append_idx < len(append_unit):
            self.global_blocks[0].append(append_unit[append_idx])
            self.block_k[0].append(append_unit_repr[append_idx])
            append_idx += 1
            self.num_global_block += 1
        o = torch.cat(o_list, dim=-2)

        global_remainder_len = self._global_remainder_ed - self._global_remainder_st
       
        assert global_remainder_len == self.global_remainder[0].size(-2)
        assert self.block_k[0].length == current_repr_token_num + len(append_unit_repr)

        return o

    def append(
        self,
        local_q, local_k, local_v,
        global_q, global_k, global_v,
    ):
        
        batch_size = local_q.size(0)
        input_length = local_q.size(-2)
        if self.perhead:
            num_heads = local_q.size(1)
            num_heads_kv = local_v.size(1)
            def repeat_kv(t):
                t = t.view(batch_size, num_heads_kv, 1, input_length, -1)
                t = t.expand(batch_size, num_heads_kv, num_heads // num_heads_kv, input_length,  -1)
                t = t.reshape(batch_size * num_heads, 1, input_length, -1)
                return t

            local_q = local_q.view(batch_size * num_heads, 1, input_length, -1)
            local_k = repeat_kv(local_k)
            local_v = repeat_kv(local_v)
            global_q = global_q.view(batch_size * num_heads , 1, input_length, -1)
            global_k = repeat_kv(global_k)
            global_v = repeat_kv(global_v)

        if not self.initialized:
            self.init(
                local_q, local_k, local_v,
                global_q, global_k, global_v
            )

        input_length = local_q.size(-2)
        
        if self.async_global_stream:
            GLOBAL_STREAM.wait_stream(torch.cuda.current_stream())
        
        # TODO: 
        # 1. flush the local cache to global cache
        # 2. retreve the kvcache from ssd
        # 3. load the kvcache to global cache 

        # append local and global tensor
        self.local_k = torch.cat((self.local_k, local_k), dim=-2)
        self.local_v = torch.cat((self.local_v, local_v), dim=-2)
        kv_length = self.local_k.size(-2)

        # append global remainder
        with torch.cuda.stream(GLOBAL_STREAM):
            self._global_remainder_st = 0
            self._global_remainder_ed = self.global_remainder[0].size(-2)

            self.global_remainder = (
                torch.cat((self.global_remainder[0], global_k), dim=-2),
                torch.cat((self.global_remainder[1], global_v), dim=-2),
            )

            self.global_remainder_local_score = torch.cat(
                (self.global_remainder_local_score, 
                torch.zeros(
                        (self.batch_size, self.num_heads, global_k.size(-2)),
                        dtype=global_k.dtype, device=global_k.device
                    )
                ),
                dim=-1
            )

        with torch.cuda.stream(GLOBAL_STREAM):
            global_q = self.position_embedding.apply_rotary_pos_emb_one_angle(
                global_q, self.n_local
            )

        use_chunk_topk = self.chunk_topk_calc is not None and input_length > 1
        self._use_chunk_topk = use_chunk_topk
        if use_chunk_topk:
            exc_block_num = input_length // self.exc_block_size
            exc_block_per_topk_chunk = self.chunk_topk_calc // self.exc_block_size
            calc_cur_list = [i * self.exc_block_size for i in range(0, exc_block_num + 1, exc_block_per_topk_chunk)]
            if calc_cur_list[-1] < input_length:
                calc_cur_list.append(input_length)
            self._topk_cur = 0
            self._topk_calc_cur = -1

        o_list = []

        for st in range(0, input_length, self.exc_block_size): 
            ed = min(st + self.exc_block_size, input_length)
            if use_chunk_topk and calc_cur_list[self._topk_calc_cur + 1] < ed:
                # calculate topk and sync with host here
                assert ed <= calc_cur_list[self._topk_calc_cur + 2]
                self._topk_calc_cur += 1
                with torch.cuda.stream(GLOBAL_STREAM):
                    self._cached_topk = self.get_batched_topk(global_q[:, :, calc_cur_list[self._topk_calc_cur]: calc_cur_list[self._topk_calc_cur + 1], :])
                self._topk_cur = 0

            kv_st = max(kv_length + st - input_length - self.n_local, 0)
            kv_ed = kv_length + ed - input_length
            # local + current 
            chunk_o, local_score = self._append(
                local_q[:, :, st:ed, :],
                self.local_k[:, :, kv_st: kv_ed, :],
                self.local_v[:, :, kv_st: kv_ed, :],
                global_q[:, :, st:ed, :]
            )
            o_list.append(chunk_o)

            # append global
            with torch.cuda.stream(GLOBAL_STREAM):
                self.append_global(ed - st, kv_ed - kv_st, local_score)

            if self.async_global_stream:
                torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)

            if use_chunk_topk:
                self._topk_cur += 1

        self.length += input_length

        # update local and global tensor
        if self.local_k.size(-2) >= self.n_local:
            self.local_k = self.local_k[:, :, -self.n_local:, :]
            self.local_v = self.local_v[:, :, -self.n_local:, :]

        assert self._global_remainder_ed == self.global_remainder[0].size(-2)
        with torch.cuda.stream(GLOBAL_STREAM):
            self.global_remainder = (
                self.global_remainder[0][:, :, self._global_remainder_st:, :],
                self.global_remainder[1][:, :, self._global_remainder_st:, :]
            )
            self.global_remainder_local_score = self.global_remainder_local_score[:, :, self._global_remainder_st:]
            self._global_remainder_st = 0
            self._global_remainder_ed = self.global_remainder[0].size(-2)
            assert self._global_remainder_ed - self._global_remainder_st == self.global_remainder[0].size(-2)

        ret = torch.cat(o_list, dim=-2)

        if self.perhead:
            ret = ret.view(batch_size, num_heads, input_length, -1)

        return ret

    def selective_recompute_block(        
        self,
        local_q, local_k, local_v,
        global_q, global_k, global_v,
        hash_str, indices, layer_idx, recompute_ratio = 0.15):
        
        #  selective the block to recompute
        self.flush_loacl_to_block()
        len_q = local_q.size(-2)
        o = self.append(
            local_q, local_k, local_v,
            global_q, global_k, global_v
        )
        self.flush_loacl_to_block()

        filename =  "kvcache/global_blocks_data" + str(hash_str) + "layer_"+ str(layer_idx) + ".bin"
        # append to the current ctm 
        _ , _ ,doc_repr = self.cpucache.get_memory_units(filename)
        repr_st = indices[0][-2] // self.block_size
        repr_ed = (indices[0][-1] + self.block_size - 1) // self.block_size 

        repr_tokens = doc_repr[repr_st:repr_ed].to(local_q.device)
        assert repr_tokens.size(0) == repr_ed - repr_st

        recompute_tokens = self.block_k[0].data[-repr_tokens.size(0) : ]

        deviation = torch.mean((repr_tokens - recompute_tokens) ** 2,dim = -1)

        recompute_block_num = int(np.ceil((repr_ed - repr_st) * recompute_ratio))
        _,topk_deviation = torch.topk(deviation, recompute_block_num, dim=0)
        topk_deviation = topk_deviation.view(-1).tolist()
        for i in topk_deviation:
            print("recompute block id", i)
        recompute_idx = [idx * self.block_size + i for idx in topk_deviation for i in range(self.block_size) if idx * self.block_size + i < len_q]
        recompute_idx.sort()
        recompute_idx_tensor = torch.tensor(recompute_idx, dtype=torch.long, device=local_q.device)
        o = o[:,:, recompute_idx_tensor,:]
        
        return  o, recompute_idx

    def selective_recompute_block_by_k_deviation(self,
        local_q, local_k, local_v,
        global_q, global_k, global_v,
        hash_str, indices, layer_idx, recompute_ratio = 0.15):
        
        len_q = local_q.size(-2)
        o = self.append(
            local_q, local_k, local_v,
            global_q, global_k, global_v
        )

        filename =  "kvcache/global_blocks_data" + str(hash_str) + "layer_"+ str(layer_idx) + ".bin"

        key = self.cpucache.get_memory_k(filename, indices[0])
        key = key.reshape(local_k.shape).to(local_q.device)
        repr_st = indices[0][-2] // self.block_size
        repr_ed = (indices[0][-1] + self.block_size - 1) // self.block_size 

        dims_to_average = [0 , 1 , -1]
        diff_per_token = torch.mean((key - local_k) ** 2, dim=dims_to_average)
        block_deviations = []

        for i in range(0 , local_q.size(-2) , self.block_size):
            st = i * self.block_size   
            ed = min(i + self.block_size, local_q.size(-1))
            block_diff = diff_per_token[st:ed]
            block_deviations.append(block_diff.sum().item())

        recompute_block_num = int(np.ceil((repr_ed - repr_st) * recompute_ratio))
        _,topk_deviation = torch.topk(torch.tensor(block_deviations), recompute_block_num, dim=0)
        topk_deviation = topk_deviation.view(-1).tolist()

        for i in topk_deviation:
            print("recompute block id", i)
        recompute_idx = [idx * self.block_size + i for idx in topk_deviation for i in range(self.block_size) if idx * self.block_size + i < len_q]
        recompute_idx.sort()
        recompute_idx_tensor = torch.tensor(recompute_idx, dtype=torch.long, device=local_q.device)
        o = o[:,:, recompute_idx_tensor,:] 
        
        return o, recompute_idx

    def size(self, *args, **kwargs):
        return self.length