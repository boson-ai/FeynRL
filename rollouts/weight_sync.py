import torch
import pickle
from misc.nccl_utils import create_nccl_process_group

class WeightSyncExtension:
    '''
        vllm WorkerExtension that enables in-place weight updates on vllm workers.
        Used with worker_extension_cls parameter when creating vllm llm instances in vllm_engine.py.
        This allows updating model weights directly in gpu memory without
        destroying and recreating the vllm engine (no disk I/O).
    '''

    def __init__(self, model_runner):
        self.model_runner = model_runner

    def update_weights(self, serialized_state):
        '''
            Update model weights in-place on this vllm worker.
            vllm's load_weights handles name remapping and tp sharding internally.
            serialized_state: file path to pickled state_dict on /dev/shm,
            or a raw dict for backward compatibility.

            Returns the number of parameters in the state_dict that were loaded.
            This is used by the caller to verify all TP workers loaded the same weights.
        '''
        if isinstance(serialized_state, str):
            with open(serialized_state, 'rb') as f:
                state_dict = pickle.load(f)

        elif isinstance(serialized_state, dict):
            state_dict = serialized_state

        else:
            raise TypeError(f"Unsupported weight payload type: {type(serialized_state)}")

        num_params = len(state_dict)

        # Sanity-check that state_dict keys have some overlap with model params.
        # vllm fuses some layers internally (e.g. q_proj + k_proj + v_proj -> qkv_proj,
        # gate_proj + up_proj -> gate_up_proj), so a strict 1:1 match is not expected.
        # load_weights handles the remapping, but if zero keys match, the naming
        # convention is completely wrong and load_weights would silently no-op.
        model_params = set(name for name, _ in self.model_runner.model.named_parameters())
        matched = sum(1 for k in state_dict if k in model_params)
        if matched == 0 and num_params > 0:
            raise RuntimeError(f"Weight sync failed: none of the {num_params} state_dict keys "
                               f"matched model parameters. This likely means the naming convention "
                               f"changed between vllm versions. "
                               f"Sample state_dict keys: {list(state_dict.keys())[:3]}, "
                               f"sample model params: {list(model_params)[:3]}")

        self.model_runner.model.load_weights(weights=state_dict.items())
        torch.cuda.synchronize()
        return num_params

    def check_weights_hash(self, param_name):
        '''
            Return a hash of a specific parameter for verification.
            Useful for confirming weights were updated correctly.
            param_name: name of the parameter to hash.
        '''
        for name, param in self.model_runner.model.named_parameters():
            if name == param_name or param_name in name:
                return param.data.float().sum().item()
        return None

    def init_weight_nccl_group(self, master_addr, master_port, rank_offset, world_size, group_name, timeout_seconds, backend="gloo"):
        '''
            Initialize a custom NCCL process group for direct gpu-to-gpu broadcast.
            This group connects training rank 0 (rank=0) to all vllm TP workers
            (rank=1..N). It is separate from DeepSpeed's and vllm's internal groups.
            master_addr: IP for TCP rendezvous, from ray.util.get_node_ip_address().
                         Must be reachable from all nodes in the cluster.
            master_port: Free port for this group (NOT the deepspeed port).
            rank_offset: Starting rank for this rollout engine in the sync group.
                         Each TP worker computes its own rank as rank_offset + tp_rank.
            world_size: 1 (training rank 0) + num_rollout_engines * tp_size.
            group_name: Unique name to namespace the TCP store and avoid key collisions
                        with deepspeed's and vllm's internal groups.
            timeout_seconds: Timeout for NCCL rendezvous and operations, from config.run.init_timeout.
        '''
        # Each TP worker determines its own rank within the sync group.
        # vllm initializes torch.distributed for TP communication (ranks 0..tp-1).
        # For TP=1, torch.distributed may not be initialized, so tp_rank defaults to 0.
        tp_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        my_rank = rank_offset + tp_rank
        self.weight_sync_rank = my_rank
        self.weight_sync_backend = backend
        self.weight_sync_group = create_nccl_process_group(init_method=f"tcp://{master_addr}:{master_port}",
                                                           rank=my_rank,
                                                           world_size=world_size,
                                                           group_name=group_name,
                                                           timeout_seconds=timeout_seconds,
                                                           backend=backend,
                                                          )
        return True

    def update_weights_nccl(self, param_name, dtype_str, shape, empty_cache=False):
        '''
            Receive a single weight tensor via NCCL broadcast from training rank 0
            and load it into the model. In particular, training rank 0 calls
            torch.distributed.broadcast as sender, and this worker calls it as
            receiver. NCCL synchronizes automatically.
            Args:
                param_name: HF parameter name, e.g. model.layers.0.self_attn.q_proj.weight.
                dtype_str: String dtype, e.g. "torch.bfloat16".
                shape: Tuple of ints for the full (unsharded) parameter shape.
                empty_cache: Whether to call torch.cuda.empty_cache() after loading.
            Returns the number of weights loaded (0 or 1).
        '''
        dtype_map = {"torch.float16":  torch.float16,
                     "torch.bfloat16": torch.bfloat16,
                     "torch.float32":  torch.float32,}
        target_dtype = dtype_map.get(dtype_str, torch.bfloat16)

        # Allocate receive buffer. Use CPU for gloo (native), CUDA for nccl.
        # Gloo handles CUDA tensors by staging through host memory, so using
        # CPU directly avoids the redundant CUDA→CPU→gloo→CPU→CUDA round-trip.
        backend = getattr(self, 'weight_sync_backend', 'gloo')
        buf_device = "cuda" if backend == "nccl" else "cpu"
        buffer = torch.empty(shape, dtype=target_dtype, device=buf_device)

        # Broadcast: rank 0 sends, all others receive
        torch.distributed.broadcast(buffer, src=0, group=self.weight_sync_group)

        # Load into model. vllm's load_weights handles tp sharding and
        # device placement internally (CPU tensors are moved to GPU).
        self.model_runner.model.load_weights(weights=[(param_name, buffer)])

        # Synchronize before deleting the buffer to ensure load_weights kernels
        # have completed reading from it. Without this, the async CUDA kernels
        # may still be reading the buffer when it is deallocated, causing
        # use-after-free and silent weight corruption.
        torch.cuda.synchronize()
        del buffer
        if empty_cache:
            torch.cuda.empty_cache()

        return 1

    def close_weight_nccl_group(self):
        '''
            Destroy the custom NCCL process group which is called during shutdown.
        '''
        if hasattr(self, 'weight_sync_group') and self.weight_sync_group is not None:
            try:
                torch.distributed.destroy_process_group(self.weight_sync_group)

            except Exception:
                pass

            self.weight_sync_group = None