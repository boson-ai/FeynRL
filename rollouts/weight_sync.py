import torch
import numpy as np
import pickle

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
            serialized_state: pickled state_dict (bytes/uint8 tensor/uint8 ndarray),
            or a raw state_dict for backward compatibility.
        '''
        if isinstance(serialized_state, dict):
            # Backward compatibility for older callers that sent state_dict directly.
            state_dict = serialized_state

        elif isinstance(serialized_state, (bytes, bytearray, memoryview)):
            state_dict = pickle.loads(serialized_state)

        elif isinstance(serialized_state, np.ndarray):
            if serialized_state.dtype != np.uint8:
                raise TypeError(f"Expected uint8 ndarray payload, got {serialized_state.dtype}")
            state_dict = pickle.loads(serialized_state.tobytes())

        elif isinstance(serialized_state, torch.Tensor):
            if serialized_state.dtype != torch.uint8:
                raise TypeError(f"Expected uint8 tensor payload, got {serialized_state.dtype}")
            state_dict = pickle.loads(serialized_state.cpu().numpy().tobytes())

        elif isinstance(serialized_state, str):
             # Handled as latin-1 string to preserve bytes
             state_dict = pickle.loads(serialized_state.encode('latin-1'))

        elif isinstance(serialized_state, list):
            # vLLM RPC might convert uint8 ndarray/tensor to a list of integers.
            # Convert back to bytes for unpickling.
            state_dict = pickle.loads(bytes(serialized_state))

        else:
            raise TypeError(f"Unsupported weight payload type: {type(serialized_state)}")

        # With pickle, we get back the original tensors, so we can pass items() directly again.
        self.model_runner.model.load_weights(weights=state_dict.items())
        torch.cuda.synchronize()

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
