import torch
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
