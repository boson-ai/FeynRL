import torch
from typing import List, Any, Dict

def compute_score(prompt_data: Dict[str, Any], response_data: Dict[str, Any]):
    '''
      input args:
        prompt_data: Dict[str, Any]
        response_data: Dict[str, Any]
      output args:
        r: torch.Tensor of length of response token ids
        is_per_token: whether the reward is per token
        correct_threshold: a response is counted as correct for pass@k
            when its scalar reward strictly exceeds this threshold. For example, for binary
            reward functions [0,1], this threshold is 0.0.
    '''
    response_ids = list(response_data.token_ids)
    finish_reason = response_data.finish_reason
    correct_threshold = 0.0

    is_per_token = False
    r = torch.zeros((len(response_ids),), dtype=torch.float32)
    
    if len(response_ids) == 0:
        return r, is_per_token, correct_threshold

    r[-1] = 1.0 if str(finish_reason) == "stop" else 0.0

    return r, is_per_token, correct_threshold