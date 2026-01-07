import torch
from typing import List, Any

def default_reward_func(prompt_ids: List[int], response_ids: List[int], finish_reason: Any):
    '''
      input args:
        prompt_ids: List[int] - list of token ids in the prompt
        response_ids: List[int] - list of token ids in the response
        finish_reason: Any - reason for the finish (e.g., stop, length, etc.)
      output args:
        r: torch.Tensor - reward tensor
        is_per_token: bool - whether the reward is per token
    '''
    is_per_token = False
    r = torch.zeros((len(response_ids),), dtype=torch.float32)
    
    if len(response_ids) == 0:
        return r, is_per_token

    r[-1] = 1.0 if str(finish_reason) == "stop" else 0.0

    return r, is_per_token
