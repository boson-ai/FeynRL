import re
from typing import Dict, Any
import torch

def extract_solution(solution_str, clip_chars=300):
    if len(solution_str) > clip_chars:
        solution_str = solution_str[-clip_chars:]

    # this also tests the formatting of the model
    solution = re.search(r"####\s*(-?[0-9.,]+)", solution_str)
    if solution is None:
        final_answer = None
    else:
        # take the last solution
        final_answer = solution.group(1).replace(",", "").replace("$", "").replace("\n", "")

    return final_answer

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
    format_score = 0.0
    score = 1.0
    correct_threshold = 0.0
    is_per_token = False

    solution_str = response_data.text
    ground_truth = prompt_data["solution"]

    r = torch.zeros((len(response_data.token_ids),), dtype=torch.float32)
    answer = extract_solution(solution_str=solution_str)

    if answer is None:
        return r, is_per_token, correct_threshold
    else:
        if answer == ground_truth:
            r[-1] = score
            return r, is_per_token, correct_threshold
        else:
            r[-1] = format_score
            return r, is_per_token, correct_threshold