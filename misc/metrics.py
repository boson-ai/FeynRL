import numpy as np    
from typing import List, Dict

def pass_at_k(n: int, c: int, k: int) -> float:
    '''
        Unbiased pass@k estimator based on figure 3 from Chen et al. (Codex), 2021
        https://arxiv.org/pdf/2107.03374.
        n: total number of samples
        c: number of correct samples
        k: k in pass@$k$
    '''
    # pass@k = 1 - C(n-c, k) / C(n, k). 
    # Uses the log-space trick to avoid overflow for large n.
    if n - c < k:
        return 1.0

    return 1.0 - np.exp(sum(np.log(n - c - i) - np.log(n - i) for i in range(k)))

def compute_pass_metrics(rewards: List[float], n_total: int, correct_threshold: float) -> Dict[str, float]:
    '''
        Compute pass@k metrics for a list of rewards.
        rewards: List of rewards for non-empty responses to the same prompt
        n_total: Total number of samples requested which is equal to n_samples.
                 Empty responses count as failed attempts for pass@k as they increase n but not c.
        correct_threshold: A response is "correct" if reward > correct_threshold.
                           Returned by the reward function's compute_score().
        Returns: Dictionary with pass@k metrics
    '''
    n = n_total
    # number of correct responses
    c = sum(1 for r in rewards if r > correct_threshold)
    pass_at_ks = {}
    for k in range(1, n + 1):
        pass_at_ks[k] = pass_at_k(n, c, k)

    # pass^k: all n samples must be correct. Empty responses are failures,
    # so require c == n (not just all entries in rewards are positive).
    pass_caret_k = float(c == n) if n > 0 else 0.0
    pass_rate = float(c / n) if n > 0 else 0.0
    group_mean_reward = float(sum(rewards) / n) if n > 0 else 0.0
    best_of_k_reward = float(max(rewards)) if len(rewards) > 0 else 0.0
    reward_std_per_prompt = float(np.std(rewards)) if len(rewards) > 0 else 0.0

    return {"pass_at_ks": pass_at_ks,
            "pass_caret_k": pass_caret_k,
            "pass_rate": pass_rate,
            "k": n,
            "group_mean_reward": group_mean_reward,
            "best_of_k_reward": best_of_k_reward,
            "reward_std_per_prompt": reward_std_per_prompt}