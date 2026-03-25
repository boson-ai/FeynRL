# RL Common (`common.py`)

`COMMON` is the base class for all policy-gradient algorithms (GRPO, CISPO, P3O, PPO). It provides shared methods that are identical across algorithms, keeping algorithm-specific files focused on loss computation and training logic.

---

## Global Token Normalization for RL

In RL training, the replay buffer produces variable-length rollouts. After collation and sharding across GPUs, each micro-batch can have a different number of valid action tokens. The same "mean of means $\neq$ global mean" problem from SFT applies here, but with an additional source of variance: **rollout lengths change every epoch** because the policy generates different-length responses as it improves.

For the full derivation of why naive per-micro-batch normalization is biased and how `dp_scale / ga_denom` corrects it, see [`algs/SFT/README.md`](../SFT/README.md). The formula and DeepSpeed interaction are identical.

### Why it matters more in RL than SFT

In SFT, sequence lengths are fixed by the dataset and only vary due to packing/padding. In RL:

- **Rollout lengths vary across epochs.** Early in training the policy may produce short responses; later it may produce longer ones. Without global normalization, epochs with longer responses would have smaller per-token gradient magnitude (more tokens $\Rightarrow$ smaller mean), creating an implicit length-dependent learning rate.
- **Replay buffer sharding is uneven.** `prepare_training_batches` pads the number of micro-batches across ranks, but the token counts within those micro-batches can differ significantly when prompts have different lengths or the policy generates variable-length completions.
- **Group-based advantages (GRPO/CISPO/P3O) amplify the imbalance.** When `n_samples > 1`, some prompts produce many short responses and others produce fewer long ones. Without global normalization, short-response micro-batches get disproportionate gradient weight.

### What `compute_global_token_denom` does

```
ga_denom = total valid action tokens across ALL micro-batches on ALL ranks
dp_scale = ga_steps * world_size
loss_for_backward = loss_sum * (dp_scale / ga_denom)
```

This single `all_reduce` (one scalar, 8 bytes) runs once before the training loop in each `train_step` call. After DeepSpeed's internal gradient averaging ($\div$ `ga_steps` $\div$ `world_size`), the net effect is:

$$\nabla_{\text{final}} = \frac{1}{N_{\text{global}}} \sum_{\text{all tokens}} m_t \cdot \nabla \ell_t$$

Every action token contributes equally to the gradient, regardless of which micro-batch or rank it landed on. When `normalize_loss=False`, the original per-micro-batch normalization is used (backward compatible).