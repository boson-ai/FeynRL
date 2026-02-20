### PPO

 Our PPO training step (`train_step`) runs on a replay shard (a list of `micro_batches`) and uses DeepSpeed for micro-batching and gradient accumulation. Before any policy/value updates, we call `precompute_gae(micro_batches)`, which runs the value model in `eval()` mode and computes returns and advantages via `compute_advantages(...)` using `gamma` and `tau` (GAE). Then, for each `micro_batch`, we update the policy with a PPO clipped objective using stored `old_logprobs` and `mask`, plus optional entropy regularization (`ent_coeff`) and optional KL-to-reference penalty (`kl_coeff` if a reference model exists). We also update the value model by regressing `values` to the precomputed `returns` with a masked MSE. If `update_only_after_full_replay=True`, we take one optimizer step at the end of the replay shard (and scale losses by `ga_steps/num_micro` to keep gradient magnitude consistent.

**Input:** initial policy parameters $\theta_0$, initial value parameters $\phi_0$, replay shards $\mathcal{B}$ (`micro_batches`)

**Hyperparams:** discount $\gamma$, GAE $\tau$, clip range $(1-\epsilon_\ell,\ 1+\epsilon_h)$, entropy weight $\beta_{\mathrm{ent}}$, KL weight $\beta_{\mathrm{kl}}$

For each training step:

1. **Precompute GAE:** for all micro-batches, compute returns $R$ and advantages $A$ using $\gamma,\tau$.

2. For each micro-batch $\mathcal{B}$:

   - Forward policy: $(\log \pi_{\theta},\, H_{\theta}) \leftarrow \pi_{\theta}(\mathcal{B})$
   - PPO ratio: $\rho \leftarrow \exp(\log \pi_{\theta} - \texttt{old\\_logprobs})$

   - Clipped policy loss (masked mean):

$$
\mathcal{L}_{\pi}
\leftarrow
-\mathrm{Mean}_{\texttt{mask}}\Big(
\min\big(
\rho A,\ \mathrm{clip}(\rho,1-\epsilon_\ell,1+\epsilon_h)\,A
\big)
\Big)
$$

   - (Optional) entropy term:

$$
\mathcal{L}_{\mathrm{ent}} \leftarrow \mathrm{Mean}_{\texttt{mask}}(H_{\theta})
$$

   - (Optional) KL penalty (vs. a reference policy $\pi_{\mathrm{ref}}$):

$$
\mathcal{L}_{\mathrm{kl}} \leftarrow \mathrm{Mean}_{\texttt{mask}}\big(\mathrm{KL}(\pi_{\theta}\,\|\,\pi_{\mathrm{ref}})\big)
$$

   - Policy loss:

$$
\mathcal{L}
\leftarrow
\mathcal{L}_{\pi}
-\beta_{\mathrm{ent}}\mathcal{L}_{\mathrm{ent}}
+\beta_{\mathrm{kl}}\mathcal{L}_{\mathrm{kl}}
$$

   - DeepSpeed backward/step for policy (grad accumulation; optionally one step at shard end)

   - Forward value: $V_{\phi} \leftarrow V_{\phi}(\mathcal{B})$

   - Value loss (masked MSE):

$$
\mathcal{L}_{V}
\leftarrow
\frac{1}{2}\,\mathrm{Mean}_{\texttt{mask}}\big((V_{\phi}-R)^2\big)
$$

   - DeepSpeed backward/step for value (same boundary)

**Return:** $\theta,\phi$