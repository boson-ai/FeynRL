import torch
import numpy as np

class PPO:
    def __init__(self,
                policy_engine,
                value_engine,
                kl_coeff: float,
                clip_low: float,
                clip_high: float,
                vf_clip: float,
                tau: float,
                gamma: float,
                entropy_coeff: float,
                use_cache: bool,
                ref_model=None,
                ):

        # policy and value engines 
        self.policy_engine = policy_engine
        self.value_engine = value_engine
        self.ref_model = ref_model
        self.use_cache = use_cache

        # policy related parameters
        self.kl_coeff = float(kl_coeff)
        self.clip_low = float(clip_low)
        self.clip_high = float(clip_high)
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.ent_coeff = float(entropy_coeff)

        # value related parameters
        self.vf_clip = float(vf_clip)

    def compute_advantages(self,
                           rewards: torch.Tensor,
                           values: torch.Tensor,
                           done: torch.Tensor,
                           mask: torch.Tensor,
                           last_val: torch.Tensor | None = None,
                          ):
        '''
            rewards, values: [B, T]
            done, mask: [B, T]
            done:    1 if t is EOS (terminal), 0 otherwise.
                     MUST be set at every packed sequence boundary so it
                     shows the boundary of each sequence.
            mask:    1 if valid token, 0 if padding.
            GAE and returns: [B, T]
            last_val: [B]
            return: rets, advs which would be both [B, T]
        '''
        # 1. Device and shape setup
        device = values.device
        dtype  = values.dtype
        B, T   = values.shape
        rets   = torch.zeros_like(values)
        advs   = torch.zeros_like(values)
        last_adv = torch.zeros(B, dtype=dtype, device=device)
        rewards  = rewards.to(dtype=dtype, device=device)

        # 2. Delay casting the mask to the same dtype for indexing and checks.
        mask  = mask.to(device=device)
        done  = done.to(device=device)
        mask  = (mask > 0.5)
        done  = (done > 0.5)

        # 3. Check for nan in rewards or values for valid tokens
        if not torch.isfinite(rewards[mask]).all() or not torch.isfinite(values[mask]).all():
            raise ValueError("rewards or values contain NaN on valid positions")

        if (done & (~mask)).any():
            raise ValueError("done flag set on padding positions")

        # 4. reject holes in padding e.g., [x1, x2, x3, pad, x4, x5] --> this is not supported
        #    we only support [x1, x2, x3, pad, pad, pad...] or [x1, x2, x3, eos, pad,..]
        if (mask[:, 1:] & (~mask[:, :-1])).any():
            raise ValueError("mask has 0->1 transitions (padding in the middle). This is unsupported.")

        # prefill val and rerward for invalid tokens (i.e., padding) as they can contain nan in padded slot
        rewards = rewards.masked_fill(~mask, 0.0)
        values  = values.detach().masked_fill(~mask, 0.0)

        # 5. empty sequences
        if T == 0:
            empty = rewards.new_zeros((B, 0))
            return empty, empty

        # 6. next value
        if last_val is not None:
            next_val = last_val.to(dtype=dtype, device=device).detach().reshape(B)

        else:
            # biased estimation espically whenre there is need for bootstrapping, i.e.,
            # no EOS in generation like [x1,x2,x3]
            next_val = torch.zeros(B, dtype=dtype, device=device)

        # 7. Using (tensor > 0.5) is safer than bool() if inputs are already floats
        # espically in case of BF16/FP16 training.
        mask  = mask.to(dtype=dtype, device=device)
        done  = done.to(dtype=dtype, device=device)

        # 8. Compute returns and advantages
        for t in reversed(range(T)): # [T-1, 0]
            # Done is 1 if EOS/Terminal, we do NOT bootstrap from t+1.
            not_done = 1.0 - done[:, t]
            is_valid = mask[:, t]

            # GAE: A[t] = delta[t] + gamma * tau * A[t+1] * (1 - done[t])
            delta = rewards[:, t] + (self.gamma * next_val * not_done) - values[:, t]
            last_adv   = is_valid * (delta + (self.gamma * self.tau * last_adv * not_done))
            advs[:, t] = last_adv

            # to avoid any leaking from padding.
            next_val = values[:, t] * is_valid

        rets = advs + values

        return rets, advs

    def compute_policy_loss(self,
                            logprobs: torch.Tensor,
                            old_logprobs: torch.Tensor,
                            advantages: torch.Tensor,
                            entropies: torch.Tensor,
                            mask: torch.Tensor,
                            ):
        '''
            logprobs, old_logprobs, advantages, mask: [B, T]
            Compute policy loss:
                1. ratio = exp(logprobs - old_logprobs)
                2. loss = -(min(ratio * adv, clip_adv * adv)) * mask
        '''
        device = logprobs.device
        dtype = logprobs.dtype
        loss_ent = 0.0

        # 1. make sure advantages are detached and convert to float32 for stability under bf16/fp16
        adv = advantages.detach().to(torch.float32)
        mask = (mask.to(device=device) > 0.5).to(dtype=dtype)
        denom = mask.sum().clamp(min=1.0)

        # 2. calculate ratio = exp(logprobs - old_logprobs)
        logratio = (logprobs - old_logprobs).to(torch.float32)
        ratio   = torch.exp(logratio)

        # 3. compute loss: -(min(ratio * adv, clip_adv)) * mask
        unclipped = ratio * adv
        clip_adv  = torch.clamp(ratio, 1.0 - self.clip_low, 1.0 + self.clip_high) * adv
        loss_pi   = -(torch.minimum(unclipped, clip_adv) * mask).sum() / denom

        # 4. compute entropy loss
        if entropies is not None and self.ent_coeff > 0.0:
            loss_ent = -(entropies * mask).sum() / denom

        loss_total = loss_pi - self.ent_coeff * loss_ent

        # 5. useful metrics
        with torch.no_grad():
            # first term too large ==> policy changed too much upward
            # second term too small ==> policy changed too much downward
            clipped_mask = (ratio > (1.0 + self.clip_high)) | (ratio < (1.0 - self.clip_low))
            # fraction of masked tokens that ratio out of ranges
            clipfrac = (clipped_mask.to(dtype=dtype) * mask).sum() / denom

            # approx KL: either E[old_logprobs - logprobs] or E[(ratio - 1) - logratio]
            approx_kl_t = (ratio - 1.0) - logratio
            approx_kl = (approx_kl_t.to(dtype=dtype) * mask).sum() / denom

            # save the metrics for debugging
            metrics = {
                'clipfrac': clipfrac,
                'approx_kl': approx_kl,
                'loss_ent': loss_ent.item(),
                'loss_pi': loss_pi.item(),
                'loss_total': loss_total.item(),
            }

        return loss_total, metrics

    def compute_value_loss(self,
                           values: torch.Tensor,
                           v_old: torch.Tensor,
                           returns: torch.Tensor,
                           mask: torch.Tensor,
                           ):
        '''
            Compute value loss:
                1. if v_old:  loss = 0.5 * (max(values, v_clipped) - rets)^2
                2. otherwise: loss = 0.5 * (values - rets)^2
        '''
        # 1. compute unclipped value loss
        rets = returns.detach()
        v_loss = (values - rets).pow(2)

        # 2. compute clipped value loss
        if  self.vf_clip > 0 and v_old is not None:
            v_old = v_old.detach()

            # 3. compute clipped value loss
            v_clipped = v_old + torch.clamp(values - v_old, -self.vf_clip, self.vf_clip)
            v_loss_clipped = (v_clipped - rets).pow(2)
            vmax =  torch.maximum(v_loss, v_loss_clipped)
            loss = 0.5 * (vmax * mask).sum() / mask.sum()

            # 4. log how much things are changed
            with torch.no_grad():
                vf_clipfrac = (values - v_old).abs() > self.vf_clip
                vf_clipfrac = (vf_clipfrac * mask).sum() / mask.sum()

        else:
            loss = 0.5 * (v_loss * mask).sum() / mask.sum()
            vf_clipfrac = 0.0

        # save the metrics for debugging
        metrics = {
            'vf_clipfrac': vf_clipfrac,
        }

        return loss, metrics

    def compute_entropy_loss(self,
                            entropies: torch.Tensor,
                            mask: torch.Tensor,
                            ):
        '''
            Compute entropy loss
        '''
        if entropies is None or self.ent_coeff == 0.0:
            return 0.0, {}

        else:
            loss = - (entropies * mask).sum() / mask.sum()
            return loss, {'ent_loss': loss}

    def policy_forward(self, batch):
        '''
            batch['seq_ids/seq_attn_mask'] are [B, T]
            batch['position_ids'] is [B, T] or None
            Returns:
                logits is [B, T-1, vocab_size]
                y is [B, T-1]
        '''
        # input_ids and att_mask are [B, T]
        input_ids = batch['seq_ids']
        att_mask  = batch['seq_attn_mask']

        # if pos_ids is not provided, HF will add that automatically.
        pos_ids   = batch.get('position_ids', None)
        if pos_ids is not None:
            pos_ids = pos_ids.to(self.att_mask.device)

        # feed data to model
        output = self.model_engine(input_ids=input_ids,
                                   attention_mask=att_mask,
                                   position_ids=pos_ids,
                                   use_cache=self.use_cache)

        # [B, T, vocab_size]
        every_token_logits = output.logits

        # label would be input_ids shifted by one (input_ids[:, 1:])
        # so the size is [B, T-1]
        y = input_ids[:, 1:].contiguous()
        # it is next token prediction, so we remove last token from logits
        logits = every_token_logits[:, :-1, :].contiguous()

        return logits, y

    def value_forward(self, batch):
        '''
            batch['seq_ids/seq_attn_mask'] are [B, T]
            batch['position_ids'] is [B, T] or None
            Returns:
                logits is [B, T-1, vocab_size]
                y is [B, T-1]
                loss_mask is [B, T-1]
        '''
        # input_ids and att_mask are [B, T]
        input_ids = batch['seq_ids']
        att_mask  = batch['seq_attn_mask']

        # if pos_ids is not provided, HF will add that automatically.
        pos_ids   = batch.get('position_ids', None)
        if pos_ids is not None:
            pos_ids = pos_ids.to(self.att_mask.device)

        # feed data to model
        output = self.model_engine(input_ids=input_ids,
                                   attention_mask=att_mask,
                                   position_ids=pos_ids,
                                   use_cache=self.use_cache)

        # [B, T, 1]
        every_token_values = output.logits

        # label would be input_ids shifted by one (input_ids[:, 1:])
        # so the size is [B, T-1]
        y = input_ids[:, 1:].contiguous()
        # it is next token prediction, so we remove last token from logits
        values = every_token_values[:, :-1, :].contiguous()
        last_value = every_token_values[:, -1, :].contiguous()

        return values, y, last_value

    def train_step(self, replay_buffer):
        '''
           This function implements a training step per rank/gpu for full replay buffer.
           The batch size for each gpu/rank should be micro_batch_size_per_gpu.
        '''
        device = self.policy_engine.device

        # 1. put model_engines in training mode
        self.policy_engine.train()
        self.value_engine.train()

        # 2. zero grads
        self.policy_engine.zero_grad()
        self.value_engine.zero_grad()

        # 3. create progress bar
        len_replay_buffer = len(replay_buffer)
        inv_num_micro = 1.0 / max(num_micro, 1)
        progress_bar = tqdm(replay_buffer, total=num_micro)

        for step, micro_batch in enumerate(progress_bar):
            is_last = (step == (num_micro - 1))
            ########
            # 1. Get the data
            ########

            # curr_data contains tokenized seq(prompt + answer), attn_mask, etc.
            curr_data   = micro_batch['data'].to(device, non_blocking=True)
            loss_mask   = micro_batch['mask'].to(device, non_blocking=True)

            ########
            # 2. Forward pass for policy and compute policy loss
            ########
            old_logprobs = micro_batch['old_logprobs'].to(device, non_blocking=True)
            advantages   = micro_batch["advantages"].to(device, non_blocking=True)
            if entropies is not None:
                entropies = entropies.to(device, non_blocking=True)

            pi_logits, pi_y     = self.policy_forward(batch=curr_data)
            pl_loss, pl_metrics = self.compute_policy_loss(logprobs=pi_logits,
                                                           old_logprobs=old_logprobs,
                                                           advantages=advantages,
                                                           mask=loss_mask)

            ########
            # 3. forward pass for value and compute value loss
            ########
            values       = micro_batch["values"].to(device, non_blocking=True)
            v_old        = micro_batch.get("v_old", None)
            if v_old is not None:
                v_old = v_old.to(device, non_blocking=True)

            v_logits, v_y, _ = self.value_forward(batch=curr_data)
            vl_loss, vl_metrics = self.compute_value_loss(values=values,
                                                          v_old=v_old,
                                                          returns=returns,
                                                          mask=v_loss_mask)

            # Mark accumulation boundary (ONLY last micro-batch updates)
            self.policy_engine.set_gradient_accumulation_boundary(is_last)
            self.value_engine.set_gradient_accumulation_boundary(is_last)

            # backward pass
            self.policy_engine.backward(pl_loss)
            self.value_engine.backward(vl_loss)

            # DeepSpeed expects step() each micro-batch
            # update happens only if boundary=True
            self.policy_engine.step()
            self.value_engine.step()