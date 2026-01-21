import torch
import numpy as np
from tqdm import tqdm
import ray
import deepspeed
from transformers import AutoModelForCausalLM, AutoConfig

@ray.remote
class PPO:
    def __init__(self,
                 policy_model_path: str,
                 policy_model_dtype: torch.dtype,
                 value_model_path:str,
                 value_model_dtype: torch.dtype,
                 trust_remote_code: bool,
                 attn_impl: str,
                 kl_coeff: float,
                 clip_low: float,
                 clip_high: float,
                 entropy_coeff: float,
                 vf_clip: float,
                 tau: float,
                 gamma: float,
                 use_cache: bool,
                 micro_batch_size_per_gpu: int,
                 update_after_full_replay: bool,
                 deepspeed_config: deepspeed.DeepSpeedConfig,
                 ref_model_path: str = None,
                 deepspeed_ref_config = None,
                 ):

        self.alg_name = self.__class__.__name__
        # model related parameters
        self.pi_model_path = policy_model_path
        self.pi_dtype = policy_model_dtype
        self.vf_model_path = value_model_path
        self.vf_dtype = value_model_dtype
        self.ref_model_path = ref_model_path
        self.use_cache = use_cache
        self.attn_impl = attn_impl
        self.trust_remote_code = trust_remote_code

        # training related parameters
        self.deepspeed_config = deepspeed_config
        self.deepspeed_ref_config = deepspeed_ref_config
        self.micro_batch_size_per_gpu = micro_batch_size_per_gpu

        # policy related parameters
        self.kl_coeff = float(kl_coeff)
        self.clip_low = float(clip_low)
        self.clip_high = float(clip_high)
        self.ent_coeff = float(entropy_coeff)

        # value model params
        self.vf_clip = float(vf_clip)

        # use cross entropy loss for policy gradient
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

        # if true, it means the update is done after seeing all samples in the reply buffer
        # treating the entire buffer as a single batch.
        self.update_only_after_full_replay = update_after_full_replay

        self.ready = False
        self.init_training_engine()
        self.ready = True

    def is_ready(self):
        '''
            Barrier method to ensure all Ray actors are initialized before DeepSpeed collective ops.
        '''
        return self.ready

    def init_training_engine(self):
        '''
            Since, we are using ray, each ray actor MUST create its own deepspeed engine.
            This is because each ray actor process is a separate process as it should be 1 actor = 1 gpu = 1 ds rank.
        '''
        # Convert pydantic model to python Dict for DeepSpeed
        ds_config_dict = self.deepspeed_config.model_dump()

        # check to avoid re-initializing distributed backend
        if not torch.distributed.is_initialized():
            # 1. Initialize distributed training engine
            deepspeed.init_distributed()

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print(f"[Alg:{self.alg_name}][Rank {rank}] Initializing training engine...")

        # 2. Load model
        pi_model, ref_model, val_model = self.load_model()
        print(f"[Alg:{self.alg_name}][Rank {rank}] Models loaded: {self.pi_model_path} {self.vf_model_path} {self.ref_model_path}")

        # 2. Initialize model engine
        self.policy_engine, self.optimizer, _, _ = deepspeed.initialize(
                                                            model=pi_model,
                                                            model_parameters=pi_model.parameters(),
                                                            config=ds_config_dict
                                                            )
        print(f"[Alg:{self.alg_name}][Rank {rank}] DeepSpeed engine initialized on device: {self.policy_engine.device}")

        self.ref_model_engine = None
        if ref_model is not None:
            ref_model.eval()
            # Use inference-only config (no optimizer needed)
            ref_ds_config = self.deepspeed_ref_config.model_dump()
            self.ref_model_engine, _, _, _ = deepspeed.initialize(
                                                            model=ref_model,
                                                            config=ref_ds_config
                                                            )
            print(f"[Alg:{self.alg_name}][Rank {rank}] Reference model initialized with DeepSpeed")

        # 3. Initialize model engine
        self.value_engine, self.optimizer, _, _ = deepspeed.initialize(
                                                                    model=val_model,
                                                                    model_parameters=val_model.parameters(),
                                                                    config=ds_config_dict
                                                                    )
        print(f"[Alg:{self.alg_name}][Rank {rank}] Value model initialized with DeepSpeed")

    def load_model(self):
        '''
            Load models and tokenizer from huggingface.
        '''
        assert self.model_dtype != 'auto', "dtype must not be auto to avoid any precision issues"
        assert self.attn_impl=='' or self.attn_impl in ['eager', 'flash_attention_2'], "attn_impl must be one of 'eager', 'flash_attention_2' or empty string"

        # 1. model and its config initialization
        policy_config = AutoConfig.from_pretrained(self.model_path)
        policy_model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                    dtype=self.model_dtype,
                                                    trust_remote_code=self.trust_remote_code,
                                                    config=policy_config,
                                                    attn_implementation=None if self.attn_impl == '' else self.attn_impl)

        # if ref model is provided to use it in kl for example.
        if self.ref_model_path and self.kl_coeff > 0.0:
            ref_config = AutoConfig.from_pretrained(self.ref_model_path)
            ref_model = AutoModelForCausalLM.from_pretrained(self.ref_model_path,
                                                            dtype=self.model_dtype,
                                                            trust_remote_code=self.trust_remote_code,
                                                            config=ref_config,
                                                            attn_implementation=None if self.attn_impl == '' else self.attn_impl)
        else:
            ref_model = None


        # load value network
        val_config = AutoConfig.from_pretrained(self.vf_model_path)
        value_model = AutoModelForCausalLM.from_pretrained(self.vf_model_path,
                                                    dtype=self.val_dtype,
                                                    trust_remote_code=self.trust_remote_code,
                                                    config=val_config,
                                                    attn_implementation=None if self.attn_impl == '' else self.attn_impl)


        return policy_model, ref_model, value_model

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

    def ref_forward(self, input_ids, att_mask, target_ids, pos_ids):
        '''
            input_ids and att_mask are [B, T]
            pos_ids is [B, T] or None
            target_ids is [B, T-1]
            Returns:
                logits is [B, T-1, vocab_size]
        '''
        # feed data to model
        with torch.no_grad():
            if pos_ids is not None:
                pos_ids = pos_ids.to(input_ids.device)

            output = self.ref_model_engine(input_ids=input_ids,
                                           attention_mask=att_mask,
                                           position_ids=pos_ids,
                                           use_cache=self.use_cache)

            # [B, T, V] -> [B, T-1, V]
            logits = output.logits[:, :-1, :].contiguous()
            B, T_minus_1, vocab_size = logits.shape

            # cross_entropy return -logprobs but we need logprobs
            # logits is [B, T-1, vocab_size] --> [B * (T-1), vocab_size]
            # target_ids is [B, T-1] --> [B * (T-1)]
            neg_logprobs = self.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1))
            ref_logprobs = -neg_logprobs.view(B, T_minus_1)

        return ref_logprobs

    def policy_forward(self, input_ids, att_mask, pos_ids):
        '''
            input_ids and att_mask are [B, T]
            pos_ids is [B, T] or None
            Returns:
                logits is [B, T-1, vocab_size]
                entropies is [B, T-1]
        '''
        # if pos_ids is not provided, HF will add that automatically.
        if pos_ids is not None:
            pos_ids = pos_ids.to(input_ids.device)

        # feed data to model
        output = self.policy_engine(input_ids=input_ids,
                                   attention_mask=att_mask,
                                   position_ids=pos_ids,
                                   use_cache=self.use_cache)

        # [B, T, V] -> [B, T-1, V]
        logits = output.logits[:, :-1, :].contiguous()
        B, T_minus_1, vocab_size = logits.shape

        # [B, T] -> [B, T-1]
        target_ids = input_ids[:, 1:].contiguous()

        # cross_entropy return -logprobs but we need logprobs
        # logits is [B, T-1, vocab_size] --> [B * (T-1), vocab_size]
        # target_ids is [B, T-1] --> [B * (T-1)]
        neg_logprobs = self.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1))
        logprobs = -neg_logprobs.view(B, T_minus_1)
        # we can also do this, but it is less efficient I guess
        #   logprobs = logits.log_softmax(dim=-1)
        #   logprobs = torch.gather(logprobs, dim=-1, index=target_ids)

        entropies = None
        if self.ent_coeff > 0.0:
            entropies = torch.distributions.Categorical(logits=logits).entropy()

        return logprobs, entropies, target_ids

    def compute_kl_distance(self, logprobs, ref_logprobs):
        '''
            Compute KL divergence between two policies.
            using var_reduced form:
            kl = E[log pi/pi_ref] + pi_ref/pi - 1
        '''
        # [B, T-1]
        log_ratio = logprobs - ref_logprobs
        # pi_ref/pi = exp(ref_logprobs - logprobs)
        ratio_inv = torch.exp(ref_logprobs - logprobs)
        kl_dist = log_ratio + ratio_inv - 1
        return kl_dist

    def compute_policy_loss(self,
                            logprobs: torch.Tensor,
                            old_logprobs: torch.Tensor,
                            advantages: torch.Tensor,
                            mask: torch.Tensor,
                            entropies: torch.Tensor,
                            ref_logprobs: torch.Tensor,
                            ):
        '''
            logprobs: [B, T-1]
            old_logprobs, advantages, mask: [B, T - 1]
            entropies: [B, T-1]
            ref_logprobs: [B, T-1]
            Compute policy loss:
                1. ratio = exp(logprobs - old_logprobs)
                2. loss = -(min(ratio * adv, clip_adv * adv)) * mask
        '''
        device = logprobs.device
        dtype = logprobs.dtype
        loss_ent = torch.tensor(0.0, device=device, dtype=dtype)
        kl_ref   = torch.tensor(0.0, device=device, dtype=dtype)

        # 1. make sure advantages are detached and
        # convert to float32 for stability under bf16/fp16
        adv = advantages.detach().to(torch.float32)
        mask = (mask.to(device=device) > 0.5).to(dtype=dtype)
        denom = mask.sum().clamp(min=1.0)

        # 2. calculate ratio = pi / pi_old = exp(logprobs - old_logprobs)
        logratio = (logprobs - old_logprobs).to(torch.float32)
        ratio   = torch.exp(logratio)

        # 3. compute loss: -(min(ratio * adv, clip_adv)) * mask
        unclipped = ratio * adv
        clip_adv  = torch.clamp(ratio, 1.0 - self.clip_low, 1.0 + self.clip_high) * adv
        loss_pi   = -(torch.minimum(unclipped, clip_adv) * mask).sum() / denom

        # 4. compute entropy loss
        if entropies is not None and self.ent_coeff > 0.0:
            loss_ent = (entropies * mask).sum() / denom

        if ref_logprobs is not None and self.kl_coeff > 0.0:
            kl_dist = self.compute_kl_distance(logprobs=logprobs, ref_logprobs=ref_logprobs)
            kl_ref  = (kl_dist * mask).sum() / denom

        loss_total = loss_pi - self.ent_coeff * loss_ent + self.kl_coeff * kl_ref

        # 5. useful metrics
        with torch.no_grad():
            # first term too large ==> policy changed too much upward
            # second term too small ==> policy changed too much downward
            clipped_mask = (ratio > (1.0 + self.clip_high)) | (ratio < (1.0 - self.clip_low))
            # fraction of masked tokens that ratio out of ranges
            clipfrac = (clipped_mask.to(dtype=dtype) * mask).sum() / denom

            # approx KL (var-reduced): log(pi/pi_old) + pi_old/pi - 1
            # logratio = log(pi/pi_old)
            ratio_inv = torch.exp(-logratio)
            approx_kl_t = logratio + ratio_inv - 1.0
            approx_kl = (approx_kl_t.to(dtype=dtype) * mask).sum() / denom

            # save the metrics for debugging
            metrics = {
                'clipfrac': clipfrac.item(),
                'kl_old': approx_kl.item(),
                'loss_ent': loss_ent.item(),
                'loss_pi': loss_pi.item(),
                'loss_total': loss_total.item(),
                'kl_ref': kl_ref.item(),
            }

        return loss_total, metrics

    def value_forward(self, input_ids, att_mask, pos_ids):
        '''
            Input:
                input_ids/att_mask: [B, T]
                pos_ids: [B, T] or None
            Returns:
                values: [B, T-1]
                last_value: [B, 1] value of the very last token
        '''
        if pos_ids is not None:
            pos_ids = pos_ids.to(input_ids.device)

        # feed data to model
        output = self.value_engine(input_ids=input_ids,
                                   attention_mask=att_mask,
                                   position_ids=pos_ids,
                                   use_cache=self.use_cache)

        # [B, T, 1] -> [B, T]
        logits = output.logits.squeeze(-1).contiguous()

        # [B, T] -> [B, T-1]
        values = logits[:, :-1].contiguous()
        # Value for terminal state (e.g., t=T-1) for bootstrapping if not EOS
        # [B, T] -> [B]
        last_value = logits[:, -1].contiguous()

        return values, last_value

    def compute_value_loss(self,
                           values: torch.Tensor,
                           v_old: torch.Tensor,
                           returns: torch.Tensor,
                           mask: torch.Tensor,
                           ):
        '''
            values/v_old/returns/mask: [B, T-1]
            Compute value loss:
                1. if v_old:  loss = 0.5 * (max(values, v_clipped) - rets)^2
                2. otherwise: loss = 0.5 * (values - rets)^2
        '''
        device = values.device
        dtype  = values.dtype

        # 1. compute unclipped value loss
        rets   = returns.detach()
        v_loss = (values - rets).pow(2)
        mask   = (mask.to(device=device) > 0.5).to(dtype=dtype)
        denom  = mask.sum().clamp(min=1.0)

        # 2. compute clipped value loss
        if  self.vf_clip > 0 and v_old is not None:
            v_old = v_old.detach()

            # 3. compute clipped value loss
            v_clipped = v_old + torch.clamp(values - v_old, -self.vf_clip, self.vf_clip)
            v_loss_clipped = (v_clipped - rets).pow(2)
            vmax =  torch.maximum(v_loss, v_loss_clipped)
            loss = 0.5 * (vmax * mask).sum() / denom

            # 4. log how much things are changed
            with torch.no_grad():
                vf_clipfrac = (values - v_old).abs() > self.vf_clip
                vf_clipfrac = (vf_clipfrac * mask).sum() / denom

        else:
            loss = 0.5 * (v_loss * mask).sum() / denom
            vf_clipfrac = 0.0

        # save the metrics for debugging
        metrics = {
            'vf_clipfrac': vf_clipfrac,
            'loss_v': loss.item(),
        }

        return loss, metrics

    def train_step(self, engine_id, micro_batches):
        '''
           This function implements a training step per rank/gpu for local_batch.
           The batch size for each gpu/rank should be micro_batch_size_per_gpu.
           micro_batches is a partition of the replay buffer (list of micro-batches) for the current rank/gpu.
        '''
        assert self.policy_engine is not None, "DeepSpeed policy_engine not initialized"
        assert self.value_engine  is not None, "DeepSpeed value_engine not initialized"

        device = self.policy_engine.device

        # 1. Models to train mode
        self.policy_engine.train()
        self.value_engine.train()

        # 2. zero grads
        self.policy_engine.zero_grad()
        self.value_engine.zero_grad()

        # 3. create progress bar
        num_micro = len(micro_batches)
        # torch.distributed.get_rank() would be the same thing as engine_id
        if engine_id == 0:
            progress_bar = tqdm(micro_batches, total=num_micro, desc="[Alg:{}] Training Step in rank {}".format(self.alg_name, engine_id))

        else:
            progress_bar = micro_batches # No tqdm for other ranks

        ga_pi_attr = getattr(self.policy_engine, 'gradient_accumulation_steps', 1)
        ga_pi = int(ga_pi_attr() if callable(ga_pi_attr) else ga_pi_attr)

        # track metrics across all micro-batches
        all_metrics = []
        for step, micro_batch in enumerate(progress_bar):
            is_last = (step == (num_micro - 1))
            is_boundary = (((step + 1) % ga_pi) == 0) or is_last

            ########
            # 1. Data from buffer
            ########
            # all are [B, T]
            # zscore is normalized rewards using the number of samples for each proompt (X -mu) / (std + eps)
            # this is a simple baseline for policy gradients (PPO in this code) as it reflects relative quality
            # among that prompt’s samples.
            rewards   = micro_batch['rewards'][:, :-1].to(device, non_blocking=True)
            done      = micro_batch['done'][:, :-1].to(device, non_blocking=True)
            mask      = micro_batch['mask'][:, :-1].to(device, non_blocking=True)
            old_logprobs = micro_batch['old_logprobs'][:, :-1].to(device, non_blocking=True)

            # PPO-specific: values and last_val from vLLM rollout
            values    = micro_batch.get('v_olds', None)
            if values is not None:
                values = values[:, :-1].to(device, non_blocking=True)
            last_val  = micro_batch.get('last_val', None)
            if last_val is not None:
                last_val = last_val[:, :-1].to(device, non_blocking=True)

            input_ids = micro_batch['input_ids'].to(device, non_blocking=True)
            att_mask  = micro_batch['attn_mask'].to(device, non_blocking=True)
            pos_ids   = micro_batch.get('position_ids', None)

            ########
            # 2. Compute adavnatage
            ########
            returns, advs = self.compute_advantages(
                                                    rewards=rewards,
                                                    values=values,
                                                    done=done,
                                                    mask=mask,
                                                    last_val=last_val)

            ########
            # 2. Compute policy loss
            ########
            # Forward pass through the policy.
            pi_logprobs, pi_entropies, target_ids = self.policy_forward(input_ids=input_ids,
                                                                        att_mask=att_mask,
                                                                        pos_ids=pos_ids)

            ref_logprobs = None
            if self.kl_coeff > 0.0 and self.ref_model_engine is not None:
                ref_logprobs = self.ref_forward(input_ids=input_ids,
                                                att_mask=att_mask,
                                                target_ids=target_ids,
                                                pos_ids=pos_ids,
                                                )

            # Compute policy loss using the current policy.
            pi_loss, pi_metrics = self.compute_policy_loss(logprobs=pi_logprobs,
                                                           old_logprobs=old_logprobs,
                                                           advantages=advs,
                                                           mask=mask,
                                                           entropies=pi_entropies,
                                                           ref_logprobs=ref_logprobs)

            # store metrics
            all_metrics.append(pi_metrics)
            if engine_id == 0:
                progress_bar.set_postfix({
                    "loss": f"{pi_loss.item():.4f}",
                    "clip": f"{pi_metrics['clipfrac']:.3f}",
                    "kl_old": f"{pi_metrics['kl_old']:.4f}",
                    "kl_ref": f"{pi_metrics['kl_ref']:.4f}"
                })

            if self.update_only_after_full_replay:
                # Accumulate gradients across all micro-batches, only update at the end
                self.policy_engine.set_gradient_accumulation_boundary(is_boundary)
            else:
                # Update after every micro-batch (treat each as a boundary)
                self.policy_engine.set_gradient_accumulation_boundary(True)

            # backward pass
            self.policy_engine.backward(pi_loss)
            self.policy_engine.step()

            ########
            # 3. Compute value loss
            ########
            # Forward pass through the value function.
            values = self.value_forward(input_ids=input_ids,
                                        att_mask=att_mask,
                                        pos_ids=pos_ids)

            # Compute value loss
            v_loss, v_metrics = self.compute_value_loss(values=values,
                                                        returns=returns,
                                                        mask=mask,
                                                        is_boundary=is_boundary,
                                                        device=device)

            # store metrics
            all_metrics.append(v_metrics)
            if engine_id == 0:
                progress_bar.set_postfix({
                    "loss": f"{v_loss.item():.4f}",
                    "clip": f"{v_metrics['clipfrac']:.3f}",
                    "kl_old": f"{v_metrics['kl_old']:.4f}",
                    "kl_ref": f"{v_metrics['kl_ref']:.4f}"
                })

            if self.update_only_after_full_replay:
                # Accumulate gradients across all micro-batches, only update at the end
                self.value_engine.set_gradient_accumulation_boundary(is_boundary)
            else:
                # Update after every micro-batch (treat each as a boundary)
                self.value_engine.set_gradient_accumulation_boundary(True)

            # backward pass
            self.value_engine.backward(v_loss)
            self.value_engine.step()

        # aggregate metrics across all micro-batches
        aggregated_metrics_policy = {}
        if all_metrics_policy:
            for key in all_metrics_policy[0].keys():
                aggregated_metrics_policy[key] = np.mean([m[key] for m in all_metrics_policy])

        aggregated_metrics_value = {}
        if all_metrics_value:
            for key in all_metrics_value[0].keys():
                aggregated_metrics_value[key] = np.mean([m[key] for m in all_metrics_value])

        return aggregated_metrics_policy, aggregated_metrics_value