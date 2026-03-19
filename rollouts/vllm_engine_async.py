import os
import sys
import torch
import gc
import ray
from vllm import SamplingParams
from typing import Optional, List, Callable, Any, Dict
import numpy as np
import pickle
import asyncio
import threading
# local imports
from misc.utils import set_random_seeds
from misc.metrics import compute_pass_metrics

@ray.remote
class VLLMRolloutEngineAsync:
    def __init__(self,
                 seed: int,
                 model_path: str,
                 trust_remote_code: bool,
                 temperature: float,
                 max_tokens: int,
                 n_samples: int,
                 top_p: float,
                 top_k: int,
                 ignore_eos: bool,
                 stop: Optional[List[str]],
                 stop_token_ids: Optional[List[int]],
                 prompt_logprobs: bool,
                 force_strict_on_policy: bool,
                 reward_func: Callable,
                 tensor_parallel_size: int,
                 eos_id: int,
                 reward_broadcast: bool,
                 eps_reward_norm: float,
                 gpu_memory_utilization: float,
                 model_dtype: str,
                 max_seq_len: int,
                 max_model_len: int | None = None,
                 engine_id: int = 0,
                 batch_invariant: bool = False,
                 ):
        # This can reduce throughput depending on model size and batch composition
        # because it forces batch-invariant kernels.
        # https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/reproducibility.py
        if batch_invariant:
            os.environ["VLLM_BATCH_INVARIANT"] = "1"

        # Seed the rollout actor's Python/NumPy/PyTorch RNGs so any
        # non-vLLM operations (reward computation, normalization) are deterministic.
        set_random_seeds(seed + engine_id)

        # Ensure current working directory is in sys.path for this actor
        # and spawned vllm workers. This is required the model so
        # worker_extension_cls resolves to local source.
        if os.getcwd() not in sys.path:
            sys.path.append(os.getcwd())

        # reward function
        self.reward_func = reward_func
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.eos_id = eos_id

        # sampling config
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.n_samples = int(n_samples)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.seed = seed
        self.ignore_eos = bool(ignore_eos)
        self.stop = stop if stop else None
        self.stop_token_ids = stop_token_ids if stop_token_ids else None
        self.prompt_logprobs = prompt_logprobs
        self.force_strict_on_policy = bool(force_strict_on_policy)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.engine_id = int(engine_id)
        self.batch_invariant = bool(batch_invariant)
        # prompt + response max length also known as context window size
        self.max_seq_len = int(max_seq_len)
        self.max_model_len = int(max_model_len) if max_model_len is not None else None

        # vllm engine config
        self.model_path = model_path
        self.model_dtype = model_dtype
        self.loaded_version = -1
        self.trust_remote_code = trust_remote_code

        # reward normalization
        self.eps_reward_norm = float(eps_reward_norm)
        # If True, broadcast a single scalar reward across all tokens in the sequence.
        self.reward_broadcast = bool(reward_broadcast)

        # Async engine requires its own event loop running in a background thread.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

        self.async_engine = None
        self._request_counter = 0
        self.load_async_engine()
        self.loaded_version = 0
        self.sampling_params = self.make_sampling_params()

    def log(self, msg: str) -> None:
        '''
            Log message only if this is the first engine to avoid clutter.
        '''
        if self.engine_id == 0:
            print(f"[VLLMEngineAsync][Rank {self.engine_id}] {msg}")

    def load_async_engine(self) -> None:
        '''
            Create the AsyncLLMEngine. Handles vllm API differences:
        '''
        # newer versions may use AsyncLLM instead of AsyncLLMEngine.
        if self.async_engine is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            except Exception:
                pass

            try:
                del self.async_engine
            except Exception as e:
                print(f"Error deleting async_engine: {e}")

            self.async_engine = None
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        engine_kwargs = dict(model=self.model_path,
                             trust_remote_code=self.trust_remote_code,
                             tensor_parallel_size=self.tensor_parallel_size,
                             gpu_memory_utilization=self.gpu_memory_utilization,
                             dtype=self.model_dtype,
                             seed=self.seed,
                             worker_extension_cls="rollouts.weight_sync.WeightSyncExtension",
                             disable_log_stats=True,
                             )
        if self.max_model_len is not None:
            engine_kwargs["max_model_len"] = self.max_model_len

        if self.batch_invariant:
            # vLLM batch invariance requires FlashAttention backend.
            # We still keep a compatibility fallback below for older vLLM versions.
            engine_kwargs["attention_backend"] = "FLASH_ATTN"

        # Try AsyncLLMEngine (vLLM >= 0.7), fall back to AsyncLLM for newer versions.
        # Wrap in create_async_engine so the attention_backend fallback
        # can catch TypeError from either path (AsyncLLMEngine or AsyncLLM).
        self.create_async_engine(engine_kwargs)

    def create_async_engine(self, engine_kwargs):
        try:
            try:
                from vllm import AsyncLLMEngine
                from vllm.engine.arg_utils import AsyncEngineArgs
                args = AsyncEngineArgs(**engine_kwargs)
                self.async_engine = AsyncLLMEngine.from_engine_args(args)
                self.log(f"Loaded AsyncLLMEngine from {self.model_path}")

            except ImportError:
                from vllm import AsyncLLM
                self.async_engine = AsyncLLM(**engine_kwargs)
                self.log(f"Loaded AsyncLLM from {self.model_path}")

        except TypeError as e:
            # Compatibility fallback: some vLLM versions do not expose
            # the attention_backend argument.
            if self.batch_invariant and "attention_backend" in str(e):
                engine_kwargs.pop("attention_backend", None)
                self.log("attention_backend unsupported in this vLLM version; retrying without it.")
                try:
                    from vllm import AsyncLLMEngine
                    from vllm.engine.arg_utils import AsyncEngineArgs
                    args = AsyncEngineArgs(**engine_kwargs)
                    self.async_engine = AsyncLLMEngine.from_engine_args(args)
                    self.log(f"Loaded AsyncLLMEngine from {self.model_path} (fallback mode)")
                
                except ImportError:
                    from vllm import AsyncLLM
                    self.async_engine = AsyncLLM(**engine_kwargs)
                    self.log(f"Loaded AsyncLLM from {self.model_path} (fallback mode)")

            else:
                raise

    def run_async(self, coro):
        '''
            Run an async coroutine on the background event loop and wait for result.
            Bridges the sync Ray actor method with async vllm calls.
        '''
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def make_sampling_params(self) -> SamplingParams:
        '''
            Create sampling params for generation.
        '''
        if self.force_strict_on_policy:
            if self.temperature != 1.0:
                raise ValueError("Strict on-policy requires temperature = 1.0 (no scaling).")

            if self.top_p != 1.0:
                raise ValueError("Strict on-policy requires top_p = 1.0 (no nucleus truncation).")

            if self.top_k != -1:
                raise ValueError("Strict on-policy requires top_k = -1 (no top-k truncation).")

            if self.n_samples < 1:
                raise ValueError("Strict on-policy requires n_samples >= 1.")

            # vllm can return empty responses for max_tokens <= 0 which will break the rest of the code.
            if self.max_tokens <= 0:
                raise ValueError("max_tokens must be > 0.")

            if self.stop is not None or self.stop_token_ids is not None or self.ignore_eos:
                raise ValueError(
                    "Strict on-policy requires stop=None, stop_token_ids=None, ignore_eos=False "
                    "(these change the trajectory distribution)."
                )
        # When batch_invariant=True, all engines use the same seed so the same
        # prompt always produces the same output regardless of which engine or
        # batch it lands in (topology-invariant).  Sharding in shard_batch_for_engines
        # ensures each prompt goes to exactly one engine, so duplicates are impossible.
        seed_base = self.seed if self.batch_invariant else (self.seed + self.engine_id * 1000)
        return SamplingParams(seed=seed_base,
                              n=self.n_samples,
                              
                              temperature=self.temperature,
                              top_p=self.top_p,
                              top_k=self.top_k,
                              min_p=0.0,
                              
                              max_tokens=self.max_tokens,
                              stop=self.stop,
                              stop_token_ids=self.stop_token_ids,
                              ignore_eos=self.ignore_eos,

                              # penalties
                              presence_penalty=0.0,
                              frequency_penalty=0.0,
                              repetition_penalty=1.0,
                              logit_bias=None,
                              allowed_token_ids=None,
                              bad_words=None,
                              logits_processors=None,

                              # setup to returns required info
                              logprobs=1,
                              prompt_logprobs=(1 if self.prompt_logprobs else None),)

    def extract_logprobs(self, response_ids: List[int], logprobs_by_pos: Any) -> torch.Tensor:
        '''
           Extract logprobs for each token in response_ids from logprobs.
           logprobs_by_pos: list of dict {token_id -> logprob_info}
        '''
        if logprobs_by_pos is None:
            raise ValueError("logprobs_by_pos must not be None.")

        if not isinstance(logprobs_by_pos, list):
            raise TypeError(f"logprobs_by_pos must be a list, got {type(logprobs_by_pos)}")

        if len(response_ids) != len(logprobs_by_pos):
            raise ValueError(f"logprobs_by_pos must have the same len as response_ids. Got {len(logprobs_by_pos)} vs {len(response_ids)}.")

        token_logprobs = []
        for t_id, lgp_dict in zip(response_ids, logprobs_by_pos):
            if lgp_dict is None:
                raise ValueError(f"No logprobs for token {t_id} in {response_ids}.")

            key = t_id
            if key not in lgp_dict and str(key) in lgp_dict:
                key = str(key)

            if key not in lgp_dict:
                raise ValueError(f"No logprobs for token {t_id} in {response_ids}.")

            # account for different formats of logprobs
            v = lgp_dict[key]
            if hasattr(v, 'logprob'):
                token_logprobs.append(float(v.logprob))

            elif isinstance(v, (int, float)):
                token_logprobs.append(float(v))

            elif isinstance(v, dict) and 'logprob' in v:
                token_logprobs.append(float(v['logprob']))

            else:
                raise TypeError(f"Unexpected logprob type: {type(v)}")

        return torch.tensor(token_logprobs, dtype=torch.float32, device='cpu')

    def score_response(self, prompt: Dict[str, Any], response: Any) -> torch.Tensor:
        '''
            Calculate the reward for each response token.
            it returns a float tensor of len(response_ids).
        '''
        with torch.no_grad():
            # per token rewards or scalar reward
            rewards, is_per_token, correct_threshold = self.reward_func(prompt, response)

        if isinstance(rewards, torch.Tensor):
            rewards = rewards.to(dtype=torch.float32, device='cpu')

        else:
            rewards = torch.tensor(rewards, dtype=torch.float32, device='cpu')

        if rewards.numel() != len(response.token_ids):
            raise ValueError(f"score_response must return len={len(response.token_ids)} rewards, got {rewards.numel()}")

        return rewards, is_per_token, correct_threshold

    def normalize_rewards(self,
                          samples: List[Dict[str, Any]],
                          stats: Dict[str, List[int]],
                          prompt_len: int,
                          is_per_token: bool) -> None:
        '''
            Normalize rewards for each group of samples for a given prompt.
            samples: list of different responses for a given prompt e.g., [{"prompt_ids": [...], "response_ids": [...],...}, ...]
            stats: {"reward": [...], "length": [...]} or {"reward": [...], "length": [...], "reward": [...], "length": [...]} if reward_broadcast is True
         '''
        denom = len(samples) # number of samples in the group
        if len(samples) > 1:
            rewards_array = np.array(stats['rewards'])
            mean_scores = rewards_array.sum() / denom
            # Bessel's correction (n-1) for unbiased sample std with small n_samples
            std_scores  = np.sqrt(((rewards_array - mean_scores)**2).sum() / max(denom - 1, 1))

        else:
            # For a single sample, we don't normalize (i.e. advantage is 0 if we subtract mean)
            # but usually for n=1 we keep the raw reward.
            mean_scores = 0.0
            std_scores  = 1.0

        if is_per_token:
            raise ValueError("per token rewards are not supported yet as normalization is done assuming per response rewards")

        # now update the rewards in the samples
        for i, sample in enumerate(samples):
            # sample['reward']: [T] where prompt tokens would get 0
            # sample['reward'][-1]: means the last token reward
            zscore = torch.zeros_like(sample['token_rewards'], dtype=torch.float)
            zscore[-1] = (sample['token_rewards'][-1] - mean_scores) / (std_scores + self.eps_reward_norm)
            sample["token_zscores"] = zscore
            if self.reward_broadcast:
                sample["token_zscores"][prompt_len:] = zscore[-1]

            # prediction-aligned zscores
            # zscore[prompt_len:] corresponds to response tokens 0..N-1
            pred_zscores = torch.zeros_like(sample['token_zscores'], dtype=torch.float)
            pred_start = prompt_len - 1
            pred_end   = len(sample['token_zscores']) - 1
            pred_zscores[pred_start:pred_end] = sample["token_zscores"][prompt_len:]
            sample["pred_zscores"] = pred_zscores

    def generate(self,
                prompts: List[Dict[str, List[int]]],
                current_iter: int,
                policy_version: int,
                log_batch_metrics: bool = False) -> List[Dict[str, Any]]:
                ''' 
                    prompts: Data provided by the dataloader. For example:
                        [{'prompt_token_ids': [2,..], 'solution': '1'}, {'prompt_token_ids': [...], 'solution': '2'}, ...]
                    Returns a list of rollout samples. length ~ B * n_samples.

                    token-aligned and prediction-aligned logprobs/mask/done are returned.
                    Prediction-aligned here means: logit position t predicts token at t+1 (SFT-style shift).
                '''
                if not isinstance(prompts, list) or len(prompts) == 0:
                    raise TypeError(f"prompts must be a non-empty list, got {type(prompts)}")

                if self.force_strict_on_policy and int(policy_version) != int(self.loaded_version):
                    raise ValueError(
                                     f"Off-policy rollout: policy_version={int(policy_version)} "
                                     f"but loaded_version={int(self.loaded_version)}. ")

                assert self.async_engine is not None, f"{self.model_path} not loaded."
                # Rotate seed each epoch so sampling RNG varies across iterations.
                # For batch invariance mode, exclude engine_id so the same prompt
                # yields the same output regardless of how many engines are used
                # (topology-invariant: 1-engine and N-engine runs match).
                epoch_offset = (current_iter + 1) * 1000000000
                if self.batch_invariant:
                    self.sampling_params.seed = self.seed + epoch_offset

                else:
                    self.sampling_params.seed = self.seed + self.engine_id * 1000 + epoch_offset

                self.log(f"Generating completions for {len(prompts)} prompts with {self.n_samples} samples each")
                generated_outputs = self.run_async(self.generate_all(prompts, self.sampling_params))
                self.log(f"Generation complete for {len(prompts)} prompts with policy version {policy_version}")

                # generated_outputs has prompt_ids and other outputs
                # this works even if n_samples >= 1
                rollout_samples = []
                batch_num_prompts = 0
                batch_num_passes_at_ks = {i: 0.0 for i in range(1, self.n_samples + 1)}
                batch_num_passes_caret_k = 0.0
                batch_pass_rate_sum = 0.0
                batch_prompt_reward_sum = 0.0
                batch_best_of_k_reward_sum = 0.0
                batch_reward_std_sum = 0.0
                for prompt_data, data in zip(prompts, generated_outputs):
                    group_samples = []
                    group_stats   = {'rewards': [], 'lengths': [], 'correct_threshold': []}
                    prompt_ids = list(data.prompt_token_ids or [])
                    prompt_len = len(prompt_ids)
                    if prompt_len == 0:
                        raise ValueError(f"No prompt token ids found in generated output: {data}")

                    # process generated responses
                    for response in data.outputs:
                        response_ids = list(response.token_ids)
                        response_len = len(response_ids)
                        finish_reason = getattr(response, "finish_reason", None)
                        stop_reason   = getattr(response, "stop_reason", None)

                        # all have length [T] and token_aligned as described above
                        seq_len = prompt_len + response_len
                        input_ids = torch.tensor(prompt_ids + response_ids, dtype=torch.int64, device='cpu')

                        token_masks      = torch.zeros((seq_len,), dtype=torch.int32, device='cpu')
                        token_dones      = torch.zeros((seq_len,), dtype=torch.int32, device='cpu')
                        token_old_logprobs = torch.zeros((seq_len,), dtype=torch.float32, device='cpu')

                        # prediction-level
                        pred_masks      = torch.zeros((seq_len,), dtype=torch.int32, device='cpu')
                        pred_dones      = torch.zeros((seq_len,), dtype=torch.int32, device='cpu')
                        pred_old_logprobs = torch.zeros((seq_len,), dtype=torch.float32, device='cpu')

                        rewards       = torch.zeros((seq_len,), dtype=torch.float32, device='cpu')
                        pred_rewards  = torch.zeros((seq_len,), dtype=torch.float32, device='cpu')

                        # Score every response (including empty) so reward_func can see them,
                        # but only responses > 0 contribute to group stats and normalization.
                        rewards_resp, is_per_token, correct_threshold = self.score_response(prompt_data, response)
                        rewards[prompt_len:] = rewards_resp
                        # correct_threshold must be collected from all responses, including empty
                        # correct_threshold is required in pass@k calculation
                        group_stats['correct_threshold'].append(correct_threshold)

                        if response_len > 0:
                            # is_per_token is False, then rewards_resp will only have value for the last element
                            group_stats['rewards'].append(rewards_resp.sum().item())
                            group_stats['lengths'].append(len(response_ids))
                            if response.logprobs is None:
                                raise ValueError("response.logprobs is None. Check if SamplingParams(logprobs=1) is set.")

                            #####
                            # token-aligned
                            #####
                            token_masks[prompt_len:] = 1 # 1 if valid token which we want to update.
                            response_logprobs = self.extract_logprobs(response_ids, response.logprobs)
                            token_old_logprobs[prompt_len:] = response_logprobs

                            #####
                            # pred-aligned
                            #####
                            # To recall how autoregressive models work:
                            # - response token j is at token index prompt_len + j in input_ids
                            # - and this is predicted by logits index prompt_len + j - 1
                            # pred_aligned which would be one we will use in policy update
                            # and to avoid any weired indexing later in the training loop.
                            pred_start = prompt_len - 1
                            pred_end   = seq_len - 1
                            pred_masks[pred_start:pred_end] = 1
                            pred_old_logprobs[pred_start:pred_end] = response_logprobs
                            pred_rewards[pred_start:pred_end] = rewards[prompt_len:]

                            # Terminal handling:
                            #  1. stop: ended due to EOS or a stop condition so done should be 1.
                            #  2. length: truncated which should not be done=1 and we need to bootstrap
                            if finish_reason == "stop":
                                token_dones[seq_len - 1] = 1

                                # pred-aligned terminal is at the logit index that predicts last token
                                # seq_len >= 2 is guaranteed since prompt_len >= 1 and response_len >= 1
                                pred_dones[seq_len - 2] = 1

                            # if stop_reason is None, it means it ended on eos
                            # see here https://docs.vllm.ai/en/stable/api/vllm/outputs/#vllm.outputs.CompletionOutput
                            eos_in_tokens = (response_ids[-1] == self.eos_id)
                            ended_on_eos  = (finish_reason == "stop" and stop_reason is None and eos_in_tokens)

                            group_samples.append({ "iter": int(current_iter),
                                                "policy_version": int(policy_version),
                                                "loaded_version": int(self.loaded_version),

                                                # token-aligned
                                                "input_ids": input_ids, #[T]
                                                "token_rewards": rewards, #[T]
                                                "token_zscores": rewards.clone(), #[T] if len(group_samples) > 1 it will be replaced in normalize_rewards
                                                "token_masks": token_masks, #[T] 1 on response/valid tokens
                                                "token_dones": token_dones, #[T] 1 on last token if terminal
                                                "token_old_logprobs": token_old_logprobs, #[T] 0 on prompt since we don't backprop on it.

                                                # pred-aligned
                                                "pred_rewards": pred_rewards, #[T]
                                                "pred_masks": pred_masks, #[T]
                                                "pred_dones": pred_dones, #[T]
                                                "pred_old_logprobs": pred_old_logprobs, #[T]
                                                "pred_zscores": pred_rewards.clone(), #[T] if len(group_samples) > 1 it will be replaced in normalize_rewards

                                                "finish_reason": finish_reason,
                                                "stop_reason": stop_reason,
                                                "ended_on_eos": ended_on_eos,

                                                "response_ids": response_ids, # list[int]
                                                "prompt_ids": prompt_ids, # list[int]
                                                "response_text": getattr(response, "text", ""),
                                                "response_len": response_len,
                                                "truncated": 1 if (prompt_len + response_len) > self.max_seq_len else 0,
                                                    })
                    self.normalize_rewards(samples=group_samples,
                                           stats=group_stats,
                                           prompt_len=prompt_len,
                                           is_per_token=is_per_token)

                    # compute pass@k metrics and update related variables
                    assert len(set(group_stats['correct_threshold'])) == 1, 'all correct_thresholds should be the same from a reward function'
                    correct_threshold = group_stats['correct_threshold'][0]
                    pass_at_k_metrics = compute_pass_metrics(group_stats['rewards'], n_total=self.n_samples,
                                                             correct_threshold=correct_threshold)

                    batch_num_prompts += 1
                    for i in range(1, self.n_samples + 1):
                        batch_num_passes_at_ks[i] += pass_at_k_metrics['pass_at_ks'].get(i, 0.0)

                    batch_num_passes_caret_k += pass_at_k_metrics['pass_caret_k']
                    batch_pass_rate_sum += pass_at_k_metrics['pass_rate']
                    batch_prompt_reward_sum += pass_at_k_metrics['group_mean_reward']
                    batch_best_of_k_reward_sum += pass_at_k_metrics['best_of_k_reward']
                    batch_reward_std_sum += pass_at_k_metrics['reward_std_per_prompt']

                    for s in group_samples:
                        s["pass_at_ks"] = pass_at_k_metrics['pass_at_ks']
                        s["pass_caret_k"] = pass_at_k_metrics['pass_caret_k']
                        s["pass_rate"] = pass_at_k_metrics['pass_rate']
                        s["k"] = pass_at_k_metrics['k']
                        s["group_mean_reward"] = pass_at_k_metrics['group_mean_reward']
                        s["best_of_k_reward"] = pass_at_k_metrics['best_of_k_reward']
                        s["reward_std_per_prompt"] = pass_at_k_metrics['reward_std_per_prompt']

                    rollout_samples.extend(group_samples)

                if log_batch_metrics and batch_num_prompts > 0:
                    pass_at_k_items = ", ".join(
                        f"avg_pass@{i}={batch_num_passes_at_ks[i] / batch_num_prompts:.4f}"
                        for i in range(1, self.n_samples + 1)
                    )
                    self.log(f"Batch metrics: prompts={batch_num_prompts}, "
                             f"{pass_at_k_items}, "
                             f"avg_pass^k={batch_num_passes_caret_k / batch_num_prompts:.4f}, "
                             f"avg_pass_rate={batch_pass_rate_sum / batch_num_prompts:.4f}, "
                             f"avg_reward_per_prompt={batch_prompt_reward_sum / batch_num_prompts:.4f}, "
                             f"avg_best_of_k_reward={batch_best_of_k_reward_sum / batch_num_prompts:.4f}, "
                             f"avg_reward_std_per_prompt={batch_reward_std_sum / batch_num_prompts:.4f}")

                return rollout_samples

    async def generate_all(self, prompts, sampling_params):
        '''
            Submit all prompts to the async engine and collect final outputs.
            Each prompt becomes an independent request, allowing the engine to
            process completions as they finish.
            Uses monotonic _request_counter so request_ids never collide across calls
            (e.g. if a previous call raised and left stale ids in the engine).
        '''
        base_id = self._request_counter
        self._request_counter += len(prompts)

        async def generate_one(request_id, prompt_data):
            prompt_token_ids = prompt_data['prompt_token_ids']
            final_output = None
            async for output in self.async_engine.generate(prompt=None,
                                                           sampling_params=sampling_params,
                                                           request_id=str(request_id),
                                                           prompt_token_ids=prompt_token_ids):
                final_output = output
            return final_output

        tasks = [generate_one(request_id=base_id + i, prompt_data=p) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)
        return results

    def update_weights_direct(self, state_dict: dict, version: int) -> bool:
        '''
            Fallback weight update via /dev/shm pickle path.
            state_dict: {param_name: cpu_tensor} from training engine rank 0.
        '''
        if self.async_engine is None:
            self.log("Cannot update weights: engine not loaded")
            return False

        if self.loaded_version == version:
            self.log(f"Model already at version {version}, skipping weight update")
            return True

        shm_path = f"/dev/shm/feynrl_weights_{os.getpid()}_v{version}.pkl"
        with open(shm_path, 'wb') as f:
            pickle.dump(state_dict, f)

        # Free the CPU state_dict now that it's persisted to /dev/shm.
        # The TP workers will read from the file, so we don't need this copy.
        del state_dict
        self.log(f"Updating weights directly to version {version}")
        try:
            results = self.async_engine.collective_rpc("update_weights", args=(shm_path,))

        finally:
            os.remove(shm_path)  

        # Verify that weights were actually updated on all tp workers.
        # collective_rpc may silently swallow errors on non-rank-0 tp workers,
        # which would leave some shards with stale weights. update_weights
        # returns the number of parameters loaded where all workers should agree.
        # collective_rpc broadcasts to all TP workers within one rollout engine
        if results is not None:
            valid = [r for r in results if r is not None and r != 0]
            # Complete failure when all workers returned None or 0
            if not valid:
                raise RuntimeError(f"Weight sync verification failed: all {len(results)} TP workers "
                                   f"returned {results} after update_weights. No parameters were loaded. "
                                   f"This likely means load_weights silently failed on all shards.")

            if len(valid) < len(results):
                # Some workers returned a count and some didn't — partial failure
                failed = [i for i, r in enumerate(results) if r is None or r == 0]
                raise RuntimeError(f"Weight sync verification failed: TP workers {failed} "
                                   f"returned {[results[i] for i in failed]} after update_weights. "
                                   f"Weights may be out of sync across TP shards.")

            # check if all tp workers loaded the same number of parameters.
            # it should be one, otherwise there is a problem
            if len(set(valid)) > 1:
                raise RuntimeError(f"Weight sync verification failed: TP workers loaded different "
                                   f"param counts: {results}. Weights may be out of sync.")

        self.loaded_version = version
        self.log(f"Weights updated to version {version}")
        return True

    def init_nccl_group(self, master_addr, master_port, rank_offset, world_size, group_name, timeout_seconds):
        '''
            Initialize NCCL weight sync group on all tp workers via collective_rpc.
            collective_rpc broadcasts to all tp workers simultaneously.
            Each worker computes its own rank as rank_offset + tp_rank internally.
        '''
        results = self.async_engine.collective_rpc( "init_weight_sync_group",
            args=(master_addr, master_port, rank_offset, world_size, group_name, timeout_seconds)
        )
        self.log(f"NCCL weight sync group initialized (rank_offset={rank_offset}, tp={self.tensor_parallel_size})")
        return all(r for r in results if r is not None)

    def update_weights_nccl(self, param_name, dtype_str, shape, empty_cache=False):
        '''
            Receive a single weight tensor via NCCL broadcast on all TP workers.
            Weight sync call order based on main_rl_nccl.py:
            1. init_nccl_group       -->   once at startup
            2. update_weights_nccl   -->   every iteration, once per parameter
            3. finalize_weight_nccl  -->   once after all parameters are received
            4. generate              -->   produce rollouts with updated weights
            Steps 2-4 repeat each iteration.
            5. close_nccl_group      -->   once at shutdown
            update_weights_direct is the alternative to steps 2-3 (used by main_rl.py,
            the non-NCCL driver). The two paths are never mixed in the same run.
        '''
        results = self.async_engine.collective_rpc("update_weights_nccl", 
                                                   args=(param_name, dtype_str, 
                                                         tuple(shape), empty_cache))
        return results

    def finalize_weight_nccl(self, version):
        '''
            Update loaded_version after all NCCL parameters have been received.
            Must be called after the last update_weights_nccl to keep version
            tracking consistent with update_weights_direct and refresh_model.
        '''
        self.loaded_version = version
        self.log(f"NCCL weight sync finalized to version {version}")
        return True

    def close_nccl_group(self):
        '''
            Destroy the NCCL weight sync group on all TP workers.
        '''
        try:
            self.async_engine.collective_rpc("close_weight_sync_group")
        except Exception:
            pass

    def refresh_model(self, model_path: str, version: int) -> bool:
        '''
            Refresh model by reloading the async engine from disk.
        '''
        if self.async_engine is not None and \
           self.loaded_version == version and \
           model_path == self.model_path:
            self.log(f"Model already at version {version}, skipping refresh")
            return False

        self.log(f"Refreshing model to version {version} from {model_path}")
        self.model_path = model_path
        self.load_async_engine()
        self.loaded_version = version
        self.log(f"Model refreshed to version {version}")
        return True

if __name__ == "__main__":
    # to test cd ~/FeynRL && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python rollouts/vllm_engine_async.py
    from transformers import AutoTokenizer
    import ray
    ray.init(local_mode=True)
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-1b-it')

    def default_reward_func(prompt, response):
        is_per_token = False
        correct_threshold = 0.0
        response_ids = response.token_ids
        finish_reason = getattr(response, "finish_reason", None)
        r = torch.zeros((len(response_ids),), dtype=torch.float32)

        if len(response_ids) == 0:
            return r, is_per_token, correct_threshold

        r[-1] = 1.0 if str(finish_reason) == "stop" else 0.0

        return r, is_per_token, correct_threshold

    vllm = VLLMRolloutEngineAsync.remote(model_path='google/gemma-3-1b-it',
                                         trust_remote_code=True,
                                         temperature=1,
                                         max_tokens=1024,
                                         n_samples=5,
                                         top_p=1,
                                         top_k=-1,
                                         seed=50,
                                         ignore_eos=False,
                                         stop=None,
                                         stop_token_ids=None,
                                         prompt_logprobs=None,
                                         force_strict_on_policy=True,
                                         reward_func=default_reward_func,
                                         tensor_parallel_size=1,
                                         eos_id=tokenizer.eos_token_id,
                                         reward_broadcast=True,
                                         eps_reward_norm=1e-8,
                                         gpu_memory_utilization=0.5,
                                         engine_id=0,
                                         max_seq_len=2048,
                                         max_model_len=32768,
                                         model_dtype='bfloat16',
                                         batch_invariant=True,
                                         )

    dummy_data = ["Hello, how are you?",
                  "Summer is the best season!",
                  "I love playing chess.",
                  ]
    samples_ids = []
    for i in dummy_data:
        prompt_ids = tokenizer.apply_chat_template(
                                        conversation= [{"role": "user", "content": i}],
                                        add_generation_prompt=True,
                                        tokenize=True,
                                        return_tensors=None,
                                        )
        samples_ids.append({"prompt_token_ids": prompt_ids})
    output_ref = vllm.generate.remote(prompts=samples_ids, current_iter=1, policy_version=0, log_batch_metrics=True)
    output = ray.get(output_ref)
    print(output)
    print('Done')
    ray.shutdown()