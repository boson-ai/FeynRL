## How to use

After setting up the environment, the next step is preparing the data. See the scripts in [`data-prep/`](https://github.com/rasoolfa/FeynRL/tree/main/data_prep) for reference implementations—you can adapt them to your own datasets.

**Data format requirement:** your final processed data must match the **exact** format produced by these scripts (the original/raw format does not matter). You need to write your own scripts simailr to the following scripts to prepare your data in the required format.

* [`data-prep/gsm8k.py`](https://github.com/rasoolfa/FeynRL/blob/main/data_prep/gsm8k.py) prepares **GSM8K** in a format suitable for **SFT** and **RL** training, and can also be used for evaluation.
* [`data-prep/hh_rlhf.py`](https://github.com/rasoolfa/FeynRL/blob/main/data_prep/hh_rlhf.py) prepares a **preference/contrastive** dataset suitable for **DPO**-style contrastive learning.

Once your data is prepared, update the **`data`** section in the relevant config file and run the corresponding entrypoint:

* SFT: `./configs/sl_args.yaml`
* Contrastive Learning (DPO, etc.): `./configs/cl_args.yaml`
* RL: `./configs/rl_args.yaml`

---

## Running on a single node

“Single node” means one machine with multiple GPUs.

### Supervised Fine-Tuning (SFT)

`main_sl.py` is the entry point for supervised learning experiments.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main_sl.py --config-file ./configs/sl_args.yaml --experiment_id myexp1
```

**Notes**

* `experiment_id` is the name of the experiment. It is used to create an output directory to store logs, checkpoints, and metrics.
* `CUDA_VISIBLE_DEVICES` selects which GPUs are visible to the run.
* `--nproc_per_node` must match the number of visible GPUs (e.g., 4 GPUs ⇒ `--nproc_per_node=4`).
* Before running, review `sl_args.yaml` and update at least:

  * model / tokenizer path
  * dataset path(s)
  * batch sizes / gradient accumulation
  * output directory / logging config
* We call it `main_sl.py` rather than `main_sft.py` because this entry point is intended to support **general supervised learning**, not just SFT.

---

### Contrastive Learning (CL) — DPO and related methods

`main_cl.py` is the entry point for contrastive/preference learning experiments (e.g., DPO).

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 main_cl.py --config-file ./configs/cl_args.yaml --experiment_id myexp2
```

**Notes**

* Ensure `cl_args.yaml` points to the processed dataset paths and that the expected fields match what the trainer expects.

---

### Reinforcement Learning (RL)

RL runs are more involved because they use **Ray** to orchestrate DeepSpeed **training** and **rollout** engines.

`main_rl.py` is the entry point for RL experiments (e.g., PPO, SGRPO, CISPO, etc.).

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main_rl.py --config-file ./configs/rl_args.yaml --experiment_id exp3
```

**Key differences vs SFT/CL**

* RL uses two types of engines:

  * **Training engine** (based on DeepSpeed)
  * **Rollout engine(s)** (inference/generation for trajectories which is based on vLLM)
* Ray schedules these workers across available GPUs.

**Config knobs**
In `rl_args.yaml`, make sure you clearly document and set:

* `rollout_gpus`: number of GPUs reserved for rollout workers
* `training_gpus`: number of GPUs reserved for training workers
* `ray_address`:

  * single node: set to `null` to start locally
  * multi node: see next section
* `ray_master_port`: port used by DeepSpeed/NCCL rendezvous.

**Single-node Ray**

* If `ray_address` is `null`, the code should start/connect to Ray on the local machine automatically.
* `ray_master_port` can be any free port on the node (example: `25000`).


---

## Running on multiple nodes

“Multi-node” means Ray spans multiple machines and schedules rollout/training workers across them.

Assume:

* Node A (head): 4 GPUs
* Node B (worker): 4 GPUs

### 1) Start Ray head on the main node

On **Node A**:

```bash
CUDA_VISIBLE_DEVICES=3,5,2,4 ray start --head --port=26789
```

Ray will print the **IP address** of the head node (example: `100.9.128.5`) and connection instructions.

### 2) Join worker nodes

On **Node B**:

```bash
CUDA_VISIBLE_DEVICES=0,1,5,7 ray start --address=100.9.128.5:26789
```

If successful, Ray will report that the node joined the cluster.

### 3) Update RL config

In `./configs/rl_args.yaml`:

* Set `ray_address: "auto"`.
* Set `ray_master_port` to a **different free port** than Ray’s port (example: `25000`).

**Why `ray_master_port` must be different**

* Ray uses `26789` for cluster coordination.
* DeepSpeed/NCCL needs its own rendezvous port (`ray_master_port`) for distributed training setup.

### 4) Run RL from the head node

On **Node A**:

```bash
python main_rl.py --config-file ./configs/rl_args.yaml --experiment_id exp4
```

**Important**

* Do **not** set `CUDA_VISIBLE_DEVICES` for the multi-node RL run. Ray discovers and manages GPUs across nodes; forcing visibility can cause mismatches and scheduling errors.

---

## Troubleshooting (DeepSpeed / Ray / multi-node)

### RL run hangs during rollout or training step

**Possible causes:**
- GPU over-allocation — `training_gpus + rollout_gpus` exceeds available cluster GPUs.
- A Ray actor crashed silently (OOM, CUDA error) and the remaining actors are stuck waiting.
- NCCL timeout on the training side (network-level issue between nodes).

**How to fix:**
1. Verify GPU budget:
   ```bash
   python -c "import ray; ray.init(); print(ray.cluster_resources())"
   ```
   Confirm that the `GPU` count ≥ `training_gpus + rollout_gpus` in your config.
2. Check Ray actor status for dead/failed actors:
   ```bash
   ray status
   ```
3. If hangs occur during training (not rollout), add `NCCL_DEBUG=INFO` to the environment to surface NCCL-level errors:
   ```bash
   NCCL_DEBUG=INFO python main_rl.py --config-file ./configs/rl_args.yaml --experiment_id debug_run
   ```

---

### Strict on-policy error (`policy_version != loaded_version`)

**Possible causes:**
- Weight sync (`direct` or `disk`) failed silently in a previous epoch, so rollout engines still hold stale weights.
- `force_strict_on_policy: True` in the config, which makes the engine reject any version mismatch.

**How to fix:**
1. Search the logs for earlier `[WeightSync]` warnings, these indicate a failed sync attempt.
2. If the problem persists, switch to `weight_sync_method: disk` in `rl_args.yaml` as a fallback (slower but more reliable).

---

### vLLM reload/update failures

**Possible causes:**
- Checkpoint directory is missing `config.json` or tokenizer files, vLLM cannot load a model without them.
- On multi-node setups, the checkpoint path is on a local disk that rollout workers on other nodes cannot see.

**How to fix:**
1. Verify the checkpoint directory contains the required files:
   ```bash
   ls <checkpoint_dir>/<experiment_id>/
   # expect: config.json, tokenizer.json, tokenizer_config.json, model*.safetensors
   ```
2. For multi-node, use a **shared filesystem** for `checkpoint_dir` so all nodes can access saved checkpoints.
3. If using `weight_sync_method: direct`, disk checkpoints are only written at save intervals, verify the sync logs show success.

---

### Unexpected zero rewards in RL

**Possible causes:**
- The reward function returns 0 for all samples (e.g., the model never produces EOS, so the default reward assigns 0).
- Responses are being truncated at `rollout.max_tokens` before the model can produce a correct answer, and the terminal reward is lost.
- `data.max_seq_len` is too small, so prompt + response gets clipped during training.

**How to fix:**
1. Inspect a few raw rollout samples to see what the model is actually generating — look at `response_len` and whether EOS is present.
2. Check the relationship between `rollout.max_tokens` and `data.max_seq_len`:
   - `max_tokens` = max generation length (response only)
   - `max_seq_len` = max total length (prompt + response)
   - `max_tokens` must be ≤ `max_seq_len`
3. Try increasing `max_tokens` to give the model more room to produce a complete answer.
4. Verify your reward function handles edge cases (empty responses, truncated responses) correctly.