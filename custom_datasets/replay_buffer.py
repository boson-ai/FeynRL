import torch
import numpy as np
from typing import Dict, Optional, Any, List
from torch.utils.data import Dataset

class ReplayBuffer(Dataset):
    '''
       Replay buffer for RL.
       It stores one trajectory per item (one sequence).
    '''
    def __init__(self,
                pad_token_id: int,
                max_seq_len: int
                ):

        self.item: List[Dict[str, Optional[torch.Tensor]]] = []
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        # shows the total number of action tokens which are not masked which
        # can be used for token-weighted scaling later.
        self.total_action_tokens = 0

    def reset(self):
        '''
            Reset the replay buffer when starting a new episode, if
            the algorithm requires it (e.g., PPO resets; P3O does not).
        '''
        self.item = []
        self.total_action_tokens = 0

    def ensure_1d(self, x: torch.Tensor, name: str):
        '''
            Sanity check to make sure the input is a 1D tensor.
        '''
        if x.dim() != 1:
            raise ValueError(f"Expected {name} to be 1D, got {x.dim()}D")

        return x

    def pad_to_max(self, x: torch.Tensor, pad_id: int, T: int) -> torch.Tensor:
        '''
            Pads the sequence to the batch max length.
            x: [T]
            T: max length of the sequence in the batch
        '''
        seq_len = x.numel()
        if seq_len > self.max_seq_len:
            # this should be handled with care to avoid any problem later
            x = x[:self.max_seq_len]

        elif seq_len < T:
            pad = torch.full((T - seq_len), pad_id, dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)

        return x

    def add(self,
            input_ids: torch.Tensor,
            attn_mask: torch.Tensor,
            old_logps: torch.Tensor,
            token_masks: torch.Tensor,
            rewards: torch.Tensor,
            dones: torch.Tensor,
            v_old: Optional[torch.Tensor] = None,
            )-> None:
        '''
            All inputs are [T]
            token_masks: 1=use token, 0=ignore (prompt/pad/etc.)
            dones: 1=eos, 0=not eos
        '''
        input_ids = self.ensure_1d(input_ids, "input_ids")
        attn_mask = self.ensure_1d(attn_mask, "attn_mask")
        old_logps = self.ensure_1d(old_logps, "old_logps")
        token_masks = self.ensure_1d(token_masks, "token_masks")
        rewards = self.ensure_1d(rewards, "rewards")
        dones = self.ensure_1d(dones, "dones")

        if v_old is not None:
            v_old = self.ensure_1d(v_old, "v_old")

        # Keep on CPU; dataLoader can pin_memory for faster H2D.
        curr_item = {
                "input_ids": input_ids.detach().cpu(),
                "attn_mask": attn_mask.detach().cpu(),
                "old_logps": old_logps.detach().cpu(),
                "token_masks": token_masks.detach().cpu(),
                "rewards": rewards.detach().cpu(),
                "dones": dones.detach().cpu(),
                "v_old": v_old.detach().cpu() if v_old is not None else None,
            }
        self.item.append(curr_item)

        self.total_action_tokens += int((token_masks > 0.5).sum()).item())

    def __len__(self) -> int:
        return len(self.item)

    def __getitem__(self, idx) -> Dict[str, Any]:
        return self.item[idx]

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        '''
            Overwrite the default collate_fn to handle padding.
        '''
        T = max([x["input_ids"].numel() for x in batch])

        # pad to batch_max_seq
        input_ids, attn_mask, old_logps = [], [], []
        token_masks, rewards, dones = [], [], []
        v_old = []
        for x in batch:
            # pad everything to zero except for input_ids which should
            # be padded to pad_token_id
            input_ids.append(self.pad_to_max(x=x["input_ids"], pad_id=self.pad_token_id, T=T))
            attn_mask.append(self.pad_to_max(x=x["attn_mask"], pad_id=0, T=T))
            old_logps.append(self.pad_to_max(x=x["old_logps"], pad_id=0, T=T))
            token_masks.append(self.pad_to_max(x=x["token_masks"], pad_id=0, T=T))
            rewards.append(self.pad_to_max(x=x["rewards"], pad_id=0, T=T))
            dones.append(self.pad_to_max(x=x["dones"], pad_id=0, T=T))

            if x["v_old"] is not None:
                v_old.append(self.pad_to_max(x["v_old"], pad_id=0, T=T))

        # convert from list of [T] to [B, T]
        input_ids = torch.stack(input_ids, dim=0)
        attn_mask = torch.stack(attn_mask, dim=0)
        old_logps = torch.stack(old_logps, dim=0)
        token_masks = torch.stack(token_masks, dim=0)
        rewards = torch.stack(rewards, dim=0)
        dones = torch.stack(dones, dim=0)

        # information for scaling later
        batch_action_tokens = int((token_masks > 0.5).sum().item())
        total_action_tokens = max(1, self.total_action_tokens)
        action_token_weight = float(batch_action_tokens) / float(total_action_tokens)

        return {
                "input_ids": input_ids, # [B, T]
                "attn_mask": attn_mask, # [B, T]
                "old_logps": old_logps, # [B, T]
                "token_masks": token_masks, # [B, T]
                "rewards": rewards, # [B, T]
                "dones": dones, # [B, T]
                "v_old": v_old, # [B, T]
                "batch_action_tokens": batch_action_tokens, # scalar int
                "action_token_weight": action_token_weight, # scalar float
                }