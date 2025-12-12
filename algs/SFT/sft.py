import torch
import numpy as np

class SFT:
    def __init__(self,
                model_engine,
                optimizer,
                device='cpu',
                use_cache=False):

        self.model_engine = model_engine
        self.optimizer = optimizer
        self.device = device
        self.use_cache = use_cache

        # use cross entropy loss
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    def compute_loss(self, logits, y, mask):
        '''
         This functions implements \sum_{i=1}^{N} log p(y_i|x_i)
         y is target label [B, T -1]
         logits is model prediction [B, T -1, vocab_size]
        '''
        # [B, T -1, vocab_size]
        _, _, vocab_size = logits.shape

        # flatten logits across batch and seq_len before computing loss
        # so logits is [B * (T -1), vocab_size]
        logits = logits.view(-1, vocab_size)
        # flatten y as well:  [B, T -1] -->  [B * (T -1)]
        y = y.view(-1)

        # per token loss
        per_token_loss = self.loss_fn(logits, y)

        # We need to apply mask to loss to remove any things 
        # which should not be considered in loss (e.g., padding tokens)
        mask = mask.view(-1).to(dtype=per_token_loss.dtype)  # [B * (T - 1)]
        masked_loss = per_token_loss * mask

        # To avoid gardient accumulation error caused by loss.mean(),
        # we use sum of loss instead but play with learning rate to account for this.
        loss = masked_loss.sum()
        return loss

    def forward(self, batch):
        '''
            This function implements a single forward pass for current batch.
            It returns logits, y, and mask.
            The size of batch['seq_ids']/batch['seq_attn_mask']/batch['loss_mask'] is [B, T]
            The size of return variables logits, y, and mask would be [B, T -1]
        '''
        # batch is a dictionary, so we want to extract things we need from it
        # input_ids/att_mask/pos_ids are [B, T]
        input_ids = batch['seq_ids'].to(self.device)
        att_mask  = batch['seq_attn_mask'].to(self.device)

        # if pos_ids is not provided, HF will add that automatically.
        pos_ids   = batch.get('position_ids', None)
        if pos_ids is not None:
            pos_ids = pos_ids.to(self.device)

        # feed data to model
        output = self.model_engine(input_ids=input_ids,
                                   attention_mask=att_mask,
                                   position_ids=pos_ids,
                                   use_cache=self.use_cache)

        # [B, T, vocab_size]
        every_token_logits = output.logits

        # Example
        #   prompt = [a, b, c]
        #   answer = [d, e, f, g, <eos>]
        # The model input (length = T) is:
        #   [a, b, c, d, e, f, g, <eos>]
        #
        # The y is the input shifted left by one (length = T-1):
        #   [b, c, d, e, f, g, <eos>]
        #
        # The model output (length = T) is if outputs correct results:
        #   [b, c, d, e, f, g, <eos>, X]
        # We discard the final token (X) since it is beyond <eos>, leaving logits of shape [T-1, vocab_size].
        #
        # The mask is length T, e.g.:
        #   [0, 0, 1, 1, 1, 1, 1, 0]
        #
        # The leading zeros correspond to prompt tokens -> no loss on them.
        # The final zero corresponds to the <eos> token in our example.
        #
        # Since we drop the last output logit, we also drop the first mask value (which is zero anyway),
        # producing a mask of shape [T-1] aligned with the logits and targets.


        # label would be input_ids shifted by one (input_ids[:, 1:])
        # so the size is [B, T-1]
        y = input_ids[:, 1:].contiguous()
        # seq_ids = [prompt_ids + answer_ids + EOS]
        # as we don't care about what is outputs of model for EOS token, we remove it from logits
        logits = every_token_logits[:, :-1, :].contiguous()

        # we need to make mask of size [B, T-1].\
        # we drop the first mask value (which is zero anyway).
        mask = batch['loss_mask'][:, 1:].contiguous().to(self.device)

        return logits, y, mask

    def eval_step(self, micro_batch):
        '''
           This function implements a single validation step per rank/gpu.
        '''
        # we need to split data into micro batches
        self.model_engine.eval()
        with torch.no_grad():
            # forward pass per gpu/rank
            logits, y, mask = self.forward(micro_batch)

            # compute loss pass
            loss = self.compute_loss(logits=logits, y=y, mask=mask)
            val_loss = loss.item()

        return {"loss": float(val_loss)}

    def train_step(self, micro_batch):
        '''
           This function implements a single training step per rank/gpu.
           The batch size for each gpu/rank should be micro_batch_size_per_gpu. 
           The DataLoader already yields micro-batches. 
        '''
        # make sure model is in training mode
        self.model_engine.train()

        # 1. forward pass per gpu/rank
        logits, y, mask = self.forward(micro_batch)

        # 2. compute loss pass
        loss = self.compute_loss(logits=logits, y=y, mask=mask)

        # 3. backward step
        # DeepSpeed backward handles gradient accumulation logic automatically.
        self.model_engine.zero_grad()
        # It aggregates gradients and only updates weights when accumulation_steps is reached.
        self.model_engine.backward(loss)

        # 4. optimizer step
        # DeepSpeed step handles optimizer updates and gradient clearing.
        self.model_engine.step()

        # return loss
        train_loss = loss.item()
        return {"loss": float(train_loss)}