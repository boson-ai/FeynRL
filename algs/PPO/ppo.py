import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class PPO:
    def __init__(self,
                config,
                model_engine,
                optimizer,
                dataloader,
                device='cpu'):
        self.config = config
        self.model_engine = model_engine
        self.optimizer = optimizer
        self.device = device
        self.dataloader = dataloader

    def compute_loss(self, logits, labels):
        '''
         This functions implements \sum_{i=1}^{N} log p(y_i|x_i)
        '''

    def train(self):
        '''
           dataloader: dataloader for training
        '''
        for epoch in range(self.config.train.epochs):
            for batch in self.dataloader:
                
