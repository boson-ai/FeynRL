import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler


class DummyDataset(Dataset):
    '''
        This is a dummy dataset to test the dataloader and pipeline in general.
    '''
    def __init__(self, tokenizer, data_path="", max_seq_len=64):
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.len_data = 1000

    def __getitem__(self, idx):
        text = " this is a dummy text to test our dataloader"
        tokenized_text = self.tokenizer(
                                        text,
                                        return_tensors="pt",
                                        padding="max_length",
                                        truncation=True,
                                        max_length=self.max_seq_len,
                                    )
        return tokenized_text

    def __len__(self):
        return self.len_data