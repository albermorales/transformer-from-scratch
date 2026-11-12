import os

from torch.utils.data import Dataset
import torch
from datasets import load_dataset

class SentenceDataset(Dataset):
    def __init__(self, tokenizer, max_length: int=65, training: bool=True):
        super().__init__()

        self.dataset = load_dataset(
            "bentrevett/multi30k",
            token=os.getenv("HF_TOKEN"),
        )['train' if training else 'validation']

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return self.dataset.num_rows


    def __getitem__(self, idx):

        english = self.tokenizer(self.dataset[idx]['en'], padding="max_length", truncation=True, max_length= self.max_length+1, return_tensors="pt")
        german  = self.tokenizer(self.dataset[idx]['de'], padding="max_length", truncation=True, max_length= self.max_length+1, return_tensors="pt")


        return {'en':
                    {
                        'token_ids': english['input_ids'][0,1:],
                        'padding_mask': english['attention_mask'][0,1:].to(bool)
                    },
                'de': {
                    'token_ids': german['input_ids'][0, :-1],
                    'padding_mask': german['attention_mask'][0, :-1].to(bool),
                },
                'target': {
                    'token_ids': german['input_ids'][0, 1:],
                    'padding_mask': german['attention_mask'][0, 1:].to(bool),
                }
        }
