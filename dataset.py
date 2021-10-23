import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class CodeDataset(Dataset):
    def init(tokenizer, model):
        global _tokenizer, _model
        _tokenizer = tokenizer
        _model = model

    def __init__(self, data):
        self.data = data
        self.tokenizer = _tokenizer
        self.model = _model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with open(self.data[idx], 'r') as f:
            tokenized_data = self.tokenizer.encode(f.read())
        return torch.tensor(tokenized_data)

    # Add a function to generate a random code snippet
    def generate_code(self, input, length=100, temperature=0.7, top_k=50):
        return self.model.generate(
            input_ids=torch.tensor(
                self.tokenizer.encode(input)
            ),
            max_length=length,
            temperature=temperature,
            top_k=top_k
        )