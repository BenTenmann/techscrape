import json
import torch
import random
import pandas as pd
from torch.utils.data import Dataset


def vocabulary(company_names: list):
    vocab = set(company_names[0])
    for name in company_names[1:]:
        if type(name) == str:
            vocab |= set(name)
    return vocab


class LSCompanies(Dataset):
    def __init__(self, file_path):
        print(
            'loading data...',
            end=' '
        )
        self.data = pd.read_csv(file_path)
        self.data.dropna(inplace=True)
        print(
            'done!'
        )
        self.vocab = {
            key: value for value, key in
            enumerate(vocabulary(self.data['company_name'].tolist()))
        }
        self.vocab_size = len(self.vocab)
        print(f'vocab size: {self.vocab_size}')

    def __len__(self):
        return self.data.shape[0]

    def _name_to_tensor(self, name):
        return torch.tensor([self.vocab[letter] for letter in name], dtype=torch.int32)

    def __getitem__(self, index):
        return (self._name_to_tensor(self.data['company_name'].iloc[index]),
                torch.tensor(self.data['is_biotech'].iloc[index], dtype=torch.int32))

    def train_test_split(self, pct_train=0.8):
        n_train = int(len(self) * pct_train)
        idx = random.sample(list(range(len(self))), len(self))
        return (CompanyDataset(self.data.iloc[idx[:n_train], :], self.vocab),
                CompanyDataset(self.data.iloc[idx[n_train:], :], self.vocab))

    def save_vocab(self, filepath):
        with open(filepath, 'w') as file:
            file.write(
                json.dumps(self.vocab, indent=4)
            )
        file.close()
        return 0

    def load_vocab(self, filepath):
        with open(filepath, 'r') as file:
            data = json.loads(file.read())
        file.close()
        self.vocab = data
        return 0


class CompanyDataset:
    def __init__(self, dataset: pd.DataFrame, vocab):
        self.dataset = dataset.reset_index()
        self.vocab = vocab

    def __len__(self):
        return self.dataset.shape[0]

    def _name_to_tensor(self, name):
        return torch.tensor([[self.vocab[letter]] for letter in name], dtype=torch.int32)

    def __getitem__(self, index):
        return (self._name_to_tensor(self.dataset['company_name'].iloc[index]),
                torch.tensor(self.dataset['is_biotech'].iloc[index], dtype=torch.float32))
