import torch
import pandas as pd
import typing
import math
import numpy as np
import os

DNA_ALPHABET = {'A': 0, 'C': 1, 'G': 2, 'T': 3} #, 'M': 4}
INDEX_TO_DNA = {v: k for k, v in DNA_ALPHABET.items()}
lookup_array = np.array([INDEX_TO_DNA[i] for i in range(len(INDEX_TO_DNA))])


def batch_dna_tokenize(batch_seq):
    """
    batch_seq: list of strings
    return: numpy array of shape [batch_size, seq_len]
    """
    tokenized_batch = np.array([[DNA_ALPHABET[c] for c in seq] for seq in batch_seq])
    return tokenized_batch


def batch_dna_detokenize(batch_seq):
    """
    batch_seq: numpy array of shape [batch_size, seq_len]
    return: list of strings
    """
    detokenized_batch = lookup_array[batch_seq]
    detokenized_batch = [''.join(seq) for seq in detokenized_batch]
    return detokenized_batch


class GosaiDataset(torch.utils.data.Dataset):
    def __init__(self, base_path=None):
        data_df = pd.read_csv(os.path.join(base_path, f'mdlm/gosai_data/processed_data/gosai_all.csv'))
        self.seqs = torch.tensor(data_df['seq'].apply(lambda x: [DNA_ALPHABET[c] for c in x]).tolist())
        self.clss = torch.tensor(data_df[['hepg2', 'k562', 'sknsh']].to_numpy())

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {'seqs': self.seqs[idx], 'clss': self.clss[idx], 'attention_mask': torch.ones(len(self.seqs[idx]))}


def get_datasets_gosai(base_path):
    return GosaiDataset(base_path)

