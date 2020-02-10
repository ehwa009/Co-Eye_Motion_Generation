import numpy as np
import torch
import pickle

from torch.utils.data import Dataset, DataLoader


# def prepare_dataloader(data, opt):
#     train_loader = DataLoader(
#         FaceLandmarksDataset()
#     )

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_torch(path):
    return torch.load(path)

class FaceLandmarksDataset(Dataset):

    def __init__(self, word2idx, idx2word, src_insts, trg_ints):
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.src_insts = src_insts
        self.trg_insts = trg_ints

    @property
    def n_insts(self):
        return len(self.src_insts)

    @property
    def vocab_size(self):
        return len(self.word2idx)

    @property
    def word2idx(self):
        return self.word2idx

    @property
    def idx2word(self):
        return self.idx2word

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        return self.src_insts[idx], self.trg_insts[idx]