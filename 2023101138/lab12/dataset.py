import numpy as np
import torch
from torch.utils.data import Dataset

class BadmintonSkeletonDataset(Dataset):
    def __init__(self, data_npy, label_npy):
        self.data = np.load(data_npy).astype(np.float32)
        self.labels = np.load(label_npy).astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.data[idx])  # [T,132]
        label = torch.tensor(self.labels[idx])
        return seq, label