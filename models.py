from torch.utils.data import Dataset
import torch
import numpy as np


class StockDataset(Dataset):
    """Wraps numpy arrays into a PyTorch Dataset for DataLoader."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y).unsqueeze(1)    # (N,) → (N,1) for loss fn

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]