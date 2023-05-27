import torch
import torch.utils.data as data
import numpy as np


class PiDataset(data.Dataset):
    def __init__(self, x, y, img, embedding_dim, device='cpu') -> None:
        super().__init__()
        self.ds = np.concatenate(
            [x.reshape((-1, 1)), y.reshape((-1, 1)), img],
            axis=1)
        self.device = device

    def __len__(self):
        return self.ds.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.ds[index]).to(self.device)
