import torch
from torch import tensor
from torch.utils.data import Dataset

from utils import draw_uniformly_in_ball


class LearningTask:
    def __init__(self, d: int = 20, g_vec_max_norm: float = 0.1, x_max_norm: float = 0.1, noise_max_norm: float = 0.1):
        self.d = d
        self.x_max_norm = x_max_norm
        self.g_vec = draw_uniformly_in_ball(d, g_vec_max_norm).squeeze()
        self.noise_max_norm = noise_max_norm

    def get_dataset(self, n_samples: int):
        X = draw_uniformly_in_ball(self.d, self.x_max_norm, n_samples)
        # noise = self.noise_min + torch.rand(n_samples) * (self.noise_max - self.noise_max)
        # Y = torch.matmul(X, self.g_vec) + noise

        noise_xi = draw_uniformly_in_ball(self.d, self.noise_max_norm, n_samples)
        Y = torch.matmul(X + noise_xi, self.g_vec)
        assert torch.all(torch.abs(Y) <= 1)
        dataset = PairsDataset(X, Y)
        return dataset


class PairsDataset(Dataset):
    def __init__(self, X: tensor, Y: tensor):
        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)
