import torch
from torch import tensor
from torch.utils.data import Dataset

from utils import draw_uniformly_in_ball


class LearningTask:
    def __init__(self, d: int = 20, g_vec_max_radius: float = 0.1, x_max_radius: float = 0.1, noise_min: float = -0.01,
                 noise_max: float = 0.01):
        self.d = d
        self.x_max_radius = x_max_radius
        self.g_vec = draw_uniformly_in_ball(d, g_vec_max_radius).squeeze()
        self.noise_min = noise_min
        self.noise_max = noise_max

    def get_dataset(self, n_samples: int):
        X = draw_uniformly_in_ball(self.d, self.x_max_radius, n_samples)
        noise = self.noise_min + torch.rand(n_samples) * (self.noise_max - self.noise_max)
        Y = torch.matmul(X, self.g_vec) + noise
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
