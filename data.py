import torch

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
        y = torch.matmul(X, self.g_vec) + noise
        assert torch.all(torch.abs(y) <= 1)
        return X, y
