
import torch
from torch import tensor

class LearningTask:
    def __init__(self, d: int):
        self.d = d
        self.g_latent = torch.randn(d)

    def get_dataset(self, n_samples: int):
        X =
        y =
        return X, y