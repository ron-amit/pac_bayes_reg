import torch
from torch import tensor
import torch.nn as nn
from utils import draw_uniformly_in_ball


def loss_func(h: tensor, X: tensor, y: tensor) -> tensor:
    assert h.n_dim == 1
    assert X.n_dim == 2  # [n_samp x d]
    assert y.n_dim == 1
    assert y.shape[0] == X.shape[0]
    return torch.square(torch.matmul(X, h) - y) / 4


class PosteriorModel(nn.Module):
    def __init__(self, d: int, mu_Q_max_radius: float, sigma_Q: float):
        super().__init__()
        self.mu_Q_max_radius = mu_Q_max_radius
        mu_Q_init = draw_uniformly_in_ball(d, mu_Q_max_radius).squeeze()
        self.mu_Q = nn.Parameter(mu_Q_init)
        self.sigma_Q = sigma_Q

    def forward(self):
        # reparameterize
        eps = torch.randn_like(self.mu_Q)
        h = self.mu_Q + eps * self.sigma_Q
        return h

def empirical_risk(model: PosteriorModel, X: tensor, y: tensor) -> tensor:
    h = model()
    return torch.mean(loss_func(h, X, y))

