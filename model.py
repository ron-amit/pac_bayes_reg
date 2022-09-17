import torch
from torch import tensor
import torch.nn as nn
from utils import draw_uniformly_in_ball
from torch import linalg

def loss_func(h: tensor, X: tensor, y: tensor) -> tensor:
    assert h.n_dim == 1
    assert X.n_dim == 2  # [n_samp x d]
    assert y.n_dim == 1
    assert y.shape[0] == X.shape[0]
    return torch.square(torch.matmul(X, h) - y) / 4


class PacBayesLinReg(nn.Module):
    def __init__(self, d: int, mu_Q_max_radius: float, sigma_Q: float):
        super().__init__()
        self.mu_Q_max_radius = mu_Q_max_radius
        mu_Q_init = draw_uniformly_in_ball(d, mu_Q_max_radius).squeeze()
        self.mu_Q = nn.Parameter(mu_Q_init)
        self.sigma_Q = sigma_Q

    def empirical_risk(self, X: tensor, Y: tensor) -> tensor:
        n_samp = X.shape[0]
        er = (torch.sum((X @ self.mu_Q - Y)**2) + self.sigma_Q**2 * torch.sum(X[:]**2))/(4 * n_samp)
        return er


    def draw_from_posterior(self):
        eps = torch.randn_like(self.mu_Q)
        h = self.mu_Q + eps * self.sigma_Q
        return h

