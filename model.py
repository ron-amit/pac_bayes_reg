import torch
import torch.nn as nn
from torch import tensor

from bounds import wpb_bound, uc_bound, kl_pb_bound
from utils import draw_uniformly_in_ball, to_device


def loss_func(h: tensor, X: tensor, y: tensor) -> tensor:
    assert h.n_dim == 1
    assert X.n_dim == 2  # [n_samp x d]
    assert y.n_dim == 1
    assert y.shape[0] == X.shape[0]
    return torch.square(torch.matmul(X, h) - y) / 4


class PacBayesLinReg(nn.Module):
    def __init__(self, d: int, r: float, mu_Q_max_norm: float, sigma_Q: float, mu_P: tensor, sigma_P: float):
        super().__init__()
        self.r = r
        self.mu_Q_max_norm = mu_Q_max_norm
        mu_Q_init = draw_uniformly_in_ball(d, mu_Q_max_norm).squeeze()
        self.mu_Q = nn.Parameter(mu_Q_init)
        self.sigma_Q = sigma_Q
        self.mu_P = mu_P
        self.sigma_P = sigma_P
        self.d = d

    def project_to_domain(self):
        with torch.no_grad():
            mu_norm_sqr = torch.sum(self.mu_Q ** 2)
            if mu_norm_sqr > self.r:
                self.mu_Q *= self.r / torch.sqrt(mu_norm_sqr)

    def empirical_risk(self, X: tensor, Y: tensor) -> tensor:
        batch_size = X.shape[0]
        return (torch.sum((X @ self.mu_Q - Y) ** 2) + self.sigma_Q ** 2 * torch.sum(X[:] ** 2)) / (4 * batch_size)

    def draw_from_posterior(self):
        eps = torch.randn_like(self.mu_Q)
        h = self.mu_Q + eps * self.sigma_Q
        return h

    def wpb_gap_bound(self, delta: float, n_samp: int) -> tensor:
        gap_bound = wpb_bound(n_samp, delta, self.mu_Q, self.sigma_Q, self.mu_P, self.sigma_P, self.d, self.r)
        return gap_bound

    def wpb_risk_bound(self, X: tensor, Y: tensor, delta: float, n_samp: int) -> tensor:
        emp_risk = self.empirical_risk(X, Y)
        gap_bound = self.wpb_gap_bound(delta, n_samp)
        return emp_risk + gap_bound

    def klpb_gap_bound(self, delta: float, n_samp: int) -> tensor:
        gap_bound = kl_pb_bound(n_samp, delta, self.mu_Q, self.sigma_Q, self.mu_P, self.sigma_P, self.d)
        return gap_bound

    def klpb_risk_bound(self, X: tensor, Y: tensor, delta: float, n_samp: int) -> tensor:
        emp_risk = self.empirical_risk(X, Y)
        gap_bound = self.klpb_gap_bound(delta, n_samp)
        return emp_risk + gap_bound

    def uc_gap_bound(self, delta: float, n_samp: int) -> tensor:
        gap_bound = uc_bound(n_samp, delta, self.d)
        return gap_bound

    def run_evaluation(self, args, data_loader):
        self.eval()
        avg_loss = 0
        n_samp = len(data_loader.dataset)
        with torch.no_grad():
            for i, (X, Y) in enumerate(data_loader):
                to_device(args.device, X, Y)
                batch_size = X.shape[0]
                loss = self.empirical_risk(X, Y)
                avg_loss += loss.item() * batch_size
        avg_loss /= n_samp
        return avg_loss
