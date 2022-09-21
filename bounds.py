import torch
from torch import tensor
from math import sqrt, log, pi
import numpy as np

def uc_bound(m: int, delta: float, d: int) -> float:
    assert delta > 0
    tmp1 = (5 * d + 2 * log(6 / delta)) / m
    bnd = sqrt(log(6 / delta) / (32 * m)) + sqrt((d + 2 * d * sqrt(log(3 / delta)) + 2 * log(3 / delta)) / (4 * m)) \
          + 8 * max(sqrt(tmp1), tmp1)
    return bnd


def uc_grad_bound(m: int, delta: float, d: int, r: float) -> float:
    assert delta > 0
    tmp1 = (5 * d + 2 * log(4 / delta)) / m
    bnd = 16 * r * max(sqrt(tmp1), tmp1) + r * sqrt((d + 2 * d * sqrt(log(2 / delta)) + 2 * log(2 / delta)) / (4 * m))
    return bnd


def wasserstein_gauss_proj(mu_q: tensor, sigma_q: tensor, mu_p: tensor, sigma_p: tensor, d: int,
                           r: float) -> tensor:
    assert mu_q.ndim == 1
    assert mu_q.shape[0] == d
    w_bnd = torch.sqrt(torch.sum((mu_q - mu_p) ** 2) + d * (sigma_q - sigma_p) ** 2) \
            + sqrt(pi / 2) * sigma_q * torch.erfc(
        (r - torch.sqrt(torch.sum(mu_q ** 2) + d * sigma_q ** 2)) / (sqrt(2) * sigma_q)) \
            + sqrt(pi / 2) * sigma_p * torch.erfc(
        (r - torch.sqrt(torch.sum(mu_p ** 2) + d * sigma_p ** 2)) / (sqrt(2) * sigma_p))
    return w_bnd


def wpb_bound(m: int, delta: float, mu_q: tensor, sigma_q: tensor, mu_p: tensor, sigma_p: tensor, d: int,
              r: float) -> tensor:
    assert m > 1
    u = uc_bound(m, delta / 4, d)
    ug = uc_grad_bound(m, delta / 4, d, r)
    w_bnd = wasserstein_gauss_proj(mu_q, sigma_q, mu_p, sigma_p, d, r)
    bnd = torch.sqrt(2 * u * ug * w_bnd + log(2 * m / delta) / (2 * (m - 1)))
    return bnd


def kl_pb_bound(m: int, delta: float, mu_q: tensor, sigma_q: tensor, mu_p: tensor, sigma_p: tensor, d: int) -> tensor:
    assert m > 1
    kl = torch.sum((mu_q - mu_p)**2) / (2 * sigma_p**2) \
          + d * (log(sigma_p / sigma_q) + sigma_q**2 / (2 * sigma_p**2) - 0.5)
    bnd = torch.sqrt((kl + log(2 * m / delta)) / (2 * (m - 1)))
    return bnd