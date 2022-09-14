import torch
from torch import sqrt, log


def uc_bound(m, delta, d):
    assert delta > 0
    bnd = sqrt(log((6 / delta) / (32 * m))) + sqrt((d * (1 + 2 * sqrt(3 / delta)) + 2 * log(3 / delta)) / (4 * m)) \
          + 00000000000000000000
    return bnd
