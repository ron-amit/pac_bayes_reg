import torch
from torch import tensor
from torch.nn.functional import normalize


def draw_uniformly_in_ball(d: int, r: float, n: int = 1) -> tensor:
    """
    Draw n samples uniformly in a ball of radius r in R^d
     (see https://blogs.sas.com/content/iml/2016/04/06/generate-points-uniformly-in-ball.html)
    :param d: dimension of the ball
    :param r: radius of the ball
    :param n: number of samples to draw
    :return: a tensor of shape [n x d] containing the samples
    """
    assert d > 0
    assert r > 0
    samp_vecs = torch.randn(n, d)
    samp_vecs = normalize(samp_vecs, dim=1)
    samp_radius = r * torch.pow(torch.rand(n, 1), 1 / d)
    samp_vecs = samp_radius * samp_vecs
    return samp_vecs
