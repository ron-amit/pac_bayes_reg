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
    samp_radius = r * torch.pow(torch.rand(n), 1 / d)
    samp_vecs *= samp_radius
    return samp_vecs


class LearningTask:
    def __init__(self, d: int = 20, g_vec_max_radius: float = 0.1, x_max_radius: float = 0.1, noise_min: float = -0.01,
                 noise_max: float = 0.01):
        self.d = d
        self.x_max_radius = x_max_radius
        self.g_vec = draw_uniformly_in_ball(d, g_vec_max_radius)
        self.noise_min = noise_min
        self.noise_max = noise_max

    def get_dataset(self, n_samples: int):
        X = draw_uniformly_in_ball(self.d, self.x_max_radius, n_samples)
        noise = self.noise_min + torch.rand(n_samples) * (self.noise_max - self.noise_max)
        y = torch.matmul(X, self.g_vec) + noise
        assert torch.all(torch.abs(y) <= 1)
        return X, y
