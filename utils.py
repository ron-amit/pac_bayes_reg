import torch
from torch.nn.functional import normalize
from torch import tensor


def draw_uniformly_in_ball(d: int, r: float, n: int = 1) -> tensor:
    """
    Draw n samples uniformly in a ball of radius r in R^d
     (see https://blogs.sas.com/content/iml/2016/04/06/generate-points-uniformly-in-ball.html)
    :param device:
    :type device:
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


def set_device(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    if args.cuda:
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)
    return device


def to_device(device, *args):
    return [arg.to(device) for arg in args]


def run_evaluation(model, args, data_loader):
    model.eval()
    avg_loss = 0
    n_samp = len(data_loader.dataset)
    with torch.no_grad():
        for i, (X, Y) in enumerate(data_loader):
            to_device(args.device, X, Y)
            loss = model.wpb_risk_bound(X, Y, args.delta)
            avg_loss += loss.item()
    avg_loss /= n_samp
    return avg_loss
