import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import tensor
from torch.nn.functional import normalize


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

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


def save_fig(run_name, base_path='./'):
    ensure_dir(base_path)
    save_path = os.path.join(base_path, run_name)
    plt.savefig(save_path + '.pdf', format='pdf', bbox_inches='tight')
    # try:
    #     plt.savefig(save_path + '.pgf', format='pgf', bbox_inches='tight')
    # except:
    #     print('Failed to save .pgf file  \n  tto allow to save pgf files -  $ sudo apt install texlive-xetex')
    print('Figure saved at ', save_path)


def set_default_plot_params():
    plt_params = {'font.size': 10,
                  'lines.linewidth': 2, 'legend.fontsize': 16, 'legend.handlelength': 2,
                  'pdf.fonttype': 42, 'ps.fonttype': 42,
                  'axes.labelsize': 18, 'axes.titlesize': 18,
                  'xtick.labelsize': 14, 'ytick.labelsize': 14}
    plt.rcParams.update(plt_params)


def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
#
