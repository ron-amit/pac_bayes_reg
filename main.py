import argparse
import torch
from utils import set_device
from learn import run_learning
# ---------------------------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = set_device(args)
args.device = device
args.d = 20
args.r = 1.
args.g_vec_max_radius = 0.1
args.x_max_radius = 0.1
args.noise_min = -0.01
args.noise_max = 0.01
args.sigma_Q = 1e-3
args.mu_Q_max_radius = 0.1
args.mu_P = torch.zeros(args.d)
args.sigma_P = 1e-3
args.batch_size = 64
args.delta = 0.05
# ---------------------------------------------------------------------------------------#
run_learning(args)
# ---------------------------------------------------------------------------------------#
