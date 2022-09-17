import argparse
import torch
from data import LearningTask
from model import PacBayesLinReg

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
# ---------------------------------------------------------------------------------------#
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

if args.cuda:
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    torch.set_default_tensor_type(torch.DoubleTensor)

# ---------------------------------------------------------------------------------------#
d = 20
r = 1.
g_vec_max_radius = 0.1
x_max_radius = 0.1
noise_min = -0.01
noise_max = 0.01
sigma_Q = 1e-3
mu_Q_max_radius = 0.1
mu_P = torch.zeros(d)
sigma_P = 1e-3

# ---------------------------------------------------------------------------------------#
task = LearningTask(d, g_vec_max_radius, x_max_radius, noise_min, noise_max)
n_train_samp = 50
X, Y = task.get_dataset(n_train_samp)

model = PacBayesLinReg(d, r, mu_Q_max_radius, sigma_Q, mu_P, sigma_P).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print(model.empirical_risk(X, Y))
# ---------------------------------------------------------------------------------------#
