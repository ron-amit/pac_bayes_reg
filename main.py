import argparse
import torch
from data import LearningTask

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
# ---------------------------------------------------------------------------------------#
task = LearningTask(d=20, g_vec_max_radius=0.1, x_max_radius=0.1, noise_min=-0.01, noise_max=0.01)
n_train_samp = 50
train_set = task.get_dataset(n_train_samp)

model = VAE().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# ---------------------------------------------------------------------------------------#
