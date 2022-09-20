import argparse
from datetime import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import set_device, set_default_plot_params, save_fig
from learn import run_learning

# ---------------------------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=2, metavar='S',
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = set_device(args)
args.device = device
args.d = 200
args.r = 1.
args.g_vec_max_norm = 0.01
args.x_max_norm = 0.1
# args.noise_min = -0.1
# args.noise_max = 0.1
args.noise_max_norm = 10.
args.sigma_Q = 1e-2
args.mu_Q_max_norm = 0.1
args.mu_P = torch.zeros(args.d)
args.sigma_P = 1e-2
args.batch_size = 50
args.delta = 0.05
args.n_train_samp = 0

torch.manual_seed(args.seed)
set_default_plot_params()
# ---------------------------------------------------------------------------------------#
n_reps = 3
n_samp_grid = [10,  20, 30, 40]
n_grid = len(n_samp_grid)
train_risk = np.zeros((n_grid, n_reps))
test_risk = np.zeros((n_grid, n_reps))
wpb_bnd = np.zeros((n_grid, n_reps))
uc_bnd = np.zeros((n_grid, n_reps))
for i_rep in range(n_reps):
    for i_grid, n_samp in enumerate(n_samp_grid):
        args.n_train_samp = n_samp
        train_risk[i_grid, i_rep], test_risk[i_grid, i_rep], wpb_bnd[i_grid, i_rep], uc_bnd[i_grid, i_rep]\
            = run_learning(args)
# ---------------------------------------------------------------------------------------#
# Risk figure
# ---------------------------------------------------------------------------------------#
mean_train_risk = train_risk.mean(axis=1)
std_train_risk = train_risk.std(axis=1)
mean_test_risk = test_risk.mean(axis=1)
std_test_risk = test_risk.std(axis=1)
mean_wpb_bnd = wpb_bnd.mean(axis=1)
std_wpb_bnd = wpb_bnd.std(axis=1)
mean_uc_bnd = uc_bnd.mean(axis=1)
std_uc_bnd = uc_bnd.std(axis=1)
ci_factor = 1.96 / math.sqrt(n_reps)  # 95% confidence interval factor
plt.figure()
plt.plot(n_samp_grid, mean_train_risk, marker='o', label='Test risk', color='blue')
plt.fill_between(n_samp_grid, mean_train_risk - std_train_risk * ci_factor,
                 mean_train_risk + std_train_risk * ci_factor,
                 color='blue', alpha=0.2)
plt.plot(n_samp_grid, mean_test_risk, marker='o', label='Train risk', color='red')
plt.fill_between(n_samp_grid, mean_test_risk - std_test_risk * ci_factor,
                 mean_test_risk + std_test_risk * ci_factor,
                 color='red', alpha=0.2)

plt.plot(n_samp_grid, mean_wpb_bnd, marker='o', label='WPB Bound', color='green')
plt.fill_between(n_samp_grid, mean_wpb_bnd - std_wpb_bnd * ci_factor,
                 mean_wpb_bnd + std_wpb_bnd * ci_factor,
                 color='green', alpha=0.2)

# plt.plot(n_samp_grid, mean_uc_bnd, marker='o', label='UC Bound', color='brown')
# plt.fill_between(n_samp_grid, mean_uc_bnd - std_uc_bnd * ci_factor,
#                  mean_uc_bnd + std_uc_bnd * ci_factor,
#                  color='brown', alpha=0.2)

plt.legend()
plt.grid(True)
plt.xlabel('Number of samples')
save_PDF = True
if save_PDF:
    save_fig(datetime.now().strftime('%Y_%m_%d__%H_%M_%S_risk'), base_path='figures')
plt.show()
