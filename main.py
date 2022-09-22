import argparse
import math
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch

from learn import run_learning
from utils import set_device, set_default_plot_params, save_fig, set_random_seed

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
parser.add_argument('--sigma_P', type=float, default=0.01, metavar='N',
                    help='STD of the prior distribution')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = set_device(args)
args.device = device
args.d = 100
args.r = 1.
args.g_vec_max_norm = 0.1
args.x_max_norm = 0.1
args.noise_max_norm = 10.
args.sigma_Q = 1e-2
args.mu_Q_max_norm = 0.1
args.mu_P = torch.zeros(args.d)
# args.sigma_P = 1e-2  # 1e-4
args.batch_size = 256
args.delta = 0.05
args.n_train_samp = 0

set_random_seed(args.seed)
set_default_plot_params()

# ---------------------------------------------------------------------------------------#
n_reps = 20
n_samp_grid = [40, 80, 120, 160]
n_grid = len(n_samp_grid)
results_labels = {'train_risk', 'test_risk', 'wpb_bnd', 'uc_bnd', 'klpb_bnd'}
results = {label: np.zeros((n_grid, n_reps)) for label in results_labels}
for i_rep in range(n_reps):
    set_random_seed(args.seed + i_rep)
    for i_grid, n_samp in enumerate(n_samp_grid):
        args.n_train_samp = n_samp
        result = run_learning(args)
        for label in results_labels:
            results[label][i_grid, i_rep] = result[label]
# ---------------------------------------------------------------------------------------#
# Risk figure
# ---------------------------------------------------------------------------------------#
mean_results = {label: np.mean(results[label], axis=1) for label in results_labels}
std_results = {label: np.std(results[label], axis=1) for label in results_labels}


ci_factor = 1.96 / math.sqrt(n_reps)  # 95% confidence interval factor
plt.figure()


def plot_line(result_name, label, color):
    plt.plot(n_samp_grid, mean_results[result_name], marker='o', label=label, color=color)
    plt.fill_between(n_samp_grid, mean_results[result_name] - std_results[result_name] * ci_factor,
                     mean_results[result_name] + std_results[result_name] * ci_factor,
                     color=color, alpha=0.2)


plot_line('train_risk', 'Train risk', 'blue')
plot_line('test_risk', 'Test risk', 'red')
plot_line('wpb_bnd', 'WPB bound', 'green')
# plot_line('uc_bnd', 'UC bound', 'orange')
plot_line('klpb_bnd', 'KLPB bound', 'purple')
plt.legend()
plt.grid(True)
plt.xlabel('Number of samples')
plt.ylabel('Loss')
plt.title(r'$\sigma_P = {}$'.format(args.sigma_P))
save_PDF = True
if save_PDF:
    save_fig(datetime.now().strftime('%Y_%m_%d__%H_%M_%S_risk'), base_path='figures')
plt.show()

df = pandas.DataFrame({"# samples": n_samp_grid} | {f"{label}": mean_results[label] for label in results_labels}
                      | {f"+/- {label}": std_results[label] for label in results_labels})
df = df.applymap(np.format_float_scientific, precision=2)
with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)
