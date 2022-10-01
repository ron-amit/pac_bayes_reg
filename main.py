import argparse
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch

from learn import run_learning
from utils import set_device, set_default_plot_params, save_fig, set_random_seed
torch.autograd.set_detect_anomaly(True)
# ---------------------------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=2, metavar='S',
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--sigma_P', type=float, default=0.01, metavar='N',
                    help='STD of the prior distribution')
parser.add_argument('--sigma_Q', type=float, default=0.001, metavar='N',
                    help='STD of the posterior distribution')
parser.add_argument('--optim_objective', type=str, default='wpb_risk_bound', metavar='N',
                    help='Optimization objective (wpb_risk_bound / klpb_risk_bound / empirical_risk)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = set_device(args)
args.device = device
args.d = 10
args.r = 1.
args.g_vec_max_norm = 0.1
args.x_max_norm = 0.1
args.noise_min = -0.5
args.noise_max = 0.5
args.mu_Q_max_norm = 0.1
args.mu_P = torch.zeros(args.d)
args.batch_size = 256
args.delta = 0.05
args.n_train_samp = 0
args.n_samp_test = 10000

set_random_seed(args.seed)
set_default_plot_params()
# ---------------------------------------------------------------------------------------#
n_reps = 10
n_samp_grid = [100, 200, 300, 400]
n_grid = len(n_samp_grid)
results_labels = ['Train risk', 'Test risk', 'UC bound', 'WPB bound', 'KLPB bound']
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
base_path = 'results'


def plot_line(result_name, label, color):
    plt.plot(n_samp_grid, mean_results[result_name], marker='o', label=label, color=color)
    plt.fill_between(n_samp_grid, mean_results[result_name] - std_results[result_name] * ci_factor,
                     mean_results[result_name] + std_results[result_name] * ci_factor,
                     color=color, alpha=0.2)


def draw_figure(f_name, show_UC=False):
    plt.figure()
    plot_line('Train risk', 'Train risk', 'blue')
    plot_line('Test risk', 'Test risk', 'red')
    if show_UC:
        plot_line('UC bound', 'UC bound', 'orange')
    plot_line('WPB bound', 'WPB bound', 'green')
    if args.sigma_P > 0:
        plot_line('KLPB bound', 'KLPB bound', 'purple')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Number of samples')
    plt.ylabel('Loss')
    plt.title(r'$\sigma_P = {}$'.format(args.sigma_P))
    save_fig(f_name, base_path=base_path)
    plt.show()


file_name = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}_sigmaP_{args.sigma_P}'.replace('.', '_')
draw_figure(f_name=file_name, show_UC=True)

# ---------------------------------------------------------------------------------------#
columns_dict = {}
for label in results_labels:
    columns_dict[f"{label}"] = [f'{mean_results[label][i]:9.4f} ({ci_factor * std_results[label][i]:5.4f})'
                                for i in range(n_grid)]

df = pandas.DataFrame({r"\# train samples": n_samp_grid} | columns_dict)
df = df[["\# train samples"] + results_labels] # Reorder columns
df.set_index(r"\# train samples", inplace=True, drop=True)
with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)

with open(os.path.join(base_path, file_name) + '.txt', 'w') as f:
    f.write(str(args))
    f.write('\n' + '-' * 100 + '\n')
    f.write(df.style.to_latex(hrules=True))

