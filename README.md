
Code for recreating the experiments in the paper ["Integral Probability Metrics PAC-Bayes Bounds ", R. Amit, B. Epstein, S. Moran, and R. Meir, NeurIPS 2022.](https://arxiv.org/abs/2207.00614)

## Pre-requisites
Tested on Python 3.10.4, conda environment with:
* matplotlib 3.5.2
* numpy 1.23.2
* pandas 1.4.3
* pytorch 1.12.1
* Jinja2 3.0.3
 

## How to re-create results
* python main.py --sigma_P 0.01   --sigma_Q 0.001  --optim_objective klpb_risk_bound
* python main.py --sigma_P 0.0001 --sigma_Q 0.001  --optim_objective klpb_risk_bound
* python main.py --sigma_P 0.     --sigma_Q 0.     --optim_objective wpb_risk_bound
