
import torch
from torch.utils.data import DataLoader

from data import LearningTask
from model import PacBayesLinReg
from utils import to_device


def run_learning(args):
    task = LearningTask(args.d, args.g_vec_max_norm, args.x_max_norm, args.noise_std)
    model = PacBayesLinReg(args.d, args.r, args.mu_Q_max_norm, args.sigma_Q, args.mu_P, args.sigma_P).to(args.device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # ---------------------------------------------------------------------------------------#
    # Training loop
    # ---------------------------------------------------------------------------------------#
    n_train_samp = args.n_train_samp
    train_data = task.get_dataset(n_train_samp)
    train_loader = None

    for epoch in range(1, args.n_epochs + 1):
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                  generator=torch.Generator(device=args.device))
        model.train()
        train_loss = 0
        for batch_idx, (X, Y) in enumerate(train_loader):
            to_device(args.device, X, Y)
            optimizer.zero_grad()
            if args.optim_objective == 'empirical_risk':
                loss = model.empirical_risk(X, Y)
            elif args.optim_objective == 'wpb_risk_bound':
                loss = model.wpb_risk_bound(X, Y, args.delta, args.n_train_samp)
            elif args.optim_objective == 'klpb_risk_bound':
                loss = model.klpb_risk_bound(X, Y, args.delta, args.n_train_samp)
            else:
                raise ValueError('Unknown optim_objective: {}'.format(args.optim_objective))
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            model.project_to_domain()
            if batch_idx and batch_idx % args.log_interval == 0:
                print(f'\rTrain Epoch: {epoch} [{batch_idx * args.batch_size}/{n_train_samp}]'
                      f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / args.batch_size:.6f}',
                      flush=True, end='')

        print(f'\r====> Epoch: {epoch} Average loss: {train_loss / n_train_samp:.4f}', flush=True, end='')
    # ---------------------------------------------------------------------------------------#
    # Evaluate final model
    # ---------------------------------------------------------------------------------------#
    print('\n')
    train_risk = model.run_evaluation(args, train_loader)
    wpb_gap_bnd = model.wpb_gap_bound(args.delta, args.n_train_samp)
    wpb_bnd = wpb_gap_bnd + train_risk
    klpb_gap_bnd = model.klpb_gap_bound(args.delta, args.n_train_samp)
    klpb_bnd = klpb_gap_bnd + train_risk
    print(f'Final training error: {train_risk:.6f}, (# training samples: {n_train_samp})')
    print(f'Final WPB bound: {wpb_bnd:.4f}')
    print(f'Final KL PB bound: {klpb_bnd:.4f}')
    uc_bnd = train_risk + model.uc_gap_bound(args.delta, args.n_train_samp)
    print(f'UC bound: {uc_bnd:.4f}')
    test_loader = DataLoader(task.get_dataset(args.n_samp_test), batch_size=args.batch_size, shuffle=False)
    test_risk = model.run_evaluation(args, test_loader)
    print(f'Final test error: {test_risk:.4f}')
    print( '-'*100)
    result = {'Train risk': train_risk, 'Test risk': test_risk, 'WPB bound': wpb_bnd,
              'UC bound': uc_bnd, 'KLPB bound': klpb_bnd}
    return result
