
import torch
from torch.utils.data import DataLoader
from utils import to_device
from data import LearningTask
from model import PacBayesLinReg


def run_learning(args):
    task = LearningTask(args.d, args.g_vec_max_radius, args.x_max_radius, args.noise_min, args.noise_max)
    model = PacBayesLinReg(args.d, args.r, args.mu_Q_max_radius, args.sigma_Q, args.mu_P, args.sigma_P).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # ---------------------------------------------------------------------------------------#
    # Training loop
    # ---------------------------------------------------------------------------------------#
    n_train_samp = args.n_train_samp
    train_data = task.get_dataset(n_train_samp)
    train_loader = None

    for epoch in range(1, args.epochs + 1):
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                  generator=torch.Generator(device=args.device))
        model.train()
        train_loss = 0
        for batch_idx, (X, Y) in enumerate(train_loader):
            to_device(args.device, X, Y)
            optimizer.zero_grad()
            loss = model.wpb_risk_bound(X, Y, args.delta)
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
    train_err = model.run_evaluation(args, train_loader)
    print(f'Final training error: {train_err:.6f}, (number of training samples: {n_train_samp})')
    n_samp_test = 10000
    test_loader = DataLoader(task.get_dataset(n_samp_test), batch_size=args.batch_size, shuffle=False)
    test_err = model.run_evaluation(args, test_loader)
    print(f'Final test error: {test_err:.6f}, (estimated from {n_samp_test} samples)')
    print( '-'*100)
    # ---------------------------------------------------------------------------------------#
    return train_err, test_err
