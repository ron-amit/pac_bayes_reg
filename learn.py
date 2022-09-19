
import torch
from torch.utils.data import DataLoader
from utils import to_device, run_evaluation
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

    print(model.empirical_risk(train_data.X, train_data.Y))

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
            if batch_idx % args.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * args.batch_size}/{n_train_samp}]'
                      f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / args.batch_size:.6f}')

        print(f'====> Epoch: {epoch} Average loss: {train_loss / n_train_samp:.4f}')
    # ---------------------------------------------------------------------------------------#
    # Evaluate final model
    # ---------------------------------------------------------------------------------------#
    print('-'*100)
    train_err = run_evaluation(model, args, train_loader)
    print(f'Final training error: {train_err}, (number of training samples: {n_train_samp})')
    n_samp_test = 10000
    test_loader = DataLoader(task.get_dataset(n_samp_test), batch_size=args.batch_size, shuffle=False)
    test_err = run_evaluation(model, args, test_loader)
    print(f'Final test error: {test_err}, (estimated from {n_samp_test} samples)')
    # ---------------------------------------------------------------------------------------#
    return train_err, test_err
