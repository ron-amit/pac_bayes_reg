import argparse
import torch
from torch.utils.data import DataLoader
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
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
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
batch_size = 64
delta = 0.05
# ---------------------------------------------------------------------------------------#
task = LearningTask(d, g_vec_max_radius, x_max_radius, noise_min, noise_max)
model = PacBayesLinReg(d, r, mu_Q_max_radius, sigma_Q, mu_P, sigma_P).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# ---------------------------------------------------------------------------------------#
# Training loop
# ---------------------------------------------------------------------------------------#
n_train_samp = 1000
train_data = task.get_dataset(n_train_samp)

print(model.empirical_risk(train_data.X, train_data.Y))

for epoch in range(1, args.epochs + 1):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    model.train()
    train_loss = 0
    for batch_idx, (X, Y) in enumerate(train_loader):
        X = X.to(device)
        Y = Y.to(device)
        optimizer.zero_grad()
        loss = model.wpb_risk_bound(X, Y, delta)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * batch_size}/{n_train_samp}]'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / batch_size:.6f}')

    print(f'====> Epoch: {epoch} Average loss: {train_loss / n_train_samp:.4f}')
# ---------------------------------------------------------------------------------------#
# Evaluate final model
# ---------------------------------------------------------------------------------------#
model.eval()
test_loss = 0
n_samp_test = 10000
test_loader = DataLoader(task.get_dataset(n_samp_test), batch_size=batch_size, shuffle=False)
with torch.no_grad():
    for i, (X, Y) in enumerate(test_loader):
        X = X.to(device)
        Y = Y.to(device)
        loss = model.wpb_risk_bound(X, Y, delta)
        test_loss += loss.item()
test_loss /= n_samp_test
print('====> Test set loss: {:.4f}'.format(test_loss))
# ---------------------------------------------------------------------------------------#