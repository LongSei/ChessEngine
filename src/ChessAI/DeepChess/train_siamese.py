from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.utils import save_image
from tqdm import tqdm

from models.siamese import Siamese

# ------------------- Argument parser ------------------- #
parser = argparse.ArgumentParser(description='Siamese Training on Chess Data')
parser.add_argument('--batch-size', type=int, default=128, metavar='N')
parser.add_argument('--lr', type=float, default=0.01, metavar='N')
parser.add_argument('--decay', type=float, default=0.99, metavar='N')
parser.add_argument('--epochs', type=int, default=10, metavar='N')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='S')
parser.add_argument('--log-interval', type=int, default=100, metavar='N')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# ------------------- Setup ------------------- #
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
lr = args.lr
decay = args.decay
batch_size = args.batch_size

# ------------------- Load Data ------------------- #
print('Loading data...')
games = np.load('./../../../data/features.npy')
wins = np.load('./../../../data/labels.npy')

p = np.random.permutation(len(wins))
games = games[p]
wins = wins[p]

split_idx = int(len(games) * 0.8)
train_games, test_games = games[:split_idx], games[split_idx:]
train_wins, test_wins = wins[:split_idx], wins[split_idx:]

train_games_wins = train_games[train_wins == 1]
train_games_losses = train_games[train_wins == -1]
test_games_wins = test_games[test_wins == 1]
test_games_losses = test_games[test_wins == -1]

# ------------------- Dataset Classes ------------------- #
class TrainSet(Dataset):
    def __init__(self, length):
        self.length = length

    def __getitem__(self, index):
        win = train_games_wins[np.random.randint(len(train_games_wins))]
        loss = train_games_losses[np.random.randint(len(train_games_losses))]
        if np.random.rand() > 0.5:
            stacked = np.hstack((win, loss))
            label = np.array([1, 0])
        else:
            stacked = np.hstack((loss, win))
            label = np.array([0, 1])
        return torch.FloatTensor(stacked), torch.FloatTensor(label)

    def __len__(self):
        return self.length

class TestSet(Dataset):
    def __init__(self, length):
        self.length = length

    def __getitem__(self, index):
        win = test_games_wins[np.random.randint(len(test_games_wins))]
        loss = test_games_losses[np.random.randint(len(test_games_losses))]
        if np.random.rand() > 0.5:
            stacked = np.hstack((win, loss))
            label = np.array([1, 0])
        else:
            stacked = np.hstack((loss, win))
            label = np.array([0, 1])
        return torch.FloatTensor(stacked), torch.FloatTensor(label)

    def __len__(self):
        return self.length

train_loader = torch.utils.data.DataLoader(TrainSet(1000000), batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(TestSet(100000), batch_size=batch_size, shuffle=True, **kwargs)

# ------------------- Model ------------------- #
print('Building model...')
model = Siamese().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# ------------------- Loss Function ------------------- #
def loss_function(pred, label):
    return F.binary_cross_entropy(pred, label, reduction='sum')

# ------------------- Training ------------------- #
def train(epoch):
    model.train()
    train_loss = 0
    loader = tqdm(train_loader, desc=f"Train Epoch {epoch}")
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = loss_function(pred, label)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            loader.set_postfix(loss=loss.item() / len(data))

    avg_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')

# ------------------- Testing ------------------- #
def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        loader = tqdm(test_loader, desc=f"Test Epoch {epoch}")
        for data, label in loader:
            data, label = data.to(device), label.to(device)
            pred = model(data)
            loss = loss_function(pred, label)
            test_loss += loss.item()
    avg_loss = test_loss / len(test_loader.dataset)
    print(f'====> Test set loss: {avg_loss:.4f}')

# ------------------- Accuracy (Optional) ------------------- #
def get_acc():
    correct = 0
    with torch.no_grad():
        for data, label in tqdm(test_loader, desc="Evaluating Accuracy"):
            pred = model(data.to(device))
            correct += np.sum((pred > 0.5).cpu().numpy() * label.numpy())
    return correct / float(len(test_loader.dataset))

# ------------------- Save Checkpoint ------------------- #
def save(epoch):
    save_dir = f'checkpoints/siamese/lr_{int(lr*1000)}_decay_{int(decay*100)}'
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch + 1
    }, os.path.join(save_dir, f'siamese_{epoch}.pth.tar'))

# ------------------- Main Loop ------------------- #
start_epoch = 1
resume = False

if resume:
    state = torch.load('./checkpoints/best_siamese.pth.tar', map_location=lambda storage, loc: storage)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    start_epoch = state['epoch']
# ------------------- Early Stopping Config ------------------- #
best_loss = float('inf')
patience = 5
counter = 0
early_stop = False

print('Begin training...')
for epoch in range(start_epoch, args.epochs + 1):
    train(epoch)
    test_loss = 0
    model.eval()
    with torch.no_grad():
        loader = tqdm(test_loader, desc=f"Evaluating Epoch {epoch}")
        for data, label in loader:
            data, label = data.to(device), label.to(device)
            pred = model(data)
            loss = loss_function(pred, label)
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader.dataset)
    print(f'====> Test Loss: {avg_test_loss:.4f}')

    # Early stopping logic
    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        counter = 0
        # Save best model
        os.makedirs('./checkpoints/siamese', exist_ok=True)
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, './checkpoints/siamese/best_model.pth.tar')
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Best test loss: {best_loss:.4f}")
            early_stop = True
            break

    # Save current epoch model
    save(epoch)

    # Decay learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay