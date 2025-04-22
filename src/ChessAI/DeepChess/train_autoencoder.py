from __future__ import print_function
import argparse, os
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from models.autoencoder import AutoEncoder

# ------------------- Args ---------------------
parser = argparse.ArgumentParser(description='AutoEncoder Chess Bitboards')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--decay', type=float, default=.95)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--cuda', action='store_true', default=False, help='Enable CUDA if available')
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

args.cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
torch.manual_seed(args.seed)

# ------------------- Data ---------------------
games = np.load('./../../../data/bitboards.npy')
np.random.shuffle(games)
split = int(len(games) * 0.8)
train_games, test_games = games[:split], games[split:]

class ChessDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float(), 1
    def __len__(self):
        return len(self.data)

train_loader = DataLoader(ChessDataset(train_games), batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(ChessDataset(test_games), batch_size=args.batch_size, shuffle=False)

# ------------------- Model & Optimizer ---------------------
model = AutoEncoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# ------------------- Loss functions ---------------------
def loss_function(recon_x, x):
    return F.binary_cross_entropy(recon_x, x.view(-1, 773), reduction='sum')

def mse_loss_function(recon_x, x):
    return F.mse_loss(recon_x, x.view(-1, 773), reduction='sum')

# ------------------- Train/Test ---------------------
def train(epoch):
    model.train()
    total_loss = 0
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon, _ = model(data)
        loss = loss_function(recon, data)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    return total_loss / len(train_loader.dataset)

def test():
    model.eval()
    total_loss, total_mse, total_diff = 0, 0, 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon, _ = model(data)
            pred = (recon.cpu().numpy() > 0.5).astype(int)
            total_diff += np.sum(data.cpu().numpy() != pred)
            total_loss += loss_function(recon, data).item()
            total_mse += mse_loss_function(recon, data).item()
    n = len(test_loader.dataset)
    return total_loss / n, total_mse / n, total_diff / n

# ------------------- Save ---------------------
def save_model(epoch):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch + 1
    }
    save_dir = f'./checkpoints/autoencoder/lr_{int(args.lr*1000)}_decay_{int(args.decay*100)}'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state, os.path.join(save_dir, f'ae_{epoch}.pth.tar'))

# ------------------- Main loop ---------------------
start_epoch = 1
resume = False
if resume:
    state = torch.load('./checkpoints/best_autoencoder.pth.tar', map_location='cpu')
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    start_epoch = state['epoch']

# ------------------- Early stopping config ---------------------
best_loss = float('inf')
patience = 5
counter = 0
early_stop = False

for epoch in tqdm(range(start_epoch, args.epochs + 1), desc="Epochs", position=0):
    train_loss = train(epoch)
    test_loss, test_mse, diff = test()

    tqdm.write(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | MSE: {test_mse:.4f} | Diff: {diff:.4f}")

    if test_loss < best_loss:
        best_loss = test_loss
        counter = 0
        # Save best model
        os.makedirs('./checkpoints/autoencoder', exist_ok=True)
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, './checkpoints/autoencoder/best_model.pth.tar')
    else:
        counter += 1
        if counter >= patience:
            tqdm.write(f"Early stopping triggered at epoch {epoch}. Best test loss: {best_loss:.4f}")
            early_stop = True
            break

    # Save current epoch model
    save_model(epoch)

    # Update LR
    for param_group in optimizer.param_groups:
        param_group['lr'] *= args.decay