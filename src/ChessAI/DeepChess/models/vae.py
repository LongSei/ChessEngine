from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(773, 600)
        self.fc2 = nn.Linear(600, 500)

        self.fc11 = nn.Linear(500, 500)
        self.fc12 = nn.Linear(500, 500)

        self.fc3 = nn.Linear(400, 200)
        self.fc41 = nn.Linear(200, 100)
        self.fc42 = nn.Linear(200, 100)
        self.fc5 = nn.Linear(100, 200)
        self.fc6 = nn.Linear(200, 400)

        self.fc7 = nn.Linear(500, 600)
        self.fc8 = nn.Linear(600, 773)
        self.bn8 = nn.BatchNorm1d(773)

    def encode(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return self.fc11(x), self.fc12(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = F.tanh(self.fc7(z))
        return F.sigmoid(self.fc8(z))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 773))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 773), size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD