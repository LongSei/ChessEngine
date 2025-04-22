from __future__ import print_function
import argparse
import os 
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, layer=[773, 600, 400, 200, 100]):
        super(AutoEncoder, self).__init__()

        # Encoder
        encoder_layers = []
        for i in range(len(layer) - 1):
            encoder_layers.append(nn.Linear(layer[i], layer[i + 1]))
            encoder_layers.append(nn.BatchNorm1d(layer[i + 1]))
            encoder_layers.append(nn.LeakyReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        for i in range(len(layer) - 1, 0, -1):
            decoder_layers.append(nn.Linear(layer[i], layer[i - 1]))
            decoder_layers.append(nn.BatchNorm1d(layer[i - 1]))
            decoder_layers.append(nn.LeakyReLU())
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.sigmoid = nn.Sigmoid()
    
    def encode(self, x):
        x = x.view(x.size(0), -1)  # Flatten input if needed
        x = self.encoder(x)
        return x
    
    def decode(self, z):
        z = self.decoder(z)
        z = self.sigmoid(z)
        return z
    
    def forward(self, x):
        encoder_out = self.encode(x)
        decoder_out = self.decode(encoder_out)
        return decoder_out, encoder_out

    def loss_function(self, recon_x, x):
        x = x.view(x.size(0), -1)
        return F.binary_cross_entropy(recon_x, x, reduction='sum')
    
class SiameseNetwork(nn.Module):
    def __init__(self, 
                 feature_extractor=AutoEncoder, 
                 feature_extractor_layers=[773, 600, 400, 200, 100], 
                 comparator_layers=[400, 200, 200, 100],
                 output_dim=2):
        super(SiameseNetwork, self).__init__()
        self.feature_extractor = feature_extractor(feature_extractor_layers)
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        comparator = []
        comparator.append(nn.Linear(feature_extractor_layers[-1] * 2, comparator_layers[0]))
        comparator.append(nn.BatchNorm1d(comparator_layers[0]))
        comparator.append(nn.LeakyReLU())

        for i in range(len(comparator_layers) - 1):
            comparator.append(nn.Linear(comparator_layers[i], comparator_layers[i + 1]))
            comparator.append(nn.BatchNorm1d(comparator_layers[i + 1]))
            comparator.append(nn.LeakyReLU())

        comparator.append(nn.Linear(comparator_layers[-1], output_dim))
        self.comparator = nn.Sequential(*comparator)
        
    def forward(self, x1, x2): 
        _, x1_encoded = self.feature_extractor(x1)
        _, x2_encoded = self.feature_extractor(x2)
        x = torch.cat((x1_encoded, x2_encoded), dim=1)
        x = self.comparator(x)
        return F.sigmoid(x)

    def loss_function(self, decode, encode):
        encode = encode.view(encode.size(0), -1)
        return F.binary_cross_entropy(decode, encode, reduction='sum')
    
    def load_pretrained(self, 
                        feature_extractor_path: str=None,
                        comparator_path: str=None, 
                        device: str='cpu'):
        try: 
            if comparator_path != None: 
                if os.path.exists(comparator_path):
                    self.comparator.load_state_dict(torch.load(comparator_path, map_location=device))
                    print(f"Loaded comparator from {comparator_path}")
                
            if feature_extractor_path != None: 
                if os.path.exists(feature_extractor_path):
                    self.feature_extractor.load_state_dict(torch.load(feature_extractor_path, map_location=device))
                    print(f"Loaded feature extractor from {feature_extractor_path}")
                    
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            raise e