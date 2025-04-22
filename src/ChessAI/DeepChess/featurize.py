from models.autoencoder import AutoEncoder
import numpy as np
import torch

model = AutoEncoder()
state = torch.load('./checkpoints/autoencoder/best_model.pth.tar', map_location=lambda storage, loc: storage)
model.load_state_dict(state['state_dict'])
games = np.load('./../../../data/bitboards.npy')

print(games.shape)
batched_games = np.split(games, 59) 

def featurize(game):
    recon, enc = model(torch.from_numpy(game).type(torch.FloatTensor))
    return enc.detach().numpy()

feat_games = [featurize(batch) for batch in batched_games]
featurized = np.vstack(feat_games)

np.save('./../../../data/features.npy', featurized)