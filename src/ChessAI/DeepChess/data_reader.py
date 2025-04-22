import numpy as np
import chess.pgn
import os
from tqdm import tqdm

def bitboard(board: chess.Board) -> np.ndarray: 
    bitboard = np.zeros(773)
    piece_idx = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}
    for index in range(64): 
        piece = board.piece_at(index)
        if piece: 
            color = int(piece.color) + 1
            bitboard[(piece_idx[piece.symbol().lower()] + index * 6) * color] = 1

    bitboard[-1] = int(board.turn)
    bitboard[-2] = int(board.has_kingside_castling_rights(True))
    bitboard[-3] = int(board.has_kingside_castling_rights(False))
    bitboard[-4] = int(board.has_queenside_castling_rights(True))
    bitboard[-5] = int(board.has_queenside_castling_rights(False))
    return bitboard

def get_result(game):
    result = game.headers['Result'].split('-')[0]
    return {'1': 1, '0': -1}.get(result, 0)

# Setup
data_folder = '../../../data/'
bitboards_path = os.path.join(data_folder, 'bitboards.npy')
labels_path = os.path.join(data_folder, 'labels.npy')

# Remove old files if any
if os.path.exists(bitboards_path):
    os.remove(bitboards_path)
if os.path.exists(labels_path):
    os.remove(labels_path)

print("Loading PGN file with batch...")
games = open('../../../data/CCRL-Chess324.pgn')

pbar = tqdm(desc="Reading PGN games")
games_with_moves = 0
batch_size = 2000000
bitboard_batch = []
label_batch = []

while True: 
    game = chess.pgn.read_game(games)
    if game is None:
        break

    result = get_result(game)
    board = game.board()

    for move in game.mainline_moves():
        board.push(move)
        bitboard_ = bitboard(board)
        if bitboard_.sum() == 0:
            continue
        bitboard_batch.append(bitboard_)
        label_batch.append(result)

    games_with_moves += 1
    pbar.update(1)
    
    if games_with_moves % 500 == 0:
        print(f"Processed {len(bitboard_batch)} bitboards...")

    if len(bitboard_batch) >= batch_size:
        print(f"Writing batch of {len(bitboard_batch)} games to disk...")
        np.save(bitboards_path, np.array(bitboard_batch), allow_pickle=False) if not os.path.exists(bitboards_path) \
            else np.save(bitboards_path, np.concatenate((np.load(bitboards_path, mmap_mode='r'), bitboard_batch)))
        np.save(labels_path, np.array(label_batch), allow_pickle=False) if not os.path.exists(labels_path) \
            else np.save(labels_path, np.concatenate((np.load(labels_path, mmap_mode='r'), label_batch)))
        bitboard_batch = []
        label_batch = []
        
    if games_with_moves % 3000 == 0:
        break

if bitboard_batch:
    np.save(bitboards_path, np.array(bitboard_batch), allow_pickle=False) if not os.path.exists(bitboards_path) \
        else np.save(bitboards_path, np.concatenate((np.load(bitboards_path, mmap_mode='r'), bitboard_batch)))
    np.save(labels_path, np.array(label_batch), allow_pickle=False) if not os.path.exists(labels_path) \
        else np.save(labels_path, np.concatenate((np.load(labels_path, mmap_mode='r'), label_batch)))

pbar.close()
print(f"Done. Bitboards saved to {bitboards_path}, Labels saved to {labels_path}")