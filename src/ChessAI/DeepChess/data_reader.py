import numpy as np
import chess.pgn
from tqdm import tqdm

def bitboard(board: chess.Board) -> np.ndarray: 
    """
    Converts a chess board to a bitboard representation.

    Args:
        board (chess.Board): A chess board object.
        
    Returns:
        np.ndarray: A bitboard representation of the chess board. (Size: 773)
        
    Examples:
        >>> board = chess.Board()
        >>> bitboard(board)
        array([ 0,  0,  0, ...,  0,  0,  0])    
    """
    
    bitboard = np.zeros(773)
    piece_idx = {
        'p': 0,
        'n': 1,
        'b': 2,
        'r': 3,
        'q': 4,
        'k': 5
    }
    
    for index in range(64): 
        if board.piece_at(index): 
            color = int(board.piece_at(index).color) + 1
            bitboard[(piece_idx[board.piece_at(index).symbol().lower()] + index * 6) * color] = 1

    bitboard[-1] = int(board.turn)
    bitboard[-2] = int(board.has_kingside_castling_rights(True))
    bitboard[-3] = int(board.has_kingside_castling_rights(False))
    bitboard[-4] = int(board.has_queenside_castling_rights(True))
    bitboard[-5] = int(board.has_queenside_castling_rights(False))

    return bitboard

def get_result(game):
    result = game.headers['Result']
    result = result.split('-')
    if result[0] == '1':
        return 1
    elif result[0] == '0':
        return -1
    else:
        return 0

games = open('../../../data/CCRL-Chess324.pgn')
bitboards = []
labels = []
num_games = 0

pbar = tqdm(desc="Reading PGN games")
while num_games < 100: 
    game = chess.pgn.read_game(games)
    if game is None:
        break  # End of file
    num_games += 1
    pbar.update(1)

    result = get_result(game)
    board = game.board()

    for move in game.mainline_moves():
        board.push(move)
        bitboard_ = bitboard(board)

        bitboards.append(bitboard_)
        labels.append(result)

pbar.close()

bitboards = np.array(bitboards)
labels = np.array(labels)


import os
data_folder = '../../../data/'
# Save the bitboards array
bitboards_path = os.path.join(data_folder, 'bitboards.npy')
np.save(bitboards_path, bitboards)
print(f"Bitboards saved to {bitboards_path}")

labels_path = os.path.join(data_folder, 'labels.npy')
np.save(labels_path, labels)
print(f"Labels saved to {labels_path}")