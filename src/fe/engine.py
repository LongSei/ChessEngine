import chess
import random

def get_engine_move(board: chess.Board) -> chess.Move:
    legal_moves = list(board.legal_moves)
    return random.choice(legal_moves)