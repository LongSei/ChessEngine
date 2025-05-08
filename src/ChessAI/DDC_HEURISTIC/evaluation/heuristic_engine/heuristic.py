from .score import ChessScore
from .utils import load_yaml
import chess

class HeuristicEngine():
    def __init__(self):
        self.config = load_yaml('./config/heuristic_configs.yaml')
        self.chess_score = ChessScore(self.config)
        
    def evaluate(self, board):
        """
        Evaluate the chess position using heuristic methods.

        Args:
            board (chess.Board): The chess board.
        Returns:
            int: The score for the current position.
            
        Example:
            >>> board = chess.Board()
            >>> engine = HeuristicEngine()
            >>> score = engine.evaluate(board, chess.WHITE)
            >>> print(score)
            ... 1234
        """
        return self.chess_score.improved_stockfish_eval(board)
    
    def minimax_with_alpha_beta(self, 
                                board: chess.Board, 
                                depth: int=3,
                                alpha: float=-float('inf'),
                                beta: float=float('inf'),
                                maximizing_player: bool=True) -> float: 
        """
        Perform the Minimax algorithm with alpha-beta pruning to evaluate the chess position.

        Args:
            board (chess.Board): The chess board to evaluate.
            depth (int, optional): Defaults to 3.
            alpha (float, optional): Defaults to -float('inf').
            beta (float, optional): Defaults to float('inf').
            maximizing_player (bool, optional): Defaults to True.

        Returns:
            float: The evaluation score for the current position.
        """
        if depth == 0 or board.is_game_over():
            return self.evaluate(board)
        best_value = -float('inf') if maximizing_player else float('inf')
        for move in board.legal_moves:
            board.push(move)
            val = self.minimax_with_alpha_beta(board, depth-1, alpha, beta, not maximizing_player)
            board.pop()
            if maximizing_player:
                best_value = max(best_value, val)
                alpha = max(alpha, best_value)
            else:
                best_value = min(best_value, val)
                beta = min(beta, best_value)
            if beta <= alpha:
                break
        return best_value
    
    def get_best_move(self, board: chess.Board, depth: int=3) -> chess.Move:
        """
        Get the best move for the current position using Minimax with alpha-beta pruning.

        Args:
            board (chess.Board): The chess board to evaluate.
            depth (int, optional): Defaults to 3.

        Returns:
            chess.Move: The best move for the current position.
        """
        best_move = None
        best_value = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            val = self.minimax_with_alpha_beta(board, depth-1, -float('inf'), float('inf'), False)
            board.pop()
            if val > best_value:
                best_value = val
                best_move = move
        return best_move
