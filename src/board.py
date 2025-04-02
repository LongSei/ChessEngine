import os 
import sys
import yaml
from abc import ABC, abstractmethod
from typing import List, Tuple
from piece import Rook, Knight, Bishop, Queen, King, Pawn, Piece

class Board():
    """
    Represents a chess board with pieces.
    """
    def __init__(self):
        self.board_size = 8
        self.board = {}
        self.setup_board()
      
    def setup_board(self):
        for i in range(self.board_size):
            self.board[(1, i)] = Pawn('BLACK')
            self.board[(6, i)] = Pawn('WHITE')
            
        self.board[(0, 0)] = Rook('BLACK')
        self.board[(0, 1)] = Knight('BLACK')
        self.board[(0, 2)] = Bishop('BLACK')
        self.board[(0, 3)] = Queen('BLACK')
        self.board[(0, 4)] = King('BLACK')
        self.board[(0, 5)] = Bishop('BLACK')
        self.board[(0, 6)] = Knight('BLACK')
        self.board[(0, 7)] = Rook('BLACK')
        
        self.board[(7, 0)] = Rook('WHITE')
        self.board[(7, 1)] = Knight('WHITE')
        self.board[(7, 2)] = Bishop('WHITE')    
        self.board[(7, 3)] = Queen('WHITE')
        self.board[(7, 4)] = King('WHITE')
        self.board[(7, 5)] = Bishop('WHITE')
        self.board[(7, 6)] = Knight('WHITE')
        self.board[(7, 7)] = Rook('WHITE')
            

    def get_piece(self, position: Tuple[int, int]) -> Piece:
        """Returns the piece at the given position on the board."""
        if not self.is_valid_position(position):
            raise ValueError("Invalid position")
        return self.board.get(position, None)

    def set_piece(self, new_position: Tuple[int, int], piece: Piece):
        """Sets a piece at the given position."""
        if not self.is_valid_position(new_position):
            raise ValueError("Invalid position")
        self.board[new_position] = piece

    def is_valid_position(self, position: Tuple[int, int]) -> bool:
        """Check if the given position is valid on an 8x8 chessboard."""
        x, y = position
        return 0 <= x < 8 and 0 <= y < 8