from typing import Tuple, Optional
from piece import Rook, Knight, Bishop, Queen, King, Pawn, Piece

class Board:
    """
    Represents a chess board with pieces.
    """
    def __init__(self):
        self.board_size = 8
        self.board = {}
        self.captured_pieces = {
            'WHITE': [],
            'BLACK': []
        }
        self.setup_board()
      
    def setup_board(self) -> None:
        """
        Initializes the board with pieces in their starting positions.
        """
        for col in range(self.board_size):
            self.set_piece((1, col), Pawn('BLACK', (1, col)))
            self.set_piece((6, col), Pawn('WHITE', (6, col)))
        
        self.set_piece((0, 0), Rook('BLACK', (0, 0)))
        self.set_piece((0, 1), Knight('BLACK', (0, 1)))
        self.set_piece((0, 2), Bishop('BLACK', (0, 2)))
        self.set_piece((0, 3), Queen('BLACK', (0, 3)))
        self.set_piece((0, 4), King('BLACK', (0, 4)))
        self.set_piece((0, 5), Bishop('BLACK', (0, 5)))
        self.set_piece((0, 6), Knight('BLACK', (0, 6)))
        self.set_piece((0, 7), Rook('BLACK', (0, 7)))
        
        self.set_piece((7, 0), Rook('WHITE', (7, 0)))
        self.set_piece((7, 1), Knight('WHITE', (7, 1)))
        self.set_piece((7, 2), Bishop('WHITE', (7, 2)))    
        self.set_piece((7, 3), Queen('WHITE', (7, 3)))
        self.set_piece((7, 4), King('WHITE', (7, 4)))
        self.set_piece((7, 5), Bishop('WHITE', (7, 5)))
        self.set_piece((7, 6), Knight('WHITE', (7, 6)))
        self.set_piece((7, 7), Rook('WHITE', (7, 7)))

    def get_piece(self, 
                  position: Tuple[int, int]) -> Optional[Piece]:
        """
        Returns the piece at the given position on the board.
        
        Args: 
            position (Tuple[int, int]): The position on the board.
            
        Returns:
            Optional[Piece]: The piece at the position, or None if empty.
        Raises:
            ValueError: If the position is invalid.
        """
        if not self.is_valid_position(position):
            raise ValueError("Invalid position")
        return self.board.get(position)

    def set_piece(self, new_position: Tuple[int, int], piece: Piece) -> None:
        """
        Places a piece at the specified position and updates its position attribute.
        
        Args:
            new_position (Tuple[int, int]): The position on the board.
            piece (Piece): The piece to place.
            
        Raises:
            ValueError: If the position is invalid.
        """
        if not self.is_valid_position(new_position):
            raise ValueError("Invalid position")
        self.board[new_position] = piece
        piece.position = new_position  

    def is_valid_position(self, position: Tuple[int, int]) -> bool:
        """
        Checks if a position is within the bounds of the chessboard.
        
        Args: 
            position (Tuple[int, int]): The position to check.
            
        Returns:
            bool: True if the position is valid, False otherwise.
        """
        x, y = position
        return 0 <= x < self.board_size and 0 <= y < self.board_size