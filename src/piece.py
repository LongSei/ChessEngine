import os 
import sys
import yaml
from abc import ABC, abstractmethod
from typing import List, Tuple

class Piece(ABC):
    """
    Abstract base class for chess pieces.
    """
    CONFIG_PATH = './config/config.yaml'

    @classmethod
    def load_config(cls):
        """Configuration loading method."""
        try:
            with open(cls.CONFIG_PATH, 'r') as file:
                config_data = yaml.safe_load(file)
                piece_config_path = config_data.get('PIECE_CONFIGURATION', '')

                if not piece_config_path:
                    raise ValueError("Missing PIECE_CONFIGURATION in config.yaml")

                piece_config_file = piece_config_path

                with open(piece_config_file, 'r') as pc_file:
                    piece_config = yaml.safe_load(pc_file)

                return piece_config
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Config file not found: {e.filename}") from e
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    def __init_subclass__(cls, **kwargs):
        """Load configuration for subclasses."""
        super().__init_subclass__(**kwargs)
        cls.CONFIG = cls.load_config()
        cls.COLOR_MAP = cls.CONFIG['PIECE_COLOR']
        cls.PIECE_TYPE_MAP = cls.CONFIG['PIECE_TYPE']
    
    def __init__(self, 
                 color: str=None, 
                 piece_type: str=None, 
                 position: Tuple[int, int]=None):
        """
        Initialize a piece with type and color.
        
        Args:
            color (str): Color of the piece ('WHITE' or 'black').
            piece_type (str): Type of the piece (e.g., 'knight', 'bishop').
            position (Tuple[int, int]): Position of the piece on the board.
            
        Example:
            >>> piece = Piece(color='WHITE', piece_type='knight', position=(0, 1))
            >>> assert piece.get_color == 'WHITE'
            ... True
            >>> assert piece.get_piece_type == 'knight'
            ... True
            >>> assert piece.get_position == (0, 1)
            ... True
        """
            
        if color not in self.COLOR_MAP:
            raise ValueError(f"Invalid color '{color}', must be one of {list(self.COLOR_MAP.keys())}")
        if piece_type not in self.PIECE_TYPE_MAP:
            raise ValueError(f"Invalid piece type '{piece_type}', must be one of {list(self.PIECE_TYPE_MAP.keys())}")

        self.color = self.COLOR_MAP[color]
        self.piece_type = self.PIECE_TYPE_MAP[piece_type]
        self.position = position
    
    def __eq__(self, other):
        """
        Check if two pieces are equal based on their type and color.
        This method overrides the default equality operator to compare pieces.

        Args:
            other (Piece): Another piece to compare with.

        Returns:
            bool: True if both pieces are equal, False otherwise.
            
        Example: 
            >>> piece1 = Piece(color='WHITE', piece_type='knight')
            >>> piece2 = Piece(color='WHITE', piece_type='knight')
            >>> piece3 = Piece(color='black', piece_type='bishop')
            >>> assert piece1 == piece2  
            ... True
            >>> assert piece1 == piece3
            ... False
        """
        if not isinstance(other, Piece):
            raise ValueError("Cannot compare Piece with non-Piece object")
        return self.piece_type == other.piece_type and self.color == other.color
    
    @property
    def get_piece_type(self) -> str: 
        return self.piece_type
    
    @property
    def get_position(self) -> Tuple[int, int]: 
        return self.position
    
    @property
    def get_color(self) -> str: 
        return self.color
    
    @abstractmethod
    def get_valid_move(self) -> List[Tuple[int, int]]: 
        """
        Get valid moves for the piece.
        
        Returns:
            List: A list of valid moves for the piece.
        """
        pass
    
    def is_valid_move(self, new_position: Tuple[int, int]) -> bool:
        """
        Check if the move to a new position is valid.

        Args:
            new_position (Tuple[int, int]): New position to check.
            
        Example:
            >>> piece = Piece(color='WHITE', piece_type='knight', position=(0, 1))
            >>> assert piece.is_valid_move((2, 2)) == True
            ... True
        """
        valid_moves = self.get_valid_move()
        return new_position in valid_moves
    
    def move_to(self, new_position: Tuple[int, int]): 
        """
        Move the piece to a new position.

        Args:
            new_position (Tuple[int, int]): New position to move the piece to.
            
        Example:
            >>> piece = Piece(color='WHITE', piece_type='knight', position=(0, 1))
            >>> piece.move_to((2, 2))
            >>> assert piece.get_position == (2, 2)
            ... True
        """
        if self.is_valid_move(new_position):
            self.position = new_position
        else:
            raise ValueError("Invalid move for this piece")
        
class Knight(Piece): 
    def __init__(self, color: str, position: Tuple[int, int]):
        super().__init__(color=color, piece_type='KNIGHT', position=position)

    def get_valid_move(self) -> List[Tuple[int, int]]:
        """
        Get valid moves for the knight piece.

        Returns:
            List[Tuple[int, int]]: A list of valid moves for the knight.
            
        Example:
            >>> knight = Knight(color='WHITE', position=(0, 1))
            >>> print(knight.get_valid_move())
            ... [(2, 2), (2, 0), ...]
        """
        x, y = self.position
        moves = [(x + 2, y + 1), (x + 2, y - 1), (x - 2, y + 1), (x - 2, y - 1),
                 (x + 1, y + 2), (x + 1, y - 2), (x - 1, y + 2), (x - 1, y - 2)]
        return [(i, j) for i, j in moves if 0 <= i < 8 and 0 <= j < 8]
    
class Bishop(Piece): 
    def __init__(self, color: str, position: Tuple[int, int]):
        super().__init__(color=color, piece_type='BISHOP', position=position)

    def get_valid_move(self) -> List[Tuple[int, int]]:
        """
        Get valid moves for the bishop piece.

        Returns:
            List[Tuple[int, int]]: A list of valid moves for the bishop.
            
        Example:
            >>> bishop = Bishop(color='WHITE', position=(0, 1))
            >>> print(bishop.get_valid_move())
            ... [(1, 2), (2, 3), ...]
        """
        x, y = self.position
        moves = []
        for i in range(1, 8):
            moves.append((x + i, y + i))
            moves.append((x + i, y - i))
            moves.append((x - i, y + i))
            moves.append((x - i, y - i))
        return [(i, j) for i, j in moves if 0 <= i < 8 and 0 <= j < 8]
    
class Queen(Piece): 
    def __init__(self, color: str, position: Tuple[int, int]):
        super().__init__(color=color, piece_type='QUEEN', position=position)

    def get_valid_move(self) -> List[Tuple[int, int]]:
        """
        Get valid moves for the queen piece.

        Returns:
            List[Tuple[int, int]]: A list of valid moves for the queen.
            
        Example:
            >>> queen = Queen(color='WHITE', position=(0, 1))
            >>> print(queen.get_valid_move())
            ... [(1, 2), (2, 3), ...]
        """
        x, y = self.position
        moves = []
        for i in range(1, 8):
            moves.append((x + i, y))
            moves.append((x - i, y))
            moves.append((x, y + i))
            moves.append((x, y - i))
            moves.append((x + i, y + i))
            moves.append((x + i, y - i))
            moves.append((x - i, y + i))
            moves.append((x - i, y - i))
        return [(i, j) for i, j in moves if 0 <= i < 8 and 0 <= j < 8]
    
class King(Piece):
    def __init__(self, color: str, position: Tuple[int, int]):
        super().__init__(color=color, piece_type='KING', position=position)
        
    def get_valid_move(self) -> List[Tuple[int, int]]:
        """
        Get valid moves for the king piece.

        Returns:
            List[Tuple[int, int]]: A list of valid moves for the king.
            
        Example:
            >>> king = King(color='WHITE', position=(0, 1))
            >>> print(king.get_valid_move())
            ... [(0, 2), (1, 2), ...]
        """
        x, y = self.position
        moves = [(x + dx, y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if (dx != 0 or dy != 0)]
        return [(i, j) for i, j in moves if 0 <= i < 8 and 0 <= j < 8]
    
class Pawn(Piece):
    def __init__(self, color: str, position: Tuple[int, int]):
        super().__init__(color=color, piece_type='PAWN', position=position)
        
    def get_valid_move(self) -> List[Tuple[int, int]]:
        """
        Get valid moves for the pawn piece.

        Returns:
            List[Tuple[int, int]]: A list of valid moves for the pawn.
            
        Example:
            >>> pawn = Pawn(color='WHITE', position=(1, 1))
            >>> print(pawn.get_valid_move())
            ... [(2, 1), ...]
        """
        x, y = self.position
        direction = 1 if self.color == 1 else -1
        return [(x + direction, y)]
    
class Rook(Piece):
    def __init__(self, color: str, position: Tuple[int, int]):
        super().__init__(color=color, piece_type='ROOK', position=position)
        
    def get_valid_move(self) -> List[Tuple[int, int]]:
        """
        Get valid moves for the rook piece.

        Returns:
            List[Tuple[int, int]]: A list of valid moves for the rook.
            
        Example:
            >>> rook = Rook(color='WHITE', position=(0, 1))
            >>> print(rook.get_valid_move())
            ... [(0, 2), (1, 1), ...]
        """
        x, y = self.position
        moves = []
        for i in range(8):
            if i != x:
                moves.append((i, y))
            if i != y:
                moves.append((x, i))
        return [(i, j) for i, j in moves if 0 <= i < 8 and 0 <= j < 8]