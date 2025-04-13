import pytest
from unittest.mock import patch
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from piece import Piece, Knight, Bishop, Queen, King, Pawn, Rook

class TestPieceInitialization:
    """Tests for Piece initialization through its subclasses."""
    
    def test_valid_initialization(self):
        knight = Knight(color='WHITE', position=(0, 0))
        assert knight.get_color() == "WHITE"
        assert knight.get_piece_type() == "KNIGHT"
    
    def test_invalid_color_raises_error(self):
        with pytest.raises(ValueError, match="Invalid color 'red'"):
            Knight(color='red', position=(0, 0))
    
    def test_invalid_piece_type_raises_error(self):
        # This test is theoretical since subclasses set their own type
        # Testing via a hypothetical subclass with invalid type
        class InvalidPiece(Piece):
            def __init__(self, color: str):
                super().__init__(color=color, piece_type='INVALID', position=(0, 0))
            def get_valid_move(self, current_position):
                return []
        with pytest.raises(ValueError, match="Invalid piece type 'INVALID'"):
            InvalidPiece(color='WHITE')

class TestPieceEquality:
    """Tests for equality checks between Piece instances."""
    
    def test_equality_same_pieces(self):
        knight1 = Knight(color='WHITE', position=(0, 0))
        knight2 = Knight(color='WHITE', position=(0, 0))
        assert knight1 == knight2
    
    def test_inequality_different_type(self):
        knight = Knight(color='WHITE', position=(0, 0))
        bishop = Bishop(color='WHITE', position=(0, 0))
        assert knight != bishop
    
    def test_inequality_different_color(self):
        knight1 = Knight(color='WHITE', position=(0, 0))
        knight2 = Knight(color='BLACK', position=(0, 0))
        assert knight1 != knight2

class TestKnightMoves:
    """Tests for Knight's valid moves."""
    
    @pytest.mark.parametrize("position, expected_moves", [
        ((0, 0), [(1, 2), (2, 1)]),
        ((3, 3), [(1, 2), (1, 4), (2, 1), (2, 5), (4, 1), (4, 5), (5, 2), (5, 4)])
    ])
    def test_valid_moves(self, position, expected_moves):
        knight = Knight(color='WHITE', position=position)
        moves = knight.get_valid_move()
        assert sorted(moves) == sorted(expected_moves)

class TestBishopMoves:
    """Tests for Bishop's valid moves."""
    
    def test_valid_moves_center(self):
        bishop = Bishop(color='WHITE', position=(3, 3))
        moves = bishop.get_valid_move()
        expected = [(4,4), (5,5), (6,6), (7,7), (4,2), (5,1), (6,0), (2,4), (1,5), (0,6), (2,2), (1,1), (0,0)]
        assert sorted(moves) == sorted(expected)
    
    def test_valid_moves_corner(self):
        bishop = Bishop(color='WHITE', position=(0, 0))
        moves = bishop.get_valid_move()
        expected = [(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7)]
        assert sorted(moves) == sorted(expected)

class TestQueenMoves:
    """Tests for Queen's valid moves."""
    
    def test_valid_moves_center(self):
        queen = Queen(color='WHITE', position=(3, 3))
        moves = queen.get_valid_move()
        # Combines rook and bishop moves from (3,3)
        expected = [
            (4,3), (5,3), (6,3), (7,3), (2,3), (1,3), (0,3),  # Horizontal
            (3,4), (3,5), (3,6), (3,7), (3,2), (3,1), (3,0),  # Vertical
            (4,4), (5,5), (6,6), (7,7), (4,2), (5,1), (6,0),  # Diagonals
            (2,4), (1,5), (0,6), (2,2), (1,1), (0,0)
        ]
        assert sorted(moves) == sorted(expected)

class TestKingMoves:
    """Tests for King's valid moves."""
    
    def test_valid_moves_center(self):
        king = King(color='WHITE', position=(3, 3))
        moves = king.get_valid_move()
        expected = [(2,2), (2,3), (2,4), (3,2), (3,4), (4,2), (4,3), (4,4)]
        assert sorted(moves) == sorted(expected)
    
    def test_valid_moves_edge(self):
        king = King(color='WHITE', position=(0, 0))
        moves = king.get_valid_move()
        expected = [(0,1), (1,0), (1,1)]
        assert sorted(moves) == sorted(expected)

class TestPawnMoves:
    """Tests for Pawn's valid moves."""
    
    def test_white_pawn_forward(self):
        pawn = Pawn(color='WHITE', position=(1, 1))
        moves = pawn.get_valid_move()
        assert moves == [(0, 1)]
    
    def test_black_pawn_forward(self):
        pawn = Pawn(color='BLACK', position=(6, 1))
        moves = pawn.get_valid_move()
        assert moves == [(7, 1)]

class TestRookMoves:
    """Tests for Rook's valid moves."""
    
    def test_valid_moves_center(self):
        rook = Rook(color='WHITE', position=(3, 3))
        moves = rook.get_valid_move()
        expected = [(i,3) for i in range(8) if i != 3] + [(3,j) for j in range(8) if j != 3]
        assert sorted(moves) == sorted(expected)

class TestMoveValidation:
    """Tests for move validation and execution."""
    
    def test_valid_move_updates_position(self):
        knight = Knight(color='WHITE', position=(0, 0))
        knight.move_to((1, 2))   
        assert knight.get_position() == (1, 2)
    
    def test_invalid_move_raises_error(self):
        knight = Knight(color='WHITE', position=(0, 0))
        with pytest.raises(ValueError, match="Invalid move"):
            knight.move_to((3, 3))