import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from board import Board
from piece import Rook, Knight, Bishop, Queen, King, Pawn

@pytest.fixture
def setup_board():
    """Fixture to set up the board before each test."""
    return Board()

def test_initialization(setup_board):
    """Test that the board is correctly initialized with pieces."""
    board = setup_board
    
    # Test pieces for BLACK side
    assert isinstance(board.get_piece((0, 0)), Rook)
    assert isinstance(board.get_piece((0, 1)), Knight)
    assert isinstance(board.get_piece((0, 2)), Bishop)
    assert isinstance(board.get_piece((0, 3)), Queen)
    assert isinstance(board.get_piece((0, 4)), King)
    assert isinstance(board.get_piece((0, 5)), Bishop)
    assert isinstance(board.get_piece((0, 6)), Knight)
    assert isinstance(board.get_piece((0, 7)), Rook)
    
    # Test pieces for WHITE side
    assert isinstance(board.get_piece((7, 0)), Rook)
    assert isinstance(board.get_piece((7, 1)), Knight)
    assert isinstance(board.get_piece((7, 2)), Bishop)
    assert isinstance(board.get_piece((7, 3)), Queen)
    assert isinstance(board.get_piece((7, 4)), King)
    assert isinstance(board.get_piece((7, 5)), Bishop)
    assert isinstance(board.get_piece((7, 6)), Knight)
    assert isinstance(board.get_piece((7, 7)), Rook)

    # Test pawns on 2nd and 7th ranks
    for col in range(8):
        assert isinstance(board.get_piece((1, col)), Pawn)
        assert isinstance(board.get_piece((6, col)), Pawn)

def test_get_piece_valid_position(setup_board):
    """Test the get_piece method on valid positions."""
    board = setup_board
    piece = board.get_piece((0, 0))  # Rook at position (0, 0)
    assert isinstance(piece, Rook)
    assert piece.position == (0, 0)

def test_get_piece_invalid_position(setup_board):
    """Test the get_piece method on invalid positions."""
    board = setup_board
    with pytest.raises(ValueError):
        board.get_piece((8, 8))  # Invalid position outside of the 8x8 board

def test_set_piece_valid_position(setup_board):
    """Test the set_piece method on valid positions."""
    board = setup_board
    new_queen = Queen('WHITE', (7, 4))
    board.set_piece((7, 4), new_queen)
    piece = board.get_piece((7, 4))
    assert isinstance(piece, Queen)
    assert piece.position == (7, 4)

def test_set_piece_invalid_position(setup_board):
    """Test the set_piece method on invalid positions."""
    board = setup_board
    new_queen = Queen('WHITE', (7, 4))
    with pytest.raises(ValueError):
        board.set_piece((8, 8), new_queen)  # Invalid position outside of the 8x8 board

def test_is_valid_position(setup_board):
    """Test the is_valid_position method."""
    board = setup_board
    assert board.is_valid_position((0, 0))  # Valid position on the board
    assert board.is_valid_position((7, 7))  # Valid position on the board
    assert not board.is_valid_position((8, 8))  # Invalid position outside of the 8x8 board
    assert not board.is_valid_position((-1, -1))  # Invalid position outside of the 8x8 board

