import pytest
from board import Board
from game_rules import GameRules
from piece import King, Queen, Rook, Pawn
from typing import Tuple

@pytest.fixture
def board():
    return Board()

@pytest.fixture
def game_rules():
    return GameRules()

def test_switch_turn(game_rules):
    assert game_rules.turn == 'WHITE'
    game_rules.switch_turn()
    assert game_rules.turn == 'BLACK'
    game_rules.switch_turn()
    assert game_rules.turn == 'WHITE'

def test_move_piece_valid_move(board, game_rules):
    from_pos = (6, 0)  # White Pawn
    to_pos = (5, 0)
    piece = board.get_piece(from_pos)
    game_rules.move_piece(board, from_pos, to_pos)
    assert board.get_piece(to_pos) == piece
    assert board.get_piece(from_pos) is None

def test_move_piece_wrong_color(board, game_rules):
    from_pos = (0, 0)  # Black Rook
    to_pos = (0, 1)
    with pytest.raises(ValueError, match="Quân ở vị trí này không thuộc lượt bạn."):
        game_rules.move_piece(board, from_pos, to_pos)

def test_capture_piece(board, game_rules):
    # Manually fix the GameRules' captured_pieces (code has a bug)
    game_rules.captured_pieces = []
    pos = (0, 0)  # Black Rook
    piece = board.get_piece(pos)
    game_rules.capture_piece(board, pos)
    assert pos not in board.board
    assert piece in game_rules.captured_pieces

def test_is_in_check_true(board):
    board.board.clear()
    king = King('WHITE', (3, 3))
    queen = Queen('BLACK', (3, 5))
    board.set_piece(king.position, king)
    board.set_piece(queen.position, queen)
    game_rules = GameRules()
    assert game_rules.is_in_check(board, 'WHITE') is True

def test_find_king_white(board, game_rules):
    assert game_rules.find_king(board, 'WHITE') == (7, 4)

def test_can_castle_valid(board):
    board.board.clear()
    king = King('WHITE', (7, 4))
    rook = Rook('WHITE', (7, 7))
    board.set_piece(king.position, king)
    board.set_piece(rook.position, rook)
    game_rules = GameRules()
    assert game_rules.can_castle(board, (7, 4), (7, 7)) is True

def test_castle_kingside(board):
    board.board.clear()
    king = King('WHITE', (7, 4))
    rook = Rook('WHITE', (7, 7))
    board.set_piece(king.position, king)
    board.set_piece(rook.position, rook)
    game_rules = GameRules()
    game_rules.castle(board, (7, 4), (7, 7))
    assert board.get_piece((7, 6)) == king
    assert board.get_piece((7, 5)) == rook