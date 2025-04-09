from typing import Tuple, Optional, List
from board import Board
from piece import Piece, King, Rook, Pawn  

class GameRules:
    def __init__(self):
        self.turn = 'WHITE'      

    def switch_turn(self) -> None:
        """
        Switch the turn to the other player.
        """
        self.turn = 'WHITE' if self.turn == 'BLACK' else 'BLACK'

    def move_piece(self, board: Board, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> None:
        """
        Move a piece from one position to another on the board.

        Args:
            board (Board): The chess board.
            from_pos (Tuple[int, int]): The starting position of the piece.
            to_pos (Tuple[int, int]): The target position for the piece.

        Code Flow:
            - Check whether the positions are valid.
            - Verify that the piece at `from_pos` belongs to the current player.
            - If the destination square contains an opponent’s piece, perform a capture.
            - Validate the move according to the rules of the piece being moved.
            - Update the board with the new position.
            - Check for check status after the move.
            - Switch the turn to the next player.
        """
        piece = board.get_piece(from_pos)
        if piece is None:
            raise ValueError(f"Không có quân nào tại vị trí {from_pos}.")

        if piece.get_color().upper() != self.turn:
            raise ValueError("Quân ở vị trí này không thuộc lượt bạn.")

        target_piece = board.get_piece(to_pos)
        if target_piece:
            if target_piece.get_color().upper() == self.turn:
                raise ValueError("Không thể di chuyển đến vị trí có quân cùng màu.")
            else:
                self.capture_piece(board, to_pos)

        if not piece.is_valid_move(to_pos):
            raise ValueError("Nước đi không hợp lệ theo luật của quân cờ.")

        board.set_piece(to_pos, piece)
        if from_pos in board.board:
            del board.board[from_pos]

        self.switch_turn()

    def capture_piece(self, board: Board, pos: Tuple[int, int]) -> None:
        """
        Capture a piece on the board.
        
        Args:
            board (Board): The chess board.
            pos (Tuple[int, int]): The position of the piece to be captured.
        """
        captured = board.get_piece(pos)
        if captured:
            self.captured_pieces.append(captured)
            if pos in board.board:
                del board.board[pos]

    def is_in_check(self, board: Board, color: str) -> bool:
        """
        Check if the king of the specified color is in check.
        
        Args:
            board (Board): The chess board.
            color (str): The color of the king to check ('WHITE' or 'BLACK').
            
        Returns:
            bool: True if the king is in check, False otherwise.
        """
        king_pos = self.find_king(board, color)
        opponent_color = 'WHITE' if color.upper() == 'BLACK' else 'BLACK'

        for pos, piece in board.board.items():
            if piece.get_color().upper() == opponent_color:
                if king_pos in piece.get_valid_move():
                    return True
        return False

    def find_king(self, board: Board, color: str) -> Tuple[int, int]:
        """
        Find the position of the king of the specified color.
        
        Args:
            board (Board): The chess board.
            color (str): The color of the king to find ('WHITE' or 'BLACK').
            
        Returns:
            Tuple[int, int]: The position of the king.
        """
        for pos, piece in board.board.items():
            if piece.get_piece_type().upper() == 'KING' and piece.get_color().upper() == color.upper():
                return pos
        raise ValueError(f"Không tìm thấy quân Vua của màu {color}")

    def can_castle(self, board: Board, king_pos: Tuple[int, int], rook_pos: Tuple[int, int]) -> bool:
        """
        Check if castling is possible between the king and rook.
        
        Args:
            board (Board): The chess board.
            king_pos (Tuple[int, int]): The position of the king.
            rook_pos (Tuple[int, int]): The position of the rook.
            
        Returns:
            bool: True if castling is possible, False otherwise.
        """
        if king_pos[0] != rook_pos[0]:
            return False

        row = king_pos[0]
        col_start = min(king_pos[1], rook_pos[1]) + 1
        col_end = max(king_pos[1], rook_pos[1])
        for col in range(col_start, col_end):
            if board.get_piece((row, col)) is not None:
                return False
        return True

    def castle(self, board: Board, king_pos: Tuple[int, int], rook_pos: Tuple[int, int]) -> None:
        """
        Perform castling between the king and rook.
        
        Args:
            board (Board): The chess board.
            king_pos (Tuple[int, int]): The position of the king.
            rook_pos (Tuple[int, int]): The position of the rook.
        """
        if not self.can_castle(board, king_pos, rook_pos):
            raise ValueError("Nhập thành không hợp lệ do điều kiện không thỏa.")

        king = board.get_piece(king_pos)
        rook = board.get_piece(rook_pos)
        if king is None or rook is None:
            raise ValueError("Không tìm thấy quân Vua hoặc Rook tại vị trí nhập thành.")

        if rook_pos[1] > king_pos[1]:
            new_king_pos = (king_pos[0], king_pos[1] + 2)
            new_rook_pos = (rook_pos[0], king_pos[1] + 1)
        else:
            new_king_pos = (king_pos[0], king_pos[1] - 2)
            new_rook_pos = (rook_pos[0], king_pos[1] - 1)

        board.set_piece(new_king_pos, king)
        if king_pos in board.board:
            del board.board[king_pos]

        board.set_piece(new_rook_pos, rook)
        if rook_pos in board.board:
            del board.board[rook_pos]