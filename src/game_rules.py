# chess_game.py

from typing import Tuple, Optional, List
from board import Board
from piece import Piece, King, Rook, Pawn  

class ChessGame:
    """
    Lớp quản lý một ván cờ vua, bao gồm bàn cờ, lượt chơi và các quy tắc cờ vua cơ bản.
    """
    def __init__(self):
        self.board = Board()      # Khởi tạo bàn cờ với trạng thái ban đầu từ board.py
        self.turn = 'WHITE'       # Lượt chơi ban đầu (WHITE luôn đi trước theo luật cờ vua)
        self.captured_pieces = [] # Danh sách lưu trữ các quân bị ăn (capture)

    def switch_turn(self) -> None:
        """
        Chuyển lượt chơi sau mỗi nước đi.
        """
        self.turn = 'WHITE' if self.turn == 'BLACK' else 'BLACK'

    def move_piece(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> None:
        """
        Di chuyển quân từ vị trí from_pos sang to_pos với các kiểm tra luật chơi:
          - Kiểm tra vị trí hợp lệ.
          - Kiểm tra quân ở from_pos thuộc lượt người chơi hiện tại.
          - Nếu nước đi đích có quân của đối phương, thực hiện capture (ăn quân).
          - Xác minh nước đi hợp lệ theo luật của quân được di chuyển.
          - Cập nhật vị trí trong bàn cờ.
          - Kiểm tra trạng thái chiếu (check) sau nước đi.
          - Chuyển lượt chơi.
        """
        # Lấy quân tại vị trí xuất phát
        piece = self.board.get_piece(from_pos)
        if piece is None:
            raise ValueError(f"Không có quân nào tại vị trí {from_pos}.")

        if piece.get_color.upper() != self.turn:
            raise ValueError("Quân ở vị trí này không thuộc lượt bạn.")

        # Nếu vị trí đích có quân đối phương, thực hiện capture
        target_piece = self.board.get_piece(to_pos)
        if target_piece:
            if target_piece.get_color.upper() == self.turn:
                raise ValueError("Không thể di chuyển đến vị trí có quân cùng màu.")
            else:
                self.capture_piece(to_pos)

        # Kiểm tra nước đi hợp lệ theo luật di chuyển của quân
        if not piece.is_valid_move(to_pos):
            raise ValueError("Nước đi không hợp lệ theo luật của quân cờ.")

        # Di chuyển quân: đặt quân vào vị trí mới và xóa quân ở vị trí cũ.
        self.board.set_piece(to_pos, piece)
        if from_pos in self.board.board:
            del self.board.board[from_pos]

        # Sau khi di chuyển, kiểm tra xem quân vua của đối phương có bị chiếu không.
        opponent = 'WHITE' if self.turn == 'BLACK' else 'BLACK'
        if self.is_in_check(opponent):
            print(f"Chú ý: Quân Vua của {opponent} đang bị chiếu!")

        # Chuyển lượt sau khi nước đi thành công.
        self.switch_turn()

    def capture_piece(self, pos: Tuple[int, int]) -> None:
        """
        Thực hiện capture (ăn quân) tại vị trí pos:
          - Lấy quân đối phương tại pos.
          - Lưu quân đó vào danh sách captured.
          - Xóa quân đó khỏi bàn cờ.
        """
        captured = self.board.get_piece(pos)
        if captured:
            self.captured_pieces.append(captured)
            if pos in self.board.board:
                del self.board.board[pos]

    def is_in_check(self, color: str) -> bool:
        """
        Kiểm tra xem quân vua của màu được chỉ định có đang bị chiếu không.
        Quy trình:
          - Tìm vị trí của quân Vua theo màu.
          - Với tất cả quân của đối phương, lấy danh sách nước đi hợp lệ.
          - Nếu vị trí Vua nằm trong nước đi của bất kỳ quân nào, trả về True.
        """
        king_pos = self.find_king(color)
        opponent_color = 'WHITE' if color.upper() == 'BLACK' else 'BLACK'

        for pos, piece in self.board.board.items():
            if piece.get_color.upper() == opponent_color:
                if king_pos in piece.get_valid_move():
                    return True
        return False

    def find_king(self, color: str) -> Tuple[int, int]:
        """
        Duyệt bàn cờ để tìm vị trí của quân Vua thuộc màu được chỉ định.
        """
        for pos, piece in self.board.board.items():
            if piece.get_piece_type.upper() == 'KING' and piece.get_color.upper() == color.upper():
                return pos
        raise ValueError(f"Không tìm thấy quân Vua của màu {color}")

    def can_castle(self, king_pos: Tuple[int, int], rook_pos: Tuple[int, int]) -> bool:
        """
        Kiểm tra điều kiện nhập thành (castling):
          - Quân Vua và Rook phải chưa di chuyển (giả sử trạng thái chưa di chuyển được kiểm soát bên trong các đối tượng).
          - Hai quân nằm trên cùng hàng.
          - Không có quân nào nằm giữa Vua và Rook.
          - (Ở ví dụ này, ta không kiểm tra chi tiết các ô mà Vua đi qua không bị chiếu.)
        """
        if king_pos[0] != rook_pos[0]:
            return False

        row = king_pos[0]
        col_start = min(king_pos[1], rook_pos[1]) + 1
        col_end = max(king_pos[1], rook_pos[1])
        for col in range(col_start, col_end):
            if self.board.get_piece((row, col)) is not None:
                return False
        return True

    def castle(self, king_pos: Tuple[int, int], rook_pos: Tuple[int, int]) -> None:
        """
        Thực hiện nhập thành nếu điều kiện được đáp ứng.
        Ví dụ:
          - Nếu vua trắng ở vị trí (7,4) và xe trắng bên phải ở (7,7): Nhập thành ngắn di chuyển Vua đến (7,6) và xe đến (7,5).
          - Nếu xe nằm bên trái, di chuyển theo hướng nhập thành dài.
        """
        if not self.can_castle(king_pos, rook_pos):
            raise ValueError("Nhập thành không hợp lệ do điều kiện không thỏa.")

        king = self.board.get_piece(king_pos)
        rook = self.board.get_piece(rook_pos)
        if king is None or rook is None:
            raise ValueError("Không tìm thấy quân Vua hoặc Rook tại vị trí nhập thành.")

        # Xác định kiểu nhập thành dựa theo vị trí của Rook so với Vua.
        if rook_pos[1] > king_pos[1]:
            # Nhập thành ngắn
            new_king_pos = (king_pos[0], king_pos[1] + 2)
            new_rook_pos = (rook_pos[0], king_pos[1] + 1)
        else:
            # Nhập thành dài
            new_king_pos = (king_pos[0], king_pos[1] - 2)
            new_rook_pos = (rook_pos[0], king_pos[1] - 1)

        # Di chuyển Vua sang vị trí mới và cập nhật trong board.
        self.board.set_piece(new_king_pos, king)
        if king_pos in self.board.board:
            del self.board.board[king_pos]

        # Di chuyển Rook sang vị trí mới.
        self.board.set_piece(new_rook_pos, rook)
        if rook_pos in self.board.board:
            del self.board.board[rook_pos]

        print("Nhập thành đã được thực hiện thành công.")

    def display_board(self) -> None:
        """
        Hiển thị trạng thái hiện tại của bàn cờ.
        In ra dạng đơn giản với ký hiệu chữ cái đầu của kiểu quân ở mỗi vị trí;
        nếu không có quân, in dấu chấm ('.').
        """
        for row in range(self.board.board_size):
            row_str = ""
            for col in range(self.board.board_size):
                piece = self.board.get_piece((row, col))
                if piece:
                    # Sử dụng ký hiệu chữ cái đầu của kiểu quân, 
                    # bạn có thể mở rộng để phân biệt màu sắc nếu cần.
                    row_str += piece.get_piece_type()[0] + " "
                else:
                    row_str += ". "
            print(row_str)

# Phần chạy thử (main)
if __name__ == "__main__":
    game = ChessGame()
    print("Bàn cờ khởi tạo:")
    game.display_board()

    # Ví dụ di chuyển quân tốt trắng từ (6,4) sang (5,4)
    try:
        print("\nDi chuyển quân trắng từ (6, 4) đến (5, 4):")
        game.move_piece((6, 4), (5, 4))
        game.display_board()
    except Exception as e:
        print("Lỗi khi di chuyển:", e)
    
    # Ví dụ thực hiện nhập thành cho quân trắng:
    # Vị trí ban đầu của Vua trắng là (7,4) và Rook trắng bên phải là (7,7)
    try:
        print("\nThực hiện nhập thành cho quân trắng (vua: (7,4), xe: (7,7)):")
        game.castle((7, 4), (7, 7))
        game.display_board()
    except Exception as e:
        print("Lỗi khi nhập thành:", e)
