import chess
import chess.svg
import math
import time
import cairosvg
import os

# Hàm đánh giá bàn cờ
def evaluate_board(board: chess.Board) -> float:
    piece_values = {
        chess.PAWN:   1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK:   5,
        chess.QUEEN:  9,
        chess.KING:   0
    }
    score = 0
    for square, piece in board.piece_map().items():
        if piece.color == chess.WHITE:
            score += piece_values[piece.piece_type]
        else:
            score -= piece_values[piece.piece_type]
    return score

# Negamax + Alpha-Beta
def negamax_alpha_beta(board: chess.Board, depth: int, alpha: float, beta: float, color: int) -> float:
    if depth == 0 or board.is_game_over():
        return color * evaluate_board(board)

    max_value = -math.inf
    for move in board.legal_moves:
        board.push(move)
        value = -negamax_alpha_beta(board, depth - 1, -beta, -alpha, -color)
        board.pop()
        max_value = max(max_value, value)
        alpha = max(alpha, value)
        if alpha >= beta:
            break
    return max_value

# Tìm nước đi tốt nhất
def find_best_move(board: chess.Board, depth: int) -> chess.Move:
    best_move = None
    best_value = -math.inf
    for move in board.legal_moves:
        board.push(move)
        move_value = -negamax_alpha_beta(board, depth - 1, -math.inf, math.inf, -1)
        board.pop()
        if move_value > best_value:
            best_value = move_value
            best_move = move
    return best_move

# Vẽ bàn cờ và lưu hình ảnh
def save_board_image(board: chess.Board, move_number: int):
    svg = chess.svg.board(board=board, size=400)
    svg_path = f"board_{move_number}.svg"
    png_path = f"board_{move_number}.png"
    with open(svg_path, "w") as f:
        f.write(svg)
    # Convert SVG to PNG
    cairosvg.svg2png(url=svg_path, write_to=png_path)
    # (Tuỳ chọn) Xoá file SVG để gọn gàng
    os.remove(svg_path)
    print(f"📷 Đã lưu ảnh bàn cờ: {png_path}")

# Main
if __name__ == "__main__":
    board = chess.Board()
    search_depth = 3

    print("Trạng thái bắt đầu:\n", board)
    count_move = 0

    while not board.is_game_over():
        start_time = time.time()
        best_move = find_best_move(board, search_depth)
        end_time = time.time()
        board.push(best_move)

        print(f"\n[{count_move+1}] Minimax_with_alpha-beta: {best_move}, Time: {end_time - start_time:.4f} seconds")
        print(board)

        save_board_image(board, count_move + 1)
        count_move += 1

        if count_move % 5 == 0 or board.is_game_over():
            print(f"==> Tạm thời sau {count_move} bước:\n{board.fen()}\n")

    print("\nTrò chơi kết thúc. Kết quả:", board.result())
