if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, default=3, help='Độ sâu tìm kiếm cho AI')
    parser.add_argument('--model_path', type=str, default='eval_model.pt', help='Đường dẫn đến file model đã train')
    args = parser.parse_args()

    # Load mô hình đã train
    model = EvalNet()
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()

    # Cập nhật hàm evaluate để dùng model này
    def evaluate(board: chess.Board) -> float:
        tensor = board_to_tensor(board).unsqueeze(0)
        with torch.no_grad():
            return model(tensor).item()

    # Bắt đầu chơi cờ
    board = chess.Board()
    render_board(board)
    while not board.is_game_over():
        m = select_move(board, args.depth)
        print('Engine đi:', m)
        board.push(m)
        render_board(board)
        if board.is_game_over():
            break
        mv = input('Bạn đi: ')
        try:
            board.push_san(mv)
        except ValueError:
            print('Nước đi không hợp lệ')
            continue
        render_board(board)
    print('Kết quả:', board.result())
