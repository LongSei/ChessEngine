import chess
import chess.svg
import chess.pgn
import chess.engine
from cairosvg import svg2png
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse

# --- Mã hóa bàn cờ thành tensor ---
def board_to_tensor(board: chess.Board) -> torch.Tensor:
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        idx = {
            (chess.PAWN, True): 0, (chess.KNIGHT, True): 1, (chess.BISHOP, True): 2,
            (chess.ROOK, True): 3, (chess.QUEEN, True): 4, (chess.KING, True): 5,
            (chess.PAWN, False): 6, (chess.KNIGHT, False): 7, (chess.BISHOP, False): 8,
            (chess.ROOK, False): 9, (chess.QUEEN, False): 10, (chess.KING, False): 11,
        }[(piece.piece_type, piece.color)]
        tensor[idx, row, col] = 1
    return torch.from_numpy(tensor)

# --- Mô hình CNN đơn giản ---
class EvalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64*8*8, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

# --- Dataset từ file PGN, đánh giá bằng Stockfish ---
class PositionDataset(Dataset):
    def __init__(self, pgn_path, max_positions=1000, eval_depth=6, engine_path='stockfish'):
        self.positions = []
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        with open(pgn_path) as f:
            while len(self.positions) < max_positions:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                board = game.board()
                for move in game.mainline_moves():
                    board.push(move)
                    info = engine.analyse(board, chess.engine.Limit(depth=eval_depth))
                    raw = info['score'].white()
                    score = raw.score(mate_score=10000) or 0
                    tensor = board_to_tensor(board)
                    self.positions.append((tensor, np.float32(score)))
                    if len(self.positions) >= max_positions:
                        break
        engine.quit()

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.positions[idx]

# --- Huấn luyện model ---
def train_model(pgn_path, epochs=5, batch_size=32, lr=1e-3, **kwargs):
    dataset = PositionDataset(pgn_path, **kwargs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EvalNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs+1):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        print(f'Epoch {epoch}/{epochs}, Loss: {total_loss/len(dataset):.4f}')
    torch.save(model.state_dict(), 'eval_model.pt')
    print('Đã lưu mô hình vào eval_model.pt')

# --- Hàm đánh giá và Negamax + Alpha-Beta ---
def evaluate(board: chess.Board) -> float:
    tensor = board_to_tensor(board).unsqueeze(0)
    model = EvalNet()
    model.load_state_dict(torch.load('eval_model.pt', map_location='cpu'))
    model.eval()
    with torch.no_grad():
        return model(tensor).item()

def negamax_ab(board, depth, alpha, beta, color):
    if depth == 0 or board.is_game_over():
        return color * evaluate(board)
    for m in board.legal_moves:
        board.push(m)
        score = -negamax_ab(board, depth-1, -beta, -alpha, -color)
        board.pop()
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    return alpha

def select_move(board, depth):
    best, alpha, beta = None, -1e9, 1e9
    for m in board.legal_moves:
        board.push(m)
        sc = -negamax_ab(board, depth-1, -beta, -alpha, -1)
        board.pop()
        if sc > alpha:
            alpha, best = sc, m
    return best

# --- Hiển thị bàn cờ đồ họa trực tiếp ---
def render_board(board, size=400):
    svg = chess.svg.board(board=board, size=size)
    png = svg2png(bytestring=svg.encode('utf-8'))
    img = Image.open(BytesIO(png))
    arr = np.array(img)
    plt.figure(figsize=(size/100, size/100))
    plt.axis('off')
    plt.imshow(arr)
    plt.show()

# --- Main ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='PGN file để train (ví dụ games.pgn)', type=str)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--engine_path', type=str, default='stockfish') 
    args = parser.parse_args()
    
    if args.train:
        train_model(
            args.train,
            epochs=args.epochs,
            batch_size=args.batch,
            max_positions=1000,
            eval_depth=6,
            engine_path=args.engine_path 
        )
    else:
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
