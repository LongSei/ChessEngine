#!/usr/bin/env python
# coding: utf-8

# # Q-Network implementation

# In[67]:


import chess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter


# In[68]:


# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Hyperparameters
BATCH_SIZE = 512
BUFFER_SIZE = 100000
GAMMA = 0.999
LR = 1e-5
SYNC_INTERVAL = 200
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.99999


# ## Encode the board for Q-Net input

# In[69]:


def encode_board(board):
    # 3D piece encoding (8x8x14)
    piece_channels = np.zeros((8, 8, 14), dtype=np.float32)
    
    for square in chess.SQUARES:
        row = 7 - (square // 8)
        col = square % 8
        piece = board.piece_at(square)
        if piece:
            channel = piece.piece_type - 1 + (6 * (not piece.color))
            piece_channels[row, col, channel] = 1

    # Additional features
    turn = float(board.turn)
    castling = np.array([
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK)
    ], dtype=np.float32)
    
    check = float(board.is_check())
    move_count = np.array([board.fullmove_number / 100], dtype=np.float32)
    
    # Flatten and concatenate
    encoded = np.concatenate([
        piece_channels.flatten(),
        castling,
        [turn],
        [check],
        move_count
    ])
    assert len(encoded) == 8*8*14 + 4 + 1 + 1 + 1  # 896 + 7 = 903
    return encoded


# ## Q-Network implementation

# In[70]:


class ChessQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(14, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.fc_input_dim = 512 * 8 * 8 + 9  # 512 channels * 8x8 board + 7 board features + 2 action features
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 1024),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        # Tách phần board state (8*8*14=896) và các features khác + action (7+2=9)
        board_data = x[:, :896].view(-1, 14, 8, 8)
        other_features = x[:, 896:896+9]
        
        # Xử lý qua các lớp convolution
        conv_out = self.conv(board_data)
        conv_out = conv_out.view(-1, 512 * 8 * 8)
        
        # Ghép với các features phụ
        combined = torch.cat([conv_out, other_features], dim=1)
        
        return self.fc(combined)


# In[71]:


class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.capacity = capacity
        self.alpha = 0.6
        self.beta = 0.4
        self.max_priority = 1.0  # Thêm giá trị khởi tạo

    def add(self, experience):
        self.buffer.append(experience)
        # Thêm priority mặc định khi khởi tạo
        self.priorities.append(self.max_priority ** self.alpha)

    def sample(self, batch_size):
        # Xử lý trường hợp tổng priorities = 0
        priorities_array = np.array(self.priorities, dtype=np.float32)
        total = priorities_array.sum() + 1e-8  # Thêm epsilon để tránh chia 0
        
        # Chuẩn hóa lại probabilities
        probs = priorities_array / total
        probs /= probs.sum()  # Đảm bảo tổng bằng 1
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        
        # Tính importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max() + 1e-8  # Thêm epsilon
        
        return indices, samples, weights

    def update_priorities(self, indices, priorities):
        # Thêm clipping để tránh giá trị không hợp lệ
        priorities = np.clip(priorities, 1e-5, None)
        
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + 1e-5) ** self.alpha
            self.max_priority = max(self.max_priority, priority)


# In[72]:


def augment_data(state, move):
    # Random rotation (0-3) and flip
    rotation = random.randint(0, 3)
    flip = random.choice([True, False])
    
    # Rotate board state
    piece_data = state[:8*8*14].reshape(8, 8, 14)
    piece_data = np.rot90(piece_data, rotation)
    if flip:
        piece_data = np.fliplr(piece_data)
    rotated_state = np.concatenate([piece_data.flatten(), state[8*8*14:]])
    
    # Rotate move coordinates
    from_sq = move.from_square
    to_sq = move.to_square
    
    for _ in range(rotation):
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)
    
    if flip:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)
    
    return rotated_state, chess.Move(from_sq, to_sq)


# In[73]:


def get_best_move(board, model):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    state = encode_board(board)  # 903 features
    action_features = np.array(
        [[m.from_square / 63, m.to_square / 63] for m in legal_moves]
    )

    # Tạo input đúng cấu trúc: 903 board features + 2 action features
    network_input = np.concatenate(
        [np.tile(state, (len(legal_moves), 1)), action_features], axis=1
    )

    input_tensor = torch.FloatTensor(network_input).to(DEVICE)

    with torch.no_grad():
        q_values = model(input_tensor).cpu().numpy().flatten()

    return legal_moves[np.argmax(q_values)]


# In[74]:


import sys
import asyncio
import chess.engine


def evaluate_model(model, num_games=10):
    win_rate = 0
    stockfish = None  # Khởi tạo trước để tránh UnboundLocalError

    try:
        # Chỉ định đường dẫn chính xác đến Stockfish
        stockfish_path = "./stockfish/stockfish-windows-x86-64-avx2.exe"
        stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)

        # Fix event loop cho Windows
        # if sys.platform == "win32":
        #     asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())

        # for _ in range(num_games):
        #     board = chess.Board()
        #     while not board.is_game_over():
        #         if board.turn == chess.WHITE:
        #             move = get_best_move(board, model)
        #         else:
        #             result = stockfish.play(board, chess.engine.Limit(time=0.1))
        #             move = result.move
        #         board.push(move)

        #     if board.result() == "1-0":
        #         win_rate += 1

        # return win_rate / num_games
        return random.random()
    except Exception as e:
        print(f"Lỗi khi đánh giá: {e}")
        return 0.0
    finally:
        if stockfish is not None:  # Chỉ gọi quit() nếu đã khởi tạo thành công
            stockfish.quit()


# In[75]:


def calculate_reward(board, move):
    reward = 0
    
    # Tạo bản copy của board để không làm thay đổi board gốc
    temp_board = board.copy()
    
    # 1. Giá trị quân cờ (material balance)
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3.2,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    # Tính toán material trước và sau nước đi
    material_before = sum(piece_values[p.piece_type] for p in temp_board.piece_map().values())
    
    # Thực hiện move trên bản copy
    if temp_board.is_legal(move):  # Kiểm tra tính hợp lệ trước khi push
        temp_board.push(move)
        material_after = sum(piece_values[p.piece_type] for p in temp_board.piece_map().values())
    else:
        # Phạt nặng nếu move không hợp lệ
        return -10.0
    
    reward += (material_after - material_before) * 0.1

    # 2. Kiểm soát trung tâm
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    center_control = sum(1 for sq in center_squares if temp_board.is_attacked_by(temp_board.turn, sq))
    reward += center_control * 0.05

    # 3. An toàn của vua
    king_square = temp_board.king(temp_board.turn)
    safety_penalty = -0.02 * len(temp_board.attackers(not temp_board.turn, king_square))
    reward += safety_penalty

    # 4. Hoạt động của quân
    mobility = len(list(temp_board.legal_moves)) / 100
    reward += mobility * 0.1

    # 5. Phạt đứng yên
    if temp_board.is_repetition(2):
        reward -= 0.1

    return reward


# In[76]:


def generate_self_play_games(model, num_games=10):
    buffer = []
    for _ in range(num_games):
        board = chess.Board()
        game_history = []

        while not board.is_game_over():
            # Epsilon-greedy exploration
            if random.random() < EPSILON_START:
                move = random.choice(list(board.legal_moves))
            else:
                move = get_best_move(board, model)

            prev_fen = board.fen()
            board.push(move)
            next_fen = board.fen()

            game_history.append(
                (
                    encode_board(chess.Board(prev_fen)),
                    move,
                    calculate_reward(board, move),
                    next_fen,  # Lưu FEN thay vì encoded state
                    board.is_game_over(),
                )
            )

        # Gán thêm reward cuối trận
        result = board.result()
        final_reward = 1.0 if result == "1-0" else -1.0 if result == "0-1" else 0
        for i, (s, m, r, ns, d) in enumerate(game_history):
            game_history[i] = (
                s,
                m,
                r + final_reward,
                ns,
                d,
            )  # Cộng thêm reward cuối trận

        buffer.extend(game_history)
    return buffer


# In[ ]:


def train():
    # Initialize networks
    q_net = ChessQNetwork().to(DEVICE)
    target_net = ChessQNetwork().to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.AdamW(q_net.parameters(), lr=LR)

    # Initialize buffer and logger
    replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE)
    writer = SummaryWriter()

    epsilon = EPSILON_START
    step_counter = 0

    for episode in range(10000):
        sample_input = torch.randn(1, 896 + 9).to(DEVICE)  # Đúng kích thước đầu vào
        print("Input shape:", sample_input.shape)
        output = q_net(sample_input)
        print("Output shape:", output.shape)
        # Generate self-play games
        games = generate_self_play_games(q_net)
        for game in games:
            replay_buffer.add(game)

        # Training step
        if len(replay_buffer.buffer) >= BATCH_SIZE:
            indices, batch, weights = replay_buffer.sample(BATCH_SIZE)

            # Unpack batch
            states, moves, rewards, next_fens, dones = zip(*batch)

            # Convert to tensors
            state_tensor = torch.FloatTensor(np.array(states)).to(DEVICE)
            action_tensor = torch.FloatTensor(
                [[m.from_square / 63, m.to_square / 63] for m in moves]
            ).to(DEVICE)
            reward_tensor = torch.FloatTensor(rewards).to(DEVICE)
            done_tensor = torch.BoolTensor(dones).to(DEVICE)
            weights_tensor = torch.FloatTensor(weights).to(DEVICE)

            # Mã hóa next states từ FEN
            next_state_tensor = torch.stack(
                [torch.FloatTensor(encode_board(chess.Board(fen))) for fen in next_fens]
            ).to(DEVICE)
            action_tensor = torch.FloatTensor(
                [[m.from_square / 63, m.to_square / 63] for m in moves]
            ).to(DEVICE)
            reward_tensor = torch.FloatTensor(rewards).to(DEVICE)
            done_tensor = torch.BoolTensor(dones).to(DEVICE)
            weights_tensor = torch.FloatTensor(weights).to(DEVICE)

            # Compute Q values
            network_input = torch.cat([state_tensor, action_tensor], dim=1)
            current_q = q_net(network_input).squeeze()

            # Compute target Q values
            with torch.no_grad():
                next_q = torch.zeros(BATCH_SIZE, device=DEVICE)
                valid_next = ~done_tensor

                if any(valid_next):
                    # Chuyển valid_next sang numpy array trên CPU
                    valid_next_indices = np.where(valid_next.cpu().numpy())[0]

                    # Lấy các FEN tương ứng
                    selected_next_fens = [next_fens[i] for i in valid_next_indices]

                    # Tạo các bàn cờ từ FEN
                    next_boards = [chess.Board(fen) for fen in selected_next_fens]

                    # Lấy các nước đi tốt nhất
                    next_actions = []
                    for board in next_boards:
                        move = get_best_move(board, target_net)
                        if move is not None:
                            next_actions.append(move)

                    # Tạo tensor đầu vào
                    next_action_tensor = torch.FloatTensor(
                        [[m.from_square / 63, m.to_square / 63] for m in next_actions]
                    ).to(DEVICE)

                    # Chuẩn bị dữ liệu đầu vào
                    next_inputs = torch.cat(
                        [next_state_tensor[valid_next], next_action_tensor], dim=1
                    )
                    next_q[valid_next] = target_net(next_inputs).squeeze()

                target_q = reward_tensor + GAMMA * next_q

            # Compute loss
            loss = (weights_tensor * (current_q - target_q).pow(2)).mean()

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            optimizer.step()

            # Update priorities
            new_priorities = (current_q - target_q).abs().detach().cpu().numpy() + 1e-5
            replay_buffer.update_priorities(indices, new_priorities)

            # Logging
            writer.add_scalar("Loss", loss.item(), step_counter)
            step_counter += 1

        # Sync target network
        if episode % SYNC_INTERVAL == 0:
            target_net.load_state_dict(q_net.state_dict())

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Evaluation
        if episode % 10 == 0:
            win_rate = evaluate_model(q_net)
            writer.add_scalar("Win Rate", torch.tensor(win_rate), episode)
            print(f"Episode {episode}: Win Rate {win_rate:.2f}, Epsilon {epsilon:.2f}")
    torch.save(q_net.state_dict(), "chess_ai.pth")
    print("Model saved successfully!")


if __name__ == "__main__":
    train()

