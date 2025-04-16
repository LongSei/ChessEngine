import chess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter
# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Hyperparameters
BATCH_SIZE = 1024
BUFFER_SIZE = 100000
GAMMA = 0.99
LR = 3e-4
SYNC_INTERVAL = 50
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.9995

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
    
    return encoded


class ChessQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # CNN for board processing (giữ nguyên)
        self.conv = nn.Sequential(
            nn.Conv2d(14, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Sửa input dimension của fc layer
        self.fc_input_dim = 256 * 8 * 8 + 7 + 2  # Thêm 2 chiều cho action
        
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),  # Đã cập nhật input dim
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # Tách board state và action features
        board_data = x[:, :8*8*14].view(-1, 14, 8, 8)
        other_features = x[:, 8*8*14:]
        
        # Xử lý board
        conv_out = self.conv(board_data).view(-1, 256*8*8)
        
        # Ghép với các features khác và action
        combined = torch.cat([conv_out, other_features], dim=1)
        
        return self.fc(combined)
    
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

def get_best_move(board, model):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    
    # Chuẩn bị batch input đúng định dạng
    state = encode_board(board)
    state_repeated = np.tile(state, (len(legal_moves), 1))
    
    # Thêm action features
    action_features = np.array([[m.from_square/63, m.to_square/63] for m in legal_moves])
    network_input = np.concatenate([state_repeated, action_features], axis=1)
    
    # Chuyển sang tensor
    input_tensor = torch.FloatTensor(network_input).to(DEVICE)
    
    with torch.no_grad():
        q_values = model(input_tensor).cpu().numpy().flatten()
    
    return legal_moves[np.argmax(q_values)]


def evaluate_model(model, num_games=10):
    wins = 0
    for _ in range(num_games):
        board = chess.Board()
        while not board.is_game_over():
            move = get_best_move(board, model) if board.turn else random.choice(list(board.legal_moves))
            board.push(move)
        if board.result() == "1-0":
            wins += 1
    return wins / num_games

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
            
            prev_state = encode_board(board)
            board.push(move)
            next_state = encode_board(board)
            
            # Store original experience
            game_history.append((prev_state, move, next_state))
            
            # Data augmentation
            for _ in range(2):  # Random flips/rotations
                rotated_state, rotated_move = augment_data(prev_state, move)
                game_history.append((rotated_state, rotated_move, next_state))
        
        # Assign final rewards
        result = board.result()
        reward = 1.0 if result == "1-0" else -1.0 if result == "0-1" else 0
        
        # Add to buffer with computed rewards
        for state, move, next_state in game_history:
            buffer.append((state, move, reward, next_state, board.is_game_over()))
        
    return buffer

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

    for episode in range(1000): 
        sample_input = torch.randn(1, 8 * 8 * 14 + 7 + 2).to(
            DEVICE
        )  # 903 (state) + 2 (action) = 905
        print("Input shape:", sample_input.shape)
        output = q_net(sample_input)
        print("Output shape:", output.shape)
        # Generate self-play games
        games = generate_self_play_games(q_net, num_games=30)
        for game in games:
            replay_buffer.add(game)

        # Training step
        if len(replay_buffer.buffer) >= BATCH_SIZE:
            indices, batch, weights = replay_buffer.sample(BATCH_SIZE)

            # Unpack batch
            states, moves, rewards, next_states, dones = zip(*batch)

            # Convert to tensors
            state_tensor = torch.FloatTensor(np.array(states)).to(DEVICE)
            action_tensor = torch.FloatTensor(
                [[m.from_square / 63, m.to_square / 63] for m in moves]
            ).to(DEVICE)
            next_state_tensor = torch.FloatTensor(np.array(next_states)).to(DEVICE)
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
                    next_actions = [
                        get_best_move(
                            chess.Board().set_fen(chess.Board().fen()), target_net
                        )
                        for _ in range(sum(valid_next))
                    ]
                    next_action_tensor = torch.FloatTensor(
                        [[m.from_square / 63, m.to_square / 63] for m in next_actions]
                    ).to(DEVICE)
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
    torch.save(q_net.state_dict(), 'chess_ai.pth')
    print("Model saved successfully!")


if __name__ == "__main__":
    train()