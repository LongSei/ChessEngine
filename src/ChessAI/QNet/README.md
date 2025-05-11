# QNet

## **1. Core Components**  

### **a) ChessQNetwork**  

- **What**: A convolutional neural network (CNN) with 3 convolutional blocks and a dense head.  
- **Key Features**:  
  - **Input**: 903 features (8x8 board encoding + game state metadata + action features).  
  - **Architecture**:  
    - `Conv2d(14,128) → BatchNorm → ReLU` (×3 conv layers)  
    - `Linear(512*8*8+9 → 1024 → 1)` (dense layers)  
  - **Purpose**: Predicts Q-values for (board state, move) pairs.  

### **b) PrioritizedReplayBuffer**  

- **What**: Experience replay with priority sampling.  
- **Key Mechanics**:  
  - **Priority**: Samples transitions with TD-error-based importance.  
  - **Annealing**: Uses `alpha=0.6` (priority exponent), `beta=0.4` (importance sampling).  
  - **Stability**: Clips priorities to avoid extremes (`np.clip(priorities, 1e-5, None)`).  

### **c) Board Encoding**  

- **What**: Converts chess boards to 903D vectors.  
- **Structure**:  
  - **Piece channels**: 14-layer 8x8 tensor (6 piece types × 2 colors + 2 empty).  
  - **Metadata**: Castling rights, turn, check, move count.  

### **d) Reward Function**  

- **Components**:  
  - Material gain/loss  
  - Center control  
  - King safety  
  - Mobility penalty  
  - Repetition penalty  
- **Design Choice**: Combines handcrafted heuristics with learned values.  

---

## **2. Training Pipeline**  

### **a) Self-Play Generation**  

- **Process**:  
  1. Uses ε-greedy exploration (`EPSILON_START=1.0 → EPSILON_END=0.1`).  
  2. Augments data with board rotations/flips.  
  3. Applies final reward based on game outcome.  

### **b) Network Update**  

- **Key Steps**:  
  1. Samples prioritized transitions from buffer.  
  2. Computes Q-values using **online network**.  
  3. Computes target Q-values using **target network** (synced every `SYNC_INTERVAL=200`).  
  4. Updates priorities based on TD errors.  

### **c) Loss & Optimization**  

- **Loss**: Prioritized MSE loss (`(weights * (Q - target_Q)^2`).  
- **Optimizer**: AdamW with learning rate `LR=1e-5`, gradient clipping (`clip_grad_norm=1.0`).  

---

## **3. Hyperparameters**  

| Parameter          | Value     | Purpose                               |  
|--------------------|-----------|---------------------------------------|  
| `BATCH_SIZE`       | 512       | Batch size for training               |  
| `BUFFER_SIZE`      | 100,000   | Experience replay capacity            |  
| `GAMMA`            | 0.999     | Discount factor for future rewards    |  
| `SYNC_INTERVAL`    | 200       | Target network sync frequency         |  
| `EPSILON_DECAY`    | 0.99999   | Exploration rate decay (per episode)  |  

---

## **4. Key Interactions**  

### **a) QNet ↔ Environment**  

- **Flow**:  
  `Board → encode_board() → QNet → Q-values → ε-greedy action → New state`  

### **b) Experience Replay ↔ Training**  

- **Loop**:  
  1. Self-play games → Buffer (with priorities).  
  2. Sample batch → Compute loss → Update QNet.  
  3. Adjust priorities → Repeat.  

### **c) Target Network Stabilization**  

- **Why**: Avoids "chasing moving targets" in Q-value estimation.  
- **How**: Delayed sync (`SYNC_INTERVAL`) of target network weights.  

---

## **5. Evaluation Challenge**  

### **Stockfish Integration Issue**  

- **Symptoms**: Fails to load Stockfish engine (path/library issues).  
- **Impact**: Cannot benchmark AI against a strong baseline.  
- **Workaround**:  
  - Temporarily uses self-play win rates (`evaluate_model()` fallback).  
  - Requires fixing engine path/OS compatibility for proper evaluation.  

---

## **6. Architecture Diagram**  

```mermaid
[Self-Play]  
   │  
   ▼  
[Prioritized Buffer] → Sampled Batch → [QNet] ←→ [Target Net]  
   ▲                      │               │  
   └──Priority Update─────┘               │  
                                   [Optimizer (AdamW)]  
```

---

## **7. Notable Design Choices**  

1. **Action Encoding**: Normalizes move squares (`from_square/63`, `to_square/63`).  
2. **Board Augmentation**: Random rotations/flips for data diversity.  
3. **Delayed Rewards**: Adds game outcome reward to all transitions in an episode.  
