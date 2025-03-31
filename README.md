**Chess Engine Development Plan**

## Requirements
- 2200 Elo
- <10s inference

## References
- [Book about Deep-Learning approaches for Chess (like AlphaZero, Leela Chess Zero and Stockfish NNUE)](https://github.com/asdfjkl/neural_network_chess?tab=readme-ov-file)
- [Minimax](https://vi.wikipedia.org/wiki/Minimax)
- [Alpha–beta pruning](https://en.wikipedia.org/w/index.php?title=Alpha–beta_pruning&oldid=1068746141)
- [Negamax](https://en.wikipedia.org/wiki/Negamax)
- [Monte Carlo Search Tree](https://en.wikipedia.org/w/index.php?title=Monte_Carlo_tree_search&oldid=1067396622)

- [Study of the Proper NNUE Dataset](https://arxiv.org/pdf/2412.17948)
- [Shuffle Chess Engine](https://github.com/ArjunBasandrai/shuffle-chess-engine/?tab=readme-ov-file)
- [Flounder: an RL Chess Agent](https://stanford.edu/~bartolo/assets/flounder.pdf)
- [Chess engine with Deep Reinforcement learning](https://github.com/zjeffer/chess-deep-rl)
- [Stockfish NNUE (Chess evaluation) trainer in Pytorch](https://github.com/official-stockfish/nnue-pytorch/tree/master)

## **Phase 0: Research and Planning**
### **1. Research Report**

### **2. Understanding Chess Engine Fundamentals**
- Study existing chess engines and their architectures.
- Review algorithms like Minimax, Negamax, and Alpha-Beta pruning.

### **3. Dataset Collection and Preparation**
- Gather chess game databases for training and evaluation.
- Process data for use in search and evaluation functions.

## **Phase 1: Core Implementation**
### **4. UI/UX**

### **5. Basic Negamax Search**
- Implement a simple Negamax search.
- Verify its correctness by checking move generation and evaluation.

### **6. Alpha-Beta Pruning**
- Integrate alpha-beta pruning to improve search efficiency.
- Test performance improvements over plain Negamax.

## **Phase 2: Enhancing Search Efficiency**
### **7. Iterative Deepening**
- Implement iterative deepening to optimize move ordering and time management.
- Ensure smooth transition between depths.

### **8. Principal Variation Search (PVS)**
- Integrate PVS to further optimize alpha-beta pruning.
- Verify its effectiveness in reducing search space.

### **9. Late Move Reductions (LMR) & Null Move Pruning (NMP)**
- Implement LMR to prioritize promising moves.
- Add NMP to prune unlikely-to-matter positions.
- Test combined impact on move ordering and search depth.

### **10. Quiescence Search**
- Implement quiescence search to mitigate the horizon effect.
- Ensure it stabilizes tactical evaluation in volatile positions.

## **Phase 3: Advanced Evaluation**
### **11. NNUE (Efficient Neural Network Evaluation)**
- Integrate NNUE for position evaluation.
- Train the neural network or use a pre-trained model.
- Optimize inference speed within the search function.

### **12. Transposition Tables**
- Implement transposition tables to cache evaluated positions.
- Optimize memory management to avoid excessive usage.

## **Phase 4: Testing & Optimization**
### **13. Testing & Tuning**
- Use tactical test suites to verify evaluation consistency.
- Compare performance against established engines.
- Tune parameters systematically for optimal results.

### **14. Time Management**
- Develop a dynamic time allocation strategy.
- Ensure the engine handles different time constraints effectively.
