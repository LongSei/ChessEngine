# 1.1. Monte Carlo Tree Search (MCTS)
## 1.1.1. Definition 
- Monte Carlo Tree Search (MCTS) is a heuristic search algorithm for decision-making processes, particularly in games with a large branching factor like Chess and Go. MCTS builds a search tree by simulating many possible games (or trajectories) from the current position, using a combination of randomness and learned evaluation. The core idea is to use random sampling of the decision space to evaluate the most promising moves, gradually improving decision accuracy with each simulation.
- MCTS has become a foundational algorithm in artificial intelligence for board games, notably forming the backbone of AlphaGo and AlphaZero. It is especially effective in environments with incomplete knowledge or large, complex state spaces where traditional brute-force search is impractical.

## 1.1.2. How MCTS works
- MCTS proceeds through four fundamental phases, repeated iteratively during simulations:
### Selection
 Starting from the root node (the current game state), the algorithm recursively selects child nodes based on a policy that balances exploration and exploitation. This is typically done using the PUCT (Predictor + Upper Confidence bound applied to Trees) formula, which incorporates both the value estimates from simulations and prior probabilities from a policy model.


### Expansion
 Once a leaf node (a node with no children) is reached, if the game is not yet finished, one or more child nodes are created. Each represents a legal move from the leaf node. This step increases the breadth of the tree.


### Simulation (Evaluation)
 Instead of simulating a random playout to the end of the game (as in traditional MCTS), modern approaches like AlphaZero replace this with a neural network evaluation. The model predicts:


A policy: a probability distribution over legal moves.


A value: an estimated outcome of the game from the current player’s perspective.


### Backpropagation
 The predicted value is then propagated back through the nodes and edges traversed in the current simulation. This updates statistics such as visit counts and cumulative values, which are used to compute the quality of each move (Q-value).
- These steps are repeated for many simulations (typically hundreds or thousands), enabling the algorithm to refine its estimate of the best move. Once all simulations are complete, the move with the highest visit count or best average value is selected.
 
## 1.1.3. MCTS implementation code
### 1. Class Initialization
```python
def __init__(self, agent: "Agent", state: str = chess.STARTING_FEN, stochastic=False):
        self.root = Node(state=state)
        self.game_path: list[Edge] = []
        self.cur_board: chess.Board = None
        self.agent = agent
        self.stochastic = stochastic
```

- Purpose: Construct a new MCTS search tree for a given board state.
- Key Parameters:
agent: The policy/value model responsible for evaluating positions.
state (default: chess.STARTING_FEN): The FEN string representing the starting board.
stochastic (bool): If True, Dirichlet noise is applied at the root for enhanced exploration.
- Attributes Initialized:
root: A Node object representing the tree’s root.
game_path: A list to track edges traversed during a single simulation.
cur_board: A chess.Board instance, updated when mapping moves.
agent and stochastic: Stored for use in later phases.

### 2. Running Simulations
```python
def run_simulations(self, n: int) -> None:
        for _ in tqdm(range(n)):
            self.game_path = []
            leaf = self.select_child(self.root)
            leaf.N += 1
 leaf = self.expand(leaf)
            leaf = self.backpropagate(leaf, leaf.value)
```

- Purpose: Execute n full MCTS simulations to accumulate search statistics.
-  Workflow:
Select a leaf node via select_child.
Expand the selected leaf, obtaining a value estimate from the model.
Backpropagate the obtained value through the path of edges.
- Significance: Repeated simulations refine the estimates of move quality, improving move selection.
   
### 3. Selection
```python
def select_child(self, node: Node) -> Node:
while not node.is_leaf():
	if not len(node.edges):
		return node
	noise = [1 for _ in range(len(node.edges))]
	if self.stochastic and node == self.root:
		noise = np.random.dirichlet([config.DIRICHLET_NOISE]*len(node.edges))
	best_edge = None
	best_score = -np.inf                
	for i, edge in enumerate(node.edges):
		if edge.upper_confidence_bound(noise[i]) > best_score:
			best_score = edge.upper_confidence_bound(noise[i])
			best_edge = edge
	if best_edge is None:
		raise Exception("No edge found")
	node = best_edge.output_node
	self.game_path.append(best_edge)
	return node
```

- Purpose: Traverse from a given node down to a leaf, choosing edges by maximizing Q + U (exploitation + exploration).
- Mechanics:
At each non-leaf node:
Compute the upper confidence bound (PUCT score) for each outgoing edge.
Choose the edge with the highest score.
Optionally add Dirichlet noise at the root to diversify opening moves.
Continue until reaching a node with no children or with unexpanded moves.
- Significance: Balances exploring unvisited moves and exploiting known strong moves.
  
### 4. Mapping Valid Moves
```python 
def map_valid_move(self, move: chess.Move) -> None:
	logging.debug("Filtering valid moves...")
	from_square = move.from_square
	to_square = move.to_square
	plane_index: int = None
	piece = self.cur_board.piece_at(from_square)
	direction = None
	if piece is None:
		raise Exception(f"No piece at {from_square}")
	if move.promotion and move.promotion != chess.QUEEN:
		piece_type, direction = Mapping.get_underpromotion_move(move.promotion, from_square, to_square)
		plane_index = Mapping.mapper[piece_type][1 - direction]
	else:
		if piece.piece_type == chess.KNIGHT:
			direction = Mapping.get_knight_move(from_square, to_square)
			plane_index = Mapping.mapper[direction]
		else:
			direction, distance = Mapping.get_queenlike_move(from_square, to_square)
			plane_index = Mapping.mapper[direction][np.abs(distance)-1]
	row = from_square % 8
	col = 7 - (from_square // 8)
	self.outputs.append((move, plane_index, row, col))
```

- Purpose: Translate a chess.Move object into the neural network’s output indexing scheme.
- Process:
Identify the piece’s origin and target squares.
Determine the corresponding plane index based on piece type and movement direction (queen-like, knight, or underpromotion).
Compute row and column offsets for the 8×8 plane.
Append a tuple (move, plane, row, col) to self.outputs.
- Significance: Aligns the model’s 73×8×8 output tensor with actual legal moves.

### 5. Converting Probabilities to Actions
```python
def probabilities_to_actions(self, probabilities: list, board: str) -> dict:
	probabilities = probabilities.reshape(config.amount_of_planes, config.n, config.n)
	actions = {}
	self.cur_board = chess.Board(board)
	valid_moves = self.cur_board.generate_legal_moves()
	self.outputs = []
	threads = []
	while True:
		try:
			move = next(valid_moves)
		except StopIteration:
			break
		thread = threading.Thread(target=self.map_valid_move, args=(move,))
		threads.append(thread)
		thread.start()
	for thread in threads:
		thread.join()
	for move, plane_index, col, row in self.outputs:
		actions[move.uci()] = probabilities[plane_index][col][row]
 	return actions
```

- Purpose: Filter the raw network output (shape: 73×8×8) to a dictionary of legal moves with their probabilities.
- Workflow:
Reshape the flat probability vector into 73 planes of 8×8.
Generate all legal moves for the current board state.
Spawn threads to call map_valid_move on each move in parallel.
For each mapped move, read the probability at (plane, row, col).
Return a dictionary: {move_uci: probability}.
- Significance: Ensures only legal moves are considered, each with its assigned prior.

### 6. Expansion 
```python
def expand(self, leaf: Node) -> Node:
	logging.debug("Expanding...")
	board = chess.Board(leaf.state)
	possible_actions = list(board.generate_legal_moves())
	if not len(possible_actions):
		assert board.is_game_over(), "Game is not over, but there are no possible moves?"
		outcome = board.outcome(claim_draw=True)
      if outcome is None:
leaf.value = 0
else:
           leaf.value = 1 if outcome.winner == chess.WHITE else -1
input_state = ChessEnv.state_to_input(leaf.state)
p, v = self.agent.predict(input_state)
actions = self.probabilities_to_actions(p, leaf.state)
logging.debug(f"Model predictions: {p}")
logging.debug(f"Value of state: {v}")
leaf.value = v
for action in possible_actions:
new_state = leaf.step(action)
    	leaf.add_child(Node(new_state), action, actions[action.uci()])
return leaf
```

- Purpose: Grow the tree by creating children for all legal moves from a leaf node.
- Steps:
If no legal moves remain, assign the terminal value (+1, –1, or 0) based on game outcome.
Otherwise, call agent.predict with the leaf’s input representation to obtain:
p: Prior probability distribution over all moves.
v: Value estimate of the position.
Convert p to a legal-move→probability map via probabilities_to_actions.
For each legal move:
Apply the move to get a new state.
Create a child Node and an Edge connecting parent → child, storing the move and its prior.
- Significance: Seeds new edges with informative priors, guiding future search

### 7. Backpropagation
```python 
def backpropagate(self, end_node: Node, value: float) -> Node:
	logging.debug("Backpropagation...")
	for edge in self.game_path:
		edge.input_node.N += 1
		edge.N += 1
		edge.W += value
	return end_node
```

- Purpose: Update visit counts and value sums along the simulation path.

- Procedure:
	+ For each Edge in self.game_path:
		1. Increment the parent Node’s visit count (N).
		2. Increment the Edge’s visit count (edge.N).
		3. Accumulate the simulation’s value into edge.W (total value).

- Significance: Aggregates experience so that edges leading to better outcomes gain higher quality (Q).

### 8. Visualization
```python 
def plot_node(self, dot: Digraph, node: Node):
	dot.node(f"{node.state}", f"N")
	for edge in node.edges:
		dot.edge(str(edge.input_node.state), 
				str(edge.output_node.state), 
				label=edge.action.uci())
	dot = self.plot_node(dot, edge.output_node)
	return dot

def plot_tree(self, save_path: str = "tests/mcts_tree.gv") -> None:
logging.debug("Plotting tree...")
  	dot = Digraph(comment='Chess MCTS Tree')
logging.info(f"# of nodes in tree:  {len(self.root.get_all_children())}")
dot = self.plot_node(dot, self.root)
  	dot.save(save_path)
```

- Purpose: Generate a visual representation of the current MCTS tree structure using Graphviz.

- Components:
1. plot_tree(save_path): Initializes a Digraph object and begins the recursive plotting process from the root node.
2. plot_node(dot, node): Recursively adds nodes and edges to the graph, labeling each edge with the move (in UCI format).

- Workflow:
1. plot_tree creates a Graphviz Digraph and logs the total number of nodes.
2. It then calls plot_node, passing the root node.
3. For each node, plot_node:
	*  Adds the node to the graph.
 	*  Iterates through its edges.
	*  Draws directed edges from parent to child nodes, labeled by the chess
          move.
	*  Recursively processes each child node.
4. The graph is saved to the specified .gv file path.


- Significance: Useful for debugging, analysis, and presenting how the search tree evolves over simulations.




