from evaluation.diffusion_engine import DiffusionEngine
from evaluation.heuristic_engine import HeuristicEngine
from evaluation.utils import load_yaml

class SuperSeiChessEngine(): 
    """
    A chess engine that combines heuristic evaluation and diffusion-based evaluation.
    
    Attributes:
        heuristic_engine (HeuristicEngine): An instance of the heuristic engine.
        diffusion_engine (DiffusionEngine): An instance of the diffusion engine.
    """
    
    def __init__(self, config_path='./config/super_sei_configs.yaml'):
        config = load_yaml(config_path)
        
        self.heuristic_engine = HeuristicEngine(config_path=config['heuristic_config'])
        self.diffusion_engine = DiffusionEngine(config_path=config['diffusion_config'])
        self.diffusion_max_attempts = config['diffusion_max_attempts']
        self.heuristic_depth = config['heuristic_depth']
        
    def get_best_move(self, board, previous_move='a1b1'): 
        move = self.diffusion_engine.get_best_move(board=board, previous_move=previous_move, max_attempts=self.diffusion_max_attempts)
        if move is None:
            move = self.heuristic_engine.get_best_move(board=board, depth=self.heuristic_depth)
        return move