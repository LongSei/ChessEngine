import torch
import torch.nn.functional as F
import numpy as np
import os
import chess
import random
import time
from transformers import AutoModelForCausalLM, AutoConfig


class DiffusionEngine:
    """
    A class that encapsulates a diffusion model for chess move prediction.
    """
    
    def __init__(self, 
                 max_length=328, 
                 model_path='../output/chess10k_gold_s_asa/DDM-s_asa-bs1024-lr3e-4-ep200-T20-20250501-145644',
                 diffusion_steps=50,
                 config_path=None):
        """
        Initialize the DiffusionEngine.
        
        Args:
            max_length (int): Maximum sequence length for input encoding
            model_path (str): Path to the model checkpoint
            diffusion_steps (int): Number of steps for the diffusion process
            config_path (str, optional): Path to a config file, if using config-based initialization
        """
        self.max_length = max_length
        self.diffusion_steps = diffusion_steps
        
        # Load from config if provided
        if config_path:
            try:
                from evaluation.utils import load_yaml
                cfg = load_yaml(config_path)
                self.max_length = cfg['diffusion_engine_configs']['max_length']
                model_path = cfg['diffusion_engine_configs']['model_to_load']
                self.diffusion_steps = cfg['diffusion_engine_configs']['diffusion_steps']
                print(f"Config loaded successfully: {cfg}")
            except Exception as e:
                print(f"Failed to load config: {e}. Using default parameters.")
        
        print(f"Initializing DiffusionEngine with: max_length={self.max_length}, "
              f"diffusion_steps={self.diffusion_steps}")
        
        # Determine the correct device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        try:
            from llmtuner.tuner.core.custom_tokenizer import CustomTokenizer
            from llmtuner.tuner.ddm.model import DiffusionModel
        except ImportError:
            try:
                # Try alternative import paths
                from .src.llmtuner.tuner.core.custom_tokenizer import CustomTokenizer
                from .src.llmtuner.tuner.ddm.model import DiffusionModel
            except ImportError:
                raise ImportError("Could not import CustomTokenizer or DiffusionModel. "
                                 "Please check your installation or provide correct import paths.")
        
        print(f"Loading model from {model_path}...")
        self.tokenizer = CustomTokenizer.from_pretrained(model_path)
        
        config = AutoConfig.from_pretrained(model_path)
        auto_model = AutoModelForCausalLM.from_config(config)
        self.model = DiffusionModel(auto_model, config, None)
        
        # Load model weights
        model_file = os.path.join(model_path, 'pytorch_model.bin')
        model_weights = torch.load(model_file, map_location=self.device)
        self.model.load_state_dict(model_weights, strict=False)
        self.model = self.model.eval()
        print("Model loaded successfully!")
    
    def topk_masking(self, scores, cutoff_len, stochastic=False, temp=1.0):
        """
        Perform top-k masking on scores.
        
        Args:
            scores (torch.Tensor): The scores to mask [batch_size, seq_len]
            cutoff_len (torch.Tensor): The cutoff length for masking [batch_size, 1]
            stochastic (bool): Whether to add noise for stochastic selection
            temp (float): Temperature for noise scaling
            
        Returns:
            torch.Tensor: Boolean mask with 1 for tokens in top-k lowest scores
        """
        if stochastic:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
            _scores = scores + temp * gumbel_noise
        else:
            _scores = scores
            
        sorted_index = _scores.sort(-1)[0]
        cutoff = sorted_index.gather(dim=-1, index=cutoff_len)
        masking = _scores < cutoff
        return masking
    
    def topk_decoding(self, x0, x0_scores, decoding_strategy, init_maskable_mask, t, max_step, noise):
        """
        Perform top-k decoding to determine which tokens to denoise.
        
        Args:
            x0 (torch.Tensor): Current token IDs
            x0_scores (torch.Tensor): Confidence scores for current tokens
            decoding_strategy (str): Strategy in format "<mode>-<schedule>" (e.g. "stochastic0.5-linear")
            init_maskable_mask (torch.Tensor): Boolean mask indicating which positions can be masked
            t (int): Current diffusion step
            max_step (int): Maximum number of diffusion steps
            noise (torch.Tensor or float): Noise to apply to masked positions
            
        Returns:
            torch.Tensor: Updated token IDs with newly masked positions
        """
        # Parse decoding strategy
        topk_mode, schedule = decoding_strategy.split("-")
        
        # Calculate rate based on schedule
        if schedule == "linear":
            rate = t / max_step
        elif schedule == "cosine":
            rate = np.cos((max_step-t) / max_step * np.pi * 0.5)
        else:
            raise NotImplementedError(f"Schedule '{schedule}' not implemented")
        
        # Compute cutoff length for denoising top-k positions
        cutoff_len = (init_maskable_mask.sum(1, keepdim=True) * rate).long()
        
        # Set scores of unmaskable positions to large value so they won't be selected
        _scores_for_topk = x0_scores.masked_fill(~init_maskable_mask, 1000.0)
        
        # Generate mask based on mode
        if topk_mode.startswith("stochastic"):
            noise_scale = float(topk_mode.replace("stochastic", ""))
            lowest_k_mask = self.topk_masking(_scores_for_topk, cutoff_len, 
                                           stochastic=True, temp=noise_scale * rate)
        elif topk_mode == "deterministic":
            lowest_k_mask = self.topk_masking(_scores_for_topk, cutoff_len, stochastic=False)
        else:
            raise NotImplementedError(f"Top-k mode '{topk_mode}' not implemented")
        
        masked_to_noise = lowest_k_mask
        if isinstance(noise, torch.Tensor):
            xt = x0.masked_scatter(masked_to_noise, noise[masked_to_noise])
        elif isinstance(noise, (int, float)):
            xt = x0.masked_fill(masked_to_noise, noise)
        else:
            raise NotImplementedError("noise should be either a tensor or a scalar")

        return xt
    
    def generate_samples(self, inputs, verbose=False):
        """
        Generate samples using the diffusion model.
        
        Args:
            inputs (dict): Input dictionary containing 'input_ids' and 'src_mask'
            verbose (bool): Whether to print intermediate steps
            
        Returns:
            torch.Tensor: Generated token IDs
        """

        self.model.cuda()
        self.model.eval()

        # Move inputs to device
        x = inputs['input_ids'].cuda()
        src_mask = inputs['src_mask'].bool().cuda()
        attention_mask = torch.ones_like(x)
        batch_size = x.size(0)
        
        # Initialize masks
        init_maskable_mask = maskable_mask = ~src_mask
        next_action_position = src_mask.int().argmin(dim=-1, keepdim=True)
        
        # Diffusion process (from T-1 to 0)
        for t in range(self.diffusion_steps-1, -1, -1):
            with torch.no_grad():
                # For first step, mask all maskable positions
                if t == self.diffusion_steps-1:
                    xt = x.masked_fill(maskable_mask, self.tokenizer.mask_token_id)
                
                if verbose:
                    print(f"t={t+1}(in):", self.tokenizer.decode(xt.tolist()[0]))
                
                # Forward pass
                t_tensor = torch.full((batch_size,), t, device=self.device)
                logits = self.model(xt, t_tensor, attention_mask=attention_mask)
                scores = torch.log_softmax(logits, dim=-1)
                scores[:,:,self.tokenizer.vocab_size:] = -1000  # Prevent generating tokens outside vocab
                x0_scores, x0 = scores.max(-1)
                
                # Deal with shift (left-most token will be replaced anyway)
                x0 = torch.cat([x[:,0:1], x0[:, :-1]], dim=1)
                x0_scores = torch.cat([x0_scores[:,0:1], x0_scores[:, :-1]], dim=1)
                
                # Replace output of non-[MASK] positions with xt
                x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])
                
                if verbose:
                    print(f"t={t+1}(out):", self.tokenizer.decode(x0.tolist()[0]))
                
                # Prepare for next step
                if t > 0:
                    # Redecide mask tokens based on confidence scores
                    xt = self.topk_decoding(
                        x0,
                        x0_scores,
                        "stochastic0.5-linear",
                        init_maskable_mask,
                        t,
                        self.diffusion_steps,
                        self.tokenizer.mask_token_id
                    )
                    # Always mask the next action position
                    xt[:, next_action_position] = self.tokenizer.mask_token_id
                else:
                    xt = x0
                    
        return xt
    
    def get_pos(self, board):
        """Convert chess board to string representation for model input"""
        def get_piece_char(piece):
            if piece is None:
                return "."
            elif piece.color == chess.WHITE:
                return piece.symbol().upper()  # White pieces are uppercase
            else:
                return piece.symbol().lower()  # Black pieces are lowercase

        # Generate the src string
        src = ""
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            src += get_piece_char(piece)

        # Add castling rights (handling both old and new python-chess versions)
        castling = ""
        if board.has_kingside_castling_rights(chess.WHITE):
            castling += "K"
        if board.has_queenside_castling_rights(chess.WHITE):
            castling += "Q"
        if board.has_kingside_castling_rights(chess.BLACK):
            castling += "k"
        if board.has_queenside_castling_rights(chess.BLACK):
            castling += "q"
        if not castling:
            castling = "-"

        # Add the rest of the src string components
        src += castling
        src += "-" if board.ep_square is None else chess.square_name(board.ep_square) + " "
        src += str(board.halfmove_clock) + "." + str(board.fullmove_number) + ".."  # Add halfmove and fullmove
        return src
    
    def display_board(self, board):
        """
        Print a nice representation of the board.
        
        Args:
            board (chess.Board): Chess board to display
        """
        print("\n" + str(board) + "\n")
    
    def get_best_move(self, board, previous_move=None, max_attempts=3):
        """
        Get a move from the diffusion model for the current board position.
        
        Args:
            board (chess.Board): The current chess board
            previous_move (str, optional): Previous move in UCI format for context
            max_attempts (int): Maximum number of attempts to get a valid move
            
        Returns:
            chess.Move or None: The selected move, or None if no valid move found
        """
        src = self.get_pos(board)
        print(f"Current position: {src}")
        
        # Check for legal moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None  # No legal moves available (game over)
        
        # Use previous move or default
        initial_move = previous_move or "a1b1"
        
        # Encode input for the model
        src_ids = self.tokenizer.encode(src)
        tgt_ids = self.tokenizer.encode(initial_move)
        input_ids = src_ids + [self.tokenizer.sep_token_id] + tgt_ids + [self.tokenizer.eos_token_id]
        input_ids = torch.tensor(input_ids)[None]
        
        # Pad to max length
        pad = torch.full((1, self.max_length-len(input_ids[0])), self.tokenizer.pad_token_id)
        encoded_input = torch.cat([input_ids, pad], dim=-1)
        
        # Create source mask
        src_mask = torch.tensor([1] * (len(src_ids) + 1) + [0] * (self.max_length-1-len(src_ids)))[None]
        pack_input = {"input_ids": encoded_input, "src_mask": src_mask}
        
        # Try to get a valid move from the model
        for attempt in range(max_attempts):
            try:
                with torch.no_grad():
                    x0 = self.generate_samples(pack_input, verbose=False)
                    res = self.tokenizer.batch_decode(x0.cpu().numpy())[0]
                    print(f"Raw model output: {res}")
                    
                    # Extract move from output
                    try:
                        move_text = res.split('[SEP]')[1].split(' ')[0][:4]
                        if move_text.startswith('.'):
                            move_text = res.split('[SEP]')[1].split(' ')[0][1:5]
                        print(f"Extracted move: {move_text}")
                        move = chess.Move.from_uci(move_text)
                        
                        # Check if move is legal
                        if move in board.legal_moves:
                            return move
                        else:
                            print(f"Move {move_text} is not legal. Retrying...")
                            
                            # Try to find a legal move starting with the same square
                            try:
                                from_square = chess.parse_square(move_text[0:2])
                                matching_legal_moves = [m for m in legal_moves if m.from_square == from_square]
                                if matching_legal_moves:
                                    print(f"Found legal move starting from same square: {matching_legal_moves[0].uci()}")
                                    return matching_legal_moves[0]
                            except Exception as e:
                                print(f"Error finding similar move: {e}")
                                
                    except Exception as e:
                        print(f"Error parsing move: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error processing model output: {e}")
                
        # If no valid move found, select a random legal move
        print("Failed to get valid move from model. Using other engine move instead.")
        return None