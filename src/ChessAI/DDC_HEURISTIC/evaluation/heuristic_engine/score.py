import chess

class ChessScore(): 
    def __init__(self, cfg):
        self._material = cfg['material']
        self._phase = cfg['phase']
        self._pst = cfg['pst']
        self._evaluation_parameters = cfg['evaluation_parameters']
        
        self.map_str_to_chess = {
            'pawn': chess.PAWN,
            'knight': chess.KNIGHT,
            'bishop': chess.BISHOP,
            'rook': chess.ROOK,
            'queen': chess.QUEEN,
            'king': chess.KING
        }
        self.map_chess_to_str = {
            chess.PAWN: 'pawn',
            chess.KNIGHT: 'knight',
            chess.BISHOP: 'bishop',
            chess.ROOK: 'rook',
            chess.QUEEN: 'queen',
            chess.KING: 'king'
        }
        
    def pawn_structure_score(self, 
                             board: chess.Board,
                             color: chess.Color) -> int: 
        """
        Calculate the pawn structure score for a given color on the board.

        Args:
            board (chess.Board): The chess board.
            color (chess.Color): The color of the pieces to evaluate (white or black).
            
        Returns:
            int: The pawn structure score for the given color.
            
        Example:
            >>> board = chess.Board()
            >>> score = pawn_structure_score(board, chess.WHITE)
            >>> print(score)
            ... 1234
        """
        score = 0
        pawn_list = board.pieces(chess.PAWN, color)
        
        for sq in pawn_list: 
            file = chess.square_file(sq) # Vertical columns (a-h)
            rank = chess.square_rank(sq) # Horizontal rows (1-8)
            
            # Isolated pawns: Pawns with no pawns on adjacent files
            adj_files = [f for f in (file - 1, file + 1) if 0 <= f < 8]
            if not any(chess.square(f, r) in pawn_list for f in adj_files for r in range(8)):
                score -= self._evaluation_parameters['isolated_pawn_penalty']
                
            # Passed: no enemy pawns ahead on same or adjacent files
            enemy_pawns = board.pieces(chess.PAWN, not color)
            direction = 1 if color else -1
            passed = True
            for f in ([file] + adj_files):
                for r in range(rank + direction, 8 if color else -1, direction):
                    if chess.square(f, r) in enemy_pawns:
                        passed = False
            if passed:
                score += self._evaluation_parameters['passed_pawn_bonus'] * (rank + 1)
        
        return score
    
    def bishop_pair_score(self, 
                          board: chess.Board, 
                          color: chess.Color) -> int:
        """
        Calculate the score for having a pair of bishops.

        Args:
            board (chess.Board): The chess board.
            color (chess.Color): The color of the pieces to evaluate (white or black).

        Returns:
            int: The score for having a pair of bishops.
            
        Example:
            >>> board = chess.Board()
            >>> score = bishop_pair_score(board, chess.WHITE)
            >>> print(score)
            ... 1234
        """
        return self._evaluation_parameters['bishop_pair_bonus'] if len(board.pieces(chess.BISHOP, color)) >= 2 else 0

    def knight_pair_score(self,
                            board: chess.Board,
                            color: chess.Color) -> int:
        """
        Calculate the score for having a pair of knights.
        
        Args:
            board (chess.Board): The chess board.
            color (chess.Color): The color of the pieces to evaluate (white or black).
            
        Returns:
            int: The score for having a pair of knights.
            
        Example:
            >>> board = chess.Board()
            >>> score = knight_pair_score(board, chess.WHITE)
            >>> print(score)
            ... 1234
        """
        return self._evaluation_parameters['knight_pair_bonus'] if len(board.pieces(chess.KNIGHT, color)) >= 2 else 0
    
    def rook_pair_score(self,
                        board: chess.Board,
                        color: chess.Color) -> int:
        """
        Calculate the score for having a pair of rooks.
        
        Args:
            board (chess.Board): The chess board.
            color (chess.Color): The color of the pieces to evaluate (white or black).
            
        Returns:
            int: The score for having a pair of rooks.
            
        Example:
            >>> board = chess.Board()
            >>> score = rook_pair_score(board, chess.WHITE)
            >>> print(score)
            ... 1234
        """
        return self._evaluation_parameters['rook_pair_bonus'] if len(board.pieces(chess.ROOK, color)) >= 2 else 0
    
    def queen_score(self,
                    board: chess.Board,
                    color: chess.Color) -> int:
        """
        Calculate the score for having a queen.

        Args:
            board (chess.Board): The chess board.
            color (chess.Color): The color of the pieces to evaluate (white or black).

        Returns:
            int: The score for having a queen.
            
        Example:
            >>> board = chess.Board()
            >>> score = queen_score(board, chess.WHITE)
            >>> print(score)
            ... 1234
        """
        return self._evaluation_parameters['queen_bonus'] if len(board.pieces(chess.QUEEN, color)) >= 1 else 0
    
    def rook_open_file_score(self, 
                             board: chess.Board, 
                             color: chess.Color) -> int:
        """
        Calculate the score for having a rook on an open file.

        Args:
            board (chess.Board): The chess board.
            color (chess.Color): The color of the pieces to evaluate (white or black).

        Returns:
            int: The score for having a rook on an open file.
            
        Example:
            >>> board = chess.Board()
            >>> score = rook_open_file_score(board, chess.WHITE)
            >>> print(score)
            ... 1234
        """
        score = 0
        pawns = board.pieces(chess.PAWN, True) | board.pieces(chess.PAWN, False)
        for sq in board.pieces(chess.ROOK, color):
            file = chess.square_file(sq)
            if not any(chess.square(file, r) in pawns for r in range(8)):
                score += self._evaluation_parameters['rook_open_file_bonus']
        return score
    
    def mobility_score(self, board: chess.Board) -> int:
        """
        Calculate the mobility score for the current position.

        Args:
            board (chess.Board): The chess board.

        Returns:
            int: The mobility score for the current position.
            
        Example:
            >>> board = chess.Board()
            >>> score = mobility_score(board)
            >>> print(score)
            ... 1234
        """
        white_moves = len(list(board.legal_moves)) if board.turn else len(list(board.copy(stack=False).legal_moves))
        board.turn = not board.turn
        black_moves = len(list(board.legal_moves))
        board.turn = not board.turn
        return (white_moves - black_moves) * self._evaluation_parameters['mobility_bonus']
    
    def game_phase_score(self, game_phase: int) -> float:
        """
        Calculate the game phase score based on the current game phase.

        Args:
            game_phase (int): The current game phase 

        Returns:
            float: The game phase score, normalized between 0 and 1.
            
        Example:
            >>> game_phase = 10
            >>> score = game_phase_score(game_phase)
            >>> print(score)
            ... 0.5
        """
        t = (game_phase - self._phase['endgame']) / (self._phase['opening'] - self._phase['endgame'])
        t = max(0.0, min(1.0, t))
        return t*t  
    
    def improved_stockfish_eval(self, board: chess.Board) -> float:
        """
        Calculate the evaluation score for a given chess board using an improved evaluation method.

        Args:
            board (chess.Board): The chess board to evaluate.

        Returns:
            float: The evaluation score for the given board.
        """
        if board.is_insufficient_material():
            return 0

        # 1) Game phase score using opening material
        gp = 0
        for pt in range(1, 6):  # 1=Pawn..5=Queen
            count = len(board.pieces(pt, True)) + len(board.pieces(pt, False))
            gp += count * self._material['opening'][pt-1]

        # 2) Scores for opening and endgame
        score_o = score_e = 0
        for sq, piece in board.piece_map().items():
            pt = piece.piece_type
            color = 1 if piece.color else -1
            # material
            score_o += self._material['opening'][pt-1] * color
            score_e += self._material['endgame'][pt-1] * color
            # positional PST
            if pt in self._pst:
                sq_eff = chess.square_mirror(sq) if not piece.color else sq
                score_o += self._pst[pt]['opening'][sq_eff] * color
                score_e += self._pst[pt]['endgame'][sq_eff] * color

        # 3) Structural & miscellanea
        for color in [True, False]:
            mult = 1 if color else -1
            score_o += self.bishop_pair_score(board, color) * mult
            score_e += self.bishop_pair_score(board, color) * mult
            score_o += self.pawn_structure_score(board, color) * mult
            score_e += self.pawn_structure_score(board, color) * mult
            score_o += self.rook_open_file_score(board, color) * mult
            score_e += self.rook_open_file_score(board, color) * mult

        # 4) Interpolate
        w_eg = self.game_phase_score(gp)
        raw_score = score_e * w_eg + score_o * (1 - w_eg)

        # 5) Mobility & castling safety bonus
        raw_score += self.mobility_score(board)

        return int(round(raw_score))