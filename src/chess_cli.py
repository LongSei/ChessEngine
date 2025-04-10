from board import Board
from game_rules import GameRules

class ChessCLI:
    def __init__(self):
        """
        Initialize the chess CLI with a board and game rules.
        """

        self.board = Board()
        self.rules = GameRules()
        self.symbol_map = {
            'PAWN': 'P',
            'KNIGHT': 'N',
            'BISHOP': 'B',
            'ROOK': 'R',
            'QUEEN': 'Q',
            'KING': 'K'
        }

    def print_board(self) -> None:
        """
        Display the current state of the chess board in a human-readable format.

        Each cell shows the piece using a symbol:
            - 'w' prefix for WHITE pieces
            - 'b' prefix for BLACK pieces
            - '.' for empty cells
        """

        print("  | a  | b  | c  | d  | e  | f  | g  | h  |")
        print("--|----|----|----|----|----|----|----|----|")
        for i in range(self.board.board_size):
            row = f"{(i + 1)} | "
            for j in range(self.board.board_size):
                piece = self.board.get_piece((i, j))
                if piece is not None:
                    color_val = piece.get_color()
                    color_symbol = 'w' if color_val == 'WHITE' else 'b' if color_val == 'BLACK' else '?'

                    piece_type_val = piece.get_piece_type()
                    symbol = self.symbol_map.get(piece_type_val, '?')
                    
                    row += color_symbol + symbol + " | "
                else:
                    row += ".  | "
            print(row)
        print("\n")

    def parse_move(self, position: str):
        """
        Parses the move position of a piece.

        Args:
            move_str (str): Position in chess notation, e.g., 'a2'.

        Returns:
            tuple: Coordinates on the board (row, col).
        Raises:
            ValueError: If move format is invalid

        Example:
        >>> parse_move('a2')
        (1, 0)
        """

        pos = position.strip()
        if len(pos) == 2 and pos[0].isalpha() and pos[1].isdigit():
            return tuple((int(pos[1]) - 1, ord(pos[0]) - ord('a')))
        else: 
            raise ValueError
        
    def run(self):
        """
        Main game loop.

        Repeatedly prompts the player for moves and applies them to the board using game rules.
        Handles invalid input and move errors gracefully.
        """

        while True:
            self.print_board()
            print(f"Turn {self.rules.turn}")
            from_input = input(f"Enter FROM posion or 'q' to quit: ")
            to_input = input(f"Enter TO position or 'q' to quit: ")

            if from_input.lower() == 'q' or to_input.lower == 'q':
                print("Quit!")
                break

            try:
                from_pos = self.parse_move(from_input)
                to_pos = self.parse_move(to_input)
            except (ValueError, IndexError):
                print("Invalid move format. Please try again.\n")
                continue

            try:
                self.rules.move_piece(self.board, from_pos, to_pos)
            except ValueError as e:
                print(f"Error: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error: {e}")
                continue
            
if __name__ == '__main__':
    game = ChessCLI()
    game.run()
