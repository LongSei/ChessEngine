from board import Board

class ChessCLI:
    def __init__(self):
        self.board = Board()
        self.current_turn = 'WHITE'
        self.symbol_map = {
            'PAWN': 'P',
            'KNIGHT': 'N',
            'BISHOP': 'B',
            'ROOK': 'R',
            'QUEEN': 'Q',
            'KING': 'K'
        }

    def print_board(self) -> None:
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

    def parse_move(self, move_str: str):
        move_str = move_str.strip()
        move_byte = move_str.encode()
        return (
            move_byte[1] - 49,  # from_x 
            move_byte[0] - 97,  # from_y 
            move_byte[4] - 49,  # to_x
            move_byte[3] - 97,  # to_y
        )

    def run(self):
        while True:
            self.print_board()
            move_input = input(f"Turn {self.current_turn}: Enter move or 'q' to quit: ")

            if move_input.lower() == 'q':
                print("Quit!")
                break

            try:
                from_x, from_y, to_x, to_y = self.parse_move(move_input)
            except (ValueError, IndexError):
                print("Invalid move format. Please try again.")
                continue

            try:
                piece = self.board.get_piece((from_x, from_y))
            except ValueError as e:
                print(e)
                continue

            if piece is None:
                print("There are no pieces in the selected position!")
                continue

            if piece.get_color() != self.current_turn:
                print(f"The piece at ({from_x}, {from_y}) is not in {self.current_turn}.")
                continue

            valid_moves = piece.get_valid_move()
            if (to_x, to_y) not in valid_moves:
                print("Invalid move!!!")
                continue

            try:
                self.board.set_piece((to_x, to_y), piece)
                self.board.board[(from_x, from_y)] = None
            except ValueError as e:
                print(e)
                continue
            
            self.current_turn = 'BLACK' if self.current_turn == 'WHITE' else 'WHITE'

if __name__ == '__main__':
    game = ChessCLI()
    game.run()
