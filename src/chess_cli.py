from board import Board

def print_board(board_obj: Board) -> None:
    symbol_map = {
        1: 'P',  # PAWN
        2: 'N',  # KNIGHT
        3: 'B',  # BISHOP
        4: 'R',  # ROOK
        5: 'Q',  # QUEEN
        6: 'K'   # KING
    }
    
    print("  | a  | b  | c  | d  | e  | f  | g  | h  |")
    print("--|----|----|----|----|----|----|----|----|")
    for i in range(board_obj.board_size):
        row = f"{(i + 1)} | "
        for j in range(board_obj.board_size):
            piece = board_obj.get_piece((i, j))
            if piece is not None:
                color_val = piece.get_color
                color_symbol = 'w' if color_val == 1 else 'b' if color_val == 2 else '?'
                
                piece_type_val = piece.get_piece_type
                symbol = symbol_map.get(piece_type_val, '?')
                
                row += color_symbol + symbol + " | "
            else:
                row += ".  | "
        print(row)
    print("\n")


def parse_move(move_str: str):
    tokens = move_str.strip().split()
    from_y = ord(tokens[0][:1]) - ord('a');  from_x = int(tokens[0][1:2]) - 1
    to_y   = ord(tokens[1][:1]) - ord('a');  to_x   = int(tokens[1][1:2]) - 1
    return from_x, from_y, to_x, to_y

def main():
    board_obj = Board()        
    current_turn = 'WHITE'      
    
    # Mapping from numeric color to string representation
    color_str_map = {1: 'WHITE', 2: 'BLACK'}
    
    while True:
        print_board(board_obj)
        move_input = input(f"Turn {current_turn}: Enter move or 'q' to quit: ")
        
        if move_input.lower() == 'q':
            print("Quit!")
            break
        
        try: 
            from_x, from_y, to_x, to_y = parse_move(move_input)
            print(from_x, from_y)
            print(to_x, to_y)
        except ValueError as e:
            print(e)
            continue
        try:
            piece = board_obj.get_piece((from_x, from_y))
        except ValueError as e:
            print(e)
            continue
        
        if piece is None:
            print("There are no pieces in the selected position!")
            continue
        
        if color_str_map.get(piece.get_color, '?') != current_turn:
            print(f"The piece at ({from_x}, {from_y}) is not in {current_turn}.")
            continue
        
        valid_moves = piece.get_valid_move()
        if (to_x, to_y) not in valid_moves:
            print("Invalid move!!!")
            continue
        
        try:
            board_obj.set_piece((to_x, to_y), piece)
            board_obj.board[(from_x, from_y)] = None
        except ValueError as e:
            print(e)
            continue

        current_turn = 'BLACK' if current_turn == 'WHITE' else 'WHITE'

if __name__ == '__main__':
    main()
