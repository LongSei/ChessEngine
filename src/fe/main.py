import pygame
import sys
import os
from pygame.locals import *

# Initialize pygame
pygame.init()
pygame.font.init()

# Font initialization simplified for English only
def init_fonts():
    font_name = "Arial"  # Default English font
    
    return (
        pygame.font.SysFont(font_name, 14),
        pygame.font.SysFont(font_name, 24),
        pygame.font.SysFont(font_name, 48)
    )

font_small, font_medium, font_large = init_fonts()

# Image loading helper
def load_image(file_name):
    try:
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        images_path = os.path.join(base_path, "images")
        path = os.path.join(images_path, file_name)
        
        if os.path.exists(path):
            return pygame.image.load(path).convert_alpha()
        
        if os.path.exists(file_name):
            return pygame.image.load(file_name).convert_alpha()
        
        raise FileNotFoundError(f"Image not found: {file_name}")
    except Exception as e:
        print(f"Error loading image {file_name}: {str(e)}")
        surf = pygame.Surface((50, 50), pygame.SRCALPHA)
        surf.fill((255, 0, 0, 128))  # Red placeholder
        return surf

# Color constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
HIGHLIGHT = (106, 168, 79)
POSSIBLE_MOVE = (255, 213, 79)
CAPTURE_MOVE = (244, 67, 54)
MODAL_BG = (0, 0, 0, 128)
MODAL_CONTENT = (255, 255, 255)

# Screen and board settings
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 800
BOARD_SIZE = 8
SQUARE_SIZE = 80
BOARD_OFFSET_X = (SCREEN_WIDTH - BOARD_SIZE * SQUARE_SIZE) // 2
BOARD_OFFSET_Y = (SCREEN_HEIGHT - BOARD_SIZE * SQUARE_SIZE) // 2

# Create screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Chess Game')

# Game state
class GameState:
    def __init__(self):
        self.current_screen = "welcome"
        self.game_mode = None
        self.current_player = "white"
        self.selected_piece = None
        self.possible_moves = []
        
        # Initial board setup
        self.board = [
            ["r", "n", "b", "q", "k", "b", "n", "r"],
            ["p", "p", "p", "p", "p", "p", "p", "p"],
            [""] * 8,
            [""] * 8,
            [""] * 8,
            [""] * 8,
            ["P", "P", "P", "P", "P", "P", "P", "P"],
            ["R", "N", "B", "Q", "K", "B", "N", "R"]
        ]
        
        self.piece_image_map = {
            "r": "black_rook.png",
            "n": "black_knight.png",
            "b": "black_bishop.png",
            "q": "black_queen.png",
            "k": "black_king.png",
            "p": "black_pawn.png",
            "R": "white_rook.png",
            "N": "white_knight.png",
            "B": "white_bishop.png",
            "Q": "white_queen.png",
            "K": "white_king.png",
            "P": "white_pawn.png"
        }
        
        self.piece_images = self._load_piece_images()
        self.background = self._load_background()

    def _load_piece_images(self):
        pieces = {}
        for piece, img_file in self.piece_image_map.items():
            try:
                img = load_image(img_file)
                if img:
                    img = pygame.transform.scale(img, (SQUARE_SIZE-10, SQUARE_SIZE-10))
                    pieces[piece] = img
                else:
                    raise Exception("Failed to load image")
            except:
                surf = pygame.Surface((SQUARE_SIZE-10, SQUARE_SIZE-10), pygame.SRCALPHA)
                is_white = piece.isupper()
                color = (255, 255, 255, 200) if is_white else (50, 50, 50, 200)
                pygame.draw.circle(surf, color, (SQUARE_SIZE//2-5, SQUARE_SIZE//2-5), SQUARE_SIZE//2-15)
                pieces[piece] = surf
        return pieces

    def _load_background(self):
        bg = load_image("background.jpg")
        if bg:
            return pygame.transform.scale(bg, (SCREEN_WIDTH, SCREEN_HEIGHT))
        else:
            bg = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            bg.fill((50, 50, 50))
            try:
                title = font_large.render("Chess Game", True, WHITE)
                bg.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, SCREEN_HEIGHT//2 - 100))
            except:
                pass
            return bg

game_state = GameState()

def draw_welcome_screen():
    screen.blit(game_state.background, (0, 0))
    
    title = font_large.render("Chess Game", True, WHITE)
    title_rect = title.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 100))
    screen.blit(title, title_rect)
    
    pygame.draw.rect(screen, (51, 51, 51), (SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT//2, 300, 50))
    ai_text = font_medium.render("Play vs AI", True, WHITE)
    ai_rect = ai_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 25))
    screen.blit(ai_text, ai_rect)
    
    pygame.draw.rect(screen, (51, 51, 51), (SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT//2 + 70, 300, 50))
    human_text = font_medium.render("Play vs Human", True, WHITE)
    human_rect = human_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 95))
    screen.blit(human_text, human_rect)

def draw_game_screen():
    screen.fill((225, 208, 181))
    
    pygame.draw.rect(screen, (176, 123, 79), 
                    (BOARD_OFFSET_X - 20, BOARD_OFFSET_Y - 60, 
                     BOARD_SIZE * SQUARE_SIZE + 40, BOARD_SIZE * SQUARE_SIZE + 80))
    
    player_text = font_medium.render(f"Player: {'🤖' if game_state.game_mode == 'AI' else '👤'}", True, WHITE)
    mode_text = font_medium.render(f"Mode: {'AI' if game_state.game_mode == 'AI' else 'Human'}", True, WHITE)
    turn_text = font_medium.render(f"Turn: {'White' if game_state.current_player == 'white' else 'Black'}", True, WHITE)
    
    screen.blit(player_text, (BOARD_OFFSET_X + 20, BOARD_OFFSET_Y - 40))
    screen.blit(mode_text, (BOARD_OFFSET_X + 200, BOARD_OFFSET_Y - 40))
    screen.blit(turn_text, (BOARD_OFFSET_X + 400, BOARD_OFFSET_Y - 40))
    
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            x = BOARD_OFFSET_X + col * SQUARE_SIZE
            y = BOARD_OFFSET_Y + row * SQUARE_SIZE
            
            color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
            pygame.draw.rect(screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))
            
            if game_state.selected_piece and game_state.selected_piece == (row, col):
                s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                s.fill((*HIGHLIGHT, 150))
                screen.blit(s, (x, y))
            
            for move in game_state.possible_moves:
                if move['row'] == row and move['col'] == col:
                    s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                    s.fill((*POSSIBLE_MOVE, 150) if move['type'] == 'move' else (*CAPTURE_MOVE, 150))
                    screen.blit(s, (x, y))
            
            piece = game_state.board[row][col]
            if piece:
                piece_img = game_state.piece_images[piece]
                screen.blit(piece_img, (x + 5, y + 5))
    
    for col in range(BOARD_SIZE):
        x = BOARD_OFFSET_X + col * SQUARE_SIZE + SQUARE_SIZE // 2
        y = BOARD_OFFSET_Y + BOARD_SIZE * SQUARE_SIZE + 5
        text = font_small.render(chr(65 + col), True, BLACK)
        text_rect = text.get_rect(center=(x, y))
        screen.blit(text, text_rect)
    
    for row in range(BOARD_SIZE):
        x = BOARD_OFFSET_X - 15
        y = BOARD_OFFSET_Y + row * SQUARE_SIZE + SQUARE_SIZE // 2
        text = font_small.render(str(8 - row), True, BLACK)
        text_rect = text.get_rect(center=(x, y))
        screen.blit(text, text_rect)
    
    pygame.draw.rect(screen, (51, 51, 51), (BOARD_OFFSET_X + BOARD_SIZE * SQUARE_SIZE - 100, BOARD_OFFSET_Y - 40, 40, 40))
    pygame.draw.rect(screen, (51, 51, 51), (BOARD_OFFSET_X + BOARD_SIZE * SQUARE_SIZE - 50, BOARD_OFFSET_Y - 40, 40, 40))
    
    restart_text = font_medium.render("↻", True, WHITE)
    settings_text = font_medium.render("⚙", True, WHITE)
    screen.blit(restart_text, (BOARD_OFFSET_X + BOARD_SIZE * SQUARE_SIZE - 80, BOARD_OFFSET_Y - 30))
    screen.blit(settings_text, (BOARD_OFFSET_X + BOARD_SIZE * SQUARE_SIZE - 30, BOARD_OFFSET_Y - 30))

def draw_modal():
    s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    s.fill(MODAL_BG)
    screen.blit(s, (0, 0))
    
    modal_width, modal_height = 300, 200
    modal_x = (SCREEN_WIDTH - modal_width) // 2
    modal_y = (SCREEN_HEIGHT - modal_height) // 2
    
    pygame.draw.rect(screen, MODAL_CONTENT, (modal_x, modal_y, modal_width, modal_height))
    pygame.draw.rect(screen, BLACK, (modal_x, modal_y, modal_width, modal_height), 2)
    
    title = font_medium.render("Settings", True, BLACK)
    title_rect = title.get_rect(center=(SCREEN_WIDTH//2, modal_y + 30))
    screen.blit(title, title_rect)
    
    pygame.draw.rect(screen, (51, 51, 51), (SCREEN_WIDTH//2 - 100, modal_y + 80, 200, 40))
    continue_text = font_medium.render("Continue", True, WHITE)
    continue_rect = continue_text.get_rect(center=(SCREEN_WIDTH//2, modal_y + 100))
    screen.blit(continue_text, continue_rect)
    
    pygame.draw.rect(screen, (51, 51, 51), (SCREEN_WIDTH//2 - 100, modal_y + 130, 200, 40))
    home_text = font_medium.render("Main Menu", True, WHITE)
    home_rect = home_text.get_rect(center=(SCREEN_WIDTH//2, modal_y + 150))
    screen.blit(home_text, home_rect)

def get_possible_moves(row, col):
    piece = game_state.board[row][col]
    moves = []
    
    if not piece:
        return moves
    
    if piece == "P":  # White pawn
        if row > 0 and not game_state.board[row - 1][col]:
            moves.append({'row': row - 1, 'col': col, 'type': 'move'})
        
        if row > 0 and col > 0 and game_state.board[row - 1][col - 1] and game_state.board[row - 1][col - 1].islower():
            moves.append({'row': row - 1, 'col': col - 1, 'type': 'capture'})
        if row > 0 and col < 7 and game_state.board[row - 1][col + 1] and game_state.board[row - 1][col + 1].islower():
            moves.append({'row': row - 1, 'col': col + 1, 'type': 'capture'})
    
    elif piece == "p":  # Black pawn
        if row < 7 and not game_state.board[row + 1][col]:
            moves.append({'row': row + 1, 'col': col, 'type': 'move'})
        
        if row < 7 and col > 0 and game_state.board[row + 1][col - 1] and game_state.board[row + 1][col - 1].isupper():
            moves.append({'row': row + 1, 'col': col - 1, 'type': 'capture'})
        if row < 7 and col < 7 and game_state.board[row + 1][col + 1] and game_state.board[row + 1][col + 1].isupper():
            moves.append({'row': row + 1, 'col': col + 1, 'type': 'capture'})
    
    elif piece.lower() == "n":  # Knight
        knight_moves = [
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2)
        ]
        
        for dr, dc in knight_moves:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                target = game_state.board[nr][nc]
                if not target or (piece.isupper() and target.islower()) or (piece.islower() and target.isupper()):
                    moves.append({
                        'row': nr,
                        'col': nc,
                        'type': 'capture' if target else 'move'
                    })
    
    return moves

def is_valid_piece(piece, player):
    return (player == 'white' and piece.isupper()) or (player == 'black' and piece.islower())

def handle_square_click(row, col):
    if game_state.selected_piece:
        move = next((m for m in game_state.possible_moves if m['row'] == row and m['col'] == col), None)
        
        if move:
            game_state.board[row][col] = game_state.board[game_state.selected_piece[0]][game_state.selected_piece[1]]
            game_state.board[game_state.selected_piece[0]][game_state.selected_piece[1]] = ""
            game_state.current_player = 'black' if game_state.current_player == 'white' else 'white'
        
        game_state.selected_piece = None
        game_state.possible_moves = []
    
    elif game_state.board[row][col] and is_valid_piece(game_state.board[row][col], game_state.current_player):
        game_state.selected_piece = (row, col)
        game_state.possible_moves = get_possible_moves(row, col)

def reset_game():
    game_state.board = [
        ["r", "n", "b", "q", "k", "b", "n", "r"],
        ["p", "p", "p", "p", "p", "p", "p", "p"],
        [""] * 8,
        [""] * 8,
        [""] * 8,
        [""] * 8,
        ["P", "P", "P", "P", "P", "P", "P", "P"],
        ["R", "N", "B", "Q", "K", "B", "N", "R"]
    ]
    game_state.current_player = 'white'
    game_state.selected_piece = None
    game_state.possible_moves = []

# Main game loop
show_modal = False
running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        
        elif event.type == MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            
            if game_state.current_screen == "welcome":
                if (SCREEN_WIDTH//2 - 150 <= mouse_pos[0] <= SCREEN_WIDTH//2 + 150 and 
                    SCREEN_HEIGHT//2 <= mouse_pos[1] <= SCREEN_HEIGHT//2 + 50):
                    game_state.current_screen = "game"
                    game_state.game_mode = "AI"
                
                elif (SCREEN_WIDTH//2 - 150 <= mouse_pos[0] <= SCREEN_WIDTH//2 + 150 and 
                      SCREEN_HEIGHT//2 + 70 <= mouse_pos[1] <= SCREEN_HEIGHT//2 + 120):
                    game_state.current_screen = "game"
                    game_state.game_mode = "Human"
            
            elif game_state.current_screen == "game" and not show_modal:
                if (BOARD_OFFSET_X <= mouse_pos[0] <= BOARD_OFFSET_X + BOARD_SIZE * SQUARE_SIZE and 
                    BOARD_OFFSET_Y <= mouse_pos[1] <= BOARD_OFFSET_Y + BOARD_SIZE * SQUARE_SIZE):
                    col = (mouse_pos[0] - BOARD_OFFSET_X) // SQUARE_SIZE
                    row = (mouse_pos[1] - BOARD_OFFSET_Y) // SQUARE_SIZE
                    handle_square_click(row, col)
                
                elif (BOARD_OFFSET_X + BOARD_SIZE * SQUARE_SIZE - 100 <= mouse_pos[0] <= BOARD_OFFSET_X + BOARD_SIZE * SQUARE_SIZE - 60 and 
                      BOARD_OFFSET_Y - 40 <= mouse_pos[1] <= BOARD_OFFSET_Y):
                    reset_game()
                
                elif (BOARD_OFFSET_X + BOARD_SIZE * SQUARE_SIZE - 50 <= mouse_pos[0] <= BOARD_OFFSET_X + BOARD_SIZE * SQUARE_SIZE - 10 and 
                      BOARD_OFFSET_Y - 40 <= mouse_pos[1] <= BOARD_OFFSET_Y):
                    show_modal = True
            
            elif show_modal:
                modal_width, modal_height = 300, 200
                modal_x = (SCREEN_WIDTH - modal_width) // 2
                modal_y = (SCREEN_HEIGHT - modal_height) // 2
                
                if (SCREEN_WIDTH//2 - 100 <= mouse_pos[0] <= SCREEN_WIDTH//2 + 100 and 
                    modal_y + 80 <= mouse_pos[1] <= modal_y + 120):
                    show_modal = False
                
                elif (SCREEN_WIDTH//2 - 100 <= mouse_pos[0] <= SCREEN_WIDTH//2 + 100 and 
                      modal_y + 130 <= mouse_pos[1] <= modal_y + 170):
                    game_state.current_screen = "welcome"
                    show_modal = False
                    reset_game()
    
    if game_state.current_screen == "welcome":
        draw_welcome_screen()
    elif game_state.current_screen == "game":
        draw_game_screen()
        if show_modal:
            draw_modal()
    
    pygame.display.flip()

pygame.quit()
sys.exit()