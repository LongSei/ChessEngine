import pygame
import sys
import os
from pygame.locals import *
import chess
from engine import get_engine_move
import asyncio
import platform

# Initialize pygame
pygame.init()
pygame.font.init()

# Font initialization simplified for English only
def init_fonts():
    font_name = "Arial"
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
        surf.fill((255, 0, 0, 128))
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
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 800
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
        self.chess_board = chess.Board()
        self.board = [[""] * 8 for _ in range(8)]
        self.sync_board()
        self.ai_thinking_time = 0  # Thời gian AI suy nghĩ (giây)
        self.last_ai_move_time = 0  # Thời điểm AI bắt đầu suy nghĩ
        self.engine_thinking = False
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
        self.icon_images = {
            "restart": self._load_icon("restart.png"),
            "settings": self._load_icon("settings.png")
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
    
    def _load_icon(self, filename):
        try:
            img = load_image(filename)
            if img:
                return pygame.transform.scale(img, (30, 30))
        except Exception as e:
            print(f"Error loading icon {filename}: {str(e)}")
            surf = pygame.Surface((30, 30), pygame.SRCALPHA)
            pygame.draw.circle(surf, (200, 200, 200), (15, 15), 14)
            return surf

    def _load_background(self):
        bg = load_image("background.png")
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

    def sync_board(self):
        self.board = [[""] * 8 for _ in range(8)]
        for square in chess.SQUARES:
            piece = self.chess_board.piece_at(square)
            if piece:
                row = 7 - (square // 8)
                col = square % 8
                self.board[row][col] = piece.symbol()

game_state = GameState()

def draw_welcome_screen():
    # Vẽ hình nền với hiệu ứng mờ hoặc gradient
    screen.blit(game_state.background, (0, 0))
    
    # Thêm lớp phủ mờ để làm nổi bật nội dung
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 128))  # Màu đen với độ trong suốt 50%
    screen.blit(overlay, (0, 0))
    
    # Tiêu đề game với hiệu ứng bóng
    title = font_large.render("CHESS MASTER", True, WHITE)
    title_shadow = font_large.render("CHESS MASTER", True, (50, 50, 50))
    title_rect = title.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//4))
    screen.blit(title_shadow, (title_rect.x+3, title_rect.y+3))
    screen.blit(title, title_rect)
    
    # Phụ đề
    subtitle = font_medium.render("Choose your opponent", True, (200, 200, 200))
    subtitle_rect = subtitle.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//4 + 70))
    screen.blit(subtitle, subtitle_rect)
    
    # Nút Play vs AI với hiệu ứng hover
    ai_button_rect = pygame.Rect(SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT//2, 300, 60)
    ai_button_color = (74, 117, 44) if ai_button_rect.collidepoint(pygame.mouse.get_pos()) else (106, 168, 79)
    pygame.draw.rect(screen, ai_button_color, ai_button_rect, border_radius=10)
    pygame.draw.rect(screen, WHITE, ai_button_rect, 2, border_radius=10)  # Viền trắng
    ai_text = font_medium.render("AI VS AI", True, WHITE)
    ai_rect = ai_text.get_rect(center=ai_button_rect.center)
    screen.blit(ai_text, ai_rect)
    
    # Nút Play vs Human với hiệu ứng hover
    human_button_rect = pygame.Rect(SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT//2 + 90, 300, 60)
    human_button_color = (44, 82, 117) if human_button_rect.collidepoint(pygame.mouse.get_pos()) else (66, 135, 245)
    pygame.draw.rect(screen, human_button_color, human_button_rect, border_radius=10)
    pygame.draw.rect(screen, WHITE, human_button_rect, 2, border_radius=10)  # Viền trắng
    human_text = font_medium.render("HUMAN VS AI", True, WHITE)
    human_rect = human_text.get_rect(center=human_button_rect.center)
    screen.blit(human_text, human_rect)
    
    # Footer với thông tin tác giả/phiên bản
    footer = font_small.render("© 2025 Chess Master | Version 1.0", True, (150, 150, 150))
    footer_rect = footer.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT - 30))
    screen.blit(footer, footer_rect)

def draw_game_screen():
    screen.fill((225, 208, 181))
    pygame.draw.rect(screen, (176, 123, 79), 
                    (BOARD_OFFSET_X - 20, BOARD_OFFSET_Y - 60, 
                     BOARD_SIZE * SQUARE_SIZE + 40, BOARD_SIZE * SQUARE_SIZE + 80))
    mode_text = font_medium.render(
        f"Mode: {'AI vs AI' if game_state.game_mode == 'AI_VS_AI' else 'Human vs AI'} | "
        f"Turn: {'WHITE' if game_state.current_player == 'white' else 'BLACK'}",
        True, WHITE
    )
    turn_text = font_medium.render(f"Turn: {'White' if game_state.current_player == 'white' else 'Black'}", True, WHITE)
    # Hiển thị thời gian suy nghĩ của AI
    time_text = font_small.render(
        f"AI Time: {game_state.ai_thinking_time:.2f}s / 10.00s",
        True, WHITE
    )
    screen.blit(mode_text, (BOARD_OFFSET_X + 200, BOARD_OFFSET_Y - 40))
    screen.blit(turn_text, (BOARD_OFFSET_X + 400, BOARD_OFFSET_Y - 40))
    screen.blit(time_text, (BOARD_OFFSET_X + 200, BOARD_OFFSET_Y - 15))
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
    restart_button_rect = pygame.Rect(BOARD_OFFSET_X + BOARD_SIZE * SQUARE_SIZE - 100, BOARD_OFFSET_Y - 40, 40, 40)
    settings_button_rect = pygame.Rect(BOARD_OFFSET_X + BOARD_SIZE * SQUARE_SIZE - 50, BOARD_OFFSET_Y - 40, 40, 40)
    pygame.draw.rect(screen, (150, 150, 150), restart_button_rect)
    pygame.draw.rect(screen, (150, 150, 150), settings_button_rect)
    restart_icon = game_state.icon_images["restart"]
    settings_icon = game_state.icon_images["settings"]
    restart_pos = (
        BOARD_OFFSET_X + BOARD_SIZE * SQUARE_SIZE - 100 + (40 - restart_icon.get_width()) // 2,
        BOARD_OFFSET_Y - 40 + (40 - restart_icon.get_height()) // 2
    )
    settings_pos = (
        BOARD_OFFSET_X + BOARD_SIZE * SQUARE_SIZE - 50 + (40 - settings_icon.get_width()) // 2,
        BOARD_OFFSET_Y - 40 + (40 - settings_icon.get_height()) // 2
    )
    screen.blit(restart_icon, restart_pos)
    screen.blit(settings_icon, settings_pos)
    if show_reset_confirmation:
        s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill(MODAL_BG)
        screen.blit(s, (0, 0))
        modal_width, modal_height = 250, 150
        modal_x = (SCREEN_WIDTH - modal_width) // 2
        modal_y = (SCREEN_HEIGHT - modal_height) // 2
        pygame.draw.rect(screen, MODAL_CONTENT, (modal_x, modal_y, modal_width, modal_height))
        pygame.draw.rect(screen, BLACK, (modal_x, modal_y, modal_width, modal_height), 2)
        line1 = font_medium.render("Are you sure you", True, BLACK)
        line2 = font_medium.render("want to reset the game?", True, BLACK)
        line1_rect = line1.get_rect(center=(modal_x + modal_width//2, modal_y + 40))
        line2_rect = line2.get_rect(center=(modal_x + modal_width//2, modal_y + 70))
        screen.blit(line1, line1_rect)
        screen.blit(line2, line2_rect)
        pygame.draw.rect(screen, (51, 51, 51), (modal_x + 50, modal_y + 90, 50, 30))
        yes_text = font_medium.render("Có", True, WHITE)
        yes_rect = yes_text.get_rect(center=(modal_x + 75, modal_y + 105))
        screen.blit(yes_text, yes_rect)
        pygame.draw.rect(screen, (51, 51, 51), (modal_x + 150, modal_y + 90, 50, 30))
        no_text = font_medium.render("Không", True, WHITE)
        no_rect = no_text.get_rect(center=(modal_x + 175, modal_y + 105))
        screen.blit(no_text, no_rect)

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

def row_col_to_square(row, col):
    file = chr(ord('a') + col)
    rank = str(8 - row)
    return f"{file}{rank}"

def square_to_row_col(square):
    file = ord(square[0]) - ord('a')
    rank = 8 - int(square[1])
    return rank, file

def get_possible_moves(row, col):
    square = row_col_to_square(row, col)
    moves = []
    for move in game_state.chess_board.legal_moves:
        move_str = move.uci()
        if move_str.startswith(square):
            dest_square = move_str[2:4]
            dest_row, dest_col = square_to_row_col(dest_square)
            move_type = 'capture' if game_state.chess_board.is_capture(move) else 'move'
            moves.append({'row': dest_row, 'col': dest_col, 'type': move_type})
    return moves

def handle_square_click(row, col):
     # Chỉ xử lý click nếu là chế độ Human vs AI và đến lượt người chơi (white)
    if game_state.game_mode == "HUMAN_VS_AI" and game_state.current_player != "white":
        return
    if game_state.selected_piece:
        move = next((m for m in game_state.possible_moves if m['row'] == row and m['col'] == col), None)
        if move:
            from_square = row_col_to_square(game_state.selected_piece[0], game_state.selected_piece[1])
            to_square = row_col_to_square(row, col)
            uci_move = f"{from_square}{to_square}"
            chess_move = chess.Move.from_uci(uci_move)
            if chess_move in game_state.chess_board.legal_moves:
                game_state.chess_board.push(chess_move)
                game_state.sync_board()
                game_state.current_player = 'black' if game_state.current_player == 'white' else 'white'
        game_state.selected_piece = None
        game_state.possible_moves = []
    else:
        piece = game_state.board[row][col]
        if piece and ((game_state.current_player == 'white' and piece.isupper()) or 
                      (game_state.current_player == 'black' and piece.islower())):
            game_state.selected_piece = (row, col)
            game_state.possible_moves = get_possible_moves(row, col)

def reset_game():
    game_state.chess_board = chess.Board()
    game_state.sync_board()
    game_state.current_player = 'white'
    game_state.selected_piece = None
    game_state.possible_moves = []
    game_state.ai_thinking_time = 0
    game_state.last_ai_move_time = 0
    game_state.engine_thinking = False

async def handle_ai_move():
    if game_state.game_mode == "AI_VS_AI" or \
       (game_state.game_mode == "HUMAN_VS_AI" and game_state.current_player == "black"):
        
        if game_state.engine_thinking:
            return
            
        game_state.engine_thinking = True
        game_state.last_ai_move_time = pygame.time.get_ticks()
        
        try:
            move = await asyncio.wait_for(
                asyncio.to_thread(get_engine_move, game_state.chess_board),
                timeout=10
            )
            
            game_state.ai_thinking_time = (pygame.time.get_ticks() - game_state.last_ai_move_time) / 1000
            
            if move in game_state.chess_board.legal_moves:
                game_state.chess_board.push(move)
                game_state.sync_board()
                game_state.current_player = 'white' if game_state.current_player == 'black' else 'black'
                
        except asyncio.TimeoutError:
            game_state.ai_thinking_time = 10.0
            print("AI took too long to move!")
            # Xử lý khi quá thời gian
        finally:
            game_state.engine_thinking = False

async def main():
    global show_modal, show_reset_confirmation, running
    show_modal = False
    show_reset_confirmation = False
    running = True
    FPS = 60

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            
            elif event.type == MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                
                if game_state.current_screen == "welcome":
                    # AI vs AI mode
                    if (SCREEN_WIDTH//2 - 150 <= mouse_pos[0] <= SCREEN_WIDTH//2 + 150 and 
                        SCREEN_HEIGHT//2 <= mouse_pos[1] <= SCREEN_HEIGHT//2 + 60):
                        game_state.current_screen = "game"
                        game_state.game_mode = "AI_VS_AI"  # Chế độ mới
                        reset_game()
                    
                    # Human vs AI mode
                    elif (SCREEN_WIDTH//2 - 150 <= mouse_pos[0] <= SCREEN_WIDTH//2 + 150 and 
                          SCREEN_HEIGHT//2 + 90 <= mouse_pos[1] <= SCREEN_HEIGHT//2 + 150):
                        game_state.current_screen = "game"
                        game_state.game_mode = "HUMAN_VS_AI"  # Chế độ mới
                        reset_game()
                
                elif game_state.current_screen == "game" and not show_modal and not show_reset_confirmation:
                    if (BOARD_OFFSET_X <= mouse_pos[0] <= BOARD_OFFSET_X + BOARD_SIZE * SQUARE_SIZE and 
                        BOARD_OFFSET_Y <= mouse_pos[1] <= BOARD_OFFSET_Y + BOARD_SIZE * SQUARE_SIZE):
                        col = (mouse_pos[0] - BOARD_OFFSET_X) // SQUARE_SIZE
                        row = (mouse_pos[1] - BOARD_OFFSET_Y) // SQUARE_SIZE
                        handle_square_click(row, col)
                    
                    elif (BOARD_OFFSET_X + BOARD_SIZE * SQUARE_SIZE - 100 <= mouse_pos[0] <= BOARD_OFFSET_X + BOARD_SIZE * SQUARE_SIZE - 60 and 
                          BOARD_OFFSET_Y - 40 <= mouse_pos[1] <= BOARD_OFFSET_Y):
                        show_reset_confirmation = True
                    
                    elif (BOARD_OFFSET_X + BOARD_SIZE * SQUARE_SIZE - 50 <= mouse_pos[0] <= BOARD_OFFSET_X + BOARD_SIZE * SQUARE_SIZE - 10 and 
                          BOARD_OFFSET_Y - 40 <= mouse_pos[1] <= BOARD_OFFSET_Y):
                        show_modal = True
                
                elif show_reset_confirmation:
                    modal_width, modal_height = 250, 150
                    modal_x = (SCREEN_WIDTH - modal_width) // 2
                    modal_y = (SCREEN_HEIGHT - modal_height) // 2
                    
                    if (modal_x + 50 <= mouse_pos[0] <= modal_x + 100 and 
                        modal_y + 90 <= mouse_pos[1] <= modal_y + 120):
                        reset_game()
                        show_reset_confirmation = False
                    
                    elif (modal_x + 150 <= mouse_pos[0] <= modal_x + 200 and 
                          modal_y + 90 <= mouse_pos[1] <= modal_y + 120):
                        show_reset_confirmation = False
                
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
        
        await handle_ai_move()
        pygame.display.flip()
        await asyncio.sleep(1.0 / FPS)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())