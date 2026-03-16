import sys
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import pygame


# Board values
EMPTY = 0
BLACK = 1
WHITE = -1

BOARD_SIZE = 8
CELL_SIZE = 80
MARGIN = 40
TOP_PANEL = 90
WIDTH = MARGIN * 2 + CELL_SIZE * BOARD_SIZE
HEIGHT = TOP_PANEL + MARGIN + CELL_SIZE * BOARD_SIZE + 70

BG_COLOR = (28, 48, 34)
BOARD_COLOR = (21, 110, 54)
GRID_COLOR = (10, 60, 30)
BLACK_PIECE = (20, 20, 20)
WHITE_PIECE = (240, 240, 240)
HINT_COLOR = (250, 220, 80)
TEXT_COLOR = (235, 235, 235)
PASS_COLOR = (240, 170, 60)
WIN_COLOR = (120, 220, 140)

DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


@dataclass
class GameState:
    board: List[List[int]]
    current_player: int
    game_over: bool = False
    winner_text: str = ""
    message: str = ""


def initial_board() -> List[List[int]]:
    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    mid = BOARD_SIZE // 2
    board[mid - 1][mid - 1] = WHITE
    board[mid][mid] = WHITE
    board[mid - 1][mid] = BLACK
    board[mid][mid - 1] = BLACK
    return board


def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def get_flips(board: List[List[int]], row: int, col: int, player: int) -> List[Tuple[int, int]]:
    if board[row][col] != EMPTY:
        return []

    flips: List[Tuple[int, int]] = []
    opponent = -player

    for dr, dc in DIRECTIONS:
        path: List[Tuple[int, int]] = []
        r, c = row + dr, col + dc

        while in_bounds(r, c) and board[r][c] == opponent:
            path.append((r, c))
            r += dr
            c += dc

        if path and in_bounds(r, c) and board[r][c] == player:
            flips.extend(path)

    return flips


def valid_moves(board: List[List[int]], player: int) -> Set[Tuple[int, int]]:
    moves: Set[Tuple[int, int]] = set()
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if get_flips(board, r, c, player):
                moves.add((r, c))
    return moves


def apply_move(board: List[List[int]], row: int, col: int, player: int) -> bool:
    flips = get_flips(board, row, col, player)
    if not flips:
        return False

    board[row][col] = player
    for r, c in flips:
        board[r][c] = player
    return True


def score(board: List[List[int]]) -> Tuple[int, int]:
    black = sum(cell == BLACK for row in board for cell in row)
    white = sum(cell == WHITE for row in board for cell in row)
    return black, white


def determine_winner_text(board: List[List[int]]) -> str:
    black, white = score(board)
    if black > white:
        return f"Game over! Black wins {black} x {white}."
    if white > black:
        return f"Game over! White wins {white} x {black}."
    return f"Game over! Draw {black} x {white}."


def board_pos_from_mouse(x: int, y: int) -> Optional[Tuple[int, int]]:
    board_x = x - MARGIN
    board_y = y - TOP_PANEL
    if board_x < 0 or board_y < 0:
        return None

    col = board_x // CELL_SIZE
    row = board_y // CELL_SIZE

    if not in_bounds(row, col):
        return None
    return int(row), int(col)


def draw_board(surface: pygame.Surface, state: GameState, font: pygame.font.Font, small_font: pygame.font.Font) -> None:
    surface.fill(BG_COLOR)

    # Header
    black, white = score(state.board)
    turn_name = "Black" if state.current_player == BLACK else "White"
    turn_color = BLACK_PIECE if state.current_player == BLACK else WHITE_PIECE

    score_text = font.render(f"Black: {black}    White: {white}", True, TEXT_COLOR)
    surface.blit(score_text, (MARGIN, 20))

    turn_text = small_font.render(f"Turn: {turn_name}", True, turn_color)
    surface.blit(turn_text, (MARGIN, 55))

    restart_text = small_font.render("R: restart    Q/Esc: quit", True, TEXT_COLOR)
    surface.blit(restart_text, (WIDTH - restart_text.get_width() - MARGIN, 55))

    # Board grid
    board_rect = pygame.Rect(MARGIN, TOP_PANEL, CELL_SIZE * BOARD_SIZE, CELL_SIZE * BOARD_SIZE)
    pygame.draw.rect(surface, BOARD_COLOR, board_rect)

    for i in range(BOARD_SIZE + 1):
        x = MARGIN + i * CELL_SIZE
        y = TOP_PANEL + i * CELL_SIZE
        pygame.draw.line(surface, GRID_COLOR, (x, TOP_PANEL), (x, TOP_PANEL + CELL_SIZE * BOARD_SIZE), 2)
        pygame.draw.line(surface, GRID_COLOR, (MARGIN, y), (MARGIN + CELL_SIZE * BOARD_SIZE, y), 2)

    # Move hints
    if not state.game_over:
        for r, c in valid_moves(state.board, state.current_player):
            cx = MARGIN + c * CELL_SIZE + CELL_SIZE // 2
            cy = TOP_PANEL + r * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.circle(surface, HINT_COLOR, (cx, cy), 8)

    # Pieces
    pad = 6
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            cell = state.board[r][c]
            if cell == EMPTY:
                continue
            cx = MARGIN + c * CELL_SIZE + CELL_SIZE // 2
            cy = TOP_PANEL + r * CELL_SIZE + CELL_SIZE // 2
            color = BLACK_PIECE if cell == BLACK else WHITE_PIECE
            pygame.draw.circle(surface, color, (cx, cy), CELL_SIZE // 2 - pad)
            pygame.draw.circle(surface, (50, 50, 50), (cx, cy), CELL_SIZE // 2 - pad, 1)

    # Footer message
    message = state.winner_text if state.game_over else state.message
    if message:
        color = WIN_COLOR if state.game_over else PASS_COLOR
        msg_surface = small_font.render(message, True, color)
        surface.blit(msg_surface, (MARGIN, TOP_PANEL + CELL_SIZE * BOARD_SIZE + 20))


def switch_or_finish(state: GameState) -> None:
    """Switch turn. If next player has no moves, pass. If both have no moves, finish game."""
    state.message = ""
    state.current_player *= -1

    next_moves = valid_moves(state.board, state.current_player)
    if next_moves:
        return

    # Forced pass
    passed_player = "Black" if state.current_player == BLACK else "White"
    state.current_player *= -1
    state.message = f"{passed_player} has no legal moves: pass."

    # If original player also has no moves, game ends
    if not valid_moves(state.board, state.current_player):
        state.game_over = True
        state.winner_text = determine_winner_text(state.board)


def reset_state() -> GameState:
    return GameState(board=initial_board(), current_player=BLACK)


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Reversi / Othello - Pygame")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("arial", 30, bold=True)
    small_font = pygame.font.SysFont("arial", 24)

    state = reset_state()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    state = reset_state()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not state.game_over:
                pos = board_pos_from_mouse(*event.pos)
                if pos is None:
                    continue

                row, col = pos
                moved = apply_move(state.board, row, col, state.current_player)
                if moved:
                    switch_or_finish(state)

        # If board is full, end game
        if not state.game_over:
            occupied = sum(cell != EMPTY for row in state.board for cell in row)
            if occupied == BOARD_SIZE * BOARD_SIZE:
                state.game_over = True
                state.winner_text = determine_winner_text(state.board)

        draw_board(screen, state, font, small_font)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
