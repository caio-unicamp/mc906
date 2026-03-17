import sys
from typing import Optional, Tuple

import pygame

try:
	from .othello_core import (
		BLACK,
		BOARD_SIZE,
		GameState,
		OthelloGame,
		WHITE,
		score,
	)
except ImportError:
	from othello_core import (
		BLACK,
		BOARD_SIZE,
		GameState,
		OthelloGame,
		WHITE,
		score,
	)


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


def board_pos_from_mouse(x: int, y: int) -> Optional[Tuple[int, int]]:
	board_x = x - MARGIN
	board_y = y - TOP_PANEL
	if board_x < 0 or board_y < 0:
		return None

	col = board_x // CELL_SIZE
	row = board_y // CELL_SIZE
	if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
		return None
	return int(row), int(col)


def draw_board(surface: pygame.Surface, state: GameState, game: OthelloGame, font: pygame.font.Font, small_font: pygame.font.Font) -> None:
	surface.fill(BG_COLOR)

	black, white = score(state.board)
	turn_name = "Black" if state.current_player == BLACK else "White"
	turn_color = BLACK_PIECE if state.current_player == BLACK else WHITE_PIECE

	score_text = font.render(f"Black: {black}    White: {white}", True, TEXT_COLOR)
	surface.blit(score_text, (MARGIN, 20))

	turn_text = small_font.render(f"Turn: {turn_name}", True, turn_color)
	surface.blit(turn_text, (MARGIN, 55))

	restart_text = small_font.render("R: restart    Q/Esc: quit", True, TEXT_COLOR)
	surface.blit(restart_text, (WIDTH - restart_text.get_width() - MARGIN, 55))

	board_rect = pygame.Rect(MARGIN, TOP_PANEL, CELL_SIZE * BOARD_SIZE, CELL_SIZE * BOARD_SIZE)
	pygame.draw.rect(surface, BOARD_COLOR, board_rect)

	for i in range(BOARD_SIZE + 1):
		x = MARGIN + i * CELL_SIZE
		y = TOP_PANEL + i * CELL_SIZE
		pygame.draw.line(surface, GRID_COLOR, (x, TOP_PANEL), (x, TOP_PANEL + CELL_SIZE * BOARD_SIZE), 2)
		pygame.draw.line(surface, GRID_COLOR, (MARGIN, y), (MARGIN + CELL_SIZE * BOARD_SIZE, y), 2)

	if not state.game_over:
		for r, c in game.legal_moves():
			cx = MARGIN + c * CELL_SIZE + CELL_SIZE // 2
			cy = TOP_PANEL + r * CELL_SIZE + CELL_SIZE // 2
			pygame.draw.circle(surface, HINT_COLOR, (cx, cy), 8)

	pad = 6
	for r in range(BOARD_SIZE):
		for c in range(BOARD_SIZE):
			cell = state.board[r][c]
			if cell == 0:
				continue
			cx = MARGIN + c * CELL_SIZE + CELL_SIZE // 2
			cy = TOP_PANEL + r * CELL_SIZE + CELL_SIZE // 2
			color = BLACK_PIECE if cell == BLACK else WHITE_PIECE
			pygame.draw.circle(surface, color, (cx, cy), CELL_SIZE // 2 - pad)
			pygame.draw.circle(surface, (50, 50, 50), (cx, cy), CELL_SIZE // 2 - pad, 1)

	message = state.winner_text if state.game_over else state.message
	if message:
		color = WIN_COLOR if state.game_over else PASS_COLOR
		msg_surface = small_font.render(message, True, color)
		surface.blit(msg_surface, (MARGIN, TOP_PANEL + CELL_SIZE * BOARD_SIZE + 20))


def main() -> None:
	pygame.init()
	pygame.display.set_caption("Reversi / Othello - Pygame")
	screen = pygame.display.set_mode((WIDTH, HEIGHT))
	clock = pygame.time.Clock()

	font = pygame.font.SysFont("arial", 30, bold=True)
	small_font = pygame.font.SysFont("arial", 24)

	game = OthelloGame()
	running = True

	while running:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False

			elif event.type == pygame.KEYDOWN:
				if event.key in (pygame.K_ESCAPE, pygame.K_q):
					running = False
				elif event.key == pygame.K_r:
					game.reset()

			elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not game.state.game_over:
				pos = board_pos_from_mouse(*event.pos)
				if pos is None:
					continue
				game.play_move(*pos)

		draw_board(screen, game.state, game, font, small_font)
		pygame.display.flip()
		clock.tick(60)

	pygame.quit()
	sys.exit(0)


if __name__ == "__main__":
	main()
