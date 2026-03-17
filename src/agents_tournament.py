import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
	import pygame as _pygame
	pygame: Any = _pygame
except ModuleNotFoundError:
	pygame: Any = None

try:
	from .alpha_beta_pruning_strategy import alphabeta_timed_decision
	from .border_control_heuristic import corner_heuristic
	from .frontier_heuristic import frontier_heuristic
	from .minimax_strategy import minimax_timed_decision
	from .mobility_heuristic import mobility_heuristic
	from .othello_core import (
		BLACK,
		BOARD_SIZE,
		EMPTY,
		GameState,
		WHITE,
		OthelloGame,
		score,
		switch_or_finish,
	)
except ImportError:
	from src.alpha_beta_pruning_strategy import alphabeta_timed_decision
	from border_control_heuristic import corner_heuristic
	from frontier_heuristic import frontier_heuristic
	from src.minimax_strategy import minimax_timed_decision
	from mobility_heuristic import mobility_heuristic
	from othello_core import (
		BLACK,
		BOARD_SIZE,
		EMPTY,
		GameState,
		WHITE,
		OthelloGame,
		score,
		switch_or_finish,
	)


CELL_SIZE = 70
MARGIN = 24
TOP_PANEL = 74
WIDTH = MARGIN * 2 + CELL_SIZE * BOARD_SIZE
HEIGHT = TOP_PANEL + MARGIN + CELL_SIZE * BOARD_SIZE + 70

BG_COLOR = (28, 48, 34)
BOARD_COLOR = (21, 110, 54)
GRID_COLOR = (10, 60, 30)
BLACK_PIECE = (20, 20, 20)
WHITE_PIECE = (240, 240, 240)
TEXT_COLOR = (235, 235, 235)
PASS_COLOR = (240, 170, 60)
WIN_COLOR = (120, 220, 140)

Board = List[List[int]]
Move = Tuple[int, int]
HeuristicFn = Callable[[Board, int], float]
PolicyFn = Callable[[Board, int, float, int, HeuristicFn], Optional[Move]]


@dataclass(frozen=True)
class AgentSpec:
	strategy_name: str
	heuristic_name: str
	policy: PolicyFn
	heuristic: HeuristicFn

	@property
	def name(self) -> str:
		return f"{self.strategy_name}+{self.heuristic_name}"


@dataclass
class PygameRenderer:
	screen: Any
	clock: Any
	font: Any
	small_font: Any
	move_delay_ms: int
	running: bool = True


def _draw_board(surface: Any, state: GameState, title: str, subtitle: str, font: Any, small_font: Any) -> None:
	surface.fill(BG_COLOR)

	black, white = score(state.board)
	turn_name = "Black" if state.current_player == BLACK else "White"
	turn_color = BLACK_PIECE if state.current_player == BLACK else WHITE_PIECE

	title_text = small_font.render(title, True, TEXT_COLOR)
	surface.blit(title_text, (MARGIN, 12))

	score_text = font.render(f"Black: {black}    White: {white}", True, TEXT_COLOR)
	surface.blit(score_text, (MARGIN, 34))

	turn_text = small_font.render(f"Turn: {turn_name}", True, turn_color)
	surface.blit(turn_text, (WIDTH - turn_text.get_width() - MARGIN, 16))

	sub_text = small_font.render(subtitle, True, TEXT_COLOR)
	surface.blit(sub_text, (WIDTH - sub_text.get_width() - MARGIN, 42))

	board_rect = pygame.Rect(MARGIN, TOP_PANEL, CELL_SIZE * BOARD_SIZE, CELL_SIZE * BOARD_SIZE)
	pygame.draw.rect(surface, BOARD_COLOR, board_rect)

	for i in range(BOARD_SIZE + 1):
		x = MARGIN + i * CELL_SIZE
		y = TOP_PANEL + i * CELL_SIZE
		pygame.draw.line(surface, GRID_COLOR, (x, TOP_PANEL), (x, TOP_PANEL + CELL_SIZE * BOARD_SIZE), 2)
		pygame.draw.line(surface, GRID_COLOR, (MARGIN, y), (MARGIN + CELL_SIZE * BOARD_SIZE, y), 2)

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

	message = state.winner_text if state.game_over else state.message
	if message:
		color = WIN_COLOR if state.game_over else PASS_COLOR
		msg_surface = small_font.render(message, True, color)
		surface.blit(msg_surface, (MARGIN, TOP_PANEL + CELL_SIZE * BOARD_SIZE + 20))


def _pump_events(renderer: PygameRenderer) -> None:
	if pygame is None:
		return
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			renderer.running = False
		elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
			renderer.running = False


def _minimax_policy(board: Board, player: int, time_limit_sec: float, max_depth: int, heuristic: HeuristicFn) -> Optional[Move]:
	return minimax_timed_decision(
		board,
		player,
		time_limit_sec=time_limit_sec,
		max_depth=max_depth,
		heuristic=heuristic,
	)


def _alphabeta_policy(board: Board, player: int, time_limit_sec: float, max_depth: int, heuristic: HeuristicFn) -> Optional[Move]:
	return alphabeta_timed_decision(
		board,
		player,
		time_limit_sec=time_limit_sec,
		max_depth=max_depth,
		heuristic=heuristic,
	)


def build_agents() -> List[AgentSpec]:
	strategies: Dict[str, PolicyFn] = {
		"minimax": _minimax_policy,
		"alphabeta": _alphabeta_policy,
	}
	heuristics: Dict[str, HeuristicFn] = {
		"mobility": mobility_heuristic,
		"corner": corner_heuristic,
		"frontier": frontier_heuristic,
	}

	agents: List[AgentSpec] = []
	for strategy_name, policy in strategies.items():
		for heuristic_name, heuristic in heuristics.items():
			agents.append(
				AgentSpec(
					strategy_name=strategy_name,
					heuristic_name=heuristic_name,
					policy=policy,
					heuristic=heuristic,
				)
			)
	return agents


def play_match(
	black_agent: AgentSpec,
	white_agent: AgentSpec,
	time_limit_sec: float,
	max_depth: int,
	renderer: Optional[PygameRenderer] = None,
	game_label: str = "",
) -> dict:
	game = OthelloGame()
	move_count = 0
	start = time.monotonic()

	if renderer is not None:
		_draw_board(
			renderer.screen,
			game.state,
			title=game_label,
			subtitle=f"{black_agent.name} vs {white_agent.name}",
			font=renderer.font,
			small_font=renderer.small_font,
		)
		pygame.display.flip()

	while not game.state.game_over and (renderer is None or renderer.running):
		if renderer is not None:
			_pump_events(renderer)
			if not renderer.running:
				break

		player = game.current_player
		legal = game.legal_moves(player)
		if not legal:
			switch_or_finish(game.state)
			move_count += 1
			if renderer is not None:
				_draw_board(
					renderer.screen,
					game.state,
					title=game_label,
					subtitle=f"{black_agent.name} vs {white_agent.name}",
					font=renderer.font,
					small_font=renderer.small_font,
				)
				pygame.display.flip()
				renderer.clock.tick(60)
				pygame.time.delay(renderer.move_delay_ms)
			continue

		agent = black_agent if player == BLACK else white_agent
		move = agent.policy(game.board, player, time_limit_sec, max_depth, agent.heuristic)
		if move not in legal:
			move = next(iter(legal))

		game.play_move(move[0], move[1])
		move_count += 1

		if renderer is not None:
			_draw_board(
				renderer.screen,
				game.state,
				title=game_label,
				subtitle=f"{black_agent.name} vs {white_agent.name}",
				font=renderer.font,
				small_font=renderer.small_font,
			)
			pygame.display.flip()
			renderer.clock.tick(60)
			pygame.time.delay(renderer.move_delay_ms)

	duration_sec = time.monotonic() - start
	black_score, white_score = score(game.board)

	if black_score > white_score:
		winner = "black"
	elif white_score > black_score:
		winner = "white"
	else:
		winner = "draw"

	return {
		"black_agent": black_agent.name,
		"white_agent": white_agent.name,
		"winner": winner,
		"black_score": black_score,
		"white_score": white_score,
		"disc_diff": black_score - white_score,
		"moves": move_count,
		"duration_sec": round(duration_sec, 4),
	}


def run_tournament(
	repetitions: int,
	time_limit_sec: float,
	max_depth: int,
	show_games: bool = False,
	move_delay_ms: int = 80,
) -> List[dict]:
	agents = build_agents()
	rows: List[dict] = []
	game_id = 1
	renderer: Optional[PygameRenderer] = None
	total_games = len(agents) * len(agents) * repetitions

	if show_games:
		if pygame is None:
			raise RuntimeError("Pygame is not installed. Install dependencies or run without --show-games.")
		pygame.init()
		pygame.display.set_caption("Othello Tournament Viewer")
		renderer = PygameRenderer(
			screen=pygame.display.set_mode((WIDTH, HEIGHT)),
			clock=pygame.time.Clock(),
			font=pygame.font.SysFont("arial", 26, bold=True),
			small_font=pygame.font.SysFont("arial", 20),
			move_delay_ms=max(0, move_delay_ms),
		)

	try:
		for black_agent in agents:
			for white_agent in agents:
				for rep in range(1, repetitions + 1):
					if renderer is not None and not renderer.running:
						print("Tournament interrupted by user.")
						return rows
					label = f"Game {game_id} | Rep {rep}/{repetitions}"
					print(
						f"[START] {label} ({game_id}/{total_games}) | "
						f"black={black_agent.name} vs white={white_agent.name}"
					)
					result = play_match(
						black_agent,
						white_agent,
						time_limit_sec,
						max_depth,
						renderer=renderer,
						game_label=label,
					)
					result["game_id"] = game_id
					result["repetition"] = rep
					rows.append(result)
					print(
						f"[END]   Game {game_id} winner={result['winner']} "
						f"score={result['black_score']}x{result['white_score']} "
						f"moves={result['moves']} time={result['duration_sec']}s"
					)
					game_id += 1
	finally:
		if renderer is not None:
			pygame.quit()
	return rows


def write_csv(rows: List[dict], output_path: str) -> None:
	os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
	fields = [
		"game_id",
		"repetition",
		"black_agent",
		"white_agent",
		"winner",
		"black_score",
		"white_score",
		"disc_diff",
		"moves",
		"duration_sec",
	]
	with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=fields)
		writer.writeheader()
		writer.writerows(rows)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run Othello agent tournament and export CSV results.")
	parser.add_argument("--repetitions", type=int, default=3, help="How many games per ordered matchup.")
	parser.add_argument("--time-limit", type=float, default=0.5, help="Time budget in seconds per move.")
	parser.add_argument("--max-depth", type=int, default=64, help="Maximum iterative deepening depth.")
	parser.add_argument("--output", type=str, default="results/agent_tournament.csv", help="CSV output path.")
	parser.add_argument("--show-games", action="store_true", help="Render tournament games with Pygame.")
	parser.add_argument("--move-delay-ms", type=int, default=80, help="Delay between rendered moves in milliseconds.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	rows = run_tournament(
		repetitions=max(1, args.repetitions),
		time_limit_sec=max(0.01, args.time_limit),
		max_depth=max(1, args.max_depth),
		show_games=args.show_games,
		move_delay_ms=max(0, args.move_delay_ms),
	)
	write_csv(rows, args.output)
	print(f"Tournament finished: {len(rows)} games")
	print(f"CSV saved at: {args.output}")


if __name__ == "__main__":
	main()
