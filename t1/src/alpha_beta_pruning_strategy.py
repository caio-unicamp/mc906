import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

try:
	from .border_control_heuristic import corner_heuristic
	from .mobility_heuristic import mobility_heuristic
	from .othello_core import (
		BLACK,
		WHITE,
		is_terminal,
		opponent,
		simulate_move,
		valid_moves,
	)
except ImportError:
	from src.border_control_heuristic import corner_heuristic
	from src.mobility_heuristic import mobility_heuristic
	from src.othello_core import (
		BLACK,
		WHITE,
		is_terminal,
		opponent,
		simulate_move,
		valid_moves,
	)

Board = List[List[int]]
Move = Tuple[int, int]
HeuristicFn = Callable[[Board, int], float]


@dataclass
class SearchStats:
	nodes_expanded: int = 0
	max_depth_reached: int = 0
	elapsed_sec: float = 0.0


def _ordered_children(
	board: Board,
	current_player: int,
	maximizing_player: int,
	quick_heuristic: HeuristicFn,
	maximizing_turn: bool,
) -> List[Tuple[Move, Board]]:
	children: List[Tuple[float, Move, Board]] = []
	for move in valid_moves(board, current_player):
		child = simulate_move(board, move[0], move[1], current_player)
		if child is None:
			continue
		quick_score = quick_heuristic(child, maximizing_player)
		children.append((quick_score, move, child))

	children.sort(key=lambda item: item[0], reverse=maximizing_turn)
	return [(move, child) for _, move, child in children]


def _print_stats(tag: str, player: int, depth: int, stats: SearchStats) -> None:
	print(
		f"[{tag}] player={player} depth={depth} "
		f"nodes={stats.nodes_expanded} reached={stats.max_depth_reached} "
		f"time={stats.elapsed_sec:.4f}s"
	)


def alphabeta_decision(
	board: Board,
	player: int,
	depth: int = 6,
	heuristic: HeuristicFn = mobility_heuristic,
	quick_order_heuristic: HeuristicFn = corner_heuristic,
) -> Tuple[Optional[Move], SearchStats]:
	stats = SearchStats()
	start = time.monotonic()
	moves = valid_moves(board, player)
	if not moves:
		stats.elapsed_sec = time.monotonic() - start
		_print_stats("alphabeta", player, depth, stats)
		return None, stats

	best_value = float("-inf")
	best_move: Optional[Move] = None
	alpha = float("-inf")
	beta = float("inf")

	ordered = _ordered_children(board, player, player, quick_order_heuristic, maximizing_turn=True)
	for move, child in ordered:
		value = _alphabeta(
			child,
			opponent(player),
			player,
			depth - 1,
			alpha,
			beta,
			heuristic,
			quick_order_heuristic,
			stats,
			current_depth=1,
		)
		if value > best_value:
			best_value = value
			best_move = move
		alpha = max(alpha, best_value)

	stats.elapsed_sec = time.monotonic() - start
	_print_stats("alphabeta", player, depth, stats)
	return best_move, stats


def _alphabeta(
	board: Board,
	current_player: int,
	maximizing_player: int,
	depth: int,
	alpha: float,
	beta: float,
	heuristic: HeuristicFn,
	quick_order_heuristic: HeuristicFn,
	stats: SearchStats,
	current_depth: int,
) -> float:
	stats.nodes_expanded += 1
	stats.max_depth_reached = max(stats.max_depth_reached, current_depth)

	if depth == 0 or is_terminal(board):
		return float(heuristic(board, maximizing_player))

	moves = valid_moves(board, current_player)
	if not moves:
		# Forced pass
		return _alphabeta(
			board,
			opponent(current_player),
			maximizing_player,
			depth - 1,
			alpha,
			beta,
			heuristic,
			quick_order_heuristic,
			stats,
			current_depth + 1,
		)

	if current_player == maximizing_player:
		value = float("-inf")
		ordered = _ordered_children(
			board,
			current_player,
			maximizing_player,
			quick_order_heuristic,
			maximizing_turn=True,
		)
		for _, child in ordered:
			value = max(
				value,
				_alphabeta(
					child,
					opponent(current_player),
					maximizing_player,
					depth - 1,
					alpha,
					beta,
					heuristic,
					quick_order_heuristic,
					stats,
					current_depth + 1,
				),
			)
			alpha = max(alpha, value)
			if beta <= alpha:
				break
		return value

	value = float("inf")
	ordered = _ordered_children(
		board,
		current_player,
		maximizing_player,
		quick_order_heuristic,
		maximizing_turn=False,
	)
	for _, child in ordered:
		value = min(
			value,
			_alphabeta(
				child,
				opponent(current_player),
				maximizing_player,
				depth - 1,
				alpha,
				beta,
				heuristic,
				quick_order_heuristic,
				stats,
				current_depth + 1,
			),
		)
		beta = min(beta, value)
		if beta <= alpha:
			break
	return value


def alphabeta_timed_decision(
	board: Board,
	player: int,
	time_limit_sec: float = 0.5,
	max_depth: int = 64,
	heuristic: HeuristicFn = mobility_heuristic,
	quick_order_heuristic: HeuristicFn = corner_heuristic,
) -> Tuple[Optional[Move], SearchStats]:
	stats = SearchStats()
	start = time.monotonic()
	deadline = time.monotonic() + time_limit_sec
	legal = valid_moves(board, player)
	if not legal:
		stats.elapsed_sec = time.monotonic() - start
		_print_stats("alphabeta-timed", player, 0, stats)
		return None, stats

	best_move: Optional[Move] = next(iter(legal))

	for depth in range(1, max_depth + 1):
		if time.monotonic() >= deadline:
			break
		try:
			candidate = _alphabeta_decision_with_deadline(
				board,
				player,
				depth,
				deadline,
				heuristic,
				quick_order_heuristic,
				stats,
			)
			if candidate is not None:
				best_move = candidate
		except TimeoutError:
			break

	stats.elapsed_sec = time.monotonic() - start
	_print_stats("alphabeta-timed", player, max_depth, stats)
	return best_move, stats


def _alphabeta_decision_with_deadline(
	board: Board,
	player: int,
	depth: int,
	deadline: float,
	heuristic: HeuristicFn,
	quick_order_heuristic: HeuristicFn,
	stats: SearchStats,
) -> Optional[Move]:
	moves = valid_moves(board, player)
	if not moves:
		return None

	best_value = float("-inf")
	best_move: Optional[Move] = None
	alpha = float("-inf")
	beta = float("inf")

	ordered = _ordered_children(board, player, player, quick_order_heuristic, maximizing_turn=True)
	for move, child in ordered:
		if time.monotonic() >= deadline:
			raise TimeoutError
		value = _alphabeta_with_deadline(
			child,
			opponent(player),
			player,
			depth - 1,
			alpha,
			beta,
			deadline,
			heuristic,
			quick_order_heuristic,
			stats,
			current_depth=1,
		)
		if value > best_value:
			best_value = value
			best_move = move
		alpha = max(alpha, best_value)

	return best_move


def _alphabeta_with_deadline(
	board: Board,
	current_player: int,
	maximizing_player: int,
	depth: int,
	alpha: float,
	beta: float,
	deadline: float,
	heuristic: HeuristicFn,
	quick_order_heuristic: HeuristicFn,
	stats: SearchStats,
	current_depth: int,
) -> float:
	if time.monotonic() >= deadline:
		raise TimeoutError
	stats.nodes_expanded += 1
	stats.max_depth_reached = max(stats.max_depth_reached, current_depth)

	if depth == 0 or is_terminal(board):
		return float(heuristic(board, maximizing_player))

	moves = valid_moves(board, current_player)
	if not moves:
		return _alphabeta_with_deadline(
			board,
			opponent(current_player),
			maximizing_player,
			depth - 1,
			alpha,
			beta,
			deadline,
			heuristic,
			quick_order_heuristic,
			stats,
			current_depth + 1,
		)

	if current_player == maximizing_player:
		value = float("-inf")
		ordered = _ordered_children(
			board,
			current_player,
			maximizing_player,
			quick_order_heuristic,
			maximizing_turn=True,
		)
		for _, child in ordered:
			value = max(
				value,
				_alphabeta_with_deadline(
					child,
					opponent(current_player),
					maximizing_player,
					depth - 1,
					alpha,
					beta,
					deadline,
					heuristic,
					quick_order_heuristic,
					stats,
					current_depth + 1,
				),
			)
			alpha = max(alpha, value)
			if beta <= alpha:
				break
		return value

	value = float("inf")
	ordered = _ordered_children(
		board,
		current_player,
		maximizing_player,
		quick_order_heuristic,
		maximizing_turn=False,
	)
	for _, child in ordered:
		value = min(
			value,
			_alphabeta_with_deadline(
				child,
				opponent(current_player),
				maximizing_player,
				depth - 1,
				alpha,
				beta,
				deadline,
				heuristic,
				quick_order_heuristic,
				stats,
				current_depth + 1,
			),
		)
		beta = min(beta, value)
		if beta <= alpha:
			break
	return value


__all__ = ["alphabeta_decision", "alphabeta_timed_decision", "BLACK", "WHITE"]