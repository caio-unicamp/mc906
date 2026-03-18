import random
from typing import Callable, List, Optional, Tuple

try:
	from .othello_core import valid_moves
except ImportError:
	from src.othello_core import valid_moves

Board = List[List[int]]
Move = Tuple[int, int]
HeuristicFn = Callable[[Board, int], float]


def random_decision(
	board: Board,
	player: int,
	depth: int = 0,
	heuristic: Optional[HeuristicFn] = None,
) -> Optional[Move]:
	"""Naive policy: choose a uniformly random legal move."""
	_ = depth
	_ = heuristic
	legal = list(valid_moves(board, player))
	if not legal:
		return None
	return random.choice(legal)


def random_timed_decision(
	board: Board,
	player: int,
	time_limit_sec: float = 0.0,
	max_depth: int = 0,
	heuristic: Optional[HeuristicFn] = None,
) -> Optional[Move]:
	"""Time/depth-compatible wrapper used by tournament agents."""
	_ = time_limit_sec
	return random_decision(board, player, depth=max_depth, heuristic=heuristic)
