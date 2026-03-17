from typing import List, Set, Tuple

from .othello_core import BOARD_SIZE, EMPTY, DIRECTIONS, valid_moves

def mobility_heuristic(board: List[List[int]], player: int) -> float:
	"""Mobility heuristic: my_moves - opponent_moves."""
	my_moves = len(valid_moves(board, player))
	opp_moves = len(valid_moves(board, -player))
	return float(my_moves - opp_moves)


__all__ = ["mobility_heuristic"]
