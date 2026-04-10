from typing import List, Set, Tuple

BOARD_SIZE = 8
EMPTY = 0

DIRECTIONS = [
	(-1, -1), (-1, 0), (-1, 1),
	(0, -1),           (0, 1),
	(1, -1),  (1, 0),  (1, 1),
]


def _in_bounds(r: int, c: int) -> bool:
	return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def _get_flips(board: List[List[int]], row: int, col: int, player: int) -> List[Tuple[int, int]]:
	if board[row][col] != EMPTY:
		return []

	flips: List[Tuple[int, int]] = []
	opp = -player
	for dr, dc in DIRECTIONS:
		path: List[Tuple[int, int]] = []
		r, c = row + dr, col + dc
		while _in_bounds(r, c) and board[r][c] == opp:
			path.append((r, c))
			r += dr
			c += dc
		if path and _in_bounds(r, c) and board[r][c] == player:
			flips.extend(path)
	return flips


def valid_moves(board: List[List[int]], player: int) -> Set[Tuple[int, int]]:
	moves: Set[Tuple[int, int]] = set()
	for r in range(BOARD_SIZE):
		for c in range(BOARD_SIZE):
			if _get_flips(board, r, c, player):
				moves.add((r, c))
	return moves

def mobility_heuristic(board: List[List[int]], player: int) -> float:
	"""Mobility heuristic: my_moves - opponent_moves."""
	my_moves = len(valid_moves(board, player))
	opp_moves = len(valid_moves(board, -player))
	return float(my_moves - opp_moves)


__all__ = ["mobility_heuristic"]
