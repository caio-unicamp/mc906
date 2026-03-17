from typing import List

from .othello_core import BOARD_SIZE, EMPTY, DIRECTIONS, in_bounds


def frontier_disks(board: List[List[int]], player: int) -> int:
	"""Count player's frontier disks (adjacent to at least one empty square)."""
	frontier = 0
	for i in range(BOARD_SIZE):
		for j in range(BOARD_SIZE):
			if board[i][j] != player:
				continue
			for di, dj in DIRECTIONS:
				ni, nj = i + di, j + dj
				if in_bounds(ni, nj) and board[ni][nj] == EMPTY:
					frontier += 1
					break
	return frontier


def frontier_heuristic(board: List[List[int]], player: int) -> float:
	"""Frontier score: opp_frontier - my_frontier (we want fewer own frontier disks)."""
	my_frontier = frontier_disks(board, player)
	opp_frontier = frontier_disks(board, -player)
	return float(opp_frontier - my_frontier)


__all__ = ["frontier_disks", "frontier_heuristic"]
