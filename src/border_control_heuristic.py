from typing import List, Tuple


Corner = Tuple[int, int]
CORNERS: List[Corner] = [(0, 0), (0, 7), (7, 0), (7, 7)]


def corner_heuristic(board: List[List[int]], player: int, normalized: bool = False) -> float:
	"""Corner-control heuristic.

	Raw: my_corners - opponent_corners
	Normalized: (my_corners - opponent_corners) / (my_corners + opponent_corners + 1)
	"""
	my_corners = sum(1 for r, c in CORNERS if board[r][c] == player)
	opp_corners = sum(1 for r, c in CORNERS if board[r][c] == -player)
	if normalized:
		return (my_corners - opp_corners) / (my_corners + opp_corners + 1)
	return float(my_corners - opp_corners)


__all__ = ["corner_heuristic"]
