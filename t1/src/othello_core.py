from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

try:
    from .border_control_heuristic import corner_heuristic as _corner_heuristic_impl
    from .mobility_heuristic import mobility_heuristic as _mobility_heuristic_impl
except ImportError:
    from border_control_heuristic import corner_heuristic as _corner_heuristic_impl
    from mobility_heuristic import mobility_heuristic as _mobility_heuristic_impl


EMPTY = 0
BLACK = 1
WHITE = -1

BOARD_SIZE = 8

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


def clone_board(board: List[List[int]]) -> List[List[int]]:
    return [row[:] for row in board]


def opponent(player: int) -> int:
    return -player


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


def has_any_valid_move(board: List[List[int]], player: int) -> bool:
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if get_flips(board, r, c, player):
                return True
    return False


def apply_move(board: List[List[int]], row: int, col: int, player: int) -> bool:
    flips = get_flips(board, row, col, player)
    if not flips:
        return False

    board[row][col] = player
    for r, c in flips:
        board[r][c] = player
    return True


def simulate_move(board: List[List[int]], row: int, col: int, player: int) -> Optional[List[List[int]]]:
    new_board = clone_board(board)
    if apply_move(new_board, row, col, player):
        return new_board
    return None


def score(board: List[List[int]]) -> Tuple[int, int]:
    black = sum(cell == BLACK for row in board for cell in row)
    white = sum(cell == WHITE for row in board for cell in row)
    return black, white


def disc_diff(board: List[List[int]], player: int) -> int:
    black, white = score(board)
    return (black - white) if player == BLACK else (white - black)


def board_full(board: List[List[int]]) -> bool:
    return all(cell != EMPTY for row in board for cell in row)


def is_terminal(board: List[List[int]]) -> bool:
    return board_full(board) or (
        not has_any_valid_move(board, BLACK) and not has_any_valid_move(board, WHITE)
    )


def determine_winner_text(board: List[List[int]]) -> str:
    black, white = score(board)
    if black > white:
        return f"Game over! Black wins {black} x {white}."
    if white > black:
        return f"Game over! White wins {white} x {black}."
    return f"Game over! Draw {black} x {white}."


def mobility_heuristic(board: List[List[int]], player: int) -> int:
    return int(_mobility_heuristic_impl(board, player))


def corner_heuristic(board: List[List[int]], player: int, normalized: bool = False) -> float:
    return _corner_heuristic_impl(board, player, normalized=normalized)


def positional_heuristic(board: List[List[int]], player: int) -> int:
    """Simple reusable heuristic for minimax and alpha-beta agents."""
    weights = [
        [100, -20, 10, 5, 5, 10, -20, 100],
        [-20, -50, -2, -2, -2, -2, -50, -20],
        [10, -2, -1, -1, -1, -1, -2, 10],
        [5, -2, -1, -1, -1, -1, -2, 5],
        [5, -2, -1, -1, -1, -1, -2, 5],
        [10, -2, -1, -1, -1, -1, -2, 10],
        [-20, -50, -2, -2, -2, -2, -50, -20],
        [100, -20, 10, 5, 5, 10, -20, 100],
    ]

    position_score = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            cell = board[r][c]
            if cell == player:
                position_score += weights[r][c]
            elif cell == opponent(player):
                position_score -= weights[r][c]

    mobility = len(valid_moves(board, player)) - len(valid_moves(board, opponent(player)))
    return position_score + (4 * mobility) + (2 * disc_diff(board, player))


def switch_or_finish(state: GameState) -> None:
    """Switch turn. If next player has no moves, pass. If both have no moves, finish game."""
    state.message = ""
    state.current_player *= -1

    if has_any_valid_move(state.board, state.current_player):
        return

    # Forced pass
    passed_player = "Black" if state.current_player == BLACK else "White"
    state.current_player *= -1
    state.message = f"{passed_player} has no legal moves: pass."

    # If original player also has no moves, game ends
    if not has_any_valid_move(state.board, state.current_player):
        state.game_over = True
        state.winner_text = determine_winner_text(state.board)


def reset_state() -> GameState:
    return GameState(board=initial_board(), current_player=BLACK)


class OthelloGame:
    """Reusable game object for UI and AI modules."""

    def __init__(self) -> None:
        self.state = reset_state()

    @property
    def board(self) -> List[List[int]]:
        return self.state.board

    @property
    def current_player(self) -> int:
        return self.state.current_player

    def legal_moves(self, player: Optional[int] = None) -> Set[Tuple[int, int]]:
        p = self.state.current_player if player is None else player
        return valid_moves(self.state.board, p)

    def play_move(self, row: int, col: int) -> bool:
        if self.state.game_over:
            return False
        moved = apply_move(self.state.board, row, col, self.state.current_player)
        if moved:
            switch_or_finish(self.state)
        if not self.state.game_over and board_full(self.state.board):
            self.state.game_over = True
            self.state.winner_text = determine_winner_text(self.state.board)
        return moved

    def reset(self) -> None:
        self.state = reset_state()
