# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Assignment T1 ("Busca Adversarial em Jogos") for Unicamp's MC906 (Intro to AI), Prof. Anderson Rocha. The group was assigned **Othello (Reversi)** based on the RA-mod-3 rule. Written in Portuguese context (README, comments).

The project implements an adversarial search agent for Othello with: Minimax, Alpha-Beta Pruning, Iterative Deepening with time limit, pluggable heuristics, move ordering, node-expansion instrumentation, and an automated tournament for comparative evaluation.

## Assignment Requirements

All core algorithms (Minimax, Alpha-Beta, heuristics, instrumentation) must be the group's own implementation — external AI/search libraries are not allowed, only data structure and visualization helpers.

**Mandatory constraints per agent:**
- Fixed time limit per move (e.g. 0.5s or 1.0s)
- Alpha-Beta pruning
- Move ordering
- Node expansion counting (instrumentation)

**Deliverables:** 5-page technical report (PDF), 30-min oral/video presentation, live demo.

**Report structure:** (1) Formal game modeling as (S,A,T,U), (2) Search algorithms with pruning impact analysis, (3) At least 2 heuristics compared, (4) Experimental evaluation (Minimax vs Alpha-Beta, heuristic comparison, avg depth, avg nodes expanded, avg time/move, win rate vs baseline), (5) Discussion of limitations and bottlenecks.

**Grading:** Agent quality 30%, Report 30%, Presentation+demo 20%, Experimental analysis 20%.

**Optional extras:** Transposition tables, internal tournament between agent versions, empirical branching factor analysis, comparison with experienced human player.

**Suggested Othello heuristics from assignment:** mobility, corner control, piece stability, frontier disks, phase-dependent weights.

## Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Interactive human vs human game (Pygame)
uv run -m src.interactive_pygame

# Run headless tournament
uv run -m src.agents_tournament --repetitions 3 --time-limit 0.5 --max-depth 64 --output results/agent_tournament.csv

# Run tournament with Pygame visualization
uv run -m src.agents_tournament --show-games --move-delay-ms 80 --repetitions 1 --time-limit 0.5 --output results/agent_tournament.csv

# Generate analytics plots from tournament CSV
uv run -m src.tournament_analytics_plots --input results/agent_tournament.csv --output-dir plots
```

## Architecture

All source lives in `src/`. Modules are run as `uv run -m src.<module>`. Imports use relative imports with a `try/except ImportError` fallback to absolute `src.*` imports.

**Core engine** (`othello_core.py`): Board representation as `List[List[int]]` with constants `BLACK=1`, `WHITE=-1`, `EMPTY=0`. `OthelloGame` class wraps game state and turn management. Also contains a `positional_heuristic` combining weighted board positions, mobility, and disc difference.

**Search strategies**: Both `minimax_strategy.py` and `alpha_beta_pruning_strategy.py` expose a `*_timed_decision` function with iterative deepening and move ordering. They share the same signature: `(board, player, time_limit_sec, max_depth, heuristic) -> (move, stats)`. `random_strategy.py` provides a baseline.

**Heuristics**: Each in its own file (`mobility_heuristic.py`, `border_control_heuristic.py`, `frontier_heuristic.py`). All follow the signature `(board, player) -> float`. Higher score = better for `player`.

**Tournament** (`agents_tournament.py`): Builds all strategy+heuristic combinations via `build_agents()`, plays round-robin matches, exports CSV to `results/`. `AgentSpec` dataclass pairs a policy function with a heuristic.

**Analytics** (`tournament_analytics_plots.py`): Reads tournament CSV, generates plots to `plots/`.

## Report & Deliverables

- `2026s1_IA_T1.pdf` — assignment spec from the professor
- `report_missing_sections.tex` — LaTeX content for sections 3 (Algoritmos de Busca), 5 (Avaliação Experimental), and 6 (Discussão e Conclusões). The main .tex report is managed externally (Overleaf); this file contains sections to paste in.
- `results/agent_tournament.csv` — tournament data (1568 games, 7 agents round-robin)
- `plots/` — generated analytics charts and `summary.txt`

## Key Conventions

- Python 3.12+, managed with `uv`
- Dependencies: `matplotlib`, `pygame`
- No test suite exists
- Board is always 8x8; `BOARD_SIZE` constant in `othello_core.py`
- Player constants: `BLACK = 1`, `WHITE = -1`
