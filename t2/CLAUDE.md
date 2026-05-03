# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Assignment **T2** for Unicamp's MC906 (Intro to AI), Prof. Anderson Rocha — *reinforcement learning for financial management* (`rl-financial-management`). Project context (group, RA-mod) is documented externally; spec is `2026s1_IA_T2.pdf`. Sibling directory `../t1` is the unrelated T1 (Othello) project — do not conflate the two; the parent `../CLAUDE.md` describes T1.

The codebase trains and compares three RL agents (Q-Learning, SARSA, Value Iteration) on a discretized stock-trading MDP built from historical OHLC data.

## Workflow

There is no `pyproject.toml`, test suite, or CLI entrypoint. Work happens in two Jupyter notebooks at the repo root, which import from the `agents/` and `env/` Python packages:

- `data-refinement.ipynb` — ETL: produces `data/amzn_transformed_{train,test}.csv` from raw ticker CSVs in `data/`.
- `training_agents.ipynb` — trains Q-Learning, SARSA, and Value Iteration agents; writes `data/training_results.csv` (~42MB; do not commit casually).

Run notebooks from the repo root so relative imports (`from agents.q_learning import ...`, `from env.env import StockMarketEnv`) resolve. Dependencies are `numpy`, `pandas`, `matplotlib` (install manually — no lockfile).

## Architecture

### State and action space (env/env.py)

`StockMarketEnv` reads a single ticker CSV and exposes a small **discrete** MDP:

- **State**: 4-tuple `(position, volatility_state, macd_state, rsi_state)` — each component is binary `{0, 1}`. State space size = `2*2*2*2 = 16`.
  - `position`: 0 = flat, 1 = long
  - `volatility_state`: 1 if 10-day rolling std of pct change < 0.015
  - `macd_state`: 1 if MACD > signal line
  - `rsi_state`: 1 if RSI > 70
- **Actions**: `{-1: sell, 0: hold, 1: buy}` (note: not `{0,1,2}` — agents map via `action_to_idx`).
- **Reward**: `log_return[t] * position - transaction_cost`, where `transaction_cost = 0.01` only when a buy/sell actually changes `position`.
- **Stochasticity**: `reset()` applies multiplicative price noise (default ±5%) so each episode sees a perturbed copy of the same series — the agent does not see fresh data, just resampled trajectories.

Indicators (MACD/RSI/volatility) are precomputed once per `_calculate_indicators()` call and cached as NumPy arrays (`_vol_state`, `_macd_state`, `_rsi_state`, `_log_returns`) for O(1) lookup in `step()`.

### Agents (agents/)

All three agents share the discrete `(2,2,2,2)` state shape and 3 actions, but use very different learning strategies:

- **`q_learning.py` — `QLearningAgent`**: tabular off-policy TD. Owns the `q_table` of shape `(2,2,2,2,3)` and the `action_to_idx`/`idx_to_action` mappings. Configurable `epsilon_decay_type` ∈ `{constant, linear, exponential}`.
- **`sarsa.py` — `SARSAAgent(QLearningAgent)`**: subclasses Q-Learning to reuse init/epsilon/action selection, overriding only `update()` to use the on-policy `Q(s', a')` instead of `max_a' Q(s', a')`. Training loop must select `next_action` *before* calling `update` (this is the `is_sarsa` branch in `train_agent`).
- **`value_iteration.py` — `ValueIterationAgent`**: model-based. Builds an empirical transition model by sweeping the env's cached indicator arrays (`_vol_state`, `_macd_state`, `_rsi_state`, `_log_returns`) and counting `(state, action) -> next_state, reward` outcomes for both possible `position` values. Then runs Bellman optimality backups on `value_table` and stores the greedy `policy_table`. Because the model is built from the env's *current* (possibly noised) prices, call `refresh_transition_model()` after `env.reset()` if you want a fresh model.

### Result of architectural choices

- The transition model in VI is empirical (counts), not analytic, so it can only see `(state, action)` pairs that occurred in the historical trajectory. `update()` returns `0.0` for unseen pairs.
- Q-Learning/SARSA's `q_table` is indexed directly by the state tuple (numpy fancy indexing), so the env's state representation and `q_table.shape[:-1]` must stay aligned. Adding a state dimension requires updating both `_get_state()` and the agent constructors.

## Conventions

- Python 3 with `numpy`/`pandas`/`matplotlib` only — no RL libraries (Gym, Stable-Baselines, etc.). Match this constraint when adding code.
- Portuguese comments and docstrings are common; keep that style for consistency.
- Player/action constants are integers `-1, 0, 1` throughout — do not rename to `SELL/HOLD/BUY` enums without updating env, agents, and notebooks together.
- Data files in `data/` are mostly raw ticker CSVs (one symbol per file, lowercase ticker name); the assignment uses `amzn_transformed_{train,test}.csv`. `training_results.csv` is generated output.
