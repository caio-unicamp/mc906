import argparse
import csv
import os
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Tuple


def load_rows(csv_path: str) -> List[dict]:
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def ensure_output_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)


def build_agent_stats(rows: List[dict]) -> Dict[str, dict]:
    stats: Dict[str, dict] = defaultdict(
        lambda: {
            "games": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "disc_diffs": [],
            "durations": [],
            "avg_nodes": [],
            "avg_depth": [],
            "avg_time": [],
        }
    )

    for r in rows:
        black = r["black_agent"]
        white = r["white_agent"]
        winner = r["winner"]
        diff = int(r["disc_diff"])
        duration_raw = r.get("duration_sec")
        duration = float(duration_raw) if duration_raw not in (None, "") else None

        b_nodes = float(r["black_avg_nodes"]) if r.get("black_avg_nodes") not in (None, "") else None
        w_nodes = float(r["white_avg_nodes"]) if r.get("white_avg_nodes") not in (None, "") else None
        b_depth = float(r["black_avg_depth"]) if r.get("black_avg_depth") not in (None, "") else None
        w_depth = float(r["white_avg_depth"]) if r.get("white_avg_depth") not in (None, "") else None
        b_time = float(r["black_avg_time"]) if r.get("black_avg_time") not in (None, "") else None
        w_time = float(r["white_avg_time"]) if r.get("white_avg_time") not in (None, "") else None

        # black perspective
        b = stats[black]
        b["games"] += 1
        b["disc_diffs"].append(diff)
        if duration is not None:
            b["durations"].append(duration)
        if b_nodes is not None: b["avg_nodes"].append(b_nodes)
        if b_depth is not None: b["avg_depth"].append(b_depth)
        if b_time is not None: b["avg_time"].append(b_time)

        if winner == "black":
            b["wins"] += 1
        elif winner == "draw":
            b["draws"] += 1
        else:
            b["losses"] += 1

        # white perspective (invert disc diff sign)
        w = stats[white]
        w["games"] += 1
        w["disc_diffs"].append(-diff)
        if duration is not None:
            w["durations"].append(duration)
        if w_nodes is not None: w["avg_nodes"].append(w_nodes)
        if w_depth is not None: w["avg_depth"].append(w_depth)
        if w_time is not None: w["avg_time"].append(w_time)

        if winner == "white":
            w["wins"] += 1
        elif winner == "draw":
            w["draws"] += 1
        else:
            w["losses"] += 1

    return stats


def _split_agent_name(agent_name: str) -> Tuple[str, str]:
    if "+" in agent_name:
        strategy, heuristic = agent_name.split("+", 1)
        return strategy, heuristic
    return agent_name, "none"


def summarize_by_group(agent_stats: Dict[str, dict], index: int) -> Dict[str, dict]:
    # index=0 strategy, index=1 heuristic from "strategy+heuristic" (or "none")
    grouped: Dict[str, dict] = defaultdict(lambda: {"games": 0, "wins": 0, "draws": 0, "disc": []})
    for agent_name, s in agent_stats.items():
        parts = _split_agent_name(agent_name)
        key = parts[index]
        grouped[key]["games"] += s["games"]
        grouped[key]["wins"] += s["wins"]
        grouped[key]["draws"] += s["draws"]
        grouped[key]["disc"] += s["disc_diffs"]
    return grouped


def build_matchup_matrix(rows: List[dict], agents: List[str]) -> List[List[float]]:
    # matrix[i][j] = winrate of agent i versus agent j (0..1), draw counts as 0.5
    idx = {a: i for i, a in enumerate(agents)}
    wins = [[0.0 for _ in agents] for _ in agents]
    games = [[0 for _ in agents] for _ in agents]

    for r in rows:
        black = r["black_agent"]
        white = r["white_agent"]
        winner = r["winner"]

        i = idx[black]
        j = idx[white]
        games[i][j] += 1
        games[j][i] += 1

        if winner == "black":
            wins[i][j] += 1.0
        elif winner == "white":
            wins[j][i] += 1.0
        else:
            wins[i][j] += 0.5
            wins[j][i] += 0.5

    matrix = [[0.0 for _ in agents] for _ in agents]
    for i in range(len(agents)):
        for j in range(len(agents)):
            if games[i][j] > 0:
                matrix[i][j] = wins[i][j] / games[i][j]
            else:
                matrix[i][j] = 0.0
    return matrix


def plot_agent_win_rates(plt, agent_stats: Dict[str, dict], output_dir: str) -> None:
    agents = sorted(agent_stats.keys())
    win_rates = [agent_stats[a]["wins"] / agent_stats[a]["games"] for a in agents]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(agents, win_rates)
    ax.set_title("Agent Win Rate")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "agent_win_rate.png"), dpi=140)
    plt.close(fig)


def plot_agent_disc_diff(plt, agent_stats: Dict[str, dict], output_dir: str) -> None:
    agents = sorted(agent_stats.keys())
    avg_diff = [mean(agent_stats[a]["disc_diffs"]) for a in agents]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(agents, avg_diff)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Average Disc Difference by Agent")
    ax.set_ylabel("Avg Disc Difference")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "agent_avg_disc_diff.png"), dpi=140)
    plt.close(fig)


def plot_group_win_rates(plt, grouped: Dict[str, dict], title: str, filename: str, output_dir: str) -> None:
    labels = sorted(grouped.keys())
    win_rates = [grouped[k]["wins"] / grouped[k]["games"] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, win_rates)
    ax.set_title(title)
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=140)
    plt.close(fig)


def plot_matchup_heatmap(plt, matrix: List[List[float]], agents: List[str], output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn")
    fig.colorbar(im, ax=ax, label="Win rate")

    ax.set_xticks(range(len(agents)))
    ax.set_yticks(range(len(agents)))
    ax.set_xticklabels(agents, rotation=30, ha="right")
    ax.set_yticklabels(agents)
    ax.set_title("Head-to-Head Win Rate Matrix")

    for i in range(len(agents)):
        for j in range(len(agents)):
            ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "head_to_head_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_duration_histogram(plt, rows: List[dict], output_dir: str) -> None:
    durations = [float(r["duration_sec"]) for r in rows if r.get("duration_sec") not in (None, "")]
    if not durations:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(durations, bins=20)
    ax.set_title("Game Duration Distribution")
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "game_duration_histogram.png"), dpi=140)
    plt.close(fig)


def write_summary_txt(rows: List[dict], agent_stats: Dict[str, dict], output_dir: str) -> None:
    winner_counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        winner_counts[r["winner"]] += 1

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"games={len(rows)}\n")
        f.write(f"winner_counts={dict(winner_counts)}\n\n")
        f.write("agent_stats\n")
        for a in sorted(agent_stats.keys()):
            s = agent_stats[a]
            win_rate = s["wins"] / s["games"] if s["games"] else 0.0
            avg_duration = mean(s["durations"]) if s["durations"] else None
            avg_nodes = mean(s["avg_nodes"]) if s["avg_nodes"] else None
            avg_depth = mean(s["avg_depth"]) if s["avg_depth"] else None
            avg_time = mean(s["avg_time"]) if s["avg_time"] else None
            
            f.write(
                f"{a}: games={s['games']} wins={s['wins']} draws={s['draws']} losses={s['losses']} "
                f"win_rate={win_rate:.3f} avg_disc_diff={mean(s['disc_diffs']):.2f} "
                f"avg_duration={(f'{avg_duration:.3f}' if avg_duration is not None else 'N/A')} "
                f"avg_nodes={(f'{avg_nodes:.2f}' if avg_nodes is not None else 'N/A')} "
                f"avg_depth={(f'{avg_depth:.2f}' if avg_depth is not None else 'N/A')} "
                f"avg_time={(f'{avg_time:.5f}' if avg_time is not None else 'N/A')}\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate analytics plots from tournament CSV.")
    parser.add_argument("--input", type=str, default="results/agent_tournament.csv", help="Tournament CSV path")
    parser.add_argument("--output-dir", type=str, default="plots", help="Directory where plots will be saved")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import importlib
        plt = importlib.import_module("matplotlib.pyplot")
    except ModuleNotFoundError:
        raise SystemExit("matplotlib is required. Run `uv sync` and try again.")

    rows = load_rows(args.input)
    if not rows:
        raise SystemExit(f"No rows found in CSV: {args.input}")

    ensure_output_dir(args.output_dir)

    agent_stats = build_agent_stats(rows)
    agents = sorted(agent_stats.keys())
    strategy_stats = summarize_by_group(agent_stats, index=0)
    heuristic_stats = summarize_by_group(agent_stats, index=1)
    matchup = build_matchup_matrix(rows, agents)

    plot_agent_win_rates(plt, agent_stats, args.output_dir)
    plot_agent_disc_diff(plt, agent_stats, args.output_dir)
    plot_group_win_rates(plt, strategy_stats, "Strategy Win Rate", "strategy_win_rate.png", args.output_dir)
    plot_group_win_rates(plt, heuristic_stats, "Heuristic Win Rate", "heuristic_win_rate.png", args.output_dir)
    plot_matchup_heatmap(plt, matchup, agents, args.output_dir)
    plot_duration_histogram(plt, rows, args.output_dir)
    write_summary_txt(rows, agent_stats, args.output_dir)

    print(f"Saved analytics in: {args.output_dir}")


if __name__ == "__main__":
    main()
