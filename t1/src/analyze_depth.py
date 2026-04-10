#!/usr/bin/env python3
"""
Analyze average depth reached during tournament games.

This script runs the tournament and captures depth statistics printed to stdout,
then computes and displays average depths per strategy/heuristic combination.
"""

import argparse
import re
import subprocess
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_depth_stats(output: str) -> List[Dict]:
	"""
	Extract depth statistics from tournament output.
	
	Looks for patterns like:
	[minimax] player=1 depth=4 nodes=523 reached=3 time=0.0234s
	[alphabeta] player=0 depth=6 nodes=123 reached=5 time=0.0145s
	"""
	pattern = r'\[(\w+)\]\s+player=(\d+)\s+depth=(\d+)\s+nodes=(\d+)\s+reached=(\d+)\s+time=([\d.]+)s'
	matches = re.findall(pattern, output)
	
	stats = []
	for match in matches:
		strategy, player, depth, nodes, reached, elapsed = match
		stats.append({
			'strategy': strategy,
			'player': int(player),
			'max_depth_param': int(depth),
			'nodes_expanded': int(nodes),
			'max_depth_reached': int(reached),
			'elapsed_sec': float(elapsed),
		})
	return stats


def analyze_stats(stats: List[Dict]) -> None:
	"""Compute and display statistics."""
	if not stats:
		print("No depth statistics found. Make sure tournament was run.")
		return
	
	# Group by strategy
	by_strategy: Dict[str, List[int]] = defaultdict(list)
	for stat in stats:
		by_strategy[stat['strategy']].append(stat['max_depth_reached'])
	
	print("\n" + "="*70)
	print("DEPTH STATISTICS SUMMARY")
	print("="*70)
	print()
	
	# Overall stats
	all_depths = [s['max_depth_reached'] for s in stats]
	print(f"Total moves analyzed: {len(stats)}")
	print(f"Overall average depth reached: {sum(all_depths)/len(all_depths):.2f}")
	print(f"Min depth: {min(all_depths)}, Max depth: {max(all_depths)}")
	print()
	
	# By strategy
	print("BY STRATEGY:")
	print("-" * 70)
	for strategy in sorted(by_strategy.keys()):
		depths = by_strategy[strategy]
		avg = sum(depths) / len(depths)
		print(f"  {strategy:12} | Avg: {avg:6.2f} | Min: {min(depths):2} | Max: {max(depths):2} | Moves: {len(depths)}")
	print()
	
	# Depth distribution
	depth_dist: Dict[int, int] = defaultdict(int)
	for depth in all_depths:
		depth_dist[depth] += 1
	
	print("DEPTH DISTRIBUTION:")
	print("-" * 70)
	for d in sorted(depth_dist.keys()):
		count = depth_dist[d]
		pct = (count / len(all_depths)) * 100
		bar = "█" * int(pct / 2)
		print(f"  Depth {d:2}: {count:4} moves ({pct:5.1f}%) {bar}")
	print()


def main():
	parser = argparse.ArgumentParser(
		description="Analyze average depth reached in tournament games"
	)
	parser.add_argument(
		"--repetitions",
		type=int,
		default=1,
		help="Number of repetitions per match (default: 1)"
	)
	parser.add_argument(
		"--time-limit",
		type=float,
		default=0.5,
		help="Time limit per move in seconds (default: 0.5)"
	)
	parser.add_argument(
		"--max-depth",
		type=int,
		default=64,
		help="Maximum iterative deepening depth (default: 64)"
	)
	parser.add_argument(
		"--quick",
		action="store_true",
		help="Run quick test with fewer repetitions"
	)
	parser.add_argument(
		"--super-quick",
		action="store_true",
		help="Run super quick test with just 1 match total"
	)
	args = parser.parse_args()
	
	if args.super_quick:
		args.repetitions = 1
		args.time_limit = 0.5
		args.max_depth = 8
	elif args.quick:
		args.repetitions = 1
		args.time_limit = 0.5
		args.max_depth = 32
	
	print(f"Running tournament with:")
	print(f"  Repetitions: {args.repetitions}")
	print(f"  Time limit: {args.time_limit}s")
	print(f"  Max depth: {args.max_depth}")
	print()
	
	# Run tournament and capture output
	cmd = [
		"uv", "run", "-m", "src.agents_tournament",
		f"--repetitions={args.repetitions}",
		f"--time-limit={args.time_limit}",
		f"--max-depth={args.max_depth}",
		"--output=/tmp/depth_analysis.csv"
	]
	
	print("Starting tournament...")
	result = subprocess.run(cmd, capture_output=True, text=True)
	
	if result.returncode != 0:
		print(f"Tournament failed with code {result.returncode}")
		print("STDERR:", result.stderr)
		return
	
	# Parse stats from output
	stats = parse_depth_stats(result.stdout)
	analyze_stats(stats)


if __name__ == "__main__":
	main()
