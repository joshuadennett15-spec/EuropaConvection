"""Entry point for autonomous autoresearch experiment loop.

Usage:
    python autoresearch/run.py --max-experiments 20 --mode latitude
    python autoresearch/run.py --max-experiments 10 --mode solver
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_cmd(cmd: list, check: bool = True):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(_REPO_ROOT), check=check)


def _create_branch() -> str:
    ts = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    branch = f"autoresearch/run-{ts}"
    _run_cmd(['git', 'checkout', '-b', branch])
    print(f"Created branch: {branch}")
    return branch


def _ensure_baseline():
    ref_dir = Path(__file__).parent / 'reference'
    if not ref_dir.exists():
        print("No reference/ found. Initializing baseline...")
        _run_cmd([sys.executable, str(Path(__file__).parent / 'harness.py'), '--init'])
        _run_cmd(['git', 'add', 'autoresearch/reference/', 'autoresearch/best.json'])
        _run_cmd(['git', 'commit', '-m', 'autoresearch: initialize baseline reference artifacts'])


def main():
    parser = argparse.ArgumentParser(description='Launch autonomous autoresearch loop')
    parser.add_argument('--max-experiments', type=int, default=20, help='Max experiments to run')
    parser.add_argument('--mode', required=True, choices=['solver', 'physics', 'latitude'])
    parser.add_argument('--n-samples', type=int, default=250)
    parser.add_argument('--n-workers', type=int, default=8)
    args = parser.parse_args()

    branch = _create_branch()
    _ensure_baseline()

    print(f"\n{'=' * 60}")
    print(f"AUTORESEARCH — Mode: {args.mode}")
    print(f"Max experiments: {args.max_experiments}")
    print(f"Branch: {branch}")
    print(f"{'=' * 60}")
    print()
    print("Ready for AI agent to begin experiment loop.")
    print("The agent should:")
    print("  1. Read autoresearch/program.md")
    print("  2. Read autoresearch/best.json")
    print("  3. Formulate hypothesis, modify code, run harness, evaluate")
    print()
    print(f"Harness command:")
    print(f"  python autoresearch/harness.py --mode {args.mode} --tag \"<description>\" "
          f"--n-samples {args.n_samples} --n-workers {args.n_workers}")


if __name__ == '__main__':
    main()
