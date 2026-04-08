"""Quick multiprocessing smoke test — run from terminal to verify mp.Pool works."""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / 'Europa2D' / 'src'))
sys.path.insert(0, str(_REPO / 'EuropaProjectDJ' / 'src'))
sys.path.insert(0, str(_REPO / 'autoresearch'))

from monte_carlo_2d import MonteCarloRunner2D

if __name__ == '__main__':
    print("Testing multiprocessing with 4 samples, 4 workers...")
    runner = MonteCarloRunner2D(
        n_iterations=4, seed=42, n_workers=4,
        n_lat=5, nx=21,
        ocean_pattern="uniform",
        verbose=True,
    )
    results = runner.run()
    print(f"\nDone! Valid: {results.n_valid}/{results.n_iterations}")
    print(f"H median range: [{results.H_median.min():.1f}, {results.H_median.max():.1f}] km")
    print("Multiprocessing works!")
