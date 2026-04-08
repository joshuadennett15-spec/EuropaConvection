"""
10,000-sample Monte Carlo with incremental checkpointing.

Kill any time with Ctrl+C — partial results are saved after every batch.
Restart and it resumes from the last checkpoint.

Usage:
    python Europa2D/scripts/run_10k_mc.py           # run (or resume)
    python Europa2D/scripts/run_10k_mc.py --plot     # plot from saved checkpoint
"""
import sys, os, signal, time, argparse, multiprocessing as mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import src

import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from monte_carlo_2d import _run_single_2d_sample

SEED = 42
N_ITER = 10_000
N_LAT = 19
NX = 21
N_WORKERS = max(1, mp.cpu_count() - 1)
CHECKPOINT_EVERY = 100  # save after this many samples
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
CHECKPOINT_PATH = os.path.join(OUT_DIR, 'mc_10k_checkpoint.npz')
FIGURE_PATH = os.path.join(OUT_DIR, 'mc_10k_uniform_summary.png')

# ── Checkpoint I/O ───────────────────────────────────────────────────

def save_checkpoint(H_list, diag_list, n_attempted, elapsed):
    """Save raw results arrays to disk."""
    np.savez(
        CHECKPOINT_PATH,
        H_profiles=np.array(H_list),
        D_cond=np.array([d['D_cond_km'] for d in diag_list]),
        D_conv=np.array([d['D_conv_km'] for d in diag_list]),
        Nu=np.array([d['Nu'] for d in diag_list]),
        lid_fraction=np.array([d['lid_fraction'] for d in diag_list]),
        latitudes_deg=np.linspace(0, 90, N_LAT),
        n_attempted=n_attempted,
        elapsed=elapsed,
    )


def load_checkpoint():
    """Load checkpoint if it exists, else return empty state."""
    if os.path.exists(CHECKPOINT_PATH):
        data = np.load(CHECKPOINT_PATH, allow_pickle=True)
        H_list = list(data['H_profiles'])
        n_attempted = int(data['n_attempted'])
        elapsed = float(data['elapsed'])
        diag_list = []
        for i in range(len(H_list)):
            diag_list.append({
                'D_cond_km': data['D_cond'][i],
                'D_conv_km': data['D_conv'][i],
                'Nu': data['Nu'][i],
                'lid_fraction': data['lid_fraction'][i],
            })
        return H_list, diag_list, n_attempted, elapsed
    return [], [], 0, 0.0


# ── Run ──────────────────────────────────────────────────────────────

def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    H_list, diag_list, n_done, elapsed_prior = load_checkpoint()

    if n_done >= N_ITER:
        print(f"Already complete: {len(H_list)} valid / {n_done} attempted")
        return

    if n_done > 0:
        print(f"Resuming from checkpoint: {len(H_list)} valid / {n_done} attempted, {elapsed_prior:.0f}s elapsed")

    remaining = N_ITER - n_done
    print(f"Running {remaining} remaining samples ({N_WORKERS} workers, checkpointing every {CHECKPOINT_EVERY})...")

    worker = partial(
        _run_single_2d_sample,
        base_seed=SEED,
        n_lat=N_LAT,
        nx=NX,
        dt=1e12,
        use_convection=True,
        max_steps=1500,
        eq_threshold=1e-12,
        initial_thickness=20e3,
        ocean_pattern='uniform',
        ocean_amplitude=None,
        q_star=None,
        rannacher_steps=4,
        coordinate_system='auto',
        grain_latitude_mode='global',
    )

    sample_ids = list(range(n_done, N_ITER))
    t0 = time.time()
    batch_count = 0

    try:
        with mp.Pool(N_WORKERS) as pool:
            for result in pool.imap_unordered(worker, sample_ids, chunksize=10):
                if result is not None:
                    H_list.append(result['H_km'])
                    diag_list.append(result)

                batch_count += 1
                total_done = n_done + batch_count
                elapsed_total = elapsed_prior + (time.time() - t0)

                if batch_count % CHECKPOINT_EVERY == 0:
                    save_checkpoint(H_list, diag_list, total_done, elapsed_total)
                    rate = batch_count / (time.time() - t0)
                    eta = (remaining - batch_count) / rate if rate > 0 else 0
                    print(
                        f"  {total_done:,}/{N_ITER:,} "
                        f"({100*total_done/N_ITER:.1f}%) | "
                        f"Valid: {len(H_list):,} | "
                        f"{rate:.1f} samples/s | "
                        f"ETA: {eta/60:.0f} min",
                        flush=True,
                    )
    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving checkpoint...")

    # Final save
    elapsed_total = elapsed_prior + (time.time() - t0)
    total_done = n_done + batch_count
    save_checkpoint(H_list, diag_list, total_done, elapsed_total)
    print(f"\nSaved: {len(H_list):,} valid / {total_done:,} attempted in {elapsed_total/60:.1f} min")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Run again to resume, or --plot to generate figure")


# ── Plot ─────────────────────────────────────────────────────────────

def plot():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"No checkpoint found at {CHECKPOINT_PATH}")
        return

    data = np.load(CHECKPOINT_PATH, allow_pickle=True)
    H = data['H_profiles']
    lats = data['latitudes_deg']
    n_valid = len(H)
    elapsed = float(data['elapsed'])
    n_attempted = int(data['n_attempted'])

    if n_valid < 10:
        print(f"Only {n_valid} valid samples — need at least 10 for a useful plot")
        return

    D_conv = data['D_conv']
    conv_frac = np.where(H > 0, D_conv / H, 0.0)

    print(f"Plotting {n_valid:,} valid samples from {n_attempted:,} attempted...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Thickness envelope
    ax = axes[0, 0]
    p16, p50, p84 = np.percentile(H, [15.87, 50, 84.13], axis=0)
    p5, p95 = np.percentile(H, [5, 95], axis=0)
    ax.fill_between(lats, p16, p84, alpha=0.3, color='tab:blue', label='1-sigma')
    ax.fill_between(lats, p5, p95, alpha=0.12, color='tab:blue', label='5-95 pctl')
    ax.plot(lats, p50, 'k-', linewidth=2, label='Median')
    ax.set_xlabel('Latitude (deg)')
    ax.set_ylabel('Ice Shell Thickness (km)')
    ax.set_title(f'(a) Thickness Profile  [N={n_valid:,}]')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(0, 90)
    ax.grid(True, alpha=0.25)

    # (b) Equatorial vs polar scatter
    ax = axes[0, 1]
    H_eq, H_pole = H[:, 0], H[:, -1]
    ax.scatter(H_eq, H_pole, s=1, alpha=0.08, color='tab:blue', rasterized=True)
    lim = max(np.percentile(H_eq, 99), np.percentile(H_pole, 99)) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Equatorial Thickness (km)')
    ax.set_ylabel('Polar Thickness (km)')
    ax.set_title('(b) Equator vs Pole')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)

    # (c) Marginal distributions
    ax = axes[1, 0]
    bins = np.linspace(0, max(80, np.percentile(H_pole, 99.5)), 80)
    ax.hist(H_eq, bins=bins, alpha=0.5, color='tab:red', density=True, label='Equator (0 deg)')
    ax.hist(H_pole, bins=bins, alpha=0.5, color='tab:blue', density=True, label='Pole (90 deg)')
    ax.axvline(np.median(H_eq), color='tab:red', linestyle='--', linewidth=1.5)
    ax.axvline(np.median(H_pole), color='tab:blue', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Ice Shell Thickness (km)')
    ax.set_ylabel('Density')
    ax.set_title('(c) Marginal Distributions')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # (d) Convective fraction
    ax = axes[1, 1]
    cf16, cf50, cf84 = np.percentile(conv_frac, [15.87, 50, 84.13], axis=0)
    ax.fill_between(lats, cf16, cf84, alpha=0.3, color='tab:orange', label='1-sigma')
    ax.plot(lats, cf50, 'k-', linewidth=2, label='Median')
    ax.set_xlabel('Latitude (deg)')
    ax.set_ylabel('Convective Fraction (D_conv / H)')
    ax.set_title('(d) Convective Sublayer Fraction')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        f'Europa 2D Monte Carlo: {n_valid:,} samples, '
        f'uniform ocean transport, implicit lateral diffusion',
        fontsize=13, y=0.98,
    )
    stats = (
        f"H_eq median: {np.median(H_eq):.1f} km  |  "
        f"H_pole median: {np.median(H_pole):.1f} km  |  "
        f"Runtime: {elapsed/60:.1f} min  |  "
        f"Valid: {n_valid:,}/{n_attempted:,}"
    )
    fig.text(0.5, 0.01, stats, ha='center', fontsize=10, color='0.4')
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {FIGURE_PATH}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true', help='Plot from checkpoint only')
    args = parser.parse_args()

    if args.plot:
        plot()
    else:
        run()
