import numpy as np
import matplotlib.pyplot as plt
import time

# ── Phase diagram (eps, alpha) for the ring model, PLUS final K/D matrix ──
# snapshots at a handful of chosen (eps, alpha) points -------------------
#
# This is your ring_phase_diagram_heatmap_100.py sweep, UNCHANGED (same
# grid, same seeds, same output files), with one addition: after the full
# sweep finishes, it re-runs a single simulation at each point listed in
# SNAPSHOT_POINTS below and saves an imshow figure of that run's FINAL
# structural connectivity (K_ij) and functional connectivity (D_ij),
# mirroring Karimian et al. 2019 Figures 2-3.
#
#   dtheta_i/dt = omega_i + (1/N) sum_j K_ij sin[theta_j(t-tau_ij) - theta_i(t)]
#   dK_ij/dt    = eps * (alpha*cos[theta_i(t) - theta_j(t-tau_ij)] - K_ij)
#   tau_ij      = d_ij / v      (d_ij = shortest ring distance between i, j)
#
# STRUCTURAL connectivity is K itself at the end of the run.
# FUNCTIONAL connectivity follows the paper's eq. 8:
#   D_ij = (1/Delta t) * integral_{t_r}^{t_r+Delta t} cos(phi_i(t) - phi_j(t)) dt
# using INSTANTANEOUS (non-delayed) phases. Here it's a proper windowed
# average over exactly the same post-settle window used for the r^2/r2^2
# order parameters (t >= settle), not a live moving average -- this script
# runs to completion before plotting, so there's no need for the
# exponential-moving-average approximation used in the live demo.
#
# NOTE ON COST: the snapshot runs are SEPARATE single simulations (one seed
# each, seed=42), not pulled out of the big sweep's internal seed sequence
# (reproducing the exact rng state the sweep reaches at an arbitrary
# (eps,alpha) cell would require replaying its whole draw order). They are
# the same model/parameters, just an independent reproducible run -- fine
# for visualizing what K/D structure that (eps, alpha, K_init) combination
# settles into. Each snapshot run costs about as much as one grid cell's
# single-seed run (~seconds to ~1 minute at N=100), so keep SNAPSHOT_POINTS
# short unless you want to wait.

# ── Parameters ────────────────────────────────────────────────────────
N          = 100        # oscillators on the ring
OMEGA_STD  = 0.1         # std of the omega distribution
DT         = 0.05
v_fixed    = 30.0          # conduction speed, held fixed for this sweep
T_steps    = 10000           # total integration steps
settle     = 2000            # discard before this step when averaging
n_seeds    = 5                 # independent random networks averaged per grid point

Kc = np.sqrt(8 / np.pi) * OMEGA_STD     # critical coupling for THIS omega spread

INITIAL_CONDITIONS = [
    ("at Kc",    1.0),   # K starts right at the critical coupling
    ("below Kc", 0.5),   # K starts sub-critical, but not zero
    ("zero",     0.0),   # K starts with no coupling at all
]

# (eps, alpha) points to snapshot K/D for, after the main sweep finishes.
# Edit this list freely -- e.g. set it to specific points from the
# r2_mean_grid heatmap that you want to look at structurally. Defaults to
# four corners of the swept grid.
SNAPSHOT_POINTS = [
    (0.0,               1.5),   # eps=0 (K frozen), alpha at max
    (0.15,              1.5),   # eps at max, alpha at max
    (0.15,              0.0),   # eps at max, alpha=0
    (0.15 / 14 * 7,     1.5),   # eps at mid-grid, alpha at max
]

print(f"Kc (sigma={OMEGA_STD}) = {Kc:.4f}")
for label, frac in INITIAL_CONDITIONS:
    print(f"  '{label}' run starts K near {frac*Kc:.4f} ({frac} x Kc)")

eps_values   = np.linspace(0, 0.15, 15)
alpha_values = np.linspace(0, 1.5, 10)     # matches your slider's 0-1.5 range

# ── Ring geometry (fixed for the whole sweep) ────────────────────────────
idx = np.arange(N)
dist = np.abs(idx[:, None] - idx[None, :])
dist = np.minimum(dist, N - dist).astype(float)
tau = dist / v_fixed
delay_steps = np.round(tau / DT).astype(int)
buffer_len = delay_steps.max() + 1
col_idx = np.arange(N)[None, :]

print(f"max delay = {tau.max():.3f} time units ({delay_steps.max()} steps), "
      f"buffer_len = {buffer_len}, v = {v_fixed}")


def run_sweep(K_init):
    """Runs the full (eps, alpha) sweep starting every seed's K near K_init.
    Returns dict of mean/std grids for r^2 and r2^2. UNCHANGED from the
    original script."""
    r_mean_grid  = np.zeros((len(alpha_values), len(eps_values)))
    r2_mean_grid = np.zeros((len(alpha_values), len(eps_values)))
    r_std_grid   = np.zeros((len(alpha_values), len(eps_values)))
    r2_std_grid  = np.zeros((len(alpha_values), len(eps_values)))

    rng = np.random.default_rng(42)   # same seed sequence for both runs
    t_start = time.time()

    for ai, alpha in enumerate(alpha_values):
        for ei, eps in enumerate(eps_values):

            r_seed_vals, r2_seed_vals = [], []

            for s in range(n_seeds):

                omega = rng.normal(0, OMEGA_STD, N)
                theta = rng.uniform(0, 2 * np.pi, N)
                if K_init == 0:
                    K = np.zeros((N, N))     # exactly zero, no jitter to speak of
                else:
                    K = np.abs(rng.normal(loc=K_init, scale=0.02 * K_init, size=(N, N)))
                np.fill_diagonal(K, 0)

                buffer = np.zeros((buffer_len, N))
                for k in range(buffer_len):
                    t_past = (k - buffer_len) * DT
                    buffer[k] = theta + omega * t_past

                r_vals, r2_vals = [], []

                for t in range(T_steps):
                    slot = t % buffer_len

                    row_idx = (slot - delay_steps) % buffer_len
                    theta_delayed = buffer[row_idx, col_idx]

                    diff = theta[:, None] - theta_delayed
                    coupling = (K / N) * np.sin(-diff)
                    dtheta = omega + coupling.sum(axis=1)

                    dK = eps * (alpha * np.cos(diff) - K)
                    np.fill_diagonal(dK, 0)

                    buffer[slot] = theta.copy()

                    theta = theta + DT * dtheta
                    K = K + DT * dK

                    if t >= settle:
                        z1 = np.mean(np.exp(1j * theta))
                        z2 = np.mean(np.exp(2j * theta))
                        r_vals.append(np.abs(z1) ** 2)
                        r2_vals.append(np.abs(z2 - z1 ** 2))

                r_seed_vals.append(np.mean(r_vals))
                r2_seed_vals.append(np.mean(r2_vals))

            r_mean_grid[ai, ei]  = np.mean(r_seed_vals)
            r2_mean_grid[ai, ei] = np.mean(r2_seed_vals)
            r_std_grid[ai, ei]   = np.std(r_seed_vals)
            r2_std_grid[ai, ei]  = np.std(r2_seed_vals)

        print(f"  alpha={alpha:.3f} done ({ai+1}/{len(alpha_values)}), "
              f"elapsed {time.time()-t_start:.0f}s")

    return dict(r_mean_grid=r_mean_grid, r2_mean_grid=r2_mean_grid,
                r_std_grid=r_std_grid, r2_std_grid=r2_std_grid)


def run_single_with_snapshot(K_init, eps, alpha, seed=42):
    """Runs ONE simulation at a fixed (eps, alpha) and returns the final K
    matrix, the post-settle-window-averaged D matrix (paper eq. 8, using
    instantaneous phases), and the mean r^2/r2^2 over that same window."""
    rng = np.random.default_rng(seed)
    omega = rng.normal(0, OMEGA_STD, N)
    theta = rng.uniform(0, 2 * np.pi, N)
    if K_init == 0:
        K = np.zeros((N, N))
    else:
        K = np.abs(rng.normal(loc=K_init, scale=0.02 * K_init, size=(N, N)))
    np.fill_diagonal(K, 0)

    buffer = np.zeros((buffer_len, N))
    for k in range(buffer_len):
        t_past = (k - buffer_len) * DT
        buffer[k] = theta + omega * t_past

    r_vals, r2_vals = [], []
    D_sum = np.zeros((N, N))
    D_count = 0

    for t in range(T_steps):
        slot = t % buffer_len
        row_idx = (slot - delay_steps) % buffer_len
        theta_delayed = buffer[row_idx, col_idx]

        diff = theta[:, None] - theta_delayed
        coupling = (K / N) * np.sin(-diff)
        dtheta = omega + coupling.sum(axis=1)

        dK = eps * (alpha * np.cos(diff) - K)
        np.fill_diagonal(dK, 0)

        buffer[slot] = theta.copy()
        theta = theta + DT * dtheta
        K = K + DT * dK

        if t >= settle:
            z1 = np.mean(np.exp(1j * theta))
            z2 = np.mean(np.exp(2j * theta))
            r_vals.append(np.abs(z1) ** 2)
            r2_vals.append(np.abs(z2 - z1 ** 2))

            diff_instant = theta[:, None] - theta[None, :]   # NOT delayed -- eq. 8
            D_sum += np.cos(diff_instant)
            D_count += 1

    D_final = D_sum / D_count
    np.fill_diagonal(D_final, 0)

    return K, D_final, np.mean(r_vals), np.mean(r2_vals)


# ── Run every initial condition ──────────────────────────────────────────
# NOTE ON RUNTIME: each full sweep (15x10 grid x 5 seeds, N=100, 10000 steps)
# takes roughly several minutes to tens of minutes depending on hardware, so
# N initial conditions means roughly N x that total. Drop n_seeds or shrink
# the grids above for a faster check.

all_results = {}
for label, frac in INITIAL_CONDITIONS:
    print(f"\nRunning '{label}' sweep (K_init = {frac} x Kc = {frac*Kc:.4f})...")
    all_results[label] = run_sweep(K_init=frac * Kc)

save_kwargs = {"eps_values": eps_values, "alpha_values": alpha_values,
                "Kc": Kc, "n_seeds": n_seeds, "v": v_fixed}
for label, results in all_results.items():
    safe_label = label.replace(" ", "_")
    save_kwargs.update({f"{safe_label}_{k}": v for k, v in results.items()})
np.savez("ring_phase_diagram_heatmap_100.npz", **save_kwargs)

# ── Figure: one row per initial condition, r^2 and r2^2 side by side ────
n_rows = len(INITIAL_CONDITIONS)
fig, axes = plt.subplots(n_rows, 2, figsize=(13, 5.2 * n_rows), squeeze=False)

for row, (label, frac) in enumerate(INITIAL_CONDITIONS):
    results = all_results[label]
    panels = [
        (axes[row, 0], results["r_mean_grid"],
         fr'One-cluster mean $r^2$ (start "{label}", $K_0$={frac*Kc:.3f})', r'$r^2$'),
        (axes[row, 1], results["r2_mean_grid"],
         fr'Two-cluster mean $r_2^2$ (start "{label}", $K_0$={frac*Kc:.3f})', r'$r_2^2$'),
    ]
    for ax, grid, title, label_cb in panels:
        im = ax.pcolormesh(eps_values, alpha_values, grid,
                            shading='auto', cmap='viridis', vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, label=label_cb)
        ax.set_xlabel(r'Learning rate $\varepsilon$')
        ax.set_ylabel(r'Learning enhancement $\alpha$')
        ax.set_title(title, fontsize=10)

fig.suptitle(fr'Ring model phase diagram, $N$={N}, $\sigma_\omega$={OMEGA_STD}, $v$={v_fixed}: '
             f'comparing initial coupling (mean over {n_seeds} seeds)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('ring_phase_diagram_heatmap_100.png', dpi=150, bbox_inches='tight')
print("Saved ring_phase_diagram_heatmap_100.png and ring_phase_diagram_heatmap_100.npz")

# ── Snapshot figures: final K and D matrices at SNAPSHOT_POINTS ─────────
# One figure per initial condition; one row per (eps, alpha) snapshot point,
# K and D shown side by side, imshow style matching the paper's Figures 2-3.
print(f"\nRunning {len(SNAPSHOT_POINTS)} K/D snapshot(s) per initial condition...")

for label, frac in INITIAL_CONDITIONS:
    K_init = frac * Kc
    n_pts = len(SNAPSHOT_POINTS)
    fig2, axes2 = plt.subplots(n_pts, 2, figsize=(10, 4.6 * n_pts), squeeze=False)

    for row, (eps, alpha) in enumerate(SNAPSHOT_POINTS):
        t0 = time.time()
        K_final, D_final, r_mean, r2_mean = run_single_with_snapshot(K_init, eps, alpha)
        print(f"  '{label}' eps={eps:.4f} alpha={alpha:.3f}: "
              f"r^2={r_mean:.3f} r2^2={r2_mean:.3f}  ({time.time()-t0:.0f}s)")

        ax_k, ax_d = axes2[row, 0], axes2[row, 1]

        im_k = ax_k.imshow(K_final, cmap='RdBu_r', vmin=-alpha_values[-1], vmax=alpha_values[-1],
                            origin='lower', interpolation='nearest')
        ax_k.set_title(fr'$K_{{ij}}$  ($\varepsilon$={eps:.3f}, $\alpha$={alpha:.2f})', fontsize=10)
        ax_k.set_xlabel('oscillator index')
        ax_k.set_ylabel('oscillator index')
        fig2.colorbar(im_k, ax=ax_k, fraction=0.046, pad=0.04, label=r'$K_{ij}$')

        im_d = ax_d.imshow(D_final, cmap='RdBu_r', vmin=-1, vmax=1,
                            origin='lower', interpolation='nearest')
        ax_d.set_title(fr'$D_{{ij}}$  ($r^2$={r_mean:.2f}, $r_2^2$={r2_mean:.2f})', fontsize=10)
        ax_d.set_xlabel('oscillator index')
        ax_d.set_ylabel('oscillator index')
        fig2.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04, label=r'$D_{ij}$')

    fig2.suptitle(fr'Final structural (K) and functional (D) connectivity, '
                  fr'start "{label}" ($K_0$={K_init:.3f}), $N$={N}',
                  fontsize=12, fontweight='bold')
    plt.tight_layout()
    safe_label = label.replace(" ", "_")
    fname = f'ring_connectivity_snapshots_100_{safe_label}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved {fname}")

print("Done.")