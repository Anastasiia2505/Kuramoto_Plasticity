import numpy as np
import matplotlib.pyplot as plt
import time

# ── Phase diagram (eps, alpha) for YOUR ring model variant ──────────────
# Mirrors your edited ring_sync_lab.py: N=30, omega ~ N(0, OMEGA_STD), ring
# distance-dependent delay tau_ij = d_ij/v.
#
#   dtheta_i/dt = omega_i + (1/N) sum_j K_ij sin[theta_j(t-tau_ij) - theta_i(t)]
#   dK_ij/dt    = eps * (alpha*cos[theta_i(t) - theta_j(t-tau_ij)] - K_ij)
#   tau_ij      = d_ij / v      (d_ij = shortest ring distance between i, j)
#
# This version computes Kc from the ACTUAL omega_std, and runs the sweep
# once per entry in INITIAL_CONDITIONS below, so you can compare several
# starting points for K:
#   "at Kc"        -- K initialized right at the (correct) critical coupling
#   "below Kc"     -- K initialized at a fraction of Kc, but not zero
#   "zero"         -- K initialized at exactly zero (no initial coupling)
# All runs use the same random seeds per grid cell, so any difference between
# the resulting phase diagrams is attributable to the initial condition, not
# seed variation.

# ── Parameters ────────────────────────────────────────────────────────
N          = 100        # oscillators on the ring
OMEGA_STD  = 0.1         # std of the omega distribution (matches sigma used elsewhere in this project)
DT         = 0.05
v_fixed    = 30.0          # conduction speed, held fixed for this sweep
T_steps    = 10000           # total integration steps (150 time units at dt=0.05)
settle     = 2000            # discard before this step when averaging order parameters
n_seeds    = 5                 # independent random networks averaged per grid point

Kc = np.sqrt(8 / np.pi) * OMEGA_STD     # critical coupling for THIS omega spread

# Each entry: (label, fraction of Kc that K is initialized near).
# Add/remove/edit entries here to change how many starting points are compared.
INITIAL_CONDITIONS = [
    ("at Kc",    1.0),   # K starts right at the critical coupling
    ("below Kc", 0.5),   # K starts sub-critical, but not zero
    ("zero",     0.0),   # K starts with no coupling at all
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
    Returns dict of mean/std grids for r^2 and r2^2."""
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


# ── Run every initial condition ──────────────────────────────────────────
# NOTE ON RUNTIME: each full sweep (15x10 grid x 5 seeds, N=30, 3000 steps)
# takes roughly 8-10 minutes on a typical laptop, so N initial conditions
# means roughly N x 8-10 minutes total (three, as configured, is ~25-30 min).
# Drop n_seeds or shrink the grids above for a faster check.

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
print("Done.")