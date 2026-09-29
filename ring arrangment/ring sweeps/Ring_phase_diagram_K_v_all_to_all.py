import numpy as np
import matplotlib.pyplot as plt
import time

# ── Phase diagram over (eps, alpha) for the RING-embedded model ────────
# Same idea as phase_diagram_multiseed_delay.py, but instead of one shared
# delay tau for every pair, oscillators sit on a 1D ring (Timms & English,
# 2014) and the delay between i and j depends on how far apart they are on
# the ring: tau_ij = d_ij / v, using the same "shortest way around" ring
# distance as ring_sync_lab.py.
#
#   dtheta_i/dt = omega_i + (1/N) sum_j K_ij sin[theta_j(t-tau_ij) - theta_i(t)]
#   dK_ij/dt    = eps * (alpha*cos[theta_i(t) - theta_j(t-tau_ij)] - K_ij)
#   tau_ij      = d_ij / v      (d_ij = shortest ring distance between i, j)
#
# For each (eps, alpha) grid point, n_seeds independent random networks are
# run and the settled order parameters r^2 and r2^2 are averaged.

# ── Parameters ────────────────────────────────────────────────────────
N       = 150      # oscillators on the ring
sigma   = 0.1        # std of frequency distribution
dt      = 0.1
T       = 2000       # total time steps
settle  = 1500        # discard first `settle` steps, average the rest
K0      = 0.75        # initial coupling
n_seeds = 5            # independent random networks averaged per grid point
v       = 8.0           # ring conduction velocity (tau_ij = d_ij / v); smaller v = longer delays

# NOTE ON RUNTIME: this is the ring version of phase_diagram_multiseed_delay.py.
# The per-step cost is similar (both do O(N^2) array work each step), so expect
# roughly the same ~20+ minutes for the full 15x10 grid x 5 seeds. If that's too
# long: drop n_seeds, shrink the eps/alpha grid below, or reduce T/settle.

Kc      = np.sqrt(8/np.pi) * sigma
alpha_c = 2 * Kc
eps_c   = 2 * sigma / np.pi
print(f"K_c = {Kc:.4f}   alpha_c = {alpha_c:.4f}   eps_c = {eps_c:.4f}   v = {v}")

eps_values   = np.linspace(0, 0.15, 15)
alpha_values = np.linspace(0, 0.6, 10)

# ── Ring geometry (fixed for the whole sweep) ───────────────────────────
idx = np.arange(N)
ring_dist = np.abs(idx[:, None] - idx[None, :])
ring_dist = np.minimum(ring_dist, N - ring_dist).astype(float)   # shortest way around

tau = ring_dist / v
delay_steps = np.round(tau / dt).astype(int)
buffer_len = delay_steps.max() + 1
col_idx = np.arange(N)[None, :]

print(f"max ring delay = {tau.max():.3f} time units ({delay_steps.max()} steps), "
      f"buffer_len = {buffer_len}")

# ── Sweep ────────────────────────────────────────────────────────────
r_mean_grid  = np.zeros((len(alpha_values), len(eps_values)))
r2_mean_grid = np.zeros((len(alpha_values), len(eps_values)))
r_std_grid   = np.zeros((len(alpha_values), len(eps_values)))
r2_std_grid  = np.zeros((len(alpha_values), len(eps_values)))

rng = np.random.default_rng(42)
t_start = time.time()

for ai, alpha in enumerate(alpha_values):
    for ei, eps in enumerate(eps_values):

        r_seed_vals, r2_seed_vals = [], []   # one value per seed, this (alpha, eps) cell

        for s in range(n_seeds):

            omega = rng.normal(0, sigma, N)
            theta = rng.uniform(0, 2*np.pi, N)
            K = np.full((N, N), K0)
            np.fill_diagonal(K, 0)

            # history buffer, seeded by free-running (uncoupled) backward extrapolation
            buffer = np.zeros((buffer_len, N))
            for k in range(buffer_len):
                t_past = (k - buffer_len) * dt
                buffer[k] = theta + omega * t_past

            r_vals, r2_vals = [], []

            for t in range(T):
                slot = t % buffer_len

                row_idx = (slot - delay_steps) % buffer_len          # (N, N)
                theta_delayed = buffer[row_idx, col_idx]               # theta_j(t - tau_ij)

                diff = theta[:, None] - theta_delayed                   # theta_i(t) - theta_j(t-tau_ij)
                coupling = (K / N) * np.sin(-diff)
                dtheta = omega + coupling.sum(axis=1)

                dK = eps * (alpha * np.cos(diff) - K)
                np.fill_diagonal(dK, 0)

                buffer[slot] = theta.copy()   # store CURRENT theta for future lookups

                theta = theta + dt * dtheta
                K = K + dt * dK

                if t >= settle:
                    z1 = np.mean(np.exp(1j * theta))
                    z2 = np.mean(np.exp(2j * theta))
                    r_vals.append(np.abs(z1)**2)
                    r2_vals.append(np.abs(z2 - z1**2))

            r_seed_vals.append(np.mean(r_vals))
            r2_seed_vals.append(np.mean(r2_vals))

        r_mean_grid[ai, ei]  = np.mean(r_seed_vals)
        r2_mean_grid[ai, ei] = np.mean(r2_seed_vals)
        r_std_grid[ai, ei]   = np.std(r_seed_vals)
        r2_std_grid[ai, ei]  = np.std(r2_seed_vals)

    print(f"alpha={alpha:.3f} done ({ai+1}/{len(alpha_values)}), "
          f"elapsed {time.time()-t_start:.0f}s")

np.savez("phase_diagram_ring.npz",
         eps_values=eps_values, alpha_values=alpha_values,
         eps_c=eps_c, alpha_c=alpha_c, n_seeds=n_seeds, v=v,
         r_mean_grid=r_mean_grid, r2_mean_grid=r2_mean_grid,
         r_std_grid=r_std_grid, r2_std_grid=r2_std_grid)

# ── Figure: means on top, seed-to-seed std on bottom ────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))
panels = [
    (axes[0, 0], r_mean_grid,  r'One-cluster mean $r^2$',       r'$r^2$',       'viridis', 0, 1),
    (axes[0, 1], r2_mean_grid, r'Two-cluster mean $r_2^2$',     r'$r_2^2$',     'viridis', 0, 1),
    (axes[1, 0], r_std_grid,   r'One-cluster std across seeds', r'std $r^2$',   'magma',   0, None),
    (axes[1, 1], r2_std_grid,  r'Two-cluster std across seeds', r'std $r_2^2$', 'magma',   0, None),
]
for ax, grid, title, label, cmap, vmin, vmax in panels:
    im = ax.pcolormesh(eps_values, alpha_values, grid,
                        shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label=label)
    ax.axvline(eps_c, color='white', linestyle='--', linewidth=1.5,
               label=fr'$\varepsilon_c \approx {eps_c:.3f}$ (no-delay, no-ring theory)')
    ax.axhline(alpha_c, color='#DC6414', linestyle='--', linewidth=1.5,
               label=fr'$\alpha_c \approx {alpha_c:.3f}$')
    ax.set_xlabel(r'Learning rate $\varepsilon$')
    ax.set_ylabel(r'Learning enhancement $\alpha$')
    ax.set_title(title)
    ax.legend(loc='upper left', framealpha=0.85, fontsize=8)

fig.suptitle(fr'Ring model phase diagram over ($\varepsilon$, $\alpha$), $v={v}$ '
             f'(N={N} on a ring), mean $\\pm$ std over {n_seeds} seeds',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('phase_diagram_ring.png', dpi=150, bbox_inches='tight')
print("Saved phase_diagram_ring.png and phase_diagram_ring.npz")
print("Done.")