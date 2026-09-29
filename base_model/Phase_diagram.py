import numpy as np
import matplotlib.pyplot as plt
import time

# ── Parameters ────────────────────────────────────────────────────────
# Same N, T, settle as the single-seed phase_diagram.py (reduced from the
# 1D sweep's N=250/T=5000 to keep the grid tractable). n_seeds adds a
# third loop, the same fix that smoothed out the 1D eps sweep.
N       = 150
sigma   = 0.1
dt      = 0.1
T       = 2000
settle  = 1500
K0      = 0.75
n_seeds = 5

# NOTE ON RUNTIME: this is n_seeds times slower than phase_diagram.py,
# which already took about 4 minutes for the 15x10 grid. With n_seeds=5,
# expect roughly 20 minutes. Drop n_seeds to 3 or shrink the grid
# (fewer eps_values / alpha_values points) if that's too long.

Kc      = np.sqrt(8/np.pi) * sigma
alpha_c = 2 * Kc
eps_c   = 2 * sigma / np.pi
print(f"K_c = {Kc:.4f}   alpha_c = {alpha_c:.4f}   eps_c = {eps_c:.4f}")

eps_values   = np.linspace(0, 0.15, 15)
alpha_values = np.linspace(0, 0.6, 10)

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
            K = np.ones((N, N)) * K0
            np.fill_diagonal(K, 0)

            r_vals, r2_vals = [], []
            for t in range(T):
                diff = np.subtract.outer(theta, theta)
                coupling = (K / N) * np.sin(-diff)
                dtheta = omega + coupling.sum(axis=1)

                dK = eps * (alpha * np.cos(-diff) - K)
                np.fill_diagonal(dK, 0)

                theta += dt * dtheta
                K += dt * dK

                if t >= settle:
                    z1 = np.mean(np.exp(1j * theta))
                    z2 = np.mean(np.exp(2j * theta))
                    r_vals.append(np.abs(z1)**2)
                    r2_vals.append(np.abs(z2 - z1**2))

            # this seed's time-averaged value for this (alpha, eps) cell
            r_seed_vals.append(np.mean(r_vals))
            r2_seed_vals.append(np.mean(r2_vals))

        # average (and spread) across the n_seeds independent networks
        r_mean_grid[ai, ei]  = np.mean(r_seed_vals)
        r2_mean_grid[ai, ei] = np.mean(r2_seed_vals)
        r_std_grid[ai, ei]   = np.std(r_seed_vals)
        r2_std_grid[ai, ei]  = np.std(r2_seed_vals)

    print(f"alpha={alpha:.3f} done ({ai+1}/{len(alpha_values)}), "
          f"elapsed {time.time()-t_start:.0f}s")

np.savez("phase_diagram_multiseed.npz",
         eps_values=eps_values, alpha_values=alpha_values,
         r_mean_grid=r_mean_grid, r2_mean_grid=r2_mean_grid,
         r_std_grid=r_std_grid, r2_std_grid=r2_std_grid,
         eps_c=eps_c, alpha_c=alpha_c, n_seeds=n_seeds)

# ── Plot: means on top, seed-to-seed std on bottom ─────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))

panels = [
    (axes[0, 0], r_mean_grid,  r'One-cluster mean $r^2$',        r'$r^2$',      'viridis', 0, 1),
    (axes[0, 1], r2_mean_grid, r'Two-cluster mean $r_2^2$',      r'$r_2^2$',    'viridis', 0, 1),
    (axes[1, 0], r_std_grid,   r'One-cluster std across seeds',  r'std $r^2$',  'magma',   0, None),
    (axes[1, 1], r2_std_grid,  r'Two-cluster std across seeds',  r'std $r_2^2$','magma',   0, None),
]

for ax, grid, title, label, cmap, vmin, vmax in panels:
    im = ax.pcolormesh(eps_values, alpha_values, grid,
                        shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label=label)
    ax.axvline(eps_c, color='white', linestyle='--', linewidth=1.5,
               label=fr'$\varepsilon_c \approx {eps_c:.3f}$')
    ax.axhline(alpha_c, color='#DC6414', linestyle='--', linewidth=1.5,
               label=fr'$\alpha_c \approx {alpha_c:.3f}$')
    ax.set_xlabel(r'Learning rate $\varepsilon$')
    ax.set_ylabel(r'Learning enhancement $\alpha$')
    ax.set_title(title)
    ax.legend(loc='upper left', framealpha=0.85, fontsize=8)

fig.suptitle(f'Phase diagram over ($\\varepsilon$, $\\alpha$), mean $\\pm$ std over {n_seeds} seeds',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('phase_diagram_multiseed.png', dpi=150, bbox_inches='tight')
print("Done.")