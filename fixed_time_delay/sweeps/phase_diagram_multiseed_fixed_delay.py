import numpy as np
import matplotlib.pyplot as plt
import time

# ── Parameters ────────────────────────────────────────────────────────
# Same as phase_diagram_multiseed.py, with an outer loop over a handful
# of fixed delay values tau. Each tau gets its own full multi-seed
# (eps, alpha) grid, the same computation you already ran for tau=0.
N       = 150
sigma   = 0.1
dt      = 0.1
T       = 2000
settle  = 1500
K0      = 0.75
n_seeds = 5

tau_values = [0.0, 0.5, 1.5]   # change/add delays here

# NOTE ON RUNTIME: phase_diagram_multiseed.py alone took ~20 minutes for
# one (eps, alpha) grid. This script runs that same grid once per tau in
# tau_values, so three delays means roughly 3x that, on the order of an
# hour. If that's too long: drop n_seeds to 3, shrink the eps/alpha grid
# below, or run fewer tau values at a time (the script saves results
# per-tau as it goes, so you don't lose earlier delays if you stop early).

Kc      = np.sqrt(8/np.pi) * sigma
alpha_c = 2 * Kc
eps_c   = 2 * sigma / np.pi
print(f"K_c = {Kc:.4f}   alpha_c = {alpha_c:.4f}   eps_c = {eps_c:.4f}")

eps_values   = np.linspace(0, 0.15, 15)
alpha_values = np.linspace(0, 0.6, 10)

results = {}   # tau -> dict of grids
t_start = time.time()

for tau in tau_values:

    delay_steps = int(round(tau / dt))

    r_mean_grid  = np.zeros((len(alpha_values), len(eps_values)))
    r2_mean_grid = np.zeros((len(alpha_values), len(eps_values)))
    r_std_grid   = np.zeros((len(alpha_values), len(eps_values)))
    r2_std_grid  = np.zeros((len(alpha_values), len(eps_values)))

    rng = np.random.default_rng(42)

    for ai, alpha in enumerate(alpha_values):
        for ei, eps in enumerate(eps_values):

            r_seed_vals, r2_seed_vals = [], []   # one value per seed, this (alpha, eps) cell

            for s in range(n_seeds):

                omega = rng.normal(0, sigma, N)
                theta = rng.uniform(0, 2*np.pi, N)
                K = np.ones((N, N)) * K0
                np.fill_diagonal(K, 0)

                r_vals, r2_vals = [], []

                if delay_steps == 0:
                    # tau = 0 falls back to the undelayed model exactly
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

                else:
                    # circular buffer of the last delay_steps theta vectors,
                    # initialized via free-running (uncoupled) backward extrapolation
                    buffer = np.zeros((delay_steps, N))
                    for k in range(delay_steps):
                        t_past = (k - delay_steps) * dt
                        buffer[k] = theta + omega * t_past

                    for t in range(T):
                        slot = t % delay_steps
                        theta_delayed = buffer[slot].copy()

                        diff_delayed = theta[:, None] - theta_delayed[None, :]

                        coupling = (K / N) * np.sin(-diff_delayed)
                        dtheta = omega + coupling.sum(axis=1)

                        dK = eps * (alpha * np.cos(diff_delayed) - K)
                        np.fill_diagonal(dK, 0)

                        buffer[slot] = theta.copy()

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

        print(f"tau={tau}  alpha={alpha:.3f} done ({ai+1}/{len(alpha_values)}), "
              f"elapsed {time.time()-t_start:.0f}s")

    results[tau] = dict(r_mean_grid=r_mean_grid, r2_mean_grid=r2_mean_grid,
                         r_std_grid=r_std_grid, r2_std_grid=r2_std_grid)

    np.savez(f"phase_diagram_multiseed_tau{tau}.npz",
             eps_values=eps_values, alpha_values=alpha_values,
             eps_c=eps_c, alpha_c=alpha_c, n_seeds=n_seeds, tau=tau,
             **results[tau])

    # ── Per-tau figure: means on top, seed-to-seed std on bottom ───────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))
    panels = [
        (axes[0, 0], results[tau]['r_mean_grid'],  r'One-cluster mean $r^2$',       r'$r^2$',       'viridis', 0, 1),
        (axes[0, 1], results[tau]['r2_mean_grid'], r'Two-cluster mean $r_2^2$',     r'$r_2^2$',     'viridis', 0, 1),
        (axes[1, 0], results[tau]['r_std_grid'],   r'One-cluster std across seeds', r'std $r^2$',   'magma',   0, None),
        (axes[1, 1], results[tau]['r2_std_grid'],  r'Two-cluster std across seeds', r'std $r_2^2$', 'magma',   0, None),
    ]
    for ax, grid, title, label, cmap, vmin, vmax in panels:
        im = ax.pcolormesh(eps_values, alpha_values, grid,
                            shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, label=label)
        ax.axvline(eps_c, color='white', linestyle='--', linewidth=1.5,
                   label=fr'$\varepsilon_c \approx {eps_c:.3f}$ (no-delay theory)')
        ax.axhline(alpha_c, color='#DC6414', linestyle='--', linewidth=1.5,
                   label=fr'$\alpha_c \approx {alpha_c:.3f}$')
        ax.set_xlabel(r'Learning rate $\varepsilon$')
        ax.set_ylabel(r'Learning enhancement $\alpha$')
        ax.set_title(title)
        ax.legend(loc='upper left', framealpha=0.85, fontsize=8)

    fig.suptitle(fr'Phase diagram over ($\varepsilon$, $\alpha$), $\tau={tau}$, '
                 f'mean $\\pm$ std over {n_seeds} seeds',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'phase_diagram_multiseed_tau{tau}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved tau={tau} figure.")

# ── Comparison figure: r2_mean_grid side by side across all tau values ──
fig, axes = plt.subplots(1, len(tau_values), figsize=(6.5*len(tau_values), 5.5), squeeze=False)
axes = axes[0]
for ax, tau in zip(axes, tau_values):
    im = ax.pcolormesh(eps_values, alpha_values, results[tau]['r2_mean_grid'],
                        shading='auto', cmap='viridis', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label=r'$r_2^2$')
    ax.axvline(eps_c, color='white', linestyle='--', linewidth=1.5)
    ax.axhline(alpha_c, color='#DC6414', linestyle='--', linewidth=1.5)
    ax.set_xlabel(r'Learning rate $\varepsilon$')
    ax.set_ylabel(r'Learning enhancement $\alpha$')
    ax.set_title(fr'$\tau={tau}$')

fig.suptitle(r'Two-cluster order parameter $r_2^2$ across delay values',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('phase_diagram_delay_comparison.png', dpi=150, bbox_inches='tight')
print("Done.")