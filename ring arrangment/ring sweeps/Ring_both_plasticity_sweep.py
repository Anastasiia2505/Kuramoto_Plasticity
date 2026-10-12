import numpy as np
import matplotlib.pyplot as plt
import time

# ── Phase diagram (eps, alpha) for the ring model WITH velocity plasticity ──
# Same (eps, alpha) sweep structure as ring_phase_diagram_heatmap_100.py, but
# v is no longer a single fixed scalar. Instead v_ij is a per-edge quantity
# that evolves under its own Hebbian rule (same form you gave for the live
# demo), using fixed EPS_V/ALPHA_V constants -- so K-plasticity is still the
# swept 2D grid, and velocity plasticity runs underneath it at a fixed rate:
#
#   dtheta_i/dt = omega_i + (1/N) sum_j K_ij sin[theta_j(t-tau_ij) - theta_i(t)]
#   dK_ij/dt    = eps   * (alpha   * cos[theta_i(t) - theta_j(t-tau_ij)] - K_ij)
#   dv_ij/dt    = eps_v * (alpha_v * cos[theta_i(t) - theta_j(t-tau_ij)] - v_ij)
#   tau_ij(t)   = d_ij / v_ij(t)      (d_ij = shortest ring distance between i, j)
#
# NOTE ON CLIPPING (same reasoning as the live-demo version): the raw rule
# dv_ij/dt = eps_v*(alpha_v*cos(diff) - v_ij) is an exponential relaxation
# toward alpha_v*cos(diff) in [-alpha_v, alpha_v]. v_ij is a conduction SPEED
# (tau_ij = d_ij / v_ij), so v_ij -> 0 sends delay to infinity and v_ij < 0
# is a non-physical negative delay. Both break the delay-buffer lookup, so
# v_ij is clipped to [V_MIN, V_MAX] after every update.
#
# NOTE ON RUNTIME: unlike fixed-v, the delay lookup table now depends on time
# (v_ij changes every step), so delay_steps must be recomputed from the N x N
# v matrix EVERY integration step instead of once before the loop. That adds
# real cost on top of the already-heavy N=100, T_steps=10000, 15x10x5=750-run
# sweep this script inherits -- expect this to run several times slower than
# the fixed-v version. Recommend a quick low-res test first (small grids /
# few seeds / fewer steps) before committing to the full run; see the sizes
# below.

# ── Parameters ────────────────────────────────────────────────────────
N          = 30        # oscillators on the ring
OMEGA_STD  = 0.1         # std of the omega distribution
DT         = 0.05
T_steps    = 10000           # total integration steps
settle     = 2000            # discard before this step when averaging order parameters
n_seeds    = 5                 # independent random networks averaged per grid point

# K-plasticity: swept dimensions (same as before)
Kc = np.sqrt(8 / np.pi) * OMEGA_STD     # critical coupling for THIS omega spread
eps_values   = np.linspace(0, 0.15, 15)
alpha_values = np.linspace(0, 1.5, 10)     # matches slider's 0-1.5 range

# velocity plasticity: fixed rate (not swept), applied underneath the K sweep
v_init_fixed = 30.0     # initial conduction speed every edge starts near
EPS_V        = 0.01     # fixed velocity learning rate
ALPHA_V      = 10.0     # fixed velocity learning-enhancement target
V_MIN        = 2.0      # physical floor -- see NOTE ON CLIPPING above
V_MAX        = 50.0     # physical ceiling, also bounds buffer_len below

INITIAL_CONDITIONS = [
    ("at Kc",    1.0),   # K starts right at the critical coupling
    ("below Kc", 0.5),   # K starts sub-critical, but not zero
    ("zero",     0.0),   # K starts with no coupling at all
]

print(f"Kc (sigma={OMEGA_STD}) = {Kc:.4f}")
for label, frac in INITIAL_CONDITIONS:
    print(f"  '{label}' run starts K near {frac*Kc:.4f} ({frac} x Kc)")

# ── Ring geometry (fixed for the whole sweep) ────────────────────────────
idx = np.arange(N)
dist = np.abs(idx[:, None] - idx[None, :])
dist = np.minimum(dist, N - dist).astype(float)
dist_safe = dist.copy()
np.fill_diagonal(dist_safe, 1.0)   # avoid 0/0 on the diagonal; diagonal is unused
col_idx = np.arange(N)[None, :]

# buffer must be sized for the WORST CASE delay, i.e. every edge sitting at
# the slowest allowed speed V_MIN -- not just the fixed-v_init case.
max_dist = dist.max()
tau_worst = max_dist / V_MIN
buffer_len = int(np.ceil(tau_worst / DT)) + 1

print(f"max ring distance = {max_dist:.0f}, worst-case delay at V_MIN={V_MIN} "
      f"is {tau_worst:.3f} time units, buffer_len = {buffer_len}")


def delay_steps_from_v(v_matrix):
    """v_matrix: N x N per-edge conduction speeds -> delay in integer steps."""
    tau = dist_safe / v_matrix
    steps = np.round(tau / DT).astype(int)
    steps = np.clip(steps, 0, buffer_len - 1)
    np.fill_diagonal(steps, 0)
    return steps


def run_sweep(K_init):
    """Runs the full (eps, alpha) sweep starting every seed's K near K_init,
    with v_ij plastic under the fixed EPS_V/ALPHA_V rate for every cell.
    Returns dict of mean/std grids for r^2 and r2^2."""
    r_mean_grid  = np.zeros((len(alpha_values), len(eps_values)))
    r2_mean_grid = np.zeros((len(alpha_values), len(eps_values)))
    r_std_grid   = np.zeros((len(alpha_values), len(eps_values)))
    r2_std_grid  = np.zeros((len(alpha_values), len(eps_values)))

    rng = np.random.default_rng(42)   # same seed sequence across all runs
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

                v = np.full((N, N), v_init_fixed)
                np.fill_diagonal(v, 0)
                delay_steps = delay_steps_from_v(
                    np.where(np.eye(N, dtype=bool), 1.0, v))

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

                    dv = EPS_V * (ALPHA_V * np.cos(diff) - v)
                    np.fill_diagonal(dv, 0)

                    buffer[slot] = theta.copy()

                    theta = theta + DT * dtheta
                    K = K + DT * dK
                    v = np.clip(v + DT * dv, V_MIN, V_MAX)
                    np.fill_diagonal(v, 0)

                    # delay changes as v changes -- recompute every step
                    v_for_lookup = np.where(np.eye(N, dtype=bool), 1.0, v)
                    delay_steps = delay_steps_from_v(v_for_lookup)

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
# NOTE ON RUNTIME: see the top-of-file note -- this is substantially slower
# per run than the fixed-v version because delay_steps is now recomputed
# every integration step. Consider testing with smaller N/T_steps/n_seeds
# or a coarser grid first.

all_results = {}
for label, frac in INITIAL_CONDITIONS:
    print(f"\nRunning '{label}' sweep (K_init = {frac} x Kc = {frac*Kc:.4f})...")
    all_results[label] = run_sweep(K_init=frac * Kc)

save_kwargs = {"eps_values": eps_values, "alpha_values": alpha_values,
                "Kc": Kc, "n_seeds": n_seeds, "v_init": v_init_fixed,
                "eps_v": EPS_V, "alpha_v": ALPHA_V,
                "v_min": V_MIN, "v_max": V_MAX}
for label, results in all_results.items():
    safe_label = label.replace(" ", "_")
    save_kwargs.update({f"{safe_label}_{k}": v for k, v in results.items()})
np.savez("ring_phase_diagram_heatmap_100_velocity.npz", **save_kwargs)

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

fig.suptitle(fr'Ring model phase diagram w/ velocity plasticity, $N$={N}, $\sigma_\omega$={OMEGA_STD}, '
             fr'$v_0$={v_init_fixed}, $\varepsilon_v$={EPS_V}, $\alpha_v$={ALPHA_V}: '
             f'comparing initial coupling (mean over {n_seeds} seeds)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('ring_phase_diagram_heatmap_30_velocity.png', dpi=150, bbox_inches='tight')
print("Saved ring_phase_diagram_heatmap_30_velocity.png and ring_phase_diagram_heatmap_30_velocity.npz")
print("Done.")