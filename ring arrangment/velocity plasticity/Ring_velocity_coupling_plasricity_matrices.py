import numpy as np
import matplotlib.pyplot as plt
import time

# ── Phase diagram (eps, alpha) for the ring model WITH velocity plasticity, ──
#    PLUS final K / v / D matrix snapshots at chosen (eps, alpha) points. ────
#
# This is ring_phase_diagram_heatmap_100_velocity.py, UNCHANGED (same grid,
# same seeds, same output files), with the same snapshot addition used in
# ring_phase_diagram_heatmap_100_with_matrices.py: after the main sweep
# finishes, it re-runs one simulation per SNAPSHOT_POINTS entry and saves
# imshow panels of that run's final K (structural), v (conduction velocity,
# the "myelin" quantity), and D (functional, paper eq. 8).
#
#   dtheta_i/dt = omega_i + (1/N) sum_j K_ij sin[theta_j(t-tau_ij) - theta_i(t)]
#   dK_ij/dt    = eps   * (alpha   * cos[theta_i(t) - theta_j(t-tau_ij)] - K_ij)
#   dv_ij/dt    = eps_v * (alpha_v * cos[theta_i(t) - theta_j(t-tau_ij)] - v_ij)
#   tau_ij(t)   = d_ij / v_ij(t)
#
# THIS IS THE DIRECT TEST of the paper's headline result: "for conditions in
# which a system limited to synaptic plasticity develops two distinct
# clusters both structurally and functionally, additional adaptive
# myelination allows for functional communication across these structural
# clusters." Concretely: pick an (eps, alpha) point where the fixed-v run
# (ring_phase_diagram_heatmap_100_with_matrices.py) gave a clean double-
# cluster checkerboard in BOTH K and D, run the same point here with
# velocity plasticity on, and check whether K stays double-clustered while
# D (and/or v) stops mirroring it, i.e. whether functional connectivity
# fills in across the structural cluster boundary even though the
# structural split in K persists. SNAPSHOT_POINTS below defaults to the
# same points used in the fixed-v comparison so the two are directly
# comparable panel-for-panel.
#
# NOTE ON COST: see ring_phase_diagram_heatmap_100_velocity.py's runtime
# note -- recomputing delay_steps every step makes this several times
# slower than the fixed-v version, and these snapshot runs inherit that
# cost on top of the main sweep. Keep SNAPSHOT_POINTS short.

# ── Parameters ────────────────────────────────────────────────────────
N          = 100        # oscillators on the ring
OMEGA_STD  = 0.1         # std of the omega distribution
DT         = 0.05
T_steps    = 10000           # total integration steps
settle     = 2000            # discard before this step when averaging
n_seeds    = 5                 # independent random networks averaged per grid point

Kc = np.sqrt(8 / np.pi) * OMEGA_STD     # critical coupling for THIS omega spread
eps_values   = np.linspace(0, 0.15, 15)
alpha_values = np.linspace(0, 1.5, 10)     # matches your slider's 0-1.5 range

# velocity plasticity: fixed rate (not swept), applied underneath the K sweep
v_init_fixed = 30.0     # initial conduction speed every edge starts near
EPS_V        = 0.01     # fixed velocity learning rate
ALPHA_V      = 10.0     # fixed velocity learning-enhancement target
V_MIN        = 2.0      # physical floor -- see NOTE ON CLIPPING in the velocity script
V_MAX        = 50.0     # physical ceiling, also bounds buffer_len below

INITIAL_CONDITIONS = [
    ("at Kc",    1.0),   # K starts right at the critical coupling
    ("below Kc", 0.5),   # K starts sub-critical, but not zero
    ("zero",     0.0),   # K starts with no coupling at all
]

# Same default points as ring_phase_diagram_heatmap_100_with_matrices.py so
# the two sets of snapshots line up panel-for-panel. Edit freely -- e.g. set
# this to the specific (eps, alpha) where you saw the cleanest double-cluster
# checkerboard in the fixed-v K/D snapshots.
SNAPSHOT_POINTS = [
    (0.0,               1.5),   # eps=0 (K frozen), alpha at max
    (0.15,              1.5),   # eps at max, alpha at max
    (0.15,              0.0),   # eps at max, alpha=0
    (0.15 / 14 * 7,     1.5),   # eps at mid-grid, alpha at max
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
    """UNCHANGED from ring_phase_diagram_heatmap_100_velocity.py."""
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
                    K = np.zeros((N, N))
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


def run_single_with_snapshot(K_init, eps, alpha, seed=42):
    """Runs ONE simulation (velocity plasticity on) at a fixed (eps, alpha)
    and returns the final K matrix, final v matrix, the post-settle-window
    averaged D matrix (paper eq. 8, instantaneous phases), and mean r^2/r2^2
    over that window. Same seed=42 caveat as the fixed-v snapshot script:
    this is an independent reproducible run at that (eps, alpha, K_init),
    not extracted from the big sweep's internal seed sequence."""
    rng = np.random.default_rng(seed)
    omega = rng.normal(0, OMEGA_STD, N)
    theta = rng.uniform(0, 2 * np.pi, N)
    if K_init == 0:
        K = np.zeros((N, N))
    else:
        K = np.abs(rng.normal(loc=K_init, scale=0.02 * K_init, size=(N, N)))
    np.fill_diagonal(K, 0)

    v = np.full((N, N), v_init_fixed)
    np.fill_diagonal(v, 0)
    delay_steps = delay_steps_from_v(np.where(np.eye(N, dtype=bool), 1.0, v))

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

        dv = EPS_V * (ALPHA_V * np.cos(diff) - v)
        np.fill_diagonal(dv, 0)

        buffer[slot] = theta.copy()
        theta = theta + DT * dtheta
        K = K + DT * dK
        v = np.clip(v + DT * dv, V_MIN, V_MAX)
        np.fill_diagonal(v, 0)

        v_for_lookup = np.where(np.eye(N, dtype=bool), 1.0, v)
        delay_steps = delay_steps_from_v(v_for_lookup)

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

    return K, v, D_final, np.mean(r_vals), np.mean(r2_vals)


# ── Run every initial condition ──────────────────────────────────────────
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
plt.savefig('ring_phase_diagram_heatmap_100_velocity.png', dpi=150, bbox_inches='tight')
print("Saved ring_phase_diagram_heatmap_100_velocity.png and ring_phase_diagram_heatmap_100_velocity.npz")

# ── Snapshot figures: final K, v, D matrices at SNAPSHOT_POINTS ─────────
# One figure per initial condition; one row per (eps, alpha) snapshot point,
# K / v / D shown side by side.
print(f"\nRunning {len(SNAPSHOT_POINTS)} K/v/D snapshot(s) per initial condition...")

for label, frac in INITIAL_CONDITIONS:
    K_init = frac * Kc
    n_pts = len(SNAPSHOT_POINTS)
    fig2, axes2 = plt.subplots(n_pts, 3, figsize=(14.5, 4.6 * n_pts), squeeze=False)

    for row, (eps, alpha) in enumerate(SNAPSHOT_POINTS):
        t0 = time.time()
        K_final, v_final, D_final, r_mean, r2_mean = run_single_with_snapshot(K_init, eps, alpha)
        print(f"  '{label}' eps={eps:.4f} alpha={alpha:.3f}: "
              f"r^2={r_mean:.3f} r2^2={r2_mean:.3f}  ({time.time()-t0:.0f}s)")

        ax_k, ax_v, ax_d = axes2[row, 0], axes2[row, 1], axes2[row, 2]

        im_k = ax_k.imshow(K_final, cmap='RdBu_r', vmin=-alpha_values[-1], vmax=alpha_values[-1],
                            origin='lower', interpolation='nearest')
        ax_k.set_title(fr'$K_{{ij}}$  ($\varepsilon$={eps:.3f}, $\alpha$={alpha:.2f})', fontsize=10)
        ax_k.set_xlabel('oscillator index')
        ax_k.set_ylabel('oscillator index')
        fig2.colorbar(im_k, ax=ax_k, fraction=0.046, pad=0.04, label=r'$K_{ij}$')

        im_v = ax_v.imshow(v_final, cmap='viridis', vmin=V_MIN, vmax=V_MAX,
                            origin='lower', interpolation='nearest')
        ax_v.set_title(fr'$v_{{ij}}$  ($\varepsilon_v$={EPS_V}, $\alpha_v$={ALPHA_V})', fontsize=10)
        ax_v.set_xlabel('oscillator index')
        ax_v.set_ylabel('oscillator index')
        fig2.colorbar(im_v, ax=ax_v, fraction=0.046, pad=0.04, label=r'$v_{ij}$')

        im_d = ax_d.imshow(D_final, cmap='RdBu_r', vmin=-1, vmax=1,
                            origin='lower', interpolation='nearest')
        ax_d.set_title(fr'$D_{{ij}}$  ($r^2$={r_mean:.2f}, $r_2^2$={r2_mean:.2f})', fontsize=10)
        ax_d.set_xlabel('oscillator index')
        ax_d.set_ylabel('oscillator index')
        fig2.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04, label=r'$D_{ij}$')

    fig2.suptitle(fr'Final K (structural) / v (conduction velocity) / D (functional), '
                  fr'start "{label}" ($K_0$={K_init:.3f}), $N$={N}, with velocity plasticity',
                  fontsize=12, fontweight='bold')
    plt.tight_layout()
    safe_label = label.replace(" ", "_")
    fname = f'ring_connectivity_snapshots_100_velocity_{safe_label}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved {fname}")

print("Done.")