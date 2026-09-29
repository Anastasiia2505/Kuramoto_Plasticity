import numpy as np
import matplotlib.pyplot as plt

# ── Real-time(-integrated) ring simulation ──────────────────────────────
# This actually integrates the delayed, adaptive Kuramoto model on a 1D
# ring (Timms & English, 2014), rather than drawing an idealized standing
# wave by hand. Oscillators sit at fixed positions around the ring, the
# delay between any pair depends on how far apart they are on the ring,
# tau_ij = d_ij / v, and the coupling K_ij evolves under the same
# Hebbian-like rule used in the unstructured (Niyogi & English) model.
#
#   dtheta_i/dt = omega_i + (1/N) sum_j K_ij sin[theta_j(t-tau_ij) - theta_i(t)]
#   dK_ij/dt    = eps * (alpha*cos[theta_i(t) - theta_j(t-tau_ij)] - K_ij)
#
# The ring geometry only changes what tau_ij is; everything else is the
# same delayed-Hebbian machinery as sweep_multiseed_delay.py.

# ── Parameters ────────────────────────────────────────────────────────
N       = 24      # oscillators on the ring (kept small so the snapshot stays legible)
sigma   = 0.1      # std of natural frequency distribution
dt      = 0.05
T       = 4000     # total integration steps
settle  = 3000     # discard before this step when averaging order parameters
K0      = 0.75     # initial coupling (all pairs)
eps     = 0.05     # learning rate for K_ij
alpha   = 0.3      # learning enhancement (try larger alpha for two-cluster states)
v       = 8.0       # conduction velocity in ring-units/time-unit (smaller v = longer delays)
seed    = 42

rng = np.random.default_rng(seed)

# ── Ring geometry and the pairwise delay matrix ─────────────────────────
idx = np.arange(N)
dist = np.abs(idx[:, None] - idx[None, :])
dist = np.minimum(dist, N - dist).astype(float)     # periodic (shortest-way-around) distance
tau = dist / v
delay_steps = np.round(tau / dt).astype(int)
max_delay = delay_steps.max()
buffer_len = max_delay + 1

print(f"Ring of N={N}: max pairwise delay = {tau.max():.3f} time units "
      f"({max_delay} steps), v={v}")

# ── Initial conditions ──────────────────────────────────────────────────
omega = rng.normal(0, sigma, N)
theta = rng.uniform(0, 2 * np.pi, N)
K = np.full((N, N), K0)
np.fill_diagonal(K, 0)

# History buffer, seeded by free-running (uncoupled) backward extrapolation
# so the delayed terms are well-defined from step 0.
buffer = np.zeros((buffer_len, N))
for k in range(buffer_len):
    t_past = (k - buffer_len) * dt
    buffer[k] = theta + omega * t_past

theta_history = np.zeros((T, N))
r_history = np.zeros(T)
r2_history = np.zeros(T)

col_idx = np.arange(N)[None, :]   # broadcasts over the N columns

# ── Integrate ────────────────────────────────────────────────────────────
for t in range(T):
    slot = t % buffer_len

    # theta_delayed[i, j] = theta_j(t - tau_ij), gathered per (i, j) pair
    # since every pair has its own delay set by ring distance
    row_idx = (slot - delay_steps) % buffer_len          # shape (N, N)
    theta_delayed = buffer[row_idx, col_idx]              # shape (N, N)

    diff_delayed = theta[:, None] - theta_delayed          # theta_i(t) - theta_j(t-tau_ij)

    coupling = (K / N) * np.sin(-diff_delayed)
    dtheta = omega + coupling.sum(axis=1)

    dK = eps * (alpha * np.cos(diff_delayed) - K)
    np.fill_diagonal(dK, 0)

    buffer[slot] = theta.copy()   # store CURRENT theta for later lookups

    theta = theta + dt * dtheta
    K = K + dt * dK

    theta_history[t] = theta
    z1 = np.mean(np.exp(1j * theta))
    z2 = np.mean(np.exp(2j * theta))
    r_history[t] = np.abs(z1) ** 2
    r2_history[t] = np.abs(z2 - z1 ** 2)

print(f"<r^2>  over settle window = {r_history[settle:].mean():.3f}")
print(f"<r2^2> over settle window = {r2_history[settle:].mean():.3f}")

np.savez("ring_simulation.npz", theta_history=theta_history,
         r_history=r_history, r2_history=r2_history,
         omega=omega, tau=tau, N=N, dt=dt)

# ── Figure 1: order parameters over time ────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 4.5))
t_axis = np.arange(T) * dt
ax1.plot(t_axis, r_history, color='steelblue', lw=1.5, label=r'$r^2$ (one cluster)')
ax1.plot(t_axis, r2_history, color='#DC6414', lw=1.5, linestyle='--', label=r'$r_2^2$ (two cluster)')
ax1.axvline(settle * dt, color='gray', linestyle=':', label='settle cutoff')
ax1.set_xlabel('time')
ax1.set_ylabel('order parameter')
ax1.set_title(f'Ring simulation: order parameters vs. time ($N$={N}, $v$={v}, $\\alpha$={alpha})',
              fontsize=11, fontweight='bold')
ax1.legend(fontsize=9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('ring_simulation_order_params.png', dpi=150, bbox_inches='tight')
print("Saved ring_simulation_order_params.png")

# ── Figure 2: final snapshot on the ring, colored by actual phase ──────
final_theta = theta_history[-1] % (2 * np.pi)
angle = 2 * np.pi * idx / N
x, y = np.cos(angle), np.sin(angle)

fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 5.5))

theta_full = np.linspace(0, 2 * np.pi, 200)
ax2.plot(np.cos(theta_full), np.sin(theta_full), color='0.85', lw=1, zorder=0)
cmap = plt.cm.hsv
colors = cmap(final_theta / (2 * np.pi))
ax2.scatter(x, y, s=380, c=colors, edgecolors='#1a1a1a', linewidths=1.8, zorder=3)
for xi, yi, ph in zip(x, y, final_theta):
    dx, dy = 0.16 * np.cos(ph), 0.16 * np.sin(ph)
    ax2.annotate('', xy=(xi + dx, yi + dy), xytext=(xi, yi),
                 arrowprops=dict(arrowstyle='-|>', color='black', lw=1.2), zorder=4)
ax2.set_xlim(-1.35, 1.35)
ax2.set_ylim(-1.35, 1.35)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title(f'Final ring state at $t={T*dt:.0f}$', fontsize=12, fontweight='bold')

# unwrap the phase pattern around the ring (relative to oscillator 0)
# to see whether a smooth ramp (integer mode) or a two-cluster zigzag
# (half-integer mode) emerged
rel_phase = np.unwrap(final_theta - final_theta[0])
ax3.plot(idx, rel_phase / (2 * np.pi), 'o-', color='#2166AC', lw=1.8, markersize=5)
ax3.set_xlabel('oscillator index $i$ around the ring')
ax3.set_ylabel(r'$(\phi_i - \phi_0)/2\pi$, unwrapped')
ax3.set_title('Phase profile around the ring\n(smooth ramp = single cluster, zigzag = two clusters)',
              fontsize=11, fontweight='bold')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('ring_simulation_snapshot.png', dpi=150, bbox_inches='tight')
print("Saved ring_simulation_snapshot.png")
print("Done.")