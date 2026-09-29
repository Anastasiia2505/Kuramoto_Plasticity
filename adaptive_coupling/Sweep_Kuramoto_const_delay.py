import numpy as np
import matplotlib.pyplot as plt
import time

# ── Parameters — same as sweep_multiseed.py, with a fixed delay added ──
N       = 250       # number of oscillators (paper: N=250 for Fig. 4)
sigma   = 0.1        # std of frequency distribution
dt      = 0.1        # Euler time step
T       = 5000        # total time steps
settle  = 4000        # discard first 4000, average last 1000
alpha   = 1.0        # learning enhancement factor (paper: "set at a large value of 1" for Fig. 4)
K0      = 0.75       # initial coupling, an absolute value (K_c ~ 0.16, so K0 is well above K_c)
n_seeds = 5          # independent random networks averaged per eps point
tau     = 0.5        # fixed transmission delay (time units) — change this to test other delays

# NOTE ON RUNTIME: this is the same cost as sweep_multiseed.py (already
# ~5x the single-run sweep), plus the small overhead of the delay buffer.
# Expect this to take at least as long as sweep_multiseed.py did for you.
# Drop n_seeds to 3, or the eps grid to ~20 points, for a faster check.

Kc      = np.sqrt(8/np.pi) * sigma
alpha_c = 2 * Kc
eps_c   = 2 * sigma / np.pi
print(f"K_c = {Kc:.4f}   alpha_c = {alpha_c:.4f}   eps_c = {eps_c:.4f}   tau = {tau}")

eps_values = np.linspace(0, 0.15, 40)

delay_steps = int(round(tau / dt))

# ── Sweep ────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

r_means, r2_means = [], []   # mean over seeds, one value per eps
r_stds,  r2_stds  = [], []   # std over seeds, for the shaded band

t_start = time.time()

for eps in eps_values:

    r_seed_vals, r2_seed_vals = [], []   # one value per seed for this eps

    for s in range(n_seeds):

        omega = rng.normal(0, sigma, N)
        theta = rng.uniform(0, 2*np.pi, N)
        K = np.ones((N, N)) * K0
        np.fill_diagonal(K, 0)

        if delay_steps == 0:
            # tau = 0 falls back to the undelayed model exactly
            r_vals, r2_vals = [], []
            for t in range(T):
                diff = np.subtract.outer(theta, theta)

                coupling = (K / N) * np.sin(-diff)
                dtheta   = omega + coupling.sum(axis=1)

                dK = eps * (alpha * np.cos(-diff) - K)
                np.fill_diagonal(dK, 0)

                theta += dt * dtheta
                K     += dt * dK

                if t >= settle:
                    mean_vec  = np.mean(np.exp(1j * theta))
                    mean_vec2 = np.mean(np.exp(2j * theta))
                    r_vals.append(np.abs(mean_vec)**2)
                    r2_vals.append(np.abs(mean_vec2 - mean_vec**2))

        else:
            # circular buffer of the last delay_steps theta vectors,
            # initialized via free-running (uncoupled) backward extrapolation
            buffer = np.zeros((delay_steps, N))
            for k in range(delay_steps):
                t_past = (k - delay_steps) * dt
                buffer[k] = theta + omega * t_past

            r_vals, r2_vals = [], []
            for t in range(T):
                slot = t % delay_steps
                theta_delayed = buffer[slot].copy()   # theta from exactly tau ago

                # diff_delayed[i, j] = theta_i(t) - theta_j(t - tau)
                diff_delayed = theta[:, None] - theta_delayed[None, :]

                coupling = (K / N) * np.sin(-diff_delayed)   # sin(theta_j(t-tau) - theta_i(t))
                dtheta   = omega + coupling.sum(axis=1)

                dK = eps * (alpha * np.cos(diff_delayed) - K)   # cos is even, sign doesn't matter
                np.fill_diagonal(dK, 0)

                buffer[slot] = theta.copy()   # store CURRENT theta; read again in delay_steps steps

                theta = theta + dt * dtheta
                K     = K + dt * dK

                if t >= settle:
                    mean_vec  = np.mean(np.exp(1j * theta))
                    mean_vec2 = np.mean(np.exp(2j * theta))
                    r_vals.append(np.abs(mean_vec)**2)
                    r2_vals.append(np.abs(mean_vec2 - mean_vec**2))

        # this seed's time-averaged value for this eps
        r_seed_vals.append(np.mean(r_vals))
        r2_seed_vals.append(np.mean(r2_vals))

    # average (and spread) across the n_seeds independent networks
    r_means.append(np.mean(r_seed_vals))
    r2_means.append(np.mean(r2_seed_vals))
    r_stds.append(np.std(r_seed_vals))
    r2_stds.append(np.std(r2_seed_vals))

    print(f"eps={eps:.4f}  r^2={r_means[-1]:.3f}+-{r_stds[-1]:.3f}  "
          f"r2^2={r2_means[-1]:.3f}+-{r2_stds[-1]:.3f}  "
          f"elapsed={time.time()-t_start:.0f}s")

r_means, r2_means = np.array(r_means), np.array(r2_means)
r_stds,  r2_stds  = np.array(r_stds),  np.array(r2_stds)

np.savez("niyogi_replication_multiseed_delay.npz",
         eps_values=eps_values, r_means=r_means, r2_means=r2_means,
         r_stds=r_stds, r2_stds=r2_stds, n_seeds=n_seeds, tau=tau)

# ── Plot ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

ax.fill_between(eps_values, r_means - r_stds, r_means + r_stds,
                 color='steelblue', alpha=0.2, linewidth=0)
ax.fill_between(eps_values, r2_means - r2_stds, r2_means + r2_stds,
                 color='#DC6414', alpha=0.2, linewidth=0)

ax.plot(eps_values, r_means,  color='steelblue',
        linewidth=2, label=r'$r^2$ (one cluster)')
ax.plot(eps_values, r2_means, color='#DC6414',
        linewidth=2, linestyle='--', label=r'$r_2^2$ (two cluster)')

ax.axvline(eps_c, color='gray', linestyle=':', linewidth=1.5,
           label=fr'$\varepsilon_c \approx {eps_c:.3f}$ (no-delay theory)')

ax.set_xlabel(r'Learning rate $\varepsilon$', fontsize=13)
ax.set_ylabel('Order parameter', fontsize=13)
ax.set_title(f'Fixed-delay replication ($\\tau={tau}$)\n'
             f'(mean $\\pm$ std over {n_seeds} seeds)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xlim(0, 0.15)
ax.set_ylim(0, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('sweep_multiseed_delay.png', dpi=150, bbox_inches='tight')
plt.show()
print("Done.")