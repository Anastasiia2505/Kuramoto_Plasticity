import numpy as np

# ── Diagnostic: does eps=0 actually freeze K, independent of alpha? ─────
# At eps=0, dK_ij/dt = eps*(alpha*cos(diff) - K_ij) = 0 for EVERY alpha, so
# K should stay exactly at its initial value the whole run, and alpha should
# have zero effect on the resulting order parameters. If this script shows
# r^2/r2^2 trending with alpha at eps=0, or K.sum() drifting away from its
# initial value, that pins the bug down to something concrete: either the
# eps/alpha values aren't reaching the sweep function the way you expect,
# or K is being updated somewhere even when eps=0.

N          = 30
OMEGA_STD  = 0.1
DT         = 0.05
v_fixed    = 30.0
T_steps    = 3000
settle     = 2000
n_seeds    = 5
eps_fixed  = 0.0                                   # <-- the case that should be alpha-independent
alpha_test_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]

idx = np.arange(N)
dist = np.abs(idx[:, None] - idx[None, :])
dist = np.minimum(dist, N - dist).astype(float)
tau = dist / v_fixed
delay_steps = np.round(tau / DT).astype(int)
buffer_len = delay_steps.max() + 1
col_idx = np.arange(N)[None, :]


def run_one(alpha, eps, K_init, seed):
    rng = np.random.default_rng(seed)
    omega = rng.normal(0, OMEGA_STD, N)
    theta = rng.uniform(0, 2 * np.pi, N)
    if K_init == 0:
        K = np.zeros((N, N))
    else:
        K = np.abs(rng.normal(loc=K_init, scale=0.02 * K_init, size=(N, N)))
    np.fill_diagonal(K, 0)
    K_initial_snapshot = K.copy()

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

    K_drift = np.max(np.abs(K - K_initial_snapshot))
    return np.mean(r_vals), np.mean(r2_vals), K_drift


print(f"eps fixed at {eps_fixed}, K_init = 0 -- expect r^2/r2^2 flat & small, K_drift = 0.0 always\n")
print(f"{'alpha':>6} | {'mean r^2':>10} | {'mean r2^2':>10} | {'max |K drift|':>14}")
for alpha in alpha_test_values:
    r_seed, r2_seed, drift_seed = [], [], []
    for s in range(n_seeds):
        r, r2, drift = run_one(alpha, eps_fixed, K_init=0.0, seed=1000 + s)
        r_seed.append(r); r2_seed.append(r2); drift_seed.append(drift)
    print(f"{alpha:6.2f} | {np.mean(r_seed):10.4f} | {np.mean(r2_seed):10.4f} | "
          f"{max(drift_seed):14.2e}")

print(f"\nFinite-N noise floor for reference (~1/N): {1/N:.4f}")
print("If mean r^2/r2^2 climb noticeably with alpha above, or max|K drift| is")
print("not ~0, that confirms the bug -- share this printout and I can help")
print("pin down exactly where eps is failing to reach the update.")