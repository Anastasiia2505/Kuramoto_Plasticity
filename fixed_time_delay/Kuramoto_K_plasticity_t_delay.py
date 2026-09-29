import numpy as np
import matplotlib.pyplot as plt
import time

# ── Model ────────────────────────────────────────────────────────────
# theta_dot_i(t) = omega_i + (1/N) sum_j K_ij(t) sin(theta_j(t - tau) - theta_i(t))
# K_dot_ij(t)    = eps * (alpha * cos(theta_i(t) - theta_j(t - tau)) - K_ij(t))
#
# tau is a single fixed scalar delay shared by all pairs (the simplest
# case, before v_ij becomes adaptive / pair-specific).

def run_delayed_sim(eps, alpha, N, K0, T, settle, sigma, dt, tau, seed):
    rng = np.random.default_rng(seed)
    omega = rng.normal(0, sigma, N)
    theta = rng.uniform(0, 2*np.pi, N)
    K = np.ones((N, N)) * K0
    np.fill_diagonal(K, 0)

    delay_steps = int(round(tau / dt))

    if delay_steps == 0:
        # no delay: identical to the undelayed model (used as a
        # regression check against the earlier scripts)
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
        return np.mean(r_vals), np.mean(r2_vals)

    # circular buffer of the last `delay_steps` theta vectors, initialized
    # via free-running (uncoupled) backward extrapolation: theta(t) = theta0 + omega*t
    buffer = np.zeros((delay_steps, N))
    for k in range(delay_steps):
        t_past = (k - delay_steps) * dt   # negative times, e.g. -delay_steps*dt ... -dt
        buffer[k] = theta + omega * t_past

    r_vals, r2_vals = [], []
    for step in range(T):
        slot = step % delay_steps
        theta_delayed = buffer[slot].copy()   # theta from exactly delay_steps ago

        # diff_delayed[i, j] = theta_i(t) - theta_j(t - tau)
        diff_delayed = theta[:, None] - theta_delayed[None, :]

        coupling = (K / N) * np.sin(-diff_delayed)   # sin(theta_j(t-tau) - theta_i(t))
        dtheta = omega + coupling.sum(axis=1)

        dK = eps * (alpha * np.cos(diff_delayed) - K)  # cos is even, sign doesn't matter
        np.fill_diagonal(dK, 0)

        buffer[slot] = theta.copy()   # store CURRENT theta; read again delay_steps steps later

        theta = theta + dt * dtheta
        K = K + dt * dK

        if step >= settle:
            z1 = np.mean(np.exp(1j * theta))
            z2 = np.mean(np.exp(2j * theta))
            r_vals.append(np.abs(z1)**2)
            r2_vals.append(np.abs(z2 - z1**2))

    return np.mean(r_vals), np.mean(r2_vals)


if __name__ == "__main__":
    N, sigma, dt = 150, 0.1, 0.1
    T, settle = 2000, 1500
    alpha, K0 = 1.0, 0.75

    # ── Sanity check: tau=0 should reproduce the undelayed model ───────
    eps_test = 0.09
    r_a, r2_a = run_delayed_sim(eps_test, alpha, N, K0, T, settle, sigma, dt, tau=0.0, seed=7)
    print(f"tau=0.0 sanity check: r^2={r_a:.3f}  r2^2={r2_a:.3f}")

    # ── Compare a few tau values across a small eps sweep ───────────────
    eps_values = np.linspace(0, 0.15, 12)
    tau_values = [0.0, 0.5, 1.5]

    results = {}
    t0 = time.time()
    for tau in tau_values:
        r2_list = []
        for eps in eps_values:
            _, r2 = run_delayed_sim(eps, alpha, N, K0, T, settle, sigma, dt, tau=tau, seed=42)
            r2_list.append(r2)
        results[tau] = np.array(r2_list)
        print(f"tau={tau} done, elapsed {time.time()-t0:.0f}s")

    np.savez("delay_comparison.npz",
             eps_values=eps_values, tau_values=tau_values,
             **{f"r2_tau{tau}": results[tau] for tau in tau_values})

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#DC6414', 'steelblue', '#2E8B57']
    for tau, color in zip(tau_values, colors):
        ax.plot(eps_values, results[tau], color=color, linewidth=2,
                 marker='o', markersize=4, label=fr'$\tau={tau}$')

    eps_c = 2 * sigma / np.pi
    ax.axvline(eps_c, color='gray', linestyle=':', linewidth=1.5,
               label=fr'$\varepsilon_c \approx {eps_c:.3f}$ (no-delay theory)')

    ax.set_xlabel(r'Learning rate $\varepsilon$', fontsize=13)
    ax.set_ylabel(r'$r_2^2$ (two-cluster order parameter)', fontsize=13)
    ax.set_title('Effect of fixed transmission delay on the transition', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(0, 0.15)
    ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('delay_comparison.png', dpi=150, bbox_inches='tight')
    print("Done.")