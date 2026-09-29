import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation

# ── Real-time ring simulation with a live epsilon slider ────────────────
# Python/matplotlib version of the browser demo: delayed, adaptive Kuramoto
# oscillators on a ring (Timms & English, 2014), with the same equations as
# ring_simulation.py, but stepped live inside a matplotlib animation so you
# can drag epsilon (and alpha, and conduction speed v) while it runs and
# watch synchronization emerge or collapse.
#
#   dtheta_i/dt = omega_i + (1/N) sum_j K_ij sin[theta_j(t-tau_ij) - theta_i(t)]
#   dK_ij/dt    = eps * (alpha*cos[theta_i(t) - theta_j(t-tau_ij)] - K_ij)
#   tau_ij      = d_ij / v      (d_ij = shortest ring distance between i, j)
#
# Needs an interactive matplotlib backend to show the sliders (the default
# backend on most desktops works; in Jupyter run `%matplotlib widget` first,
# or `%matplotlib qt` / `%matplotlib tk` if that's installed).

# ── Fixed model parameters ───────────────────────────────────────────────
N          = 30       # oscillators on the ring
SIGMA      = 0.1       # std of natural frequency distribution
DT         = 0.05
# K0         = 0.5      # initial coupling
SUBSTEPS   = 10         # integration steps per animation frame
BUFFER_LEN = 400        # long enough to hold history for the slowest allowed v
HIST_MAX   = 400        # points kept in the rolling order-parameter chart

# Random seed for reproducibility
rng = np.random.default_rng(42)
# Natural frequencies — each oscillator has its own
omega = rng.normal(loc=0, scale=0.4, size=N)
# Initial phases — random starting positions on circle
theta = rng.uniform(low=0, high=2*np.pi, size=N)


# # K is now an N×N matrix — starts small and uniform
# K = np.ones((N, N)) * 0.1
# np.fill_diagonal(K, 0)   # no self-coupling

# K is now an N×N matrix — starts small and uniform
sigma = 0.1
Kc = np.sqrt(8/np.pi) * sigma   # ≈1.596 if sigma=1

K = np.abs(rng.normal(loc=Kc, scale=0.02*Kc, size=(N, N)))
np.fill_diagonal(K, 0)   # no self-coupling

# live-adjustable parameters, mutated by the sliders while the animation runs
params = {"eps": 0.01, "alpha": 0.1, "v": 30.0}

rng = np.random.default_rng()

# ── Ring geometry: fixed positions, distance-dependent delay ────────────
idx = np.arange(N)
dist = np.abs(idx[:, None] - idx[None, :])
dist = np.minimum(dist, N - dist).astype(float)   # periodic (shortest-way) distance


def delay_steps_for(v):
    tau = dist / v
    steps = np.round(tau / DT).astype(int)
    return np.clip(steps, 0, BUFFER_LEN - 1)


delay_steps = delay_steps_for(params["v"])
col_idx = np.arange(N)[None, :]

# ── State (reset on "Reseed") ────────────────────────────────────────────
state = {}


def reseed():
    omega = rng.normal(0, SIGMA, N)
    theta = rng.uniform(0, 2 * np.pi, N)
    # K = np.full((N, N), K)
    np.fill_diagonal(K, 0)

    # history buffer seeded by free-running (uncoupled) backward extrapolation
    buffer = np.zeros((BUFFER_LEN, N))
    for k in range(BUFFER_LEN):
        t_past = (k - BUFFER_LEN) * DT
        buffer[k] = theta + omega * t_past

    state.update(omega=omega, theta=theta, K=K, buffer=buffer,
                 step=0, t_hist=[], r_hist=[], r2_hist=[])


reseed()


def step_sim():
    eps, alpha = params["eps"], params["alpha"]
    theta, K, buffer, omega = state["theta"], state["K"], state["buffer"], state["omega"]
    slot = state["step"] % BUFFER_LEN

    row_idx = (slot - delay_steps) % BUFFER_LEN
    theta_delayed = buffer[row_idx, col_idx]        # theta_delayed[i,j] = theta_j(t - tau_ij)

    diff = theta[:, None] - theta_delayed             # theta_i(t) - theta_j(t-tau_ij)
    coupling = (K / N) * np.sin(-diff)
    dtheta = omega + coupling.sum(axis=1)

    dK = eps * (alpha * np.cos(diff) - K)
    np.fill_diagonal(dK, 0)

    buffer[slot] = theta.copy()          # store CURRENT theta for future lookups
    state["theta"] = theta + DT * dtheta
    state["K"] = K + DT * dK
    state["step"] += 1


def order_params(theta):
    z1 = np.mean(np.exp(1j * theta))
    z2 = np.mean(np.exp(2j * theta))
    r_sq = np.abs(z1) ** 2
    r2_sq = np.abs(z2 - z1 ** 2)
    return r_sq, r2_sq, z1


# ── Figure layout ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10.5, 7.8))
ax_ring  = fig.add_axes([0.06, 0.38, 0.42, 0.55])
ax_chart = fig.add_axes([0.55, 0.56, 0.40, 0.37])
ax_eps   = fig.add_axes([0.10, 0.25, 0.35, 0.03])
ax_alpha = fig.add_axes([0.10, 0.19, 0.35, 0.03])
ax_v     = fig.add_axes([0.10, 0.13, 0.35, 0.03])
ax_play  = fig.add_axes([0.55, 0.16, 0.12, 0.05])
ax_reset = fig.add_axes([0.70, 0.16, 0.14, 0.05])
ax_stats = fig.add_axes([0.55, 0.05, 0.40, 0.09])
ax_stats.axis("off")

# ring panel
angle = 2 * np.pi * idx / N
ring_x, ring_y = np.cos(angle), np.sin(angle)
guide = np.linspace(0, 2 * np.pi, 200)
ax_ring.plot(np.cos(guide), np.sin(guide), color="0.85", lw=1, zorder=0)
scatter = ax_ring.scatter(ring_x, ring_y, s=220, c=np.zeros(N), cmap="hsv",
                           vmin=0, vmax=2 * np.pi, edgecolors="black",
                           linewidths=1.2, zorder=3)
mean_line, = ax_ring.plot([0, 0], [0, 0], color="#2a78d6", lw=2.5, zorder=4)
ax_ring.set_xlim(-1.3, 1.3)
ax_ring.set_ylim(-1.3, 1.3)
ax_ring.set_aspect("equal")
ax_ring.axis("off")
ax_ring.set_title("Ring state", fontsize=12, fontweight="bold")

# order-parameter strip chart
line_r,  = ax_chart.plot([], [], color="#2a78d6", lw=2, label=r"$r^2$ (one cluster)")
line_r2, = ax_chart.plot([], [], color="#eb6834", lw=2, ls="--", label=r"$r_2^2$ (two cluster)")
ax_chart.set_xlim(0, 20)
ax_chart.set_ylim(0, 1)
ax_chart.set_xlabel("time")
ax_chart.set_ylabel("order parameter")
ax_chart.legend(loc="upper left", fontsize=9)
ax_chart.spines["top"].set_visible(False)
ax_chart.spines["right"].set_visible(False)

stats_text = ax_stats.text(0, 0.5, "", fontsize=11, family="monospace", va="center")

# sliders
s_eps   = Slider(ax_eps,   r"$\varepsilon$", 0.0, 0.15, valinit=params["eps"],   valstep=0.001)
s_alpha = Slider(ax_alpha, r"$\alpha$",      0.0, 1.5,  valinit=params["alpha"], valstep=0.01)
s_v     = Slider(ax_v,     "v",              1.0, 40.0, valinit=params["v"],    valstep=0.5)


def on_eps(val):
    params["eps"] = val


def on_alpha(val):
    params["alpha"] = val


def on_v(val):
    global delay_steps
    params["v"] = val
    delay_steps = delay_steps_for(val)


s_eps.on_changed(on_eps)
s_alpha.on_changed(on_alpha)
s_v.on_changed(on_v)

playing = {"on": True}


def toggle_play(event):
    playing["on"] = not playing["on"]
    btn_play.label.set_text("Pause" if playing["on"] else "Play")


def do_reset(event):
    reseed()


btn_play = Button(ax_play, "Pause")
btn_reset = Button(ax_reset, "Reseed")
btn_play.on_clicked(toggle_play)
btn_reset.on_clicked(do_reset)


def update(frame):
    if playing["on"]:
        for _ in range(SUBSTEPS):
            step_sim()

        theta = state["theta"]
        r_sq, r2_sq, z1 = order_params(theta)
        t_now = state["step"] * DT

        state["t_hist"].append(t_now)
        state["r_hist"].append(r_sq)
        state["r2_hist"].append(r2_sq)
        if len(state["t_hist"]) > HIST_MAX:
            state["t_hist"].pop(0)
            state["r_hist"].pop(0)
            state["r2_hist"].pop(0)

        # ring colors + mean-field vector
        scatter.set_array(theta % (2 * np.pi))
        mean_mag, mean_angle = np.abs(z1), np.angle(z1)
        mean_line.set_data([0, mean_mag * np.cos(mean_angle)],
                            [0, mean_mag * np.sin(mean_angle)])

        # strip chart
        t_hist = state["t_hist"]
        line_r.set_data(t_hist, state["r_hist"])
        line_r2.set_data(t_hist, state["r2_hist"])
        if t_hist:
            ax_chart.set_xlim(max(0, t_hist[-1] - 20), max(20, t_hist[-1]))

        mean_k = (state["K"].sum()) / (N * N - N)
        if r_sq > 0.85:
            status = "synchronized"
        elif r_sq > 0.4:
            status = "partially synchronized"
        else:
            status = "desynchronized"
        stats_text.set_text(
            f"t = {t_now:6.1f}   r^2 = {r_sq:.3f}   r2^2 = {r2_sq:.3f}\n"
            f"mean K = {mean_k:.3f}   max delay = {(delay_steps.max()*DT):.2f}   status: {status}"
        )

    return scatter, mean_line, line_r, line_r2, stats_text


anim = FuncAnimation(fig, update, interval=33, blit=False, cache_frame_data=False)

plt.show()