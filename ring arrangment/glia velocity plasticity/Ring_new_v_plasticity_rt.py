import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.animation import FuncAnimation

# ── Real-time ring simulation: K-plasticity + GLIAL (source-axon) velocity
#    plasticity + live imshow of K / v / D, mirroring Karimian et al. 2019
#    Figures 2-3 for K and D. ──────────────────────────────────────────────
#
#   dtheta_i/dt = omega_i + (1/N) sum_j K_ij sin[theta_j(t-tau_ij) - theta_i(t)]
#   dK_ij/dt    = eps_K * (alpha_K*cos[theta_i(t) - theta_j(t-tau_ij)] - K_ij)
#   tau_ij(t)   = d_ij / v_ij(t)      (d_ij = shortest ring distance between i, j)
#
# VELOCITY PLASTICITY HERE IS GLIAL, NOT SYNAPTIC-STYLE HEBBIAN. The earlier
# version of this demo used a pairwise rule, dv_ij/dt = eps_v*(alpha_v*
# cos(theta_i-theta_j(t-tau_ij)) - v_ij), i.e. velocity between a SPECIFIC
# pair relaxed toward a target set by that pair's own phase coherence, same
# functional form as K. That's a reasonable first pass, but it's not quite
# how activity-dependent myelination works biologically: oligodendrocyte
# precursor cells (OPCs) wrap a given axon and respond to THAT axon's own
# activity (depolarization, glutamate release along its length), not to a
# precise phase-locking relationship with one specific downstream partner.
# So here, the plasticity target for v_ij depends on oscillator i (the
# source) ALONE, not on the pair (i, j):
#
#   Z_i(t)   = (1/N) sum_j K_ij(t) * exp(i*theta_j(t-tau_ij))   [the local
#               K-weighted field oscillator i is being driven by -- a proxy
#               for how much coordinated activity this axon's local
#               environment is carrying right now]
#   A_i(t)   = |Z_i(t)|
#   dv_ij/dt = eps_v * (alpha_v * A_i(t) - v_ij)
#
# Every outgoing edge from oscillator i relaxes toward the SAME
# activity-driven target. Concretely this means v_ij develops visible ROW
# structure (one velocity per source oscillator) instead of the old
# symmetric pairwise pattern -- watch for that in the v panel once velocity
# plasticity is on for a while.
#
# FUNCTIONAL connectivity (D) still follows the paper's eq. 8:
#   D_ij = (1/Delta t) * integral cos(phi_i(t) - phi_j(t)) dt
# using INSTANTANEOUS (non-delayed) phases, tracked live as an exponential
# moving average (time constant D_TAU) since there's no fixed "done" point.
#
# NOTE ON CLIPPING: v_ij is clipped to [V_MIN, V_MAX] after every update.
# A_i is non-negative by construction (it's a magnitude), so the glial
# target itself never goes unphysical the way the old cos(diff) target
# could, but v_ij can still start below V_MIN or drift there if A_i is
# small, so the floor/ceiling stays as a safety clip, not just a sign-fix.
#
# NOTE ON ALPHA_V SCALE: A_i is bounded by roughly the local K magnitude
# (so typically O(0.1-1.5) once K has built up), much smaller than the old
# rule's cos(diff) in [-1,1]. ALPHA_V defaults to 20 here so alpha_v*A_i
# lands in a velocity-like range comparable to v_init=30 -- watch the "A="
# readout in the stats line and retune the alpha_v slider if v keeps
# pinning at the floor or ceiling.
#
# Velocity plasticity defaults to ON (checkbox starts checked); toggle it
# off on screen to compare against fixed-v behavior.
#
# Needs an interactive matplotlib backend to show the sliders (the default
# backend on most desktops works; in Jupyter run `%matplotlib widget` first,
# or `%matplotlib qt` / `%matplotlib tk` if that's installed).

# ── Fixed model parameters ───────────────────────────────────────────────
N          = 30       # oscillators on the ring
SIGMA      = 0.1       # std of natural frequency distribution
DT         = 0.05
SUBSTEPS   = 10         # integration steps per animation frame
BUFFER_LEN = 400        # long enough to hold history for the slowest allowed v
HIST_MAX   = 400        # points kept in the rolling order-parameter chart
D_TAU      = 5.0        # time constant (time units) of the D moving average

# velocity bounds -- see NOTE ON CLIPPING above.
V_MIN = 2.0
V_MAX = 50.0

# alpha_K slider max -- also used to fix the K imshow color scale so the
# colorbar doesn't rescale every frame (matches the paper's fixed -alpha_s
# to alpha_s convention in Figure 2).
ALPHA_K_MAX = 1

# Random seed for reproducibility
rng = np.random.default_rng(42)
omega = rng.normal(loc=0, scale=0.4, size=N)
theta = rng.uniform(low=0, high=2 * np.pi, size=N)

sigma = 0.1
Kc = np.sqrt(8 / np.pi) * sigma   # ≈1.596 if sigma=1

K = np.abs(rng.normal(loc=Kc, scale=0.02 * Kc, size=(N, N)))
np.fill_diagonal(K, 0)

# live-adjustable parameters, mutated by the sliders while the animation runs
params = {
    "eps": 0.01, "alpha": 0.1,           # K-plasticity
    "v_init": 30.0,                       # initial/reset conduction speed
    "eps_v": 0.01, "alpha_v": 20.0,       # glial v-plasticity (alpha_v scales A_i)
    "v_plastic": True,                    # toggle: velocity plasticity on/off
}

rng = np.random.default_rng()

# ── Ring geometry: fixed positions, per-edge distance ────────────────────
idx = np.arange(N)
dist = np.abs(idx[:, None] - idx[None, :])
dist = np.minimum(dist, N - dist).astype(float)   # periodic (shortest-way) distance
dist_safe = dist.copy()
np.fill_diagonal(dist_safe, 1.0)   # avoid 0/0 on the diagonal; diagonal is unused anyway

col_idx = np.arange(N)[None, :]


def delay_steps_from_v(v_matrix):
    """v_matrix: N x N array of per-edge conduction speeds -> delay in steps."""
    tau = dist / v_matrix
    steps = np.round(tau / DT).astype(int)
    steps = np.clip(steps, 0, BUFFER_LEN - 1)
    np.fill_diagonal(steps, 0)
    return steps


# ── State (reset on "Reseed") ────────────────────────────────────────────
state = {}


def reseed():
    omega = rng.normal(0, SIGMA, N)
    theta = rng.uniform(0, 2 * np.pi, N)
    K_local = np.abs(rng.normal(loc=Kc, scale=0.02 * Kc, size=(N, N)))
    np.fill_diagonal(K_local, 0)

    v_local = np.full((N, N), params["v_init"])
    np.fill_diagonal(v_local, 0)   # diagonal unused, kept 0 for display cleanliness

    delay_steps = delay_steps_from_v(np.where(np.eye(N, dtype=bool), 1.0, v_local))

    D_local = np.zeros((N, N))   # functional connectivity moving average, starts blank
    A_local = np.zeros(N)        # per-oscillator glial activity signal, for the readout

    # history buffer seeded by free-running (uncoupled) backward extrapolation
    buffer = np.zeros((BUFFER_LEN, N))
    for k in range(BUFFER_LEN):
        t_past = (k - BUFFER_LEN) * DT
        buffer[k] = theta + omega * t_past

    state.update(omega=omega, theta=theta, K=K_local, v=v_local, D=D_local, A=A_local,
                 delay_steps=delay_steps, buffer=buffer,
                 step=0, t_hist=[], r_hist=[], r2_hist=[])


reseed()


def step_sim():
    eps, alpha = params["eps"], params["alpha"]
    eps_v, alpha_v = params["eps_v"], params["alpha_v"]
    theta = state["theta"]
    K_local = state["K"]
    v_local = state["v"]
    D_local = state["D"]
    buffer = state["buffer"]
    omega = state["omega"]
    delay_steps = state["delay_steps"]
    slot = state["step"] % BUFFER_LEN

    row_idx = (slot - delay_steps) % BUFFER_LEN
    theta_delayed = buffer[row_idx, col_idx]        # theta_delayed[i,j] = theta_j(t - tau_ij)

    diff = theta[:, None] - theta_delayed             # theta_i(t) - theta_j(t-tau_ij), DELAYED
    coupling = (K_local / N) * np.sin(-diff)
    dtheta = omega + coupling.sum(axis=1)

    dK = eps * (alpha * np.cos(diff) - K_local)
    np.fill_diagonal(dK, 0)

    # functional connectivity (paper eq. 8): uses INSTANTANEOUS phases, no delay
    diff_instant = theta[:, None] - theta[None, :]
    decay = 1.0 - np.exp(-DT / D_TAU)
    D_new = D_local + decay * (np.cos(diff_instant) - D_local)
    np.fill_diagonal(D_new, 0)

    buffer[slot] = theta.copy()          # store CURRENT theta for future lookups
    state["theta"] = theta + DT * dtheta
    state["K"] = K_local + DT * dK
    state["D"] = D_new

    if params["v_plastic"]:
        # GLIAL source-axon rule: target depends on oscillator i's own
        # K-weighted local field magnitude, not on pair (i,j)
        Z_i = (K_local * np.exp(1j * theta_delayed)).sum(axis=1) / N
        A_i = np.abs(Z_i)
        state["A"] = A_i
        dv = eps_v * (alpha_v * A_i[:, None] - v_local)
        np.fill_diagonal(dv, 0)
        v_new = v_local + DT * dv
        v_new = np.clip(v_new, V_MIN, V_MAX)   # see NOTE ON CLIPPING at top
        np.fill_diagonal(v_new, 0)
        state["v"] = v_new
        # delay changes as v changes -- recompute the lookup table each step
        v_for_lookup = np.where(np.eye(N, dtype=bool), 1.0, v_new)
        state["delay_steps"] = delay_steps_from_v(v_for_lookup)

    state["step"] += 1


def order_params(theta):
    z1 = np.mean(np.exp(1j * theta))
    z2 = np.mean(np.exp(2j * theta))
    r_sq = np.abs(z1) ** 2
    r2_sq = np.abs(z2 - z1 ** 2)
    return r_sq, r2_sq, z1


# ── Figure layout ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16.5, 11.5))
ax_ring   = fig.add_axes([0.05, 0.70, 0.28, 0.27])
ax_chart  = fig.add_axes([0.40, 0.73, 0.55, 0.24])
ax_K      = fig.add_axes([0.05, 0.40, 0.26, 0.26])
ax_v      = fig.add_axes([0.38, 0.40, 0.26, 0.26])
ax_D      = fig.add_axes([0.71, 0.40, 0.26, 0.26])
ax_eps    = fig.add_axes([0.06, 0.305, 0.35, 0.020])
ax_alpha  = fig.add_axes([0.06, 0.270, 0.35, 0.020])
ax_vinit  = fig.add_axes([0.06, 0.235, 0.35, 0.020])
ax_epsv   = fig.add_axes([0.06, 0.200, 0.35, 0.020])
ax_alphav = fig.add_axes([0.06, 0.165, 0.35, 0.020])
ax_toggle = fig.add_axes([0.06, 0.095, 0.20, 0.045])
ax_play   = fig.add_axes([0.55, 0.125, 0.12, 0.045])
ax_reset  = fig.add_axes([0.70, 0.125, 0.14, 0.045])
ax_stats  = fig.add_axes([0.46, 0.02, 0.50, 0.09])
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

# structural (K), conduction velocity (v), and functional (D) connectivity
# matrices, imshow style matching the paper's Figure 2/3 for K and D
im_K = ax_K.imshow(state["K"], cmap="RdBu_r", vmin=-ALPHA_K_MAX, vmax=ALPHA_K_MAX,
                    origin="lower", interpolation="nearest")
ax_K.set_title(r"Structural connectivity $K_{ij}$", fontsize=11, fontweight="bold")
ax_K.set_xlabel("oscillator index")
ax_K.set_ylabel("oscillator index")
fig.colorbar(im_K, ax=ax_K, fraction=0.046, pad=0.04, label=r"$K_{ij}$")

im_v = ax_v.imshow(state["v"], cmap="viridis", vmin=V_MIN, vmax=V_MAX,
                    origin="lower", interpolation="nearest")
ax_v.set_title(r"Conduction velocity $v_{ij}$ (glial)", fontsize=11, fontweight="bold")
ax_v.set_xlabel("oscillator index")
ax_v.set_ylabel("oscillator index (row = source axon $i$)")
fig.colorbar(im_v, ax=ax_v, fraction=0.046, pad=0.04, label=r"$v_{ij}$")

im_D = ax_D.imshow(state["D"], cmap="RdBu_r", vmin=-1, vmax=1,
                    origin="lower", interpolation="nearest")
ax_D.set_title(r"Functional connectivity $D_{ij}$", fontsize=11, fontweight="bold")
ax_D.set_xlabel("oscillator index")
ax_D.set_ylabel("oscillator index")
fig.colorbar(im_D, ax=ax_D, fraction=0.046, pad=0.04, label=r"$D_{ij}$")

stats_text = ax_stats.text(0, 0.5, "", fontsize=10.5, family="monospace", va="center")

# sliders
s_eps    = Slider(ax_eps,    r"$\varepsilon_K$",  0.0, 0.15, valinit=params["eps"],    valstep=0.001)
s_alpha  = Slider(ax_alpha,  r"$\alpha_K$",       0.0, ALPHA_K_MAX, valinit=params["alpha"],  valstep=0.01)
s_vinit  = Slider(ax_vinit,  r"$v_{init}$",       V_MIN, V_MAX, valinit=params["v_init"], valstep=0.5)
s_epsv   = Slider(ax_epsv,   r"$\varepsilon_v$",  0.0, 0.15, valinit=params["eps_v"],  valstep=0.001)
s_alphav = Slider(ax_alphav, r"$\alpha_v$ (glial scale)", 0.0, 50.0, valinit=params["alpha_v"], valstep=0.5)


def on_eps(val):
    params["eps"] = val


def on_alpha(val):
    params["alpha"] = val


def on_vinit(val):
    params["v_init"] = val   # takes effect on next Reseed


def on_epsv(val):
    params["eps_v"] = val


def on_alphav(val):
    params["alpha_v"] = val


s_eps.on_changed(on_eps)
s_alpha.on_changed(on_alpha)
s_vinit.on_changed(on_vinit)
s_epsv.on_changed(on_epsv)
s_alphav.on_changed(on_alphav)

check = CheckButtons(ax_toggle, ["velocity plasticity"], [params["v_plastic"]])


def on_toggle(label):
    params["v_plastic"] = not params["v_plastic"]


check.on_clicked(on_toggle)

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

        scatter.set_array(theta % (2 * np.pi))
        mean_mag, mean_angle = np.abs(z1), np.angle(z1)
        mean_line.set_data([0, mean_mag * np.cos(mean_angle)],
                            [0, mean_mag * np.sin(mean_angle)])

        t_hist = state["t_hist"]
        line_r.set_data(t_hist, state["r_hist"])
        line_r2.set_data(t_hist, state["r2_hist"])
        if t_hist:
            ax_chart.set_xlim(max(0, t_hist[-1] - 20), max(20, t_hist[-1]))

        im_K.set_data(state["K"])
        im_v.set_data(state["v"])
        im_D.set_data(state["D"])

        mean_k = state["K"].sum() / (N * N - N)
        mean_v = state["v"].sum() / (N * N - N)
        A = state["A"]
        max_delay = state["delay_steps"].max() * DT
        mean_delay = state["delay_steps"].mean() * DT
        if r_sq > 0.85:
            status = "sync"
        elif r_sq > 0.4:
            status = "partial"
        else:
            status = "desync"
        v_flag = "p" if params["v_plastic"] else "f"
        stats_text.set_text(
            f"t={t_now:6.1f}  r^2={r_sq:.3f}  r2^2={r2_sq:.3f}\n"
            f"K={mean_k:.3f}  v={mean_v:.1f}({v_flag})  A=[{A.min():.2f},{A.max():.2f}]  "
            f"delay(max/mean)={max_delay:.2f}/{mean_delay:.2f}  {status}"
        )

    return scatter, mean_line, line_r, line_r2, im_K, im_v, im_D, stats_text


anim = FuncAnimation(fig, update, interval=33, blit=False, cache_frame_data=False)

plt.show()