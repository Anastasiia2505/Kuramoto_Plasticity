"""
Kuramoto Model — Basic Animation
=================================
N oscillators on a unit circle.
Watch them go from incoherent to synchronized as K increases.

Controls:
- Press SPACE to pause/unpause
- Press UP arrow to increase K
- Press DOWN arrow to decrease K
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ── Parameters ──────────────────────────────────────────────────────
N   = 100          # number of oscillators (keep small for clarity)
dt  = 0.05        # time step
alpha = 0.5
eps_K   = 0.01        # learning rate for coupling

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
sigma = 0.5
Kc = np.sqrt(8/np.pi) * sigma   # ≈1.596 if sigma=1

K = np.abs(rng.normal(loc=Kc, scale=0.02*Kc, size=(N, N)))
np.fill_diagonal(K, 0)   # no self-coupling



# ── Figure setup ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.axis('off')
ax.set_facecolor('white')

# Draw the unit circle
ring = np.linspace(0, 2*np.pi, 300)
ax.plot(np.cos(ring), np.sin(ring), color='lightgray', linewidth=1.5)

# Center dot
ax.plot(0, 0, 'o', color='gray', markersize=4)

# Oscillator dots
dots, = ax.plot([], [], 'o', color='#DC6414', markersize=10, zorder=3)

# Mean vector arrow (shows r)
mean_arrow = ax.annotate(
    '', xy=(0, 0), xytext=(0, 0),
    arrowprops=dict(arrowstyle='->', color='black', lw=3, mutation_scale=20),
    zorder=4
)

# Text labels
r_label   = ax.text(0, -1.35, '', ha='center', fontsize=13,
                    color='#DC6414', fontweight='bold')
k_label   = ax.text(0,  1.35, '', ha='center', fontsize=12, color='gray')
state_label = ax.text(0,  1.2,  '', ha='center', fontsize=11,
                      color='black', style='italic')

ax.set_title('Kuramoto Model\nPress ↑↓ to change K, SPACE to pause',
             fontsize=11, pad=10)

# ── State variables ──────────────────────────────────────────────────
paused = [False]
eps_K_val = [0.01]

# ── Key controls ─────────────────────────────────────────────────────
def on_key(event):
    if event.key == ' ':
        paused[0] = not paused[0]
    elif event.key == 'up':
        eps_K_val[0] = min(eps_K_val[0] + 0.01, 1.0)
    elif event.key == 'down':
        eps_K_val[0] = max(eps_K_val[0] - 0.01, 0.0)

fig.canvas.mpl_connect('key_press_event', on_key)

# ── Update function ──────────────────────────────────────────────────
def update(frame):
    global theta, K

    if paused[0]:
        return dots, mean_arrow, r_label, k_label, state_label


    # ── Kuramoto equation ──────────────────────────────────────────
    # diff[i,j] = theta_i - theta_j
    diff = np.subtract.outer(theta, theta)
    # print(diff)

    # now each pair has its own K_ij
    coupling = (K / N) * np.sin(-diff)      # N×N matrix
    dtheta   = omega + coupling.sum(axis=1) # sum over j for each i

    # ── Hebbian plasticity rule ───────────────────────────────────
    # dK_ij/dt = eps_K * cos(θ_j - θ_i)
    dK = eps_K_val * (alpha * np.cos(-diff) - K)             # N×N matrix
    np.fill_diagonal(dK, 0)                 # no self-coupling


    # ── Euler steps ───────────────────────────────────────────────
    theta += dt * dtheta
    K     += dt * dK


    # ── Compute positions on unit circle ───────────────────────────
    x = np.cos(theta)
    y = np.sin(theta)

    # ── Order parameter ────────────────────────────────────────────
    # r = |mean of e^{i*theta}|
    mean_vec = np.mean(np.exp(1j * theta))
    r = np.abs(mean_vec)

    # ── Update plot elements ───────────────────────────────────────
    # Oscillator dots
    dots.set_data(x, y)

    # Mean vector arrow — points in direction of collective phase
    mx = np.real(mean_vec) * 0.9
    my = np.imag(mean_vec) * 0.9
    mean_arrow.xy     = (mx, my)
    mean_arrow.xytext = (0,  0)

    # Labels
    r_label.set_text(f'r = {r:.2f}')
    k_label.set_text(f'K_mean = {K.mean():.2f}   ε_K = {eps_K_val[0]:.2f}')

    return dots, mean_arrow, r_label, k_label, state_label


# ── Run ──────────────────────────────────────────────────────────────
ani = animation.FuncAnimation(
    fig,
    update,
    interval=30,       # milliseconds between frames (~33 fps)
    blit=False,        # redraw whole figure each frame (simpler)
    cache_frame_data=False
)

plt.tight_layout()
plt.show()