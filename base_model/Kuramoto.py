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
K   = 0.0         # coupling strength (you control this)

# Random seed for reproducibility
rng = np.random.default_rng(42)

# Natural frequencies — each oscillator has its own
omega = rng.normal(loc=0, scale=1, size=N)

# Initial phases — random starting positions on circle
theta = rng.uniform(low=0, high=2*np.pi, size=N)

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
K_val  = [0.0]

# ── Key controls ─────────────────────────────────────────────────────
def on_key(event):
    if event.key == ' ':
        paused[0] = not paused[0]
    elif event.key == 'up':
        K_val[0] = min(K_val[0] + 0.2, 5.0)
    elif event.key == 'down':
        K_val[0] = max(K_val[0] - 0.2, 0.0)

fig.canvas.mpl_connect('key_press_event', on_key)

# ── Update function ──────────────────────────────────────────────────
def update(frame):
    global theta

    if paused[0]:
        return dots, mean_arrow, r_label, k_label, state_label

    K = K_val[0]

    # ── Kuramoto equation ──────────────────────────────────────────
    # diff[i,j] = theta_i - theta_j
    diff = np.subtract.outer(theta, theta)
    # print(diff)

    # dtheta_i = omega_i + (K/N) * sum_j sin(theta_j - theta_i)
    #                                         = sin(-diff[i,:])
    dtheta = omega + (K / N) * np.sin(-diff).sum(axis=1)

    # Euler step
    theta += dt * dtheta

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
    k_label.set_text(f'K = {K:.1f}   (K_c ≈ 1.60)')

    # # State description
    # if K < 1.2:
    #     state_label.set_text('incoherent')
    #     state_label.set_color('steelblue')
    # elif K < 2.0:
    #     state_label.set_text('near critical point')
    #     state_label.set_color('darkorange')
    # else:
    #     state_label.set_text('synchronized')
    #     state_label.set_color('green')

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