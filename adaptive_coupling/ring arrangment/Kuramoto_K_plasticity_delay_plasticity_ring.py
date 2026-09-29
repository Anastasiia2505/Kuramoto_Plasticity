import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

# ── Visualizing the Timms & English (2014) ring embedding ──────────────
# Section 2.2 of the proposal: oscillators sit on a 1D ring with periodic
# boundaries, pairwise delay tau_ij = d_ij / v depends on ring distance,
# and the network settles into "standing-wave modes" where phase advances
# by a fixed step from one oscillator to the next around the ring —
# indexed by integers m for a single cluster, and by half-integers once
# fast learning splits the population into two interleaved clusters.

N = 24  # number of oscillators on the ring (kept small so it's legible)

def ring_positions(N, R=1.0):
    idx = np.arange(N)
    angle = 2 * np.pi * idx / N
    x, y = R * np.cos(angle), R * np.sin(angle)
    return idx, angle, x, y

def standing_wave_phase(idx, N, m):
    # phase advances by a fixed step 2*pi*m/N from one oscillator to the next
    return 2 * np.pi * m * idx / N

def draw_ring_panel(ax, m, title, two_cluster=False):
    idx, pos_angle, x, y = ring_positions(N)
    phase = standing_wave_phase(idx, N, m) % (2 * np.pi)

    # faint ring guide
    theta_full = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta_full), np.sin(theta_full), color='0.85', lw=1, zorder=0)

    # color each oscillator by its phase (cyclic colormap)
    cmap = plt.cm.hsv
    colors = cmap(phase / (2 * np.pi))

    if two_cluster:
        # interleave two sub-populations (even/odd ring positions) to show
        # how a half-integer mode requires alternating cluster membership:
        # circles for cluster A, squares for cluster B
        even_mask = (idx % 2 == 0)
        ax.scatter(x[even_mask], y[even_mask], s=380, c=colors[even_mask],
                   marker='o', edgecolors='#1a1a1a', linewidths=1.8, zorder=3)
        ax.scatter(x[~even_mask], y[~even_mask], s=380, c=colors[~even_mask],
                   marker='s', edgecolors='#1a1a1a', linewidths=1.8, zorder=3)
    else:
        ax.scatter(x, y, s=380, c=colors, marker='o',
                   edgecolors='#1a1a1a', linewidths=1.8, zorder=3)

    # little arrows at each node showing instantaneous phase direction
    for xi, yi, ph in zip(x, y, phase):
        dx, dy = 0.16 * np.cos(ph), 0.16 * np.sin(ph)
        ax.annotate('', xy=(xi + dx, yi + dy), xytext=(xi, yi),
                    arrowprops=dict(arrowstyle='-|>', color='black', lw=1.2), zorder=4)

    # connect a couple of representative pairs to show distance-dependent delay
    pairs = [(0, 1), (0, N // 2)]
    for (i, j) in pairs:
        xs, ys = [x[i], x[j]], [y[i], y[j]]
        d = min(abs(i - j), N - abs(i - j))
        ax.plot(xs, ys, color='#555555', lw=1.0 + 1.8 * (d / (N / 2)),
                linestyle='--', alpha=0.6, zorder=1)

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold')


fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))

draw_ring_panel(axes[0], m=3, title=r'Integer mode $m=3$' + '\n(single phase cluster)',
                two_cluster=False)
draw_ring_panel(axes[1], m=2.5, title=r'Half-integer mode $m=5/2$' + '\n(two interleaved clusters)',
                two_cluster=True)

# shared legend explaining the diagram elements
circle_marker = plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='0.7',
                            markeredgecolor='#1a1a1a', markersize=13,
                            label='Cluster A (even sites, half-integer mode only)')
square_marker = plt.Line2D([0], [0], marker='s', color='none', markerfacecolor='0.7',
                            markeredgecolor='#1a1a1a', markersize=11,
                            label='Cluster B (odd sites, half-integer mode only)')
dashed_line = plt.Line2D([0], [0], color='#555555', lw=1.5, linestyle='--',
                          label=r'Pairwise delay $\tau_{ij}=d_{ij}/v$ (thicker = larger $d_{ij}$)')

fig.text(0.5, 0.065, 'Node color = instantaneous phase $\\phi_i$ (hue, cyclic colormap)',
          ha='center', fontsize=10)
fig.legend(handles=[circle_marker, square_marker, dashed_line],
           loc='lower center', ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.01))

fig.suptitle('Spatial embedding: oscillators on a 1D ring (Timms & English, 2014)',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.savefig('ring_visualization.png', dpi=150, bbox_inches='tight')
print("Saved ring_visualization.png")


# ── Second figure: phase profile "unrolled" around the ring ────────────
# Shows the standing-wave phase as a function of position index, making
# the integer-vs-half-integer winding number visually obvious.
fig2, ax2 = plt.subplots(figsize=(8, 5))

idx = np.arange(N + 1)  # +1 to show wraparound back to start
for m, style, label, color in [
    (3, '-', r'$m=3$ (integer, one cluster)', '#2166AC'),
    (2.5, '--', r'$m=5/2$ (half-integer, two clusters)', '#B2182B'),
]:
    phase = 2 * np.pi * m * idx / N
    ax2.plot(idx, phase / (2 * np.pi), style, color=color, lw=2.2, label=label, marker='o', markersize=4)

ax2.axhline(0, color='0.7', lw=0.8)
ax2.set_xlabel('Oscillator index $i$ around the ring')
ax2.set_ylabel(r'Phase $\phi_i / 2\pi$')
ax2.set_title('Standing-wave phase profile: winding number $m$ per full loop of the ring',
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('ring_phase_profile.png', dpi=150, bbox_inches='tight')
print("Saved ring_phase_profile.png")