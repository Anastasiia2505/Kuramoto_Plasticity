import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')

# Draw the circle
ring = np.linspace(0, 2*np.pi, 300)
ax.plot(np.cos(ring), np.sin(ring), 'lightgray')

# One dot — starts at (0,0)
dot, = ax.plot([], [], 'o', color='orange', markersize=15)
dot1, = ax.plot([], [], 'o', color='purple', markersize=15)
label = ax.text(-1.2, -1.2, '')


# One line creates ALL dots at once
N = 10
dots, = ax.plot([], [], 'o', color='pink', markersize=10)
# Each dot has its own speed
speeds = np.linspace(0.02, 0.1, N)  # speeds from slow to fast



def update(frame):
    # Move dot around circle
    angle = frame * 0.05   # angle increases each frame
    x = np.cos(angle)
    y = np.sin(angle)
    dot.set_data([x], [y])

    # Second dot — goes counter-clockwise (negative angle)
    angle2 = -frame * 0.05
    x2 = np.cos(angle2)
    y2 = np.sin(angle2)
    dot1.set_data([x2], [y2])

    label.set_text(f'phase = {x:.2f}')

    # Array of N angles
    angles = np.array([frame * 0.05 * (i+1) for i in range(N)])
    
    x = np.cos(angles)
    y = np.sin(angles)
    
    # Move ALL dots at once with one line
    dots.set_data(x, y)
    
    return dots,

    return dot, dot1


ani = animation.FuncAnimation(fig, update, interval=30, blit=False)
plt.show()