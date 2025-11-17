"""
Local Potential-Field Path Planning (elliptical obstacles)

Run to plot a local path from start to goal around elliptical obstacles 
using an attractive + repulsive potential field.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation

# ===============================
# INITIAL SETUP
# ===============================
q_goal = np.array([10, 10])
q = np.array([0.0, 0.0])

obstacles_true = np.array([[3, 3.5],
                           [6, 7],
                           [9, 9],
                           [8, 4],
                           [5, 5]])

sigma = 0.1
obstacles_noisy = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)

obstacle_speeds = np.array([[0.1, 0.2],
                            [-0.2, -0.1],
                            [0.1, -0.1],
                            [-0.1, 0.1],
                            [0.2, 0.1]])

obstacle_speeds = obstacle_speeds * 10

k_att, k_rep, d0, dt = 2.0, 10.0, 2.0, 0.01
max_rep_force = 14.0

path_data = [q.copy()]

# Ellipse shape parameters
a0 = 2.0
b0 = 1.0
alpha = 1.2
beta  = 0.3

sizes = np.array([1.0, 1.3, 0.8, 1.6, 1.1])
a_base = a0 * sizes
b_base = b0 * sizes

# ===============================
# FORCE FUNCTIONS
# ===============================

def attractive_force(q, q_goal):
    return -k_att * (q - q_goal)

def repulsive_force(q, obstacles_noisy, obstacle_speeds):
    F_rep_total = np.zeros(2)

    for i, obs in enumerate(obstacles_noisy):
        vx, vy = obstacle_speeds[i]
        vmag = np.sqrt(vx**2 + vy**2)

        theta = np.arctan2(vy, vx + 1e-12)

        a = a_base[i] + alpha * vmag
        b = b_base[i] + beta * vmag

        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, s],
                      [-s, c]])

        Q0 = np.diag([1/a**2, 1/b**2])
        Q  = R @ Q0 @ R.T

        v = q - obs
        dE = np.sqrt(float(v.T @ Q @ v) + 1e-12)

        if dE < d0:
            F_mag = k_rep * (1/dE - 1/d0) * (1/dE**2)
            grad = Q @ (q - obs) / (dE + 1e-12)
            F_rep = F_mag * grad

            # Limit force
            if np.linalg.norm(F_rep) > max_rep_force:
                F_rep = F_rep / np.linalg.norm(F_rep) * max_rep_force
        else:
            F_rep = np.zeros(2)

        F_rep_total += F_rep

    return F_rep_total

def potential(q, q_goal, obstacles_noisy, obstacle_speeds):
    U_rep_total = 0
    U_att = 0.5 * k_att * np.linalg.norm(q - q_goal)**2
    
    for i, obs in enumerate(obstacles_noisy):
        vx, vy = obstacle_speeds[i]
        vmag = np.sqrt(vx**2 + vy**2) 
        theta = np.arctan2(vy, vx + 1e-12)
        a = a_base[i] + alpha * vmag 
        b = b_base[i] + beta  * vmag 
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, s],
                      [-s,  c]])
        Q0 = np.diag([1/a**2, 1/b**2])
        Q = R @ Q0 @ R.T
        v = q - obs
        dE = np.sqrt(float(v.T @ Q @ v) + 1e-12)
        if dE < 1e-6:  # avoid division by zero
            dE = 1e-6
        U_rep = 0.5 * k_rep * (1/dE - 1/d0)**2 if dE < d0 else 0
        U_rep_total += U_rep
    return U_att + U_rep_total

def total_force(q, q_goal, obstacles_noisy, obstacle_speeds):
    return attractive_force(q, q_goal) + repulsive_force(q, obstacles_noisy, obstacle_speeds)

# ===============================
# ANIMATION SETUP
# ===============================

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-5, 15)
ax.set_ylim(-5, 15)
ax.set_title("Dynamic Obstacle Avoidance Animation")

# ----- BACKGROUND POTENTIAL FIELD -----
x_range = np.linspace(-5, 15, 60)
y_range = np.linspace(-5, 15, 60)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)
U = np.zeros_like(X)
V = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pos = np.array([X[i,j], Y[i,j]])

        # calculate potentia; for each frame
        Z[i,j] = potential(pos, q_goal, obstacles_noisy, obstacle_speeds)

        # force field
        Fx, Fy = total_force(pos, q_goal, obstacles_noisy, obstacle_speeds)
        U[i,j] = Fx
        V[i,j] = Fy

# draw the potential field
contour_bg = ax.contourf(X, Y, Z, levels=80, cmap='viridis', alpha=0.6)

# COLORBAR
cbar = fig.colorbar(contour_bg, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Potential Energy")

# normalize quiver vectors
mag = np.sqrt(U**2 + V**2) + 1e-12
U_norm = U / mag
V_norm = V / mag

# add quiver field
quiver_bg = ax.quiver(X, Y, U_norm, V_norm, color='white', alpha=0.6)

path_line, = ax.plot([], [], 'r-', linewidth=2)
robot_dot, = ax.plot([], [], 'ro', markersize=6)

true_scatter = ax.scatter([], [], c='black', s=40)
noisy_scatter = ax.scatter([], [], c='red', s=40)

goal_dot, = ax.plot(q_goal[0], q_goal[1], 'go', markersize=8, label="Goal")

ellipse_patches = []

def init():
    path_line.set_data([], [])
    robot_dot.set_data([], [])
    true_scatter.set_offsets(np.empty((0, 2)))
    noisy_scatter.set_offsets(np.empty((0, 2)))
    return path_line, robot_dot, true_scatter, noisy_scatter, goal_dot


# ===============================
# UPDATE FUNCTION
# ===============================

def update(frame):
    global q

    # --- 1) Move obstacles ---
    obstacles_true[:,0] += obstacle_speeds[:,0] * dt
    obstacles_true[:,1] += obstacle_speeds[:,1] * dt

    # --- 2) Noise sample ---
    obstacles_noisy[:] = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)

    # --- 3) Robot force ---
    F = total_force(q, q_goal, obstacles_noisy, obstacle_speeds)

    # --- 4) Robot motion ---
    q[:] = q + F * dt
    path_data.append(q.copy())

    # --- 5) Update path ---
    arr = np.array(path_data)
    path_line.set_data(arr[:,0], arr[:,1])
    robot_dot.set_data(q[0], q[1])

    # --- 6) Obstacle scatter ---
    true_scatter.set_offsets(obstacles_true)
    noisy_scatter.set_offsets(obstacles_noisy)

    # --- 7) Remove old ellipses ---
    for e in ellipse_patches:
        e.remove()
    ellipse_patches.clear()

    # --- 8) Draw ellipses (WITHOUT d0 scaling!) ---
    for i, obs in enumerate(obstacles_true):
        vx, vy = obstacle_speeds[i]
        vmag = np.sqrt(vx**2 + vy**2)
        theta = np.degrees(np.arctan2(vy, vx))

        a = a_base[i] + alpha * vmag
        b = b_base[i] + beta * vmag

        ellipse = Ellipse(
            xy=(obs[0], obs[1]),
            width=2*a,
            height=2*b,
            angle=theta,
            edgecolor='black',
            facecolor='cyan',
            alpha=0.15,       # şeffaflık
            linestyle='--',
            linewidth=1.2
        )
        ax.add_patch(ellipse)
        ellipse_patches.append(ellipse)

    return path_line, robot_dot, true_scatter, noisy_scatter, goal_dot


# ===============================
# RUN ANIMATION
# ===============================
ani = animation.FuncAnimation(
    fig,
    update,
    frames=400,
    init_func=init,
    interval=40,
    blit=False
)

plt.show()
