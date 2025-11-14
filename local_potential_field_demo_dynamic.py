"""
Local Potential-Field Path Planning (elliptical obstacles)

Run to plot a local path from start to goal around elliptical obstacles 
using an attractive + repulsive potential field.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse


k_att, k_rep, d0, dt = 2.0, 10.0, 2.0, 0.01
q_goal = np.array([10, 10])
max_rep_force = 14.0
obstacles_true = np.array([[3, 3.5], [6,7], [9,9], [8,4], [5,5]])
sigma = 0.1  # 10 cm uncertainity
obstacles_noisy = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)
obstacle_speeds = np.array([[0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1]])
q = np.array([0.0, 0.0]) # starting position of the robot
path_data = [q.copy()]
initial_obstacles = obstacles_true.copy()

# elliptical obstacles
a = 2.0 # major axis
b = 1.0 # minor axis


def attractive_force(q, q_goal):
    F_att = -k_att * (q - q_goal)
    return F_att

def repulsive_force(q, obstacles_noisy, obstacle_speeds):
    F_rep_total = np.array([0.0, 0.0])
    for i, obs in enumerate(obstacles_noisy):
        vx, vy = obstacle_speeds[i]
        theta = np.arctan2(vy, vx + 1e-12)  # obstacle motion direction

        # rotate into obstacle frame and calculate the Q matrix
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s],
                  [s,  c]])
        Q0 = np.diag([1/a**2, 1/b**2])
        Q  = R @ Q0 @ R.T

        # elliptical distance
        dE = np.sqrt(float((q - obs).T @ Q @ (q - obs)) + 1e-12)

        if dE < d0:
            F_mag = k_rep * (1/dE - 1/d0) * (1/dE**2)
            # yön vektörü (normalize edilmiş fark)
            grad_Dq = Q @ (q - obs) / (dE + 1e-12)
            # toplam kuvvet
            F_rep = F_mag * grad_Dq

            if np.linalg.norm(F_rep) > max_rep_force: 
                F_rep = F_rep/(np.linalg.norm(F_rep) + 1e-12)* max_rep_force
        else:
            F_rep = np.array([0.0, 0.0])
        F_rep_total += F_rep
    return F_rep_total


def potential(q, q_goal, obstacles_noisy, obstacle_speeds):
    U_rep_total = 0
    U_att = 0.5 * k_att * np.linalg.norm(q - q_goal)**2
    
    for i, obs in enumerate(obstacles_noisy):
        vx, vy = obstacle_speeds[i]
        theta = np.arctan2(vy, vx + 1e-12)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s],
                      [s,  c]])
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
    """
    Calculate the total force on the robot
    """
    F_att = attractive_force(q, q_goal)
    F_rep = repulsive_force(q, obstacles_noisy, obstacle_speeds)
    return F_att + F_rep

x_range = np.linspace(-5, 15, 50) #-5 ile 15 arasinda 50 esit parca olustur.
y_range = np.linspace(-5, 15, 50)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)
U = np.zeros_like(X)
V = np.zeros_like(Y)

# simulate the path
max_steps = 1000
tolerance = 0.1

for step in range(max_steps):
    # 1) move the real obstacles
    obstacles_true[:,0] += obstacle_speeds[:,0]*dt
    obstacles_true[:,1] += obstacle_speeds[:,1]*dt

    # 2) robot observes obstacles (noisy)
    obstacles_noisy = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)

    # 3) force calculation
    F = total_force(q, q_goal, obstacles_noisy, obstacle_speeds)

    # 4) robot moves
    q = q + F * dt

    # 5) path
    path_data.append(q.copy())

    # stop condition: close enough to goal
    if np.linalg.norm(q - q_goal) < tolerance:
        print(f"Reached goal in {step} steps!")
        break

path = np.array(path_data)

# ==========================================================
# Compute field AFTER full motion simulation
# ==========================================================

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pos = np.array([X[i,j], Y[i,j]])
        
        # potential uses noisy obstacles
        Z[i,j] = potential(pos, q_goal, obstacles_noisy, obstacle_speeds)

        # force field also uses noisy obstacles
        F = total_force(pos, q_goal, obstacles_noisy, obstacle_speeds)
        U[i,j], V[i,j] = F[0], F[1]

# ==========================================================
#                     PLOTTING SECTION
# ==========================================================

# ---- 1️⃣ 3D Potential Surface ----
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9)

# REAL OBSTACLES (truth)
ax.scatter(obstacles_true[:,0], obstacles_true[:,1],
            np.max(Z)*0.8, color='black', s=50, label='True Obstacle Centers')

# NOISY (sensor-detected) OBSTACLES
ax.scatter(obstacles_noisy[:,0], obstacles_noisy[:,1],
            np.max(Z)*0.8, color='red', s=50, label='Noisy Detected Centers')

ax.scatter(q_goal[0], q_goal[1],
            np.min(Z), color='green', s=50, label='Goal')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Potential Energy')
ax.set_title('3D Potential Field Surface')
ax.legend()
plt.show()


# ---- 2️⃣ 2D Contour + Force Field ----
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
contour = axs[0].contourf(X, Y, Z, levels=100, cmap='viridis')

axs[0].plot(path[:,0], path[:,1], 'w-', label='Path')

# true vs noisy
axs[0].plot(obstacles_true[:,0], obstacles_true[:,1], 'ko', label='True Centers')
axs[0].plot(obstacles_noisy[:,0], obstacles_noisy[:,1], 'ro', label='Noisy Centers')

axs[0].plot(q_goal[0], q_goal[1], 'go', label='Goal')


# --- draw ellipses for TRUE obstacles
for i, obs in enumerate(obstacles_true):
    vx, vy = obstacle_speeds[i]
    theta = np.degrees(np.arctan2(vy, vx + 1e-12))
    ellipse = Ellipse(
        xy=(obs[0], obs[1]),
        width=2*a, height=2*b,
        angle=theta,
        edgecolor='white', facecolor='none',
        linestyle='--', linewidth=1.5
    )
    axs[0].add_patch(ellipse)

axs[0].set_title("Potential Energy Map with Elliptical Obstacles")
axs[0].legend()
plt.colorbar(contour, ax=axs[0])


# FORCE FIELD PLOT
axs[1].quiver(X, Y, U, V, color='black', alpha=0.6)
axs[1].plot(path[:,0], path[:,1], 'r-', linewidth=2)
axs[1].set_title("Force Field (Gradient of Potential)")
axs[1].set_xlabel("X")
axs[1].set_ylabel("Y")

plt.tight_layout()
plt.show()


# ---- 3️⃣ 3D Path over Potential Surface ----
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

# Path with correct potential computation
ax.plot(path[:,0], path[:,1],
        [potential(p, q_goal, obstacles_noisy, obstacle_speeds) for p in path],
        color='red', linewidth=2, label='Path')

ax.scatter(q_goal[0], q_goal[1], 0, color='green', s=50, label='Goal')

# true obstacles 
ax.scatter(obstacles_true[:,0], obstacles_true[:,1],
           np.zeros(len(obstacles_true)),  
           color='black', s=50, label='True Centers')

# noisy centers
ax.scatter(obstacles_noisy[:,0], obstacles_noisy[:,1],
           np.zeros(len(obstacles_noisy)),
           color='red', s=50, label='Noisy Centers')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Potential')
ax.set_title('3D Potential Field with Path')
ax.legend()
plt.show()
