"""
Local Potential-Field Path Planning (elliptical obstacles)

Run to plot a local path from start to goal around elliptical obstacles 
using an attractive + repulsive potential field.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse

os.makedirs("figures", exist_ok=True)

q_goal = np.array([10, 10]) # goal position
q = np.array([0.0, 0.0]) # starting position of the robot

# obstacle coordinates 
obstacles_true = np.array([[3, 3.5], [6,7], [9,9], [8,4], [5,5]])
sigma = 0.1  # 10 cm uncertainity
obstacles_noisy = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)

# obstacle speeds
obstacle_speeds = np.array([[0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1]])
obstacle_speeds = obstacle_speeds * 5

# APF parameters
k_att, k_rep, d0, dt = 2.0, 40.0, 2.0, 0.01
max_rep_force = 14.0
path_data = [q.copy()]
initial_obstacles = obstacles_true.copy()

# elliptical obstacles
a0 = 2.0 # major axis (along velocity direction)
b0 = 1.0 # minor axis (perpendicular to velocity direction)
alpha = 1.2   # major scaling (large)
beta  = 0.3   # minor scaling (small)

# each obstacle has a size factor: 1 = normal, >1 = large, <1 = small
sizes = np.array([1.0, 1.3, 0.8, 1.6, 1.1])
# incorporate static size
a_base = a0 * sizes
b_base = b0 * sizes


"""
Dynamic obstacle modeling:
1) Velocity is aligned with the major axis of the ellipse. Frep is higher. Hiz yonunde daha fazla itecek.
2) Buyuk engelden daha itici guc gelecek, a ve b buyuyecek 1/a ve 1/b kuculecek, dE kuculecek, 1/dE buyuyecek.
Frep artacak.
3) Hizli engelden daha itici guc gelecek.
4) Engelin gercek yerini bilmiyoruz. Gaussian noise ekledik. Her zaman adiminda bu pdf'ten bir sample aliyoruz.  
"""

def attractive_force(q, q_goal):
    F_att = -k_att * (q - q_goal)
    return F_att

def repulsive_force(q, obstacles_noisy, obstacle_speeds):
    F_rep_total = np.array([0.0, 0.0])
    for i, obs in enumerate(obstacles_noisy):
        vx, vy = obstacle_speeds[i]
        vmag = np.sqrt(vx**2 + vy**2) 
        theta = np.arctan2(vy, vx + 1e-12)  # obstacle motion direction

        # dynamic scaling with speed
        a = a_base[i] + alpha * vmag # major axis (velocity direction)
        b = b_base[i] + beta  * vmag # minor axis (perpendicular)

        # rotate into obstacle frame and calculate the Q matrix
        c, s = np.cos(theta), np.sin(theta)
        
        # rotation matrix that aligns major axis with velocity
        R = np.array([[c, s],
                  [-s,  c]])
        
        # Q0 (major axis = a, minor axis = b)
        Q0 = np.diag([1/a**2, 1/b**2])
        Q  = R @ Q0 @ R.T

        # elliptical distance
        dE = np.sqrt(float((q - obs).T @ Q @ (q - obs)) + 1e-12)

        # dynamic avoidance boundary
        d0_i = 1.2 * max(a, b)

        if dE < d0_i:
            F_mag = k_rep * (1/dE - 1/d0_i) * (1/dE**2)
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
        
        # dynamic avoidance boundary
        d0_i = 1.2 * max(a, b)

        if dE < 1e-6:  # avoid division by zero
            dE = 1e-6
        U_rep = 0.5 * k_rep * (1/dE - 1/d0_i)**2 if dE < d0_i else 0
        U_rep_total += U_rep
    return U_att + U_rep_total

def total_force(q, q_goal, obstacles_noisy, obstacle_speeds):
    """
    Calculate the total force on the robot
    """
    F_att = attractive_force(q, q_goal)
    F_rep = repulsive_force(q, obstacles_noisy, obstacle_speeds)
    return F_att + F_rep

x_range = np.linspace(-25, 25, 50) #-5 ile 15 arasinda 50 esit parca olustur.
y_range = np.linspace(-25, 25, 50)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)
U = np.zeros_like(X)
V = np.zeros_like(Y)

# simulate the path
max_steps = 2000
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
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Surface + colorbar
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label("Potential Energy", fontsize=14)
cbar.ax.tick_params(labelsize=12)

# Path with correct potential computation
ax.plot(path[:,0], path[:,1],
        [potential(p, q_goal, obstacles_noisy, obstacle_speeds) for p in path],
        color='red', linewidth=2, label='Path')

# Start position (⭐)
ax.scatter(path[0,0], path[0,1],
           potential(path[0], q_goal, obstacles_noisy, obstacle_speeds),
           color='cyan', marker='x', s=120, linewidths=3, label='Start')

# REAL OBSTACLES (truth)
ax.scatter(obstacles_true[:,0], obstacles_true[:,1],
            np.max(Z)*0.8, color='black', s=80, label='True Obstacle Centers')

# NOISY OBSTACLES (sensor-detected) 
ax.scatter(obstacles_noisy[:,0], obstacles_noisy[:,1],
            np.max(Z)*0.8, color='red', s=80, label='Noisy Detected Centers')

# Goal position
ax.scatter(q_goal[0], q_goal[1], np.min(Z),
           color='orange', s=80, marker='x', linewidths=3, label='Goal')

# FONT SIZE SETTINGS
ax.set_xlabel('X', fontsize=16)
ax.set_ylabel('Y', fontsize=16)
ax.set_zlabel('Potential Energy', fontsize=16)

ax.set_title('3D Potential Field Surface', fontsize=18)

# Tick label size
ax.tick_params(axis='both', labelsize=12)

# Legend font size
ax.legend(fontsize=14)

plt.savefig("figures/fig_3d_potentialsurface.png", dpi=300, bbox_inches='tight')
plt.show()


# ---- 2️⃣ 2D Contour + Force Field ----
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
contour = axs[0].contourf(X, Y, Z, levels=100, cmap='viridis')

# --- Start Position ---
start_x, start_y = path[0]
axs[0].scatter(start_x, start_y, marker='x', s=120, color='cyan', linewidths=3, label='Start')

axs[0].plot(path[:,0], path[:,1], 'w-', label='Path')

# true vs noisy obstacles
axs[0].plot(obstacles_true[:,0], obstacles_true[:,1], 'ko', label='True Centers')
axs[0].plot(obstacles_noisy[:,0], obstacles_noisy[:,1], 'ro', label='Noisy Centers')

# --- Goal Position ---
axs[0].scatter(q_goal[0], q_goal[1],
               marker='x', s=120, color='orange', linewidths=3, label='Goal')

# --- draw ellipses for TRUE obstacles
for i, obs in enumerate(obstacles_true):
    vx, vy = obstacle_speeds[i]
    vmag = np.sqrt(vx**2 + vy**2)
    theta = np.degrees(np.arctan2(vy, vx + 1e-12))

    # 1) Physical ellipse (dE = 1 boundary)
    a = a_base[i] + alpha * vmag
    b = b_base[i] + beta  * vmag
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
plt.savefig("figures/fig_contour_force.png", dpi=300, bbox_inches='tight')
plt.show()

