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
obstacles = np.array([[3, 3.5], [6,7], [9,9], [8,4], [5,5]])
obstacle_speeds = np.array([[0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1]])
q = np.array([0.0, 0.0]) # starting position of the robot
path_data = [q.copy()]
initial_obstacles = obstacles.copy()

# elliptical obstacles
a = 2.0 # major axis
b = 1.0 # minor axis


def attractive_force(q, q_goal):
    F_att = -k_att * (q - q_goal)
    return F_att

def repulsive_force(q, obstacles, obstacle_speeds):
    F_rep_total = np.array([0.0, 0.0])
    for i, obs in enumerate(obstacles):
        # calculate the position difference: x-x0, y-y0
        # this is the vector strating from obstacle position and going towards robot position
        delta_x = (q - obs)[0]
        delta_y = (q - obs)[1]
        vx, vy = obstacle_speeds[i]
        theta = np.arctan2(vy, vx + 1e-6)  # obstacle motion direction

        # rotate into obstacle frame
        x_prime = delta_x*np.cos(theta) + delta_y*np.sin(theta)
        y_prime = -delta_x*np.sin(theta) + delta_y*np.cos(theta)

        # elliptical distance
        dE = np.sqrt((x_prime/a)**2 + (y_prime/b)**2)

        if dE < d0:
            F_mag = k_rep * (1/dE - 1/d0) * (1/dE**3)
            # yön vektörü (normalize edilmiş fark)
            grad = (q - obs) / np.linalg.norm((q - obs))
            # toplam kuvvet
            F_rep = F_mag * grad

            if np.linalg.norm(F_rep) > max_rep_force: 
                F_rep = F_rep/np.linalg.norm(F_rep) * max_rep_force
        else:
            F_rep = np.array([0.0, 0.0])
        F_rep_total += F_rep
    return F_rep_total


def potential(q, q_goal, obstacles):
    U_rep_total = 0
    U_att = 0.5 * k_att * np.linalg.norm(q - q_goal)**2
    
    for obs in obstacles:
        d = np.linalg.norm(q - obs)
        if d < 1e-6:  # avoid division by zero
            d = 1e-6
        U_rep = 0.5 * k_rep * (1/d - 1/d0)**2 if d < d0 else 0
        U_rep_total += U_rep
    return U_att + U_rep_total

def total_force(q, q_goal, obstacles, obstacle_speeds):
    """
    Calculate the total force on the robot
    """
    F_att = attractive_force(q, q_goal)
    F_rep = repulsive_force(q, obstacles, obstacle_speeds)
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
    F = total_force(q, q_goal, obstacles, obstacle_speeds)
    q = q + F * dt
    obstacles[:,0] += obstacle_speeds[:,0]*np.sin(0.05*step)
    obstacles[:,1] += obstacle_speeds[:,1]*np.cos(0.05*step)    
    path_data.append(q.copy())

    # stop condition: close enough to goal
    if np.linalg.norm(q - q_goal) < tolerance:
        print(f"Reached goal in {step} steps!")
        break

path = np.array(path_data)

# compute field after motion
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pos = np.array([X[i,j], Y[i,j]])
        Z[i,j] = potential(pos, q_goal, obstacles)
        F = total_force(pos, q_goal, obstacles, obstacle_speeds)
        U[i,j], V[i,j] = F[0], F[1]

# ==========================================================
# Show plots
# ==========================================================

# ---- 1️⃣ 3D Potential Surface ----
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9)
ax.scatter(obstacles[:,0], obstacles[:,1],
            np.max(Z)*0.8, color='red', s=50, label='Obstacle Centers')
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
axs[0].plot(q_goal[0], q_goal[1], 'go', label='Goal')
axs[0].plot(obstacles[:,0], obstacles[:,1], 'ro', label='Obstacle Centers')
axs[0].plot(initial_obstacles[:,0], initial_obstacles[:,1], 'ko', label='Initial Obstacle Centers')

# --- elipsleri çiz ---
for i, obs in enumerate(obstacles):
    vx, vy = obstacle_speeds[i]
    theta = np.degrees(np.arctan2(vy, vx + 1e-6))  # derece cinsinden açı
    ellipse = Ellipse(
        xy=(obs[0], obs[1]),  # elipsin merkezi
        width=2*a,            # major axis (x yönünde)
        height=2*b,           # minor axis (y yönünde)
        angle=theta,          # hız yönüyle hizalama
        edgecolor='r', facecolor='none', linestyle='--', linewidth=1.5
    )
    axs[0].add_patch(ellipse)

axs[0].set_title("Potential Energy Map with Elliptical Obstacles")
axs[0].legend()
plt.colorbar(contour, ax=axs[0])

axs[1].quiver(X, Y, U, V, color='black', alpha=0.6)
axs[1].plot(path[:,0], path[:,1], 'r-')
axs[1].set_title("Force Field (Gradient of Potential)")
axs[1].set_xlabel("X")
axs[1].set_ylabel("Y")
plt.tight_layout()
plt.show()

# ---- 3️⃣ 3D Path over Potential Surface ----
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax.plot(path[:,0], path[:,1],
        [potential(p, q_goal, obstacles) for p in path],
        color='red', linewidth=2, label='Path')
ax.scatter(q_goal[0], q_goal[1],
            potential(q_goal, q_goal, obstacles), color='green', s=50, label='Goal')
ax.scatter(obstacles[:,0], obstacles[:,1],
            [potential(o, q_goal, obstacles) for o in obstacles],
            color='red', s=50, label='Obstacle Centers')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Potential')
ax.set_title('3D Potential Field with Path')
ax.legend()
plt.show()
