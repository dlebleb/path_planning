"""
Local Potential-Field Path Planning (elliptical obstacles)

Run to plot a local path from start to goal around elliptical obstacles 
using an attractive + repulsive potential field.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


k_att, k_rep, d0, dt = 2.0, 10.0, 2.0, 0.01
q_goal = np.array([10, 10])
max_rep_force = 14.0
obstacles = np.array([[3, 3.5], [6,7], [9,9], [8,4], [5,5]])
obstacle_speeds = np.array([[0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1]])
q = np.array([0.0, 0.0])
path_data = [q.copy()]


def attractive_force(q, q_goal):
    F_att = -k_att * (q - q_goal)
    return F_att

def repulsive_force(q, obstacles):
    F_rep_total = np.array([0.0, 0.0])
    for obs in obstacles:
        d = np.linalg.norm(q - obs)
        if d < d0:
            F_rep = k_rep * (1/d - 1/d0) * (1/d**3) * (q - obs)
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

def total_force(q, q_goal, obstacles):
    """
    Calculate the total force on the robot
    """
    F_att = attractive_force(q, q_goal)
    F_rep = repulsive_force(q, obstacles)
    return F_att + F_rep

x_range = np.linspace(-5, 15, 50) #-5 ile 15 arasinda 50 esit parca olustur.
y_range = np.linspace(-5, 15, 50)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)
U = np.zeros_like(X)
V = np.zeros_like(Y)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pos = np.array([X[i, j], Y[i, j]])
        Z[i, j] = potential(pos, q_goal, obstacles)
        force = total_force(pos, q_goal, obstacles)
        U[i, j] = force[0]
        V[i, j] = force[1]

# form the figure
fig, ax  = plt.subplots()

# axis limits
ax.set_xlim(-5, 15)
ax.set_ylim(-5, 15)

# position of the target and obstacles
goal_plot, = ax.plot(q_goal[0], q_goal[1], 'go', label='Goal')
obstacles_plot, = ax.plot(obstacles[:, 0], obstacles[:, 1], 'ro', label='Obstacles')

# robot path and current position
path_plot, = ax.plot([], [], 'b-', label='Robot Path')
robot_plot, = ax.plot([], [], 'bo')

# potential energy height map-high potential energy areas have darker colour
potential_contour = ax.contourf(X, Y, Z, levels=100, cmap='viridis')

# force x and y directions in each point
quiver_all = ax.quiver(X, Y, U, V, color='white', alpha=0.5)
# force x and y directions athe current position of the robot
quiver_robot = ax.quiver([q[0]], [q[1]], [0], [0],
                         color='red',
                         scale=10,           
                         scale_units='xy',   
                         angles='xy') 

def init():
    path_plot.set_data([], [])
    robot_plot.set_data([], [])
    quiver_robot.set_UVC(0, 0)
    return path_plot, robot_plot, quiver_robot


def update(frame):
    global q
    force = total_force(q, q_goal, obstacles)
    q = q + force * dt
    path_data.append(q.copy())

    path = np.array(path_data)
    path_plot.set_data(path[:, 0], path[:, 1])
    robot_plot.set_data(q[0], q[1])

    quiver_robot.set_offsets(q)
    quiver_robot.set_UVC(force[0], force[1])
    return path_plot, robot_plot, quiver_robot

ani = animation.FuncAnimation(fig, update, frames=2000, init_func=init, blit=True, interval=100, repeat=False)

ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Dynamic Potential Field')
ax.grid()

plt.show()

# ==========================================================
# After animation: show additional plots
# ==========================================================
q_new = np.array([0.0, 0.0]) # starting position of the robot
path_data_new = [q_new.copy()] 

max_steps = 1000
tolerance = 0.1

for step in range(max_steps):
    F = total_force(q_new, q_goal, obstacles)
    q_new = q_new + F * dt
    path_data_new.append(q_new.copy())

    # stop condition: close enough to goal
    if np.linalg.norm(q_new - q_goal) < tolerance:
        print(f"Reached goal in {step} steps!")
        break

path_new = np.array(path_data_new)

# ---- 1️⃣ 3D Potential Surface ----
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9)
ax.scatter(obstacles[:,0], obstacles[:,1],
            np.max(Z)*0.8, color='red', s=50, label='Obstacles')
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
axs[0].plot(path_new[:,0], path_new[:,1], 'w-', label='Path')
axs[0].plot(q_goal[0], q_goal[1], 'go')
axs[0].plot(obstacles[:,0], obstacles[:,1], 'ro')
axs[0].set_title("Potential Energy Map")
axs[0].legend()
plt.colorbar(contour, ax=axs[0])

axs[1].quiver(X, Y, U, V, color='black', alpha=0.6)
axs[1].plot(path_new[:,0], path_new[:,1], 'r-')
axs[1].set_title("Force Field (Gradient of Potential)")
axs[1].set_xlabel("X")
axs[1].set_ylabel("Y")
plt.tight_layout()
plt.show()

# ---- 3️⃣ 3D Path over Potential Surface ----
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax.plot(path_new[:,0], path_new[:,1],
        [potential(p, q_goal, obstacles) for p in path_new],
        color='red', linewidth=2, label='Path')
ax.scatter(q_goal[0], q_goal[1],
            potential(q_goal, q_goal, obstacles), color='green', s=50, label='Goal')
ax.scatter(obstacles[:,0], obstacles[:,1],
            [potential(o, q_goal, obstacles) for o in obstacles],
            color='red', s=50, label='Obstacles')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Potential')
ax.set_title('3D Potential Field with Path')
ax.legend()
plt.show()
