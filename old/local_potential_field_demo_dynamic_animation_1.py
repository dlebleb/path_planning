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
q = np.array([-40.0, -40.0])
v_robot = 25

# obstacle coordinates 
obstacles_true = np.array([[-18.0,-10.0], [18,-20], [18, 8], [22,26], [23,15], [-23,15], [5,5], [-40, -30], [15, -10], [10, 5], [-30, 0], [-20, -20], [0, 0]])
sigma = 0.1  # 10 cm uncertainity
obstacles_noisy = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)

# obstacle speeds
obstacle_speeds = np.array([[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1], [-0.1, 0.1], [-0.2, 0.1], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2]])
obstacle_speeds = obstacle_speeds * 80
#angles = np.deg2rad([135.0, 135.0, 63.43, -153.43, -45.0, 135.0, 26.57, 135.0, 153.43])
#vmag = np.array([11.3, 22.6, 17.88, 17.88, 11.3, 11.3, 17.88, 11.3, 17.88])
vmag_max = 24.5
vmag_min = 5

# APF parameters
#k_att, k_rep, d0, dt = 10.0, 500.0, 3.0, 0.001
k_att, k_rep, d0, dt = 10.0, 10.0, 10.0, 0.01
max_rep_force = np.inf
path_data = [q.copy()]
initial_obstacles = obstacles_true.copy()

# elliptical obstacles
a0 = 2.0 # major axis (along velocity direction)
b0 = 1.0 # minor axis (perpendicular to velocity direction)
alpha = 0.2   # major scaling (large)
beta  = 0.1   # minor scaling (small)

# each obstacle has a size factor: 1 = normal, >1 = large, <1 = small
sizes = np.array([1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1, 1.0, 1.2, 1.8, 2.0, 1.8, 1.9])
# incorporate static size                          
a_base = a0 * sizes 
b_base = b0 * sizes

a_max = 7
b_max = 5

# ===============================
# FORCE FUNCTIONS
# ===============================

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
        b = b_base[i] + beta  * vmag# minor axis (perpendicular)

        # clamp to max
        a = min(a, a_max)
        b = min(b, b_max)

        # move obstacle center to origin and transform
        obs_x, obs_y = obs[0], obs[1]
        # x2, y2 = -obs_x/a, -obs_y/b

        eps = 1e-12
        # move robot center to origin and transform
        q_x, q_y = (q[0]-obs_x)/(a+eps), (q[1]-obs_y)/(b+eps)

        # distance of the robot to origin -1
        dE = np.sqrt(q_x**2 + q_y**2) - 1

        # dynamic avoidance boundary
        # d0_i = d0 * max(a, b)

        if dE < d0:
            F_mag = k_rep * (1/dE - 1/d0) * (1/dE**2)
            # yön vektörü (normalize edilmiş fark)
            grad_Dq = (q - obs) / (dE + 1e-12)
            # toplam kuvvet
            F_rep = F_mag * grad_Dq

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
       
        a = a_base[i] + alpha * vmag 
        b = b_base[i] + beta  * vmag 
        a = min(a, a_max)
        b = min(b, b_max)
       
        obs_x, obs_y = obs[0], obs[1]

        eps = 1e-12
        q_x, q_y = (q[0]-obs_x)/(a+eps), (q[1]-obs_y)/(b+eps)
        
        dE = np.sqrt(q_x**2 + q_y**2) - 1

        # dynamic avoidance boundary
        # d0_i = d0 * max(a, b)

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

# def resolve_obstacle_collisions(obstacles_true, obstacle_speeds):
#     N = len(obstacles_true)

#     # --- compute dynamic radii ---
#     radii = np.zeros(N)
#     for i in range(N):
#         vx, vy = obstacle_speeds[i]
#         vmag = np.sqrt(vx**2 + vy**2)

#         a_i = a_base[i] + alpha * vmag
#         b_i = b_base[i] + beta  * vmag

#         # realistic effective radius = max(axis lengths)
#         radii[i] = max(a_i, b_i)

#     # --- pairwise collision check ---
#     for i in range(N):
#         for j in range(i+1, N):

#             difference = obstacles_true[i] - obstacles_true[j]
#             distance = np.linalg.norm(difference)
#             allowed_distance = radii[i] + radii[j]

#             if distance < allowed_distance and distance > 1e-9:  # collision
#                 penetration = allowed_distance - distance
#                 direction = difference / distance  # normalized
#                 obstacles_true[i] += direction * (penetration / 2)
#                 obstacles_true[j] -= direction * (penetration / 2)

def apply_stochastic_maneuver(obstacle_speeds, maneuver_prob=0.25,
                              magnitude_sigma=0.05, turn_sigma=0.02):
    """
    Modify obstacle velocities by adding stochastic maneuvers.
    """
    new_speeds = obstacle_speeds.copy()

    for i in range(len(new_speeds)):
        vx, vy = new_speeds[i]
        vmag = np.sqrt(vx**2 + vy**2)
        
        # 1) Small continuous jitter (small speed increase & decrease, 5%) (1 ± 0.15)
        # [1 ± 0.15] contains 99.7% 
        # vmag *= np.random.normal(1, magnitude_sigma)
        scale = np.random.normal(1, magnitude_sigma)
        scale = np.clip(scale, 0.85, 1.15)   # 3σ limit
        vmag *= scale
        vmag = np.clip(vmag, vmag_min, vmag_max)
       
        # 2) Slight random drift in direction(2%)
        theta = np.arctan2(vy, vx)
        theta += np.random.normal(0, turn_sigma)
        
        # reconstruct velocity
        vx = vmag * np.cos(theta)
        vy = vmag * np.sin(theta)

        # 3) Occasional large maneuver (%25 probability)
        if np.random.rand() < maneuver_prob:
            big_turn = np.random.uniform(-np.pi/2, np.pi/2)
            theta += big_turn
            vx = vmag * np.cos(theta)
            vy = vmag * np.sin(theta)

        new_speeds[i] = [vx, vy]

    return new_speeds

def is_collision_check(q, obstacles_noisy, obstacle_speeds):
    collided_indices = []
    counter = 0

    for i, obs in enumerate(obstacles_noisy):
        vx, vy = obstacle_speeds[i]
        vmag = np.sqrt(vx**2 + vy**2) 
        theta = np.arctan2(vy, vx + 1e-12)
               
        a = a_base[i] + alpha * vmag 
        b = b_base[i] + beta  * vmag
        a = min(a, a_max)
        b = min(b, b_max)
        
       # move obstacle center to origin and transform
        obs_x, obs_y = obs[0], obs[1]
        # x2, y2 = -obs_x/a, -obs_y/b

        eps = 1e-12
        # move robot center to origin and transform
        q_x, q_y = (q[0]-obs_x)/(a+eps), (q[1]-obs_y)/(b+eps)

        # distance of the robot to origin -1
        dE = np.sqrt(q_x**2 + q_y**2) - 1
        
        # COLLISION condition: robot is inside the ellipse
        if dE < 0:
            counter += 1
            collided_indices.append(i)
    
    return counter, collided_indices

    U_rep_total = 0
    U_att = 0.5 * k_att * np.linalg.norm(q - q_goal)**2
    
    for i, obs in enumerate(obstacles_noisy):
        vx, vy = obstacle_speeds[i]
        vmag = np.sqrt(vx**2 + vy**2) 
        theta = np.arctan2(vy, vx + 1e-12)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, s],
                      [-s,  c]])
        Q0 = np.diag([1/a**2, 1/b**2])
        Q = R @ Q0 @ R.T
        v = q - obs
        dE = np.sqrt(float(v.T @ Q @ v) + 1e-12)
        
        # dynamic avoidance boundary
        d0_i = d0 * max(a, b)

        if dE < 1e-6:  # avoid division by zero
            dE = 1e-6
        U_rep = 0.5 * k_rep * (1/dE - 1/d0_i)**2 if dE < d0_i else 0
        U_rep_total += U_rep
    return U_att + U_rep_total

# ===============================
# ANIMATION SETUP
# ===============================

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_title("Dynamic Obstacle Avoidance Animation")

# ----- BACKGROUND POTENTIAL FIELD -----
x_range = np.linspace(-50, 50, 50)
y_range = np.linspace(-50, 50, 50)
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
#contour_bg = ax.contourf(X, Y, Z, levels=80, cmap='viridis', alpha=0.6)

# COLORBAR
# cbar = fig.colorbar(contour_bg, ax=ax, fraction=0.046, pad=0.04)
# cbar.set_label("Potential Energy")
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.grid(color='gray', linestyle='--', linewidth=0.4, alpha=0.7)


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

tolerance = 1
# ===============================
# UPDATE FUNCTION
# ===============================

def update(frame):
    global q
    global obstacle_speeds
    global ani

    # 0) random maneuver (new speeds)
    obstacle_speeds = apply_stochastic_maneuver(obstacle_speeds)

    # --- 1) Move obstacles ---
    obstacles_true[:,0] += obstacle_speeds[:,0] * dt
    obstacles_true[:,1] += obstacle_speeds[:,1] * dt

    #print("Obstacle 0 position:", obstacles_true[0])
    # 1.1) resolve collisions between obstacles (prevent overlap)
    #resolve_obstacle_collisions(obstacles_true, obstacle_speeds)

    # --- 2) Noise sample ---
    obstacles_noisy[:] = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)

    # --- 3) Robot force ---
    F = total_force(q, q_goal, obstacles_noisy, obstacle_speeds)

    # --- 4) Robot motion ---
    F_norm = np.linalg.norm(F)
    direction = F/F_norm
    q[:] = q + direction * v_robot * dt
    path_data.append(q.copy())

    # --- STOP CONDITION: close enough to goal ---
    if np.linalg.norm(q - q_goal) < tolerance:
        print(f"Reached goal at frame {frame}")
        ani.event_source.stop()
        return path_line, robot_dot, true_scatter, noisy_scatter, goal_dot

    #. check the collision
    count, hits = is_collision_check(q, obstacles_noisy, obstacle_speeds)
    if count > 0:
        print("Collision detected with obstacle:", hits)

    # --- 5) Update path ---
    arr = np.array(path_data)
    path_line.set_data(arr[:,0], arr[:,1])
    robot_dot.set_data([q[0]], [q[1]])

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
        b = b_base[i] + beta  * vmag
        a = min(a, a_max)
        b = min(b, b_max)

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