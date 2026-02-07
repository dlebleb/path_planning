"""
RRT + Potential Field Path Planning Animation (baseline / simple version)

Same as the full version but WITHOUT stall-prevention and related improvements:
- No stall detection or stall-triggered replan
- No multi-run RRT (single path per replan)
- No obstacle speed/ellipse caps
- No goal region or reduced repulsion near goal
- No logging (stall, collision, progress)

Run to see: RRT plans waypoints, potential field follows them toward the goal.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
from typing import List, Optional

# Import RRT planner from simulation folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'simulation'))
from TSP import Point, Obstacle
from rrt_planner import RRTPlanner

# ===============================
# INITIAL SETUP
# ===============================
q_goal = np.array([10, 12])
q = np.array([-40.0, -40.0])
v_robot = 12

obstacles_true = np.array([[-18.0,-10.0], [18,-20], [18, 8], [22,26], [23,15], [-23,15], [5,5]])
sigma = 0.1
obstacles_noisy = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)

obstacle_speeds = np.array([[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1]])
obstacle_speeds = obstacle_speeds * 80

k_att, k_rep, d0, dt = 4.0, 10.0, 10.0, 0.01
path_data = [q.copy()]

a0, b0 = 2.0, 1.0
alpha, beta = 0.2, 0.1
sizes = np.array([1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1])
a_base = a0 + sizes
b_base = b0 + sizes
a_max = 7
b_max = 5

# ===============================
# FORCE FUNCTIONS
# ===============================

def attractive_force(q, q_goal):
    return -k_att * (q - q_goal)

def repulsive_force(q, obstacles_noisy, obstacle_speeds):
    F_rep_total = np.array([0.0, 0.0])
    for i, obs in enumerate(obstacles_noisy):
        vx, vy = obstacle_speeds[i]
        vmag = np.sqrt(vx**2 + vy**2)
        a = a_base[i] + alpha * vmag
        b = b_base[i] + beta * vmag
        a_max = 7
        b_max = 5
        obs_x, obs_y = obs[0], obs[1]
        eps = 1e-12
        q_x, q_y = (q[0]-obs_x)/(a+eps), (q[1]-obs_y)/(b+eps)
        dE = np.sqrt(q_x**2 + q_y**2) - 1

        if dE < d0:
            diff = q - obs
            dist_raw = np.linalg.norm(diff) + 1e-12
            outward = diff / dist_raw
            if dE < 0:
                F_rep = k_rep * 2.0 * outward
            else:
                F_mag = k_rep * (1/dE - 1/d0) * (1/dE**2)
                F_rep = F_mag * outward
        else:
            F_rep = np.array([0.0, 0.0])
        F_rep_total += F_rep
    return F_rep_total

def potential(q, q_goal, obstacles_noisy, obstacle_speeds):
    U_att = 0.5 * k_att * np.linalg.norm(q - q_goal)**2
    U_rep_total = 0
    for i, obs in enumerate(obstacles_noisy):
        vx, vy = obstacle_speeds[i]
        vmag = np.sqrt(vx**2 + vy**2)
        a = a_base[i] + alpha * vmag
        b = b_base[i] + beta * vmag
        a_max = 7
        b_max = 5
        obs_x, obs_y = obs[0], obs[1]
        eps = 1e-12
        q_x, q_y = (q[0]-obs_x)/(a+eps), (q[1]-obs_y)/(b+eps)
        dE = np.sqrt(q_x**2 + q_y**2) - 1
        if dE < 1e-6:
            dE = 1e-6
        U_rep = 0.5 * k_rep * (1/dE - 1/d0)**2 if dE < d0 else 0
        U_rep_total += U_rep
    return U_att + U_rep_total

def total_force(q, q_goal, obstacles_noisy, obstacle_speeds):
    return attractive_force(q, q_goal) + repulsive_force(q, obstacles_noisy, obstacle_speeds)

def apply_stochastic_maneuver(obstacle_speeds, maneuver_prob=0.25, magnitude_sigma=0.05, turn_sigma=0.02):
    new_speeds = obstacle_speeds.copy()
    for i in range(len(new_speeds)):
        vx, vy = new_speeds[i]
        vmag = np.sqrt(vx**2 + vy**2)
        vmag *= np.random.normal(1, magnitude_sigma)
        theta = np.arctan2(vy, vx) + np.random.normal(0, turn_sigma)
        vx, vy = vmag * np.cos(theta), vmag * np.sin(theta)
        if np.random.rand() < maneuver_prob:
            theta += np.random.uniform(-np.pi/2, np.pi/2)
            vx, vy = vmag * np.cos(theta), vmag * np.sin(theta)
        new_speeds[i] = [vx, vy]
    return new_speeds

def min_clearance(q, obstacles_noisy, obstacle_speeds):
    min_dE = 1e9
    for i, obs in enumerate(obstacles_noisy):
        vx, vy = obstacle_speeds[i]
        vmag = np.sqrt(vx**2 + vy**2)
        a = a_base[i] + alpha * vmag
        b = b_base[i] + beta * vmag
        a_max = 7
        b_max = 5
        obs_x, obs_y = obs[0], obs[1]
        eps = 1e-12
        q_x = (q[0] - obs_x) / (a + eps)
        q_y = (q[1] - obs_y) / (b + eps)
        dE = np.sqrt(q_x**2 + q_y**2) - 1
        min_dE = min(min_dE, dE)
    return min_dE

def is_collision_check(q, obstacles_noisy, obstacle_speeds):
    collided_indices = []
    for i, obs in enumerate(obstacles_noisy):
        vx, vy = obstacle_speeds[i]
        vmag = np.sqrt(vx**2 + vy**2)
        a = a_base[i] + alpha * vmag
        b = b_base[i] + beta * vmag
        a_max = 7
        b_max = 5
        obs_x, obs_y = obs[0], obs[1]
        eps = 1e-12
        q_x, q_y = (q[0]-obs_x)/(a+eps), (q[1]-obs_y)/(b+eps)
        dE = np.sqrt(q_x**2 + q_y**2) - 1
        if dE < 0:
            collided_indices.append(i)
    return len(collided_indices), collided_indices

# ===============================
# RRT INTEGRATION
# ===============================

def convert_obstacles_for_rrt(obstacles_true, obstacle_speeds, safety_margin: float = 0.5):
    obstacles_rrt = []
    for i, obs_pos in enumerate(obstacles_true):
        vx, vy = obstacle_speeds[i] if i < len(obstacle_speeds) else [0, 0]
        vmag = np.sqrt(vx**2 + vy**2)
        a = a_base[i] + alpha * vmag
        b = b_base[i] + beta * vmag
        a_max = 7
        b_max = 5
        radius = max(a, b) + safety_margin
        obstacles_rrt.append(Obstacle(Point(obs_pos[0], obs_pos[1]), radius))
    return obstacles_rrt

rrt_planner = RRTPlanner(step_size=1.0, max_iterations=1000)
rrt_waypoints: Optional[List[Point]] = None
current_rrt_index = 0
rrt_replan_interval = 50
last_rrt_replan = 0
waypoint_threshold = 0.5
tolerance = 1.5

# ===============================
# ANIMATION SETUP
# ===============================

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_title("RRT + Potential Field (Baseline)")
ax.legend(loc='upper right')

x_range = np.linspace(-50, 50, 50)
y_range = np.linspace(-50, 50, 50)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)
U, V = np.zeros_like(X), np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pos = np.array([X[i,j], Y[i,j]])
        Z[i,j] = potential(pos, q_goal, obstacles_noisy, obstacle_speeds)
        Fx, Fy = total_force(pos, q_goal, obstacles_noisy, obstacle_speeds)
        U[i,j], V[i,j] = Fx, Fy
mag = np.sqrt(U**2 + V**2) + 1e-12
quiver_bg = ax.quiver(X, Y, U/mag, V/mag, color='white', alpha=0.6)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

path_line, = ax.plot([], [], 'r-', linewidth=2, label="Robot Path", zorder=10)
robot_dot, = ax.plot([], [], 'ro', markersize=6, zorder=11)
true_scatter = ax.scatter([], [], c='black', s=40, label="True Obstacles")
noisy_scatter = ax.scatter([], [], c='red', s=40, label="Noisy Obstacles")
goal_dot, = ax.plot(q_goal[0], q_goal[1], 'go', markersize=8, label="Goal")
rrt_path_line, = ax.plot([], [], 'b--', linewidth=1.5, alpha=0.7, label="RRT Path")
rrt_scatter = ax.scatter([], [], c='blue', s=30, marker='o', alpha=0.5, label="RRT Waypoints")
current_target_scatter = ax.scatter([], [], c='yellow', s=100, marker='*', label="Current Target")
guide_arrow = ax.quiver([0], [0], [0], [0], color='gold', scale_units='xy', scale=1,
                        width=0.06, headwidth=4, headlength=3, alpha=0.9, label="Guide")
ellipse_patches = []

def init():
    path_line.set_data([], [])
    robot_dot.set_data([], [])
    true_scatter.set_offsets(np.empty((0, 2)))
    noisy_scatter.set_offsets(np.empty((0, 2)))
    rrt_path_line.set_data([], [])
    rrt_scatter.set_offsets(np.empty((0, 2)))
    current_target_scatter.set_offsets(np.empty((0, 2)))
    guide_arrow.set_offsets(np.array([[0, 0]]))
    guide_arrow.set_UVC(np.array([[0]]), np.array([[0]]))
    return path_line, robot_dot, true_scatter, noisy_scatter, goal_dot, rrt_path_line, rrt_scatter, current_target_scatter, guide_arrow

def update(frame):
    global q, obstacle_speeds, ani, rrt_waypoints, current_rrt_index, last_rrt_replan

    obstacle_speeds = apply_stochastic_maneuver(obstacle_speeds)
    obstacles_true[:,0] += obstacle_speeds[:,0] * dt
    obstacles_true[:,1] += obstacle_speeds[:,1] * dt
    obstacles_noisy[:] = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)

    obstacles_rrt = convert_obstacles_for_rrt(obstacles_true, obstacle_speeds)
    if (frame - last_rrt_replan) >= rrt_replan_interval or rrt_waypoints is None:
        start_point = Point(q[0], q[1])
        goal_point = Point(q_goal[0], q_goal[1])
        new_waypoints = rrt_planner.plan_path(start_point, goal_point, obstacles_rrt, bounds=(-50, 50, -50, 50))
        if new_waypoints:
            rrt_waypoints = new_waypoints
            current_rrt_index = 0
        last_rrt_replan = frame

    if rrt_waypoints and current_rrt_index < len(rrt_waypoints):
        target_point = rrt_waypoints[current_rrt_index]
        q_target = np.array([target_point.x, target_point.y])
        if np.linalg.norm(q - q_target) < waypoint_threshold:
            current_rrt_index += 1
            if current_rrt_index >= len(rrt_waypoints):
                current_rrt_index = len(rrt_waypoints) - 1
    else:
        q_target = q_goal.copy()

    F_att = attractive_force(q, q_target)
    F_rep = repulsive_force(q, obstacles_noisy, obstacle_speeds)
    F = F_att + F_rep

    F_norm = np.linalg.norm(F)
    direction = F / F_norm if F_norm > 1e-6 else np.array([0.0, 0.0])
    step = direction * v_robot * dt
    clearance = min_clearance(q, obstacles_noisy, obstacle_speeds)
    if 0 < clearance < 1.0:
        step_scale = max(0.2, min(1.0, clearance))
        step = step * step_scale
    q[:] = q + step
    path_data.append(q.copy())

    arr = np.array(path_data)
    path_line.set_data(arr[:,0], arr[:,1])
    robot_dot.set_data([q[0]], [q[1]])

    if np.linalg.norm(q - q_goal) < tolerance:
        print("Reached goal at frame", frame)
        ani.event_source.stop()
        return path_line, robot_dot, true_scatter, noisy_scatter, goal_dot, rrt_path_line, rrt_scatter, current_target_scatter, guide_arrow

    if rrt_waypoints:
        rrt_x = [wp.x for wp in rrt_waypoints]
        rrt_y = [wp.y for wp in rrt_waypoints]
        rrt_path_line.set_data(rrt_x, rrt_y)
        rrt_scatter.set_offsets(np.array([[wp.x, wp.y] for wp in rrt_waypoints]))
        if current_rrt_index < len(rrt_waypoints):
            ct = rrt_waypoints[current_rrt_index]
            current_target_scatter.set_offsets([[ct.x, ct.y]])
        else:
            current_target_scatter.set_offsets(np.empty((0, 2)))
    else:
        rrt_path_line.set_data([], [])
        rrt_scatter.set_offsets(np.empty((0, 2)))
        current_target_scatter.set_offsets(np.empty((0, 2)))

    dx = q_target[0] - q[0]
    dy = q_target[1] - q[1]
    dist = np.sqrt(dx*dx + dy*dy) + 1e-12
    u = (dx / dist) * 4.0
    v = (dy / dist) * 4.0
    guide_arrow.set_offsets(np.array([[q[0], q[1]]]))
    guide_arrow.set_UVC(np.array([[u]]), np.array([[v]]))

    true_scatter.set_offsets(obstacles_true)
    noisy_scatter.set_offsets(obstacles_noisy)
    for e in ellipse_patches:
        e.remove()
    ellipse_patches.clear()
    for i, obs in enumerate(obstacles_true):
        vx, vy = obstacle_speeds[i]
        vmag = np.sqrt(vx**2 + vy**2)
        theta = np.degrees(np.arctan2(vy, vx))
        a = a_base[i] + alpha * vmag
        b = b_base[i] + beta * vmag
        a_max = 7
        b_max = 5
        ell = Ellipse(xy=(obs[0], obs[1]), width=2*a, height=2*b, angle=theta,
                      edgecolor='black', facecolor='cyan', alpha=0.15, linestyle='--', linewidth=1.2, zorder=1)
        ax.add_patch(ell)
        ellipse_patches.append(ell)

    return path_line, robot_dot, true_scatter, noisy_scatter, goal_dot, rrt_path_line, rrt_scatter, current_target_scatter, guide_arrow

ani = animation.FuncAnimation(fig, update, frames=400, init_func=init, interval=40, blit=False)
plt.show()
