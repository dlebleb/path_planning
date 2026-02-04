"""
A* Path Planning Animation (no potential field)

Same setup as RRT animation: start, goal, obstacles, speed, moving ellipses.
Robot follows the A* path directly (move toward next waypoint at constant speed).
No attraction/repulsion forces — pure A* path following.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'simulation'))
from astar_planner import astar

# ===============================
# INITIAL SETUP (same as RRT animation)
# ===============================
q_goal = np.array([5, 0])
q = np.array([-40.0, -40.0])
v_robot = 10

obstacles_true = np.array([[-18.0,-10.0], [18,-20], [18, 8], [22,26], [23,15], [-23,15], [5,5]])
sigma = 0.1
obstacles_noisy = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)

obstacle_speeds = np.array([[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1]])
obstacle_speeds = obstacle_speeds * 80

dt = 0.01
path_data = [q.copy()]

a0, b0 = 2.0, 1.0
alpha, beta = 0.2, 0.1
sizes = np.array([1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1])
a_base = a0 + sizes
b_base = b0 + sizes

MAX_OBSTACLE_SPEED = 12.0
MIN_OBSTACLE_SPEED = 6.0
MAX_ELLIPSE_AXIS = 8.0
BOUNDS = (-50, 50, -50, 50)
ASTAR_STEP = 0.5
SAFETY_MARGIN = 0.5

def get_obstacle_axes(i, vmag):
    v = min(vmag, MAX_OBSTACLE_SPEED)
    a = min(a_base[i] + alpha * v, MAX_ELLIPSE_AXIS)
    b = min(b_base[i] + beta * v, MAX_ELLIPSE_AXIS)
    return a, b

def build_obstacles_for_astar(obstacles_true, obstacle_speeds) -> List[Tuple[float, float, float]]:
    out = []
    for i, obs_pos in enumerate(obstacles_true):
        vx, vy = obstacle_speeds[i] if i < len(obstacle_speeds) else [0, 0]
        vmag = np.sqrt(vx**2 + vy**2)
        a, b = get_obstacle_axes(i, vmag)
        r = max(a, b) + SAFETY_MARGIN
        out.append((float(obs_pos[0]), float(obs_pos[1]), r))
    return out

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
        if vmag > MAX_OBSTACLE_SPEED:
            f = MAX_OBSTACLE_SPEED / (vmag + 1e-12)
            vx, vy = vx * f, vy * f
            vmag = np.sqrt(vx**2 + vy**2)
        if vmag < MIN_OBSTACLE_SPEED:
            if vmag < 1e-6:
                theta = np.random.uniform(0, 2 * np.pi)
                vx = MIN_OBSTACLE_SPEED * np.cos(theta)
                vy = MIN_OBSTACLE_SPEED * np.sin(theta)
            else:
                f = MIN_OBSTACLE_SPEED / vmag
                vx, vy = vx * f, vy * f
        new_speeds[i] = [vx, vy]
    return new_speeds

astar_waypoints: Optional[List[Tuple[float, float]]] = None
current_waypoint_index = 0
replan_interval = 50
last_replan = 0
waypoint_threshold = 0.5
tolerance = 1.5

# ===============================
# ANIMATION SETUP
# ===============================

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_title("A* Path Planning Animation")
ax.legend(loc='upper right')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

path_line, = ax.plot([], [], 'r-', linewidth=2, label="Robot Path", zorder=10)
robot_dot, = ax.plot([], [], 'ro', markersize=6, zorder=11)
true_scatter = ax.scatter([], [], c='black', s=40, label="True Obstacles")
noisy_scatter = ax.scatter([], [], c='red', s=40, label="Noisy Obstacles")
goal_dot, = ax.plot(q_goal[0], q_goal[1], 'go', markersize=8, label="Goal")
planned_path_line, = ax.plot([], [], 'b--', linewidth=1.5, alpha=0.7, label="A* Path")
planned_scatter = ax.scatter([], [], c='blue', s=30, marker='o', alpha=0.5, label="A* Waypoints")
current_target_scatter = ax.scatter([], [], c='yellow', s=100, marker='*', label="Current Target")
guide_arrow = ax.quiver([0], [0], [0], [0], color='gold', scale_units='xy', scale=1,
                        width=0.06, headwidth=4, headlength=3, alpha=0.9, label="Direction to target")
ellipse_patches = []

def init():
    path_line.set_data([], [])
    robot_dot.set_data([], [])
    true_scatter.set_offsets(np.empty((0, 2)))
    noisy_scatter.set_offsets(np.empty((0, 2)))
    planned_path_line.set_data([], [])
    planned_scatter.set_offsets(np.empty((0, 2)))
    current_target_scatter.set_offsets(np.empty((0, 2)))
    guide_arrow.set_offsets(np.array([[0, 0]]))
    guide_arrow.set_UVC(np.array([[0]]), np.array([[0]]))
    return path_line, robot_dot, true_scatter, noisy_scatter, goal_dot, planned_path_line, planned_scatter, current_target_scatter, guide_arrow

def update(frame):
    global q, obstacle_speeds, ani, astar_waypoints, current_waypoint_index, last_replan

    obstacle_speeds = apply_stochastic_maneuver(obstacle_speeds)
    obstacles_true[:,0] += obstacle_speeds[:,0] * dt
    obstacles_true[:,1] += obstacle_speeds[:,1] * dt
    obstacles_noisy[:] = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)

    obstacles_circ = build_obstacles_for_astar(obstacles_true, obstacle_speeds)
    if (frame - last_replan) >= replan_interval or astar_waypoints is None:
        start = (float(q[0]), float(q[1]))
        goal = (float(q_goal[0]), float(q_goal[1]))
        path = astar(start, goal, obstacles_circ, bounds=BOUNDS, step=ASTAR_STEP)
        if path:
            astar_waypoints = path
            current_waypoint_index = 0
        last_replan = frame

    if astar_waypoints and current_waypoint_index < len(astar_waypoints):
        wx, wy = astar_waypoints[current_waypoint_index]
        q_target = np.array([wx, wy])
    else:
        q_target = q_goal.copy()

    # Pure A*: move directly toward current target (no potential field)
    to_target = q_target - q
    dist = np.linalg.norm(to_target) + 1e-12
    step_size = v_robot * dt
    if dist <= step_size:
        q[:] = q_target.copy()
    else:
        direction = to_target / dist
        q[:] = q + direction * step_size
    path_data.append(q.copy())

    # Advance to next waypoint if we reached current target
    if astar_waypoints and current_waypoint_index < len(astar_waypoints):
        if np.linalg.norm(q - q_target) < waypoint_threshold:
            current_waypoint_index += 1
            if current_waypoint_index >= len(astar_waypoints):
                current_waypoint_index = len(astar_waypoints) - 1

    arr = np.array(path_data)
    path_line.set_data(arr[:,0], arr[:,1])
    robot_dot.set_data([q[0]], [q[1]])

    if np.linalg.norm(q - q_goal) < tolerance:
        print(f"[GOAL] Reached goal at frame {frame}")
        ani.event_source.stop()
        return path_line, robot_dot, true_scatter, noisy_scatter, goal_dot, planned_path_line, planned_scatter, current_target_scatter, guide_arrow

    if astar_waypoints:
        px = [p[0] for p in astar_waypoints]
        py = [p[1] for p in astar_waypoints]
        planned_path_line.set_data(px, py)
        planned_scatter.set_offsets(np.array(astar_waypoints))
        if current_waypoint_index < len(astar_waypoints):
            cx, cy = astar_waypoints[current_waypoint_index]
            current_target_scatter.set_offsets([[cx, cy]])
        else:
            current_target_scatter.set_offsets(np.empty((0, 2)))
    else:
        planned_path_line.set_data([], [])
        planned_scatter.set_offsets(np.empty((0, 2)))
        current_target_scatter.set_offsets(np.empty((0, 2)))

    dx = q_target[0] - q[0]
    dy = q_target[1] - q[1]
    d = np.sqrt(dx*dx + dy*dy) + 1e-12
    u, v = (dx / d) * 4.0, (dy / d) * 4.0
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
        a, b = get_obstacle_axes(i, vmag)
        ell = Ellipse(xy=(obs[0], obs[1]), width=2*a, height=2*b, angle=theta,
                      edgecolor='black', facecolor='cyan', alpha=0.15, linestyle='--', linewidth=1.2, zorder=1)
        ax.add_patch(ell)
        ellipse_patches.append(ell)

    return path_line, robot_dot, true_scatter, noisy_scatter, goal_dot, planned_path_line, planned_scatter, current_target_scatter, guide_arrow

ani = animation.FuncAnimation(fig, update, frames=400, init_func=init, interval=40, blit=False)
plt.show()
