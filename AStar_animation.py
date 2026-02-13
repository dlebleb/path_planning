"""
A* Path Planning Animation — same obstacle animation setup as main1.py.

- A* algorithm: plans collision-free path from current position to current target (one of the stops or goal).
- Obstacles (from main1.py): 13 elliptical obstacles (positions, speeds, sizes), 6 moving rectangles with
  expanded buffers recomputed each frame. Ellipse dimensions and drawing match main1/APF1.
- Robot visits the same 5 stations then goal.
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.animation as animation
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "simulation"))
from astar_planner import astar

# =============================================================================
# COPIED FROM TSP10 / MAIN SCRIPT — for animation (same world, same stops)
# =============================================================================
q_start = np.array([-40.0, -40.0], dtype=float)
q_goal  = np.array([10.0,  12.0], dtype=float)
q = q_start.copy()
v_robot = 24.0

# From main1.py: 13 elliptical obstacles (same positions, speeds, sizes as main1)
obstacles_true = np.array([
    [-18.0, -10.0], [18, -20], [18, 8], [22, 26], [23, 15], [-23, 15], [5, 5],
    [-40, -30], [15, -10], [10, 5], [-30, 0], [-20, -20], [0, 0]
], dtype=float)
sigma = 0.1
obstacles_noisy = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)
obstacle_speeds = np.array([
    [-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1],
    [0.2, 0.1], [-0.1, 0.1], [-0.2, 0.1], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2]
], dtype=float) * 80.0
vmag_max, vmag_min = 24.5, 5.0

dt = 0.01
path_data = [q.copy()]

# From main1/APF1: ellipse dimensions (a_base = a0 * sizes, b_base = b0 * sizes)
a0, b0 = 2.0, 1.0
alpha, beta = 0.2, 0.1
sizes = np.array([1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1, 1.0, 1.2, 1.8, 2.0, 1.8, 1.9])
a_base = a0 * sizes
b_base = b0 * sizes
a_max, b_max = 7.0, 5.0

# From main1/APF1 init_environment(): rectangular obstacles (x_min, y_min, width, height); mutable for animation
RECT_OBSTACLES_INIT = [
    [-40, -30, 12, 3],
    [20, -35, 6, 3],
    [-45, 15, 10, 4],
    [5, 25, 3, 2],
    [-6, -6.5, 12, 3],
    [-13.5, -23.0, 12, 6],
]
rect_obstacles = [list(r) for r in RECT_OBSTACLES_INIT]
# Rectangle speeds (main1: same scale 80); expanded_rects recomputed each frame in update()
rectangle_speeds = np.array([
    [-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1]
], dtype=float) * 80.0
expanded_rects = []  # filled each frame by compute_expanded_rects(rect_obstacles, buffer=1.0)

# From TSP10: stations to stop at (post office, hospital, grocery store, home, cafe) — same exact stops
STATIONS = np.array([
    [-30, -35],
    [25, -40],
    [-48, 20],
    [7, 30],
    [15, -2],
], dtype=float)
# Ordered list of targets: station 0, ..., station 4, then goal (same sequence of stops as TSP10)
STOPS_LIST = [tuple(STATIONS[i]) for i in range(len(STATIONS))] + [(q_goal[0], q_goal[1])]

# =============================================================================
# A* AND ANIMATION PARAMETERS
# =============================================================================
BOUNDS = (-50, 50, -50, 50)
ASTAR_STEP = 0.5
SAFETY_MARGIN = 0.5
REPLAN_INTERVAL = 80
waypoint_threshold = 0.5
goal_tolerance = 2.5

# A* algorithm state: current path to current stop, and which stop we're heading to
astar_waypoints: Optional[List[Tuple[float, float]]] = None
current_waypoint_index = 0
current_stop_index = 0   # 0..len(STOPS_LIST)-1; last stop is goal

# ===============================
# HELPERS (ellipse axes and obstacle list for A* — match main1/APF1)
# ===============================
def compute_expanded_rects(rect_obstacles, buffer=1.0):
    """Recompute expanded rectangles from current rect positions (main1/APF1)."""
    rect_obstacles = np.array(rect_obstacles)
    expanded = np.zeros_like(rect_obstacles)
    expanded[:, 0] = rect_obstacles[:, 0] - buffer
    expanded[:, 1] = rect_obstacles[:, 1] - buffer
    expanded[:, 2] = rect_obstacles[:, 2] + 2 * buffer
    expanded[:, 3] = rect_obstacles[:, 3] + 2 * buffer
    return expanded

def get_obstacle_axes(i: int, vmag: float) -> Tuple[float, float]:
    """From main1/APF1: ellipse axis lengths from speed (a_base + alpha*vmag, clamped to a_max, b_max)."""
    vmag = float(np.clip(vmag, vmag_min, vmag_max))
    a = min(a_base[i] + alpha * vmag, a_max)
    b = min(b_base[i] + beta * vmag, b_max)
    return float(a), float(b)

def build_obstacles_for_astar(
    obstacles_pos: np.ndarray,
    obstacle_speeds_: np.ndarray,
    expanded_rects_,
) -> List[Tuple[float, float, float]]:
    """
    A* algorithm: build list of (x, y, radius) for collision checking.
    Ellipses → circles; rectangles → circumcircles of expanded rects (current frame).
    """
    out: List[Tuple[float, float, float]] = []
    for i, obs_pos in enumerate(obstacles_pos):
        vx, vy = obstacle_speeds_[i] if i < len(obstacle_speeds_) else (0.0, 0.0)
        vmag = float(np.hypot(vx, vy))
        a, b = get_obstacle_axes(i, vmag)
        r = max(a, b) + SAFETY_MARGIN
        out.append((float(obs_pos[0]), float(obs_pos[1]), float(r)))
    for x, y, w, h in expanded_rects_:
        cx = x + w / 2
        cy = y + h / 2
        r = math.sqrt((w / 2) ** 2 + (h / 2) ** 2) + SAFETY_MARGIN
        out.append((cx, cy, r))
    return out

def apply_stochastic_maneuver(
    obstacle_speeds_: np.ndarray,
    maneuver_prob: float = 0.25,
    magnitude_sigma: float = 0.05,
    turn_sigma: float = 0.02,
) -> np.ndarray:
    """
    Randomly perturbs velocity direction/magnitude (main1/APF1 style).
    Enforces vmag_min and vmag_max (24.5, 5).
    """
    new_speeds = np.array(obstacle_speeds_, dtype=float)
    for i in range(len(new_speeds)):
        vx, vy = new_speeds[i]
        vmag = float(np.hypot(vx, vy))
        scale = np.clip(np.random.normal(1.0, magnitude_sigma), 0.85, 1.15)
        vmag *= scale
        vmag = np.clip(vmag, vmag_min, vmag_max)
        theta = float(np.arctan2(vy, vx) + np.random.normal(0.0, turn_sigma))
        if np.random.rand() < maneuver_prob:
            theta += float(np.random.uniform(-np.pi / 2, np.pi / 2))
        vx = vmag * np.cos(theta)
        vy = vmag * np.sin(theta)
        new_speeds[i] = [vx, vy]
    return new_speeds

def closest_waypoint_index(qpos: np.ndarray, waypoints: List[Tuple[float, float]]) -> int:
    """Return index of waypoint closest to current robot position."""
    if not waypoints:
        return 0
    d = [float(np.hypot(w[0] - qpos[0], w[1] - qpos[1])) for w in waypoints]
    return int(np.argmin(d))

def path_is_blocked(remaining_waypoints: List[Tuple[float, float]], obstacles_circ: List[Tuple[float, float, float]]) -> bool:
    """
    Conservative "is the remaining path blocked?" check:
    If any waypoint lies inside any obstacle circle => consider path blocked and trigger replanning.
    """
    if not remaining_waypoints:
        return True
    for (px, py) in remaining_waypoints:
        p = np.array([px, py], dtype=float)
        for (ox, oy, r) in obstacles_circ:
            if np.hypot(p[0] - ox, p[1] - oy) < r:
                return True
    return False

def plan_astar_from(
    current_pos: np.ndarray,
    target: Tuple[float, float],
    obstacles_circ: List[Tuple[float, float, float]],
) -> Optional[List[Tuple[float, float]]]:
    """A* algorithm: single-shot path from current_pos to target."""
    start = (float(current_pos[0]), float(current_pos[1]))
    return astar(start, target, obstacles_circ, bounds=BOUNDS, step=ASTAR_STEP)

# ===============================
# ANIMATION SETUP — from TSP10: same figure style, Start/Goal, stations as purple stars
# ===============================
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_aspect("equal")
ax.set_title("A* Path Planning Animation")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.grid(True)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Start = red, Goal = green
path_line, = ax.plot([], [], "r-", linewidth=3, label="Robot Path", zorder=10)
robot_dot, = ax.plot([], [], "ko", markersize=8, zorder=11, label="Robot")
start_dot, = ax.plot(q_start[0], q_start[1], "ro", markersize=8, label="Start")
goal_dot,  = ax.plot(q_goal[0],  q_goal[1],  "go", markersize=8, label="Goal")

# From TSP10: stations (stops) as purple stars — same as "Uğranacak noktalar"
stations_scatter = ax.scatter(STATIONS[:, 0], STATIONS[:, 1], color="purple", marker="*", s=120, label="Stations (stops)", zorder=9)

true_scatter = ax.scatter([], [], c="black", s=40, marker="x", label="True obstacles")
noisy_scatter = ax.scatter([], [], c="red", s=40, marker="o", alpha=0.5, label="Noisy obstacles")

# A* path to current stop
planned_path_line, = ax.plot([], [], "b--", linewidth=1.5, alpha=0.7, label="A* Path")
planned_scatter = ax.scatter([], [], c="blue", s=30, marker="o", alpha=0.5, label="A* Waypoints")

ellipse_patches: List[Ellipse] = []
rect_patches: List[Rectangle] = []
expanded_rect_patches: List[Rectangle] = []   # main1: buffered rects (orange dashed)

ax.legend(loc="upper right")

def init():
    global q, astar_waypoints, current_waypoint_index, current_stop_index, path_data, rect_obstacles, expanded_rects

    q[:] = q_start.copy()
    path_data = [q.copy()]
    current_stop_index = 0
    for i, r0 in enumerate(RECT_OBSTACLES_INIT):
        rect_obstacles[i][:] = r0
    expanded_rects = compute_expanded_rects(rect_obstacles, buffer=1.0)

    path_line.set_data([], [])
    robot_dot.set_data([q[0]], [q[1]])
    true_scatter.set_offsets(np.empty((0, 2)))
    noisy_scatter.set_offsets(np.empty((0, 2)))
    planned_path_line.set_data([], [])
    planned_scatter.set_offsets(np.empty((0, 2)))

    obstacles_circ = build_obstacles_for_astar(obstacles_true, obstacle_speeds, expanded_rects)
    target = STOPS_LIST[current_stop_index]
    path = plan_astar_from(q, target, obstacles_circ)
    if path:
        astar_waypoints = path
        current_waypoint_index = closest_waypoint_index(q, path)
        px = [p[0] for p in astar_waypoints]
        py = [p[1] for p in astar_waypoints]
        planned_path_line.set_data(px, py)
        planned_scatter.set_offsets(np.array(astar_waypoints))

    return path_line, robot_dot, start_dot, goal_dot, true_scatter, noisy_scatter, planned_path_line, planned_scatter


def update(frame):
    global q, obstacle_speeds, rectangle_speeds, rect_obstacles, expanded_rects, astar_waypoints, current_waypoint_index, current_stop_index, path_data

    # main1/APF1: stochastic maneuver and move ellipses
    obstacle_speeds = apply_stochastic_maneuver(obstacle_speeds)
    obstacles_true[:, 0] += obstacle_speeds[:, 0] * dt
    obstacles_true[:, 1] += obstacle_speeds[:, 1] * dt
    obstacles_noisy[:] = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)

    # main1/APF1: move rectangles and recompute expanded rects each frame
    rectangle_speeds = apply_stochastic_maneuver(rectangle_speeds)
    for i in range(len(rect_obstacles)):
        rect_obstacles[i][0] += rectangle_speeds[i][0] * dt
        rect_obstacles[i][1] += rectangle_speeds[i][1] * dt
    expanded_rects = compute_expanded_rects(rect_obstacles, buffer=1.0)

    obstacles_circ = build_obstacles_for_astar(obstacles_true, obstacle_speeds, expanded_rects)
    current_target_tuple = STOPS_LIST[current_stop_index]

    # A* algorithm: replan when path blocked, no path, or periodically
    remaining = astar_waypoints[current_waypoint_index:] if astar_waypoints and current_waypoint_index < len(astar_waypoints) else []
    should_replan = (
        not astar_waypoints
        or path_is_blocked(remaining, obstacles_circ)
        or (frame > 0 and frame % REPLAN_INTERVAL == 0)
    )
    if should_replan:
        path = plan_astar_from(q, current_target_tuple, obstacles_circ)
        if path:
            astar_waypoints = path
            current_waypoint_index = closest_waypoint_index(q, path)
            if frame > 0:
                print(f"[Replan] frame {frame} — path to stop {current_stop_index + 1}/{len(STOPS_LIST)}, {len(path)} waypoints")

    # Current target (next A* waypoint along path to current stop)
    if astar_waypoints and current_waypoint_index < len(astar_waypoints):
        wx, wy = astar_waypoints[current_waypoint_index]
        q_target = np.array([wx, wy], dtype=float)
    else:
        q_target = np.array([current_target_tuple[0], current_target_tuple[1]], dtype=float)

    # Move robot toward target
    to_target = q_target - q
    dist = float(np.linalg.norm(to_target)) + 1e-12
    step_size = v_robot * dt
    if dist <= step_size:
        q[:] = q_target
    else:
        q[:] = q + (to_target / dist) * step_size
    path_data.append(q.copy())

    # Advance along A* path waypoint when reached
    if astar_waypoints and current_waypoint_index < len(astar_waypoints):
        if np.linalg.norm(q - q_target) < waypoint_threshold:
            current_waypoint_index += 1

    # Reached current stop? (same stations as TSP10) — advance to next stop and replan
    if np.linalg.norm(q - np.array([current_target_tuple[0], current_target_tuple[1]])) < waypoint_threshold:
        if current_stop_index < len(STOPS_LIST) - 1:
            current_stop_index += 1
            path = plan_astar_from(q, STOPS_LIST[current_stop_index], obstacles_circ)
            if path:
                astar_waypoints = path
                current_waypoint_index = closest_waypoint_index(q, path)
                print(f"[Stop reached] Advancing to stop {current_stop_index + 1}/{len(STOPS_LIST)}")
        # else: current_stop_index is already goal; goal_tolerance check below will stop animation

    # Update plot
    arr = np.array(path_data)
    path_line.set_data(arr[:, 0], arr[:, 1])
    robot_dot.set_data([q[0]], [q[1]])
    true_scatter.set_offsets(obstacles_true)
    noisy_scatter.set_offsets(obstacles_noisy)

    if astar_waypoints:
        px = [p[0] for p in astar_waypoints]
        py = [p[1] for p in astar_waypoints]
        planned_path_line.set_data(px, py)
        planned_scatter.set_offsets(np.array(astar_waypoints))
    else:
        planned_path_line.set_data([], [])
        planned_scatter.set_offsets(np.empty((0, 2)))

    # Ellipses (same style as TSP10)
    for e in ellipse_patches:
        e.remove()
    ellipse_patches.clear()
    for i, obs in enumerate(obstacles_true):
        vx, vy = obstacle_speeds[i]
        vmag = float(np.hypot(vx, vy))
        theta_deg = np.degrees(np.arctan2(vy, vx))
        a, b = get_obstacle_axes(i, vmag)
        ell = Ellipse(
            (obs[0], obs[1]),
            width=2 * a,
            height=2 * b,
            angle=theta_deg,
            edgecolor="black",
            facecolor="cyan",
            alpha=0.15,
            linestyle="--",
            linewidth=1.2,
            zorder=1,
        )
        ax.add_patch(ell)
        ellipse_patches.append(ell)

    # main1: rectangular obstacles (black) + expanded/buffered rects (orange dashed) — no circumcircles
    for r in rect_patches:
        r.remove()
    rect_patches.clear()
    for r in expanded_rect_patches:
        r.remove()
    expanded_rect_patches.clear()
    for x, y, w, h in rect_obstacles:
        rect = Rectangle((x, y), w, h, fill=False, linewidth=2, edgecolor="black", zorder=1)
        ax.add_patch(rect)
        rect_patches.append(rect)
    for x, y, w, h in expanded_rects:
        rect2 = Rectangle((x, y), w, h, fill=False, linewidth=1.5, edgecolor="orange", linestyle="--", zorder=1)
        ax.add_patch(rect2)
        expanded_rect_patches.append(rect2)

    # Goal reached: last stop is goal; within margin (same idea as TSP10)
    if current_stop_index == len(STOPS_LIST) - 1 and np.linalg.norm(q - q_goal) < goal_tolerance:
        ax.set_title("A* Path Planning — Goal Reached!", fontsize=14)
        print(f"[Goal Reached] Robot within {goal_tolerance} units of goal at frame {frame}")
        ani.event_source.stop()

    return path_line, robot_dot, start_dot, goal_dot, true_scatter, noisy_scatter, planned_path_line, planned_scatter


ani = animation.FuncAnimation(fig, update, frames=2000, init_func=init, interval=40, blit=False)
plt.show()
