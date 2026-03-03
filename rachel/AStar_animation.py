"""
A* Path Planning Animation.

Runs the animation (figure, update loop, drawing). Algorithm and environment logic
live in Astar.py — same separation as main1.py (animation) and APF1.py (logic).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.animation as animation
from typing import List, Optional

from Astar import (
    init_environment,
    compute_expanded_rects,
    get_obstacle_axes,
    build_obstacles_for_astar,
    apply_stochastic_maneuver,
    is_collision_check,
    is_collision_free_rect,
    segment_to_target_blocked,
    closest_waypoint_index,
    is_waypoint_in_obstacle,
    plan_astar_from,
    dt,
    RECT_OBSTACLES_INIT,
    STATIONS,
    OBSTACLES_TRUE_INIT,
    OBSTACLE_SPEEDS_INIT,
    RECTANGLE_SPEEDS_INIT,
    sigma,
)
from animation_summary import (
    init_summary,
    set_route,
    log_path_point,
    log_collision,
    log_replan,
    log_obstacles,
    end_summary,
    write_summary,
)

# =============================================================================
# ANIMATION-ONLY SETUP (start, goal, robot speed, thresholds)
# =============================================================================
q_start = np.array([-40.0, -40.0], dtype=float)
q_goal = np.array([10.0, 12.0], dtype=float)
q = q_start.copy()
v_robot = 60.0

REPLAN_INTERVAL = 80
waypoint_threshold = 0.5
goal_tolerance = 2.5

# World state (mutated each frame) — from Astar
rect_obstacles, _ = init_environment()
obstacles_true = np.array(OBSTACLES_TRUE_INIT, dtype=float)
obstacles_noisy = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)
obstacle_speeds = np.array(OBSTACLE_SPEEDS_INIT, dtype=float)
rectangle_speeds = np.array(RECTANGLE_SPEEDS_INIT, dtype=float)
expanded_rects = compute_expanded_rects(rect_obstacles, buffer=1.0)

path_data = [q.copy()]
STOPS_LIST = [tuple(STATIONS[i]) for i in range(len(STATIONS))] + [(q_goal[0], q_goal[1])]
WAYPOINT_LABELS = [f"Station {i+1}" for i in range(len(STATIONS))] + ["Goal"]

# Summary logging (run report CSV + trajectory image)
init_summary("AStar_animation", dt=dt, obstacle_log_interval=10)
set_route(q_start, q_goal, STOPS_LIST, WAYPOINT_LABELS)

# A* state
astar_waypoints: Optional[List[tuple]] = None
current_waypoint_index = 0
current_stop_index = 0
summary_written = False

# =============================================================================
# FIGURE AND ARTISTS
# =============================================================================
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

path_line, = ax.plot([], [], "r-", linewidth=3, label="Robot Path", zorder=10)
robot_dot, = ax.plot([], [], "ko", markersize=8, zorder=11, label="Robot")
start_dot, = ax.plot(q_start[0], q_start[1], "ro", markersize=8, label="Start")
goal_dot, = ax.plot(q_goal[0], q_goal[1], "go", markersize=8, label="Goal")
stations_scatter = ax.scatter(STATIONS[:, 0], STATIONS[:, 1], color="purple", marker="*", s=120, label="Stations (stops)", zorder=9)
true_scatter = ax.scatter([], [], c="black", s=40, marker="x", label="True obstacles")
noisy_scatter = ax.scatter([], [], c="red", s=40, marker="o", alpha=0.5, label="Noisy obstacles")
planned_path_line, = ax.plot([], [], "b--", linewidth=1.5, alpha=0.7, label="A* Path")
planned_scatter = ax.scatter([], [], c="blue", s=30, marker="o", alpha=0.5, label="A* Waypoints")

ellipse_patches: List[Ellipse] = []
rect_patches: List[Rectangle] = []
expanded_rect_patches: List[Rectangle] = []
ax.legend(loc="upper right")


def init():
    global q, astar_waypoints, current_waypoint_index, current_stop_index, path_data, rect_obstacles, expanded_rects

    q[:] = q_start.copy()
    path_data[:] = [q.copy()]
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

    log_path_point(0, q[0], q[1])

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
    global q, obstacle_speeds, rectangle_speeds, rect_obstacles, expanded_rects
    global astar_waypoints, current_waypoint_index, current_stop_index, path_data, summary_written

    obstacle_speeds = apply_stochastic_maneuver(obstacle_speeds)
    obstacles_true[:, 0] += obstacle_speeds[:, 0] * dt
    obstacles_true[:, 1] += obstacle_speeds[:, 1] * dt
    obstacles_noisy[:] = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)

    rectangle_speeds = apply_stochastic_maneuver(rectangle_speeds)
    for i in range(len(rect_obstacles)):
        rect_obstacles[i][0] += rectangle_speeds[i][0] * dt
        rect_obstacles[i][1] += rectangle_speeds[i][1] * dt
    expanded_rects = compute_expanded_rects(rect_obstacles, buffer=1.0)

    obstacles_circ = build_obstacles_for_astar(obstacles_true, obstacle_speeds, expanded_rects)
    current_target_tuple = STOPS_LIST[current_stop_index]

    remaining = astar_waypoints[current_waypoint_index:] if astar_waypoints and current_waypoint_index < len(astar_waypoints) else []
    should_replan = (
        not astar_waypoints
        or is_waypoint_in_obstacle(remaining, obstacles_true, obstacle_speeds, rect_obstacles)
        or (frame > 0 and frame % REPLAN_INTERVAL == 0)
    )
    if should_replan:
        if not astar_waypoints:
            replan_reason = "No path (segment blocked or collision)"
        elif is_waypoint_in_obstacle(remaining, obstacles_true, obstacle_speeds, rect_obstacles):
            replan_reason = "Waypoint in obstacle"
        else:
            replan_reason = "Periodic (80 frames)"
        path = plan_astar_from(q, current_target_tuple, obstacles_circ)
        if path:
            astar_waypoints = path
            current_waypoint_index = closest_waypoint_index(q, path)
            log_replan(frame, replan_reason)
            if frame > 0:
                print(f"[Replan] frame {frame} — path to stop {current_stop_index + 1}/{len(STOPS_LIST)}, {len(path)} waypoints")

    if astar_waypoints and current_waypoint_index < len(astar_waypoints):
        wx, wy = astar_waypoints[current_waypoint_index]
        q_target = np.array([wx, wy], dtype=float)
    else:
        q_target = np.array([current_target_tuple[0], current_target_tuple[1]], dtype=float)

    if segment_to_target_blocked(q, q_target, obstacles_true, obstacle_speeds, rect_obstacles):
        astar_waypoints = None
        current_waypoint_index = 0
        if frame > 0:
            log_replan(frame, "Segment blocked")
            print("[Wait] Segment to next waypoint blocked — replanning next frame")
    else:
        to_target = q_target - q
        dist = float(np.linalg.norm(to_target)) + 1e-12
        step_size = v_robot * dt
        if dist <= step_size:
            q[:] = q_target
        else:
            q[:] = q + (to_target / dist) * step_size
        path_data.append(q.copy())
        log_path_point(frame, q[0], q[1])

    if astar_waypoints and current_waypoint_index < len(astar_waypoints):
        if np.linalg.norm(q - q_target) < waypoint_threshold:
            current_waypoint_index += 1

    if np.linalg.norm(q - np.array([current_target_tuple[0], current_target_tuple[1]])) < waypoint_threshold:
        if current_stop_index < len(STOPS_LIST) - 1:
            current_stop_index += 1
            path = plan_astar_from(q, STOPS_LIST[current_stop_index], obstacles_circ)
            if path:
                astar_waypoints = path
                current_waypoint_index = closest_waypoint_index(q, path)
                print(f"[Stop reached] Advancing to stop {current_stop_index + 1}/{len(STOPS_LIST)}")

    ell_count, ell_hits = is_collision_check(q, obstacles_true, obstacle_speeds)
    in_rect = not is_collision_free_rect(q, rect_obstacles)
    if ell_count > 0:
        log_collision(frame, "ellipse", f"obstacle indices {ell_hits}")
        print(f"Collision detected with ellipse obstacle(s): {ell_hits}")
    if in_rect:
        log_collision(frame, "rectangle", "")
        print("Collision detected with rectangular obstacle")
    if ell_count > 0 or in_rect:
        log_replan(frame, "Collision; clear path")
        astar_waypoints = None
        current_waypoint_index = 0

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

    for e in ellipse_patches:
        e.remove()
    ellipse_patches.clear()
    for i, obs in enumerate(obstacles_true):
        vx, vy = obstacle_speeds[i]
        vmag = float(np.hypot(vx, vy))
        theta_deg = np.degrees(np.arctan2(vy, vx))
        a, b = get_obstacle_axes(i, vmag)
        ell = Ellipse(
            (obs[0], obs[1]), width=2 * a, height=2 * b, angle=theta_deg,
            edgecolor="black", facecolor="cyan", alpha=0.15, linestyle="--", linewidth=1.2, zorder=1,
        )
        ax.add_patch(ell)
        ellipse_patches.append(ell)

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

    if current_stop_index == len(STOPS_LIST) - 1 and np.linalg.norm(q - q_goal) < goal_tolerance:
        ax.set_title("A* Path Planning — Goal Reached!", fontsize=14)
        print(f"[Goal Reached] Robot within {goal_tolerance} units of goal at frame {frame}")
        end_summary(success=True)
        if write_summary() is not None:
            summary_written = True
        ani.event_source.stop()

    if frame % 10 == 0:
        log_obstacles(frame, obstacles_true, obstacle_speeds, rect_obstacles)

    return path_line, robot_dot, start_dot, goal_dot, true_scatter, noisy_scatter, planned_path_line, planned_scatter


ani = animation.FuncAnimation(fig, update, frames=2000, init_func=init, interval=40, blit=False)
try:
    plt.show()
finally:
    if not summary_written:
        end_summary(success=False)
        write_summary()