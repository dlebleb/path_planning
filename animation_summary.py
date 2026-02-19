"""
Animation run summary: one CSV plus trajectory image.

Call from main1.py or AStar_animation.py at the end of a run. Writes a single CSV with:
  - At top: clear statistics (animation type, total frames, time, success, num_collisions, num_replans, path_length, image path)
  - Collisions: time_s, frame, type, what was triggered
  - Replans: time_s, frame, short reason (e.g. segment blocked, waypoint in obstacle)
  - Obstacle log: time_s, frame, then all ellipse/rect positions and velocities

Plus: trajectory image (start, goal, labeled waypoints). Professional, no emojis.
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Optional, Tuple

_DT = 0.01
_animation_type = ""
_start = None
_goal = None
_waypoints = []  # list of (x, y)
_waypoint_labels = []  # e.g. ["Station 1", "Station 2", ..., "Goal"]
_path_frames = []
_path_points = []  # list of (x, y)
_collisions = []  # list of (time_s, frame, type, detail)
_replans = []  # list of (time_s, frame, reason)
_obstacle_log = []  # list of (time_s, frame, ellipse_pos, ellipse_vel, rect_list)
_success = False
_ended = False
_obstacle_log_interval = 10


def init_summary(animation_type: str, dt: float = 0.01, obstacle_log_interval: int = 10):
    """Call at start of animation. animation_type must be 'main1' or 'AStar_animation'."""
    global _DT, _animation_type, _path_frames, _path_points, _collisions, _replans
    global _obstacle_log, _success, _ended, _obstacle_log_interval
    _DT = dt
    _animation_type = animation_type
    _obstacle_log_interval = obstacle_log_interval
    _path_frames = []
    _path_points = []
    _collisions = []
    _replans = []
    _obstacle_log = []
    _success = False
    _ended = False


def set_route(
    start: np.ndarray,
    goal: np.ndarray,
    waypoints: List[Tuple[float, float]],
    waypoint_labels: Optional[List[str]] = None,
):
    """Set start, goal, and waypoints (with optional labels for the figure)."""
    global _start, _goal, _waypoints, _waypoint_labels
    _start = np.asarray(start)
    _goal = np.asarray(goal)
    _waypoints = list(waypoints)
    if waypoint_labels is not None:
        _waypoint_labels = list(waypoint_labels)
    else:
        _waypoint_labels = [f"Waypoint {i+1}" for i in range(len(_waypoints))]


def log_path_point(frame: int, x: float, y: float):
    """Log one robot position (call each frame or when position updates)."""
    _path_frames.append(frame)
    _path_points.append((float(x), float(y)))


def log_collision(frame: int, collision_type: str, detail: str = ""):
    """Log a collision event. collision_type e.g. 'ellipse', 'rectangle'. detail e.g. obstacle indices."""
    t = frame * _DT
    _collisions.append((t, frame, collision_type, detail))


def log_replan(frame: int, reason: str):
    """Log a replan event. reason e.g. 'Segment blocked', 'Waypoint in obstacle', 'Periodic'."""
    t = frame * _DT
    _replans.append((t, frame, reason))


def log_obstacles(
    frame: int,
    ellipse_positions: np.ndarray,
    ellipse_velocities: np.ndarray,
    rect_obstacles: List,
):
    """Log obstacle positions and velocities at this frame (with timestamp). Sampled every obstacle_log_interval frames."""
    t = frame * _DT
    _obstacle_log.append((t, frame, np.asarray(ellipse_positions).copy(), np.asarray(ellipse_velocities).copy(), list(rect_obstacles)))


def end_summary(success: bool = True):
    """Call when animation stops (goal reached or window closed)."""
    global _success, _ended
    _success = success
    _ended = True


def write_summary(output_dir: str = ".") -> Optional[str]:
    """
    Write CSV files and trajectory image. Call after animation ends (e.g. after plt.show()).
    Returns the prefix of written files, or None if nothing was logged.
    """
    global _path_points, _path_frames
    if not _path_points and not _collisions and not _replans:
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_type = _animation_type.replace(" ", "_")
    prefix = os.path.join(output_dir, f"animation_run_{ts}_{safe_type}")

    path_points = list(_path_points)
    if _start is not None and path_points and (path_points[0][0] != _start[0] or path_points[0][1] != _start[1]):
        path_points.insert(0, (float(_start[0]), float(_start[1])))

    total_frames = max(_path_frames) if _path_frames else 0
    total_time_s = total_frames * _DT
    path_length = 0.0
    if len(path_points) >= 2:
        pts = np.array(path_points)
        path_length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    # ---- Single CSV: summary at top, then collisions, replans, obstacle log ----
    csv_path = prefix + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Section 1: Summary statistics at the top (clear key-value rows)
        w.writerow(["# Animation run summary"])
        w.writerow(["animation_type", _animation_type])
        w.writerow(["total_frames", total_frames])
        w.writerow(["total_time_s", f"{total_time_s:.4f}"])
        w.writerow(["success", _success])
        w.writerow(["num_collisions", len(_collisions)])
        w.writerow(["num_replans", len(_replans)])
        w.writerow(["path_length", f"{path_length:.4f}"])
        w.writerow(["trajectory_image", os.path.basename(prefix) + "_trajectory.png"])
        w.writerow([])
        # Section 2: Collisions (timestamps + what was triggered)
        w.writerow(["# Collisions (time_s, frame, type, detail)"])
        w.writerow(["time_s", "frame", "type", "detail"])
        for t, fr, ctype, detail in _collisions:
            w.writerow([f"{t:.4f}", fr, ctype, detail])
        w.writerow([])
        # Section 3: Replans (timestamps + short reason)
        w.writerow(["# Replans (time_s, frame, reason)"])
        w.writerow(["time_s", "frame", "reason"])
        for t, fr, reason in _replans:
            w.writerow([f"{t:.4f}", fr, reason])
        w.writerow([])
        # Section 4: Obstacle positions and velocities with timestamps
        w.writerow(["# Obstacle log (positions and velocities per timestamp)"])
        if _obstacle_log:
            t0, f0, epos, evel, rects = _obstacle_log[0]
            n_ell = len(epos)
            n_rect = len(rects)
            header = ["time_s", "frame"]
            for i in range(n_ell):
                header.extend([f"e{i}_x", f"e{i}_y", f"e{i}_vx", f"e{i}_vy"])
            for i in range(n_rect):
                header.extend([f"r{i}_x", f"r{i}_y", f"r{i}_w", f"r{i}_h"])
            w.writerow(header)
            for t, fr, epos, evel, rects in _obstacle_log:
                row = [f"{t:.4f}", fr]
                for i in range(len(epos)):
                    row.extend([f"{epos[i][0]:.4f}", f"{epos[i][1]:.4f}", f"{evel[i][0]:.4f}", f"{evel[i][1]:.4f}"])
                for r in rects:
                    row.extend([f"{r[0]:.4f}", f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.4f}"])
                w.writerow(row)

    # ---- Trajectory image ----
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_aspect("equal")
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    if path_points:
        path = np.array(path_points)
        ax.plot(path[:, 0], path[:, 1], "k-", linewidth=1.5, label="Trajectory")
    if _start is not None:
        ax.plot(_start[0], _start[1], "s", color="red", markersize=10, markeredgecolor="black", markeredgewidth=1, label="Start")
        ax.text(_start[0], _start[1] - 2.5, "Start", fontsize=10, ha="center")
    if _goal is not None:
        ax.plot(_goal[0], _goal[1], "s", color="green", markersize=10, markeredgecolor="black", markeredgewidth=1, label="Goal")
        ax.text(_goal[0], _goal[1] + 2.5, "Goal", fontsize=10, ha="center")
    for i, (wx, wy) in enumerate(_waypoints):
        label = _waypoint_labels[i] if i < len(_waypoint_labels) else f"W{i+1}"
        if label == "Goal":
            continue  # already drawn as green square above
        ax.plot(wx, wy, "o", color="steelblue", markersize=6, markeredgecolor="black", markeredgewidth=0.5)
        ax.text(wx, wy + 1.5, label, fontsize=9, ha="center")
    ax.set_title(f"Trajectory: {_animation_type}")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(prefix + "_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return prefix