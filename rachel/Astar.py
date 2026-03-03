"""
A* path planning algorithm and environment logic.

- Environment: elliptical and rectangular obstacles, waypoints (stations), same setup as main1/APF1.
- Collision: point-in-ellipse, point-in-rect, segment-vs-ellipse, segment-vs-rect.
- Planning: build circular obstacles for A* planner, plan path from current position to target.

Used by AStar_animation.py for the animation; this module contains no plotting or animation code.
"""

import os
import sys
import math
import numpy as np
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "simulation"))
from astar_planner import astar

# =============================================================================
# ENVIRONMENT CONSTANTS (same world as main1/APF1)
# =============================================================================
BOUNDS = (-50, 50, -50, 50)
ASTAR_STEP = 0.5
SAFETY_MARGIN = 0.5
dt = 0.01

# Ellipse parameters (main1/APF1)
a0, b0 = 2.0, 1.0
alpha, beta = 0.2, 0.1
sizes = np.array([1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1, 1.0, 1.2, 1.8, 2.0, 1.8, 1.9])
a_base = a0 * sizes
b_base = b0 * sizes
a_max, b_max = 7.0, 5.0
vmag_max, vmag_min = 24.5, 5.0

# Initial obstacle positions and speeds (13 ellipses, 6 rects)
OBSTACLES_TRUE_INIT = np.array([
    [-18.0, -10.0], [18, -20], [18, 8], [22, 26], [23, 15], [-23, 15], [5, 5],
    [-40, -30], [15, -10], [10, 5], [-30, 0], [-20, -20], [0, 0]
], dtype=float)
OBSTACLE_SPEEDS_INIT = np.array([
    [-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1],
    [0.2, 0.1], [-0.1, 0.1], [-0.2, 0.1], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2]
], dtype=float) * 80.0

RECT_OBSTACLES_INIT = [
    [-40, -30, 12, 3],
    [20, -35, 6, 3],
    [-45, 15, 10, 4],
    [5, 25, 3, 2],
    [-6, -6.5, 12, 3],
    [-13.5, -23.0, 12, 6],
]
RECTANGLE_SPEEDS_INIT = np.array([
    [-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1]
], dtype=float) * 80.0

STATIONS = np.array([
    [-30, -35],
    [25, -40],
    [-48, 20],
    [7, 30],
    [15, -2],
], dtype=float)

sigma = 0.1


# =============================================================================
# ENVIRONMENT INIT
# =============================================================================
def init_environment():
    """
    Returns mutable rect obstacles and waypoints (stations).
    rect_obstacles is a list of lists so the animation can update positions each frame.
    """
    rect_obstacles = [list(r) for r in RECT_OBSTACLES_INIT]
    waypoints = STATIONS.copy()
    return rect_obstacles, waypoints


def compute_expanded_rects(rect_obstacles, buffer=1.0):
    """Recompute expanded rectangles from current rect positions (main1/APF1)."""
    rect_obstacles = np.array(rect_obstacles)
    expanded = np.zeros_like(rect_obstacles)
    expanded[:, 0] = rect_obstacles[:, 0] - buffer
    expanded[:, 1] = rect_obstacles[:, 1] - buffer
    expanded[:, 2] = rect_obstacles[:, 2] + 2 * buffer
    expanded[:, 3] = rect_obstacles[:, 3] + 2 * buffer
    return expanded


# =============================================================================
# ELLIPSE AXES AND A* OBSTACLE LIST
# =============================================================================
def get_obstacle_axes(i: int, vmag: float) -> Tuple[float, float]:
    """Ellipse axis lengths from speed (a_base + alpha*vmag, clamped to a_max, b_max)."""
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
    Build list of (x, y, radius) for the A* planner.
    Ellipses → circles; rectangles → circumcircles of expanded rects.
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
    """Randomly perturbs velocity direction/magnitude (main1/APF1 style). Enforces vmag_min, vmag_max."""
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


# =============================================================================
# COLLISION: POINT
# =============================================================================
def is_collision_check(q, obstacles_noisy, obstacle_speeds):
    """
    Check if point q is inside any elliptical obstacle (same geometry as main1/APF1).
    Returns (count of collisions, list of collided obstacle indices).
    """
    collided_indices = []
    counter = 0
    eps = 1e-12
    for i, obs in enumerate(obstacles_noisy):
        vx, vy = obstacle_speeds[i] if i < len(obstacle_speeds) else (0.0, 0.0)
        vmag = np.sqrt(vx**2 + vy**2)
        a = a_base[i] + alpha * vmag
        b = b_base[i] + beta * vmag
        a = min(a, a_max)
        b = min(b, b_max)
        obs_x, obs_y = obs[0], obs[1]
        q_x = (q[0] - obs_x) / (a + eps)
        q_y = (q[1] - obs_y) / (b + eps)
        dE = np.sqrt(q_x**2 + q_y**2) - 1
        if dE < 0:
            counter += 1
            collided_indices.append(i)
    return counter, collided_indices


def is_collision_free_rect(point, rect_obstacles: List) -> bool:
    """True if point is outside all rectangular obstacles. point: (x,y) or length-2 array."""
    X = point[0] if hasattr(point, "__getitem__") else point.x
    Y = point[1] if hasattr(point, "__getitem__") else point.y
    for rect in rect_obstacles:
        x, y, w, h = rect[0], rect[1], rect[2], rect[3]
        if (x <= X <= x + w) and (y <= Y <= y + h):
            return False
    return True


# =============================================================================
# COLLISION: SEGMENT (for wait/replan instead of driving through)
# =============================================================================
def _segment_intersects_ellipse(
    pA: np.ndarray, pB: np.ndarray, obs_center: np.ndarray, a: float, b: float
) -> bool:
    """True if segment pA->pB intersects the axis-aligned ellipse centered at obs_center with semi-axes a, b."""
    eps = 1e-12
    cx, cy = obs_center[0], obs_center[1]
    ax = (pA[0] - cx) / (a + eps)
    ay = (pA[1] - cy) / (b + eps)
    bx = (pB[0] - cx) / (a + eps)
    by = (pB[1] - cy) / (b + eps)
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom < 1e-20:
        return ax * ax + ay * ay <= 1.0
    t = np.clip(-(ax * dx + ay * dy) / denom, 0.0, 1.0)
    px = ax + t * dx
    py = ay + t * dy
    return (px * px + py * py) <= 1.0


def _segment_intersects_rect(
    pA: np.ndarray, pB: np.ndarray, x: float, y: float, w: float, h: float
) -> bool:
    """True if segment pA->pB intersects the rectangle [x, y, w, h]."""
    if not is_collision_free_rect(pA, [[x, y, w, h]]):
        return True
    if not is_collision_free_rect(pB, [[x, y, w, h]]):
        return True
    def seg_intersect(a1, a2, b1, b2):
        oa = a2[0] - a1[0], a2[1] - a1[1]
        ob = b2[0] - b1[0], b2[1] - b1[1]
        d = oa[0] * ob[1] - oa[1] * ob[0]
        if abs(d) < 1e-12:
            return False
        t = ((b1[0] - a1[0]) * ob[1] - (b1[1] - a1[1]) * ob[0]) / d
        u = ((b1[0] - a1[0]) * oa[1] - (b1[1] - a1[1]) * oa[0]) / d
        return 0 <= t <= 1 and 0 <= u <= 1
    edges = [
        ((x, y), (x + w, y)),
        ((x + w, y), (x + w, y + h)),
        ((x + w, y + h), (x, y + h)),
        ((x, y + h), (x, y)),
    ]
    for e1, e2 in edges:
        if seg_intersect(pA, pB, np.array(e1), np.array(e2)):
            return True
    return False


def segment_to_target_blocked(
    q_from: np.ndarray,
    q_to: np.ndarray,
    obstacles_true_: np.ndarray,
    obstacle_speeds_: np.ndarray,
    rect_obstacles_: List,
) -> bool:
    """True if the segment from q_from to q_to is blocked by any ellipse or rectangle."""
    for i, obs in enumerate(obstacles_true_):
        vx, vy = obstacle_speeds_[i] if i < len(obstacle_speeds_) else (0.0, 0.0)
        vmag = float(np.hypot(vx, vy))
        a = min(a_base[i] + alpha * vmag, a_max)
        b = min(b_base[i] + beta * vmag, b_max)
        if _segment_intersects_ellipse(q_from, q_to, obs, a, b):
            return True
    for rect in rect_obstacles_:
        x, y, w, h = rect[0], rect[1], rect[2], rect[3]
        if _segment_intersects_rect(q_from, q_to, x, y, w, h):
            return True
    return False


# =============================================================================
# PATH BLOCKED AND PLANNING
# =============================================================================
def closest_waypoint_index(qpos: np.ndarray, waypoints: List[Tuple[float, float]]) -> int:
    """Index of waypoint closest to qpos."""
    if not waypoints:
        return 0
    d = [float(np.hypot(w[0] - qpos[0], w[1] - qpos[1])) for w in waypoints]
    return int(np.argmin(d))


def is_waypoint_in_obstacle(
    remaining_waypoints: List[Tuple[float, float]],
    obstacles_true_: np.ndarray,
    obstacle_speeds_: np.ndarray,
    rect_obstacles_: List,
) -> bool:
    """True if any remaining waypoint lies inside an ellipse or rectangle (path blocked)."""
    if not remaining_waypoints:
        return True
    for (px, py) in remaining_waypoints:
        p = np.array([px, py], dtype=float)
        count, _ = is_collision_check(p, obstacles_true_, obstacle_speeds_)
        if count > 0:
            return True
        if not is_collision_free_rect(p, rect_obstacles_):
            return True
    return False


def plan_astar_from(
    current_pos: np.ndarray,
    target: Tuple[float, float],
    obstacles_circ: List[Tuple[float, float, float]],
) -> Optional[List[Tuple[float, float]]]:
    """Compute A* path from current_pos to target. Returns list of (x,y) waypoints or None."""
    start = (float(current_pos[0]), float(current_pos[1]))
    return astar(start, target, obstacles_circ, bounds=BOUNDS, step=ASTAR_STEP)