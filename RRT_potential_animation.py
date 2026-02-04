"""
RRT + Potential Field Path Planning Animation (elliptical obstacles)

Run to see animated path planning using:
- RRT for global collision-free path planning
- Potential fields for local navigation with dynamic obstacles
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
from typing import List, Optional

# -----------------------------
# Logging (stall + collision)
# -----------------------------
def log_stall(frame: int, q: np.ndarray, q_goal: np.ndarray, q_target: np.ndarray,
              clearance: float, F_att: np.ndarray, F_rep: np.ndarray, F: np.ndarray,
              current_rrt_index: int, n_waypoints: int, in_goal_region: bool):
    """Print one short stall line (throttled by STALL_LOG_INTERVAL)."""
    d_goal = np.linalg.norm(q - q_goal)
    print(f"[STALL] frame={frame} pos=({q[0]:.1f},{q[1]:.1f}) dist_goal={d_goal:.1f} wp={current_rrt_index}/{n_waypoints}")

def log_collision(frame: int, obstacle_indices: List[int], q: np.ndarray):
    """Print collision event (throttled)."""
    print(f"[COLLISION] frame={frame} robot=({q[0]:.2f}, {q[1]:.2f}) obstacle_indices={obstacle_indices}")

def log_progress(frame: int, q: np.ndarray, q_goal: np.ndarray, current_rrt_index: int, n_waypoints: int):
    """Optional periodic progress line."""
    d = np.linalg.norm(q - q_goal)
    print(f"[PROGRESS] frame={frame} dist_to_goal={d:.2f} waypoint={current_rrt_index}/{n_waypoints}")

# Import RRT planner from simulation folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'simulation'))
from TSP import Point, Obstacle
from rrt_planner import RRTPlanner

# ===============================
# INITIAL SETUP
# ===============================
q_goal = np.array([5, 0])   # goal: same x, lower y
q = np.array([-40.0, -40.0])
v_robot = 10

# obstacle coordinates 
obstacles_true = np.array([[-18.0,-10.0], [18,-20], [18, 8], [22,26], [23,15], [-23,15], [5,5]])
sigma = 0.1  # 10 cm uncertainity
obstacles_noisy = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)

# obstacle speeds
obstacle_speeds = np.array([[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1]])
obstacle_speeds = obstacle_speeds * 80

# APF parameters
#k_att, k_rep, d0, dt = 10.0, 500.0, 3.0, 0.001
k_att, k_rep, d0, dt = 4.0, 10.0, 10.0, 0.01
max_rep_force = np.inf
path_data = [q.copy()]
initial_obstacles = obstacles_true.copy()

# elliptical obstacles
a0 = 2.0 # major axis (along velocity direction)
b0 = 1.0 # minor axis (perpendicular to velocity direction)
alpha = 0.2   # major scaling (large)
beta  = 0.1   # minor scaling (small)

# each obstacle has a size factor: 1 = normal, >1 = large, <1 = small
sizes = np.array([1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1])
# incorporate static size
a_base = a0 + sizes
b_base = b0 + sizes

# Caps so obstacles don't go crazy between runs (optional: set seed for same run every time)
# np.random.seed(42)
MAX_OBSTACLE_SPEED = 12.0       # clamp obstacle velocity magnitude so they don't explode
MIN_OBSTACLE_SPEED = 6.0        # ensure every obstacle keeps moving so they eventually get out of the way (higher = more motion)
MAX_ELLIPSE_AXIS = 8.0          # clamp ellipse a,b so obstacles never become huge

def get_obstacle_axes(i, vmag):
    """Return (a, b) for obstacle i with speed vmag, capped so ellipses stay reasonable."""
    v = min(vmag, MAX_OBSTACLE_SPEED)
    a = min(a_base[i] + alpha * v, MAX_ELLIPSE_AXIS)
    b = min(b_base[i] + beta * v, MAX_ELLIPSE_AXIS)
    return a, b

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

        # dynamic scaling with speed (capped so obstacles don't grow huge)
        a, b = get_obstacle_axes(i, vmag)

        # move obstacle center to origin and transform
        obs_x, obs_y = obs[0], obs[1]
        # x2, y2 = -obs_x/a, -obs_y/b

        eps = 1e-12
        # move robot center to origin and transform
        q_x, q_y = (q[0]-obs_x)/(a+eps), (q[1]-obs_y)/(b+eps)

        # distance of the robot to origin -1 (negative = inside ellipse)
        dE = np.sqrt(q_x**2 + q_y**2) - 1

        # dynamic avoidance boundary
        # d0_i = d0 * max(a, b)

        if dE < d0:
            # Outward direction from obstacle to robot (always use for repulsion)
            diff = q - obs
            dist_raw = np.linalg.norm(diff) + 1e-12
            outward = diff / dist_raw

            if dE < 0:
                # Robot inside obstacle: force strong outward push (otherwise formula
                # gives inward force and robot gets stuck)
                F_mag = k_rep * 2.0  # strong constant escape force
                F_rep = F_mag * outward
            else:
                F_mag = k_rep * (1/dE - 1/d0) * (1/dE**2)
                # Repulsion must point outward; when dE > 0, (q-obs) already does
                F_rep = F_mag * outward
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
       
        a, b = get_obstacle_axes(i, vmag)

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
        
        # 1) Small continuous jitter (small speed increase & decrease, 5%)
        vmag *= np.random.normal(1, magnitude_sigma)
        
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

def min_clearance(q, obstacles_noisy, obstacle_speeds):
    """Minimum signed distance to any obstacle (ellipse boundary). Negative = inside."""
    min_dE = 1e9
    for i, obs in enumerate(obstacles_noisy):
        vx, vy = obstacle_speeds[i]
        vmag = np.sqrt(vx**2 + vy**2)
        a, b = get_obstacle_axes(i, vmag)
        obs_x, obs_y = obs[0], obs[1]
        eps = 1e-12
        q_x = (q[0] - obs_x) / (a + eps)
        q_y = (q[1] - obs_y) / (b + eps)
        dE = np.sqrt(q_x**2 + q_y**2) - 1
        min_dE = min(min_dE, dE)
    return min_dE

def is_collision_check(q, obstacles_noisy, obstacle_speeds):
    collided_indices = []
    counter = 0

    for i, obs in enumerate(obstacles_noisy):
        vx, vy = obstacle_speeds[i]
        vmag = np.sqrt(vx**2 + vy**2) 
        theta = np.arctan2(vy, vx + 1e-12)
               
        a, b = get_obstacle_axes(i, vmag)

        # move obstacle center to origin and transform
        obs_x, obs_y = obs[0], obs[1]

        eps = 1e-12
        q_x, q_y = (q[0]-obs_x)/(a+eps), (q[1]-obs_y)/(b+eps)

        dE = np.sqrt(q_x**2 + q_y**2) - 1

        # COLLISION condition: robot is inside the ellipse
        if dE < 0:
            counter += 1
            collided_indices.append(i)
    
    return counter, collided_indices

# ===============================
# RRT INTEGRATION
# ===============================

def convert_obstacles_for_rrt(obstacles_true, obstacle_speeds, safety_margin: float = 0.5):
    """
    Convert dynamic elliptical obstacles to circular obstacles for RRT collision checking.
    """
    obstacles_rrt = []
    for i, obs_pos in enumerate(obstacles_true):
        vx, vy = obstacle_speeds[i] if i < len(obstacle_speeds) else [0, 0]
        vmag = np.sqrt(vx**2 + vy**2)
        a, b = get_obstacle_axes(i, vmag)

        # Use maximum axis as radius (conservative approach)
        radius = max(a, b) + safety_margin
        
        obstacles_rrt.append(Obstacle(Point(obs_pos[0], obs_pos[1]), radius))
    
    return obstacles_rrt

# Initialize RRT system
rrt_planner = RRTPlanner(step_size=1.0, max_iterations=1000)
rrt_waypoints: Optional[List[Point]] = None
current_rrt_index = 0
rrt_replan_interval = 50  # Replan every 50 frames
last_rrt_replan = 0
waypoint_threshold = 0.5

# Multi-run RRT: run RRT several times when replanning and pick the best path
RRT_REPLAN_TRIALS = 5   # number of paths to try before picking one

def path_min_clearance(path: List[Point], obstacles_rrt: List) -> float:
    """Minimum distance from any waypoint to any obstacle boundary (clearance). Higher = safer path."""
    if not path or not obstacles_rrt:
        return 0.0
    min_clear = float("inf")
    for wp in path:
        for obs in obstacles_rrt:
            d = wp.distance_to(obs.position) - obs.radius
            min_clear = min(min_clear, d)
    return min_clear if min_clear != float("inf") else 0.0

def path_total_length(path: List[Point]) -> float:
    """Total length of path (sum of segment lengths). Shorter can mean faster."""
    if not path or len(path) < 2:
        return 0.0
    return sum(path[i].distance_to(path[i + 1]) for i in range(len(path) - 1))

# Stall / collision logging (intervals set high to keep terminal quiet)
STALL_WINDOW = 20          # frames to look back for stall
STALL_THRESHOLD = 0.35     # max movement over STALL_WINDOW to count as stalled
STALL_LOG_INTERVAL = 120   # log stall at most every N frames when stalled
COLLISION_LOG_INTERVAL = 100
PROGRESS_LOG_INTERVAL = 150  # log progress every N frames
last_stall_log_frame = -999
last_collision_log_frame = -999
was_in_collision_prev = False

# When robot is within this distance of the goal, we target the goal directly
# and reduce repulsive force so an obstacle near the goal doesn't block arrival.
GOAL_REGION_RADIUS = 5.0
REPULSION_SCALE_NEAR_GOAL = 0.15  # scale down repulsion so robot can reach goal

# ===============================
# ANIMATION SETUP
# ===============================

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_title("RRT + Potential Field Path Planning Animation")
ax.legend(loc='upper right')

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

# normalize quiver vectors
mag = np.sqrt(U**2 + V**2) + 1e-12
U_norm = U / mag
V_norm = V / mag

# add quiver field
quiver_bg = ax.quiver(X, Y, U_norm, V_norm, color='white', alpha=0.6)

path_line, = ax.plot([], [], 'r-', linewidth=2, label="Robot Path", zorder=10)
robot_dot, = ax.plot([], [], 'ro', markersize=6, zorder=11)

true_scatter = ax.scatter([], [], c='black', s=40, label="True Obstacles")
noisy_scatter = ax.scatter([], [], c='red', s=40, label="Noisy Obstacles")

goal_dot, = ax.plot(q_goal[0], q_goal[1], 'go', markersize=8, label="Goal")

# RRT visualization elements
rrt_path_line, = ax.plot([], [], 'b--', linewidth=1.5, alpha=0.7, label="RRT Path")
rrt_scatter = ax.scatter([], [], c='blue', s=30, marker='o', alpha=0.5, label="RRT Waypoints")
current_target_scatter = ax.scatter([], [], c='yellow', s=100, marker='*', label="Current Target")

# Yellow arrow: intended direction (toward current target). The red dot follows the
# *total* force (attraction + repulsion), so it can diverge from this arrow when
# repulsive forces from obstacles push it aside.
guide_arrow = ax.quiver([0], [0], [0], [0], color='gold', scale_units='xy', scale=1,
                        width=0.06, headwidth=4, headlength=3, alpha=0.9,
                        label="Guide (intended direction)")

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

# Consider "goal reached" when within this distance (slightly larger so obstacle
# near goal doesn't prevent success)
tolerance = 1.5

# ===============================
# UPDATE FUNCTION
# ===============================

def update(frame):
    global q
    global obstacle_speeds
    global ani
    global rrt_waypoints, current_rrt_index, last_rrt_replan
    global last_stall_log_frame, last_collision_log_frame, was_in_collision_prev

    # 0) random maneuver (new speeds), then clamp so obstacles always move but don't go crazy
    obstacle_speeds = apply_stochastic_maneuver(obstacle_speeds)
    for i in range(len(obstacle_speeds)):
        vx, vy = obstacle_speeds[i]
        vmag = np.sqrt(vx**2 + vy**2)
        if vmag > MAX_OBSTACLE_SPEED:
            f = MAX_OBSTACLE_SPEED / (vmag + 1e-12)
            vx, vy = vx * f, vy * f
            vmag = np.sqrt(vx**2 + vy**2)
        if vmag < MIN_OBSTACLE_SPEED:
            # Keep obstacles moving so they eventually get out of the way (no stuck-at-waypoint)
            if vmag < 1e-6:
                theta = np.random.uniform(0, 2 * np.pi)
                vx = MIN_OBSTACLE_SPEED * np.cos(theta)
                vy = MIN_OBSTACLE_SPEED * np.sin(theta)
            else:
                f = MIN_OBSTACLE_SPEED / vmag
                vx, vy = vx * f, vy * f
        obstacle_speeds[i] = [vx, vy]

    # --- 1) Move obstacles ---
    obstacles_true[:,0] += obstacle_speeds[:,0] * dt
    obstacles_true[:,1] += obstacle_speeds[:,1] * dt

    # --- 2) Noise sample ---
    obstacles_noisy[:] = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)

    # --- 3) RRT Planning ---
    obstacles_rrt = convert_obstacles_for_rrt(obstacles_true, obstacle_speeds)
    
    # Replan RRT if needed (keep previous path if RRT fails so path doesn't disappear).
    # When stalled we force replan next frame so RRT can try a different route (e.g. via open area).
    # Why RRT often picks the blocked route instead of "go down into open area":
    # - Goal bias (10%): RRT samples the goal 10% of the time, so the tree grows toward the goal.
    #   From (-15,-15) toward (7,0) is right/up, which goes toward the obstacle; the open area is the other way.
    # - First path wins: RRT returns the first collision-free path to the goal, not the nicest. So it may
    #   return a path that skirts the obstacle before it finds one that goes around via the open area.
    if (frame - last_rrt_replan) >= rrt_replan_interval or rrt_waypoints is None:
        start_point = Point(q[0], q[1])
        goal_point = Point(q_goal[0], q_goal[1])
        candidates = []
        for _ in range(RRT_REPLAN_TRIALS):
            wp = rrt_planner.plan_path(start_point, goal_point, obstacles_rrt,
                                       bounds=(-50, 50, -50, 50))
            if wp:
                candidates.append(wp)
        if candidates:
            # Pick path with largest minimum clearance (prefer routes that stay away from obstacles)
            best = max(candidates, key=lambda p: path_min_clearance(p, obstacles_rrt))
            rrt_waypoints = best
            current_rrt_index = 0
        last_rrt_replan = frame
    
    # Get current RRT target (or goal when already in goal region)
    dist_to_goal = np.linalg.norm(q - q_goal)
    in_goal_region = dist_to_goal < GOAL_REGION_RADIUS

    if in_goal_region:
        # Near goal: target the goal directly so we can actually reach it
        q_target = q_goal.copy()
    elif rrt_waypoints and current_rrt_index < len(rrt_waypoints):
        target_point = rrt_waypoints[current_rrt_index]
        q_target = np.array([target_point.x, target_point.y])
        # Check if reached current waypoint
        if np.linalg.norm(q - q_target) < waypoint_threshold:
            current_rrt_index += 1
            if current_rrt_index >= len(rrt_waypoints):
                current_rrt_index = len(rrt_waypoints) - 1
    else:
        q_target = q_goal.copy()

    # --- 4) Robot force: attraction to target + repulsion from obstacles ---
    F_att = attractive_force(q, q_target)
    F_rep = repulsive_force(q, obstacles_noisy, obstacle_speeds)
    if in_goal_region:
        # Reduce repulsion so an obstacle near the goal doesn't block arrival
        F = F_att + REPULSION_SCALE_NEAR_GOAL * F_rep
    else:
        F = F_att + F_rep

    # --- 5) Robot motion ---
    F_norm = np.linalg.norm(F)
    direction = F/F_norm if F_norm > 1e-6 else np.array([0, 0])
    step = direction * v_robot * dt
    # Avoid overshooting into an obstacle: take smaller steps when very close
    clearance = min_clearance(q, obstacles_noisy, obstacle_speeds)
    if 0 < clearance < 1.0:
        step_scale = max(0.2, min(1.0, clearance))
        step = step * step_scale
    q[:] = q + step
    path_data.append(q.copy())

    # --- 6) Update path (always, so path never disappears under other elements) ---
    arr = np.array(path_data)
    path_line.set_data(arr[:,0], arr[:,1])
    robot_dot.set_data([q[0]], [q[1]])

    # --- STOP CONDITION: close enough to goal (after updating path so last segment shows) ---
    if np.linalg.norm(q - q_goal) < tolerance:
        print(f"[GOAL] Reached goal at frame {frame}")
        ani.event_source.stop()
        return path_line, robot_dot, true_scatter, noisy_scatter, goal_dot, rrt_path_line, rrt_scatter, current_target_scatter, guide_arrow

    # --- 6b) Stall detection and logging; trigger replan so next path can use open area ---
    if len(path_data) >= STALL_WINDOW:
        movement = np.linalg.norm(np.array(path_data[-1]) - np.array(path_data[-STALL_WINDOW]))
        if movement < STALL_THRESHOLD:
            if (frame - last_stall_log_frame) >= STALL_LOG_INTERVAL:
                n_wp = len(rrt_waypoints) if rrt_waypoints else 0
                clearance_now = min_clearance(q, obstacles_noisy, obstacle_speeds)
                log_stall(frame, q, q_goal, q_target, clearance_now, F_att, F_rep, F,
                          current_rrt_index, n_wp, in_goal_region)
                last_stall_log_frame = frame
                # Force a replan next frame so RRT can try a different route (e.g. down into open area)
                last_rrt_replan = frame - rrt_replan_interval

    # --- 6c) Collision logging (throttled; first time and then every N frames) ---
    count, hits = is_collision_check(q, obstacles_noisy, obstacle_speeds)
    if count > 0:
        if not was_in_collision_prev or (frame - last_collision_log_frame) >= COLLISION_LOG_INTERVAL:
            log_collision(frame, hits, q)
            last_collision_log_frame = frame
        was_in_collision_prev = True
    else:
        was_in_collision_prev = False

    # --- 6d) Periodic progress (every PROGRESS_LOG_INTERVAL frames) ---
    if frame > 0 and frame % PROGRESS_LOG_INTERVAL == 0:
        n_wp = len(rrt_waypoints) if rrt_waypoints else 0
        log_progress(frame, q, q_goal, current_rrt_index, n_wp)

    # --- 7) Update RRT visualization ---
    if rrt_waypoints:
        rrt_x = [wp.x for wp in rrt_waypoints]
        rrt_y = [wp.y for wp in rrt_waypoints]
        rrt_path_line.set_data(rrt_x, rrt_y)
        rrt_scatter.set_offsets(np.array([[wp.x, wp.y] for wp in rrt_waypoints]))
        
        # Highlight current target
        if current_rrt_index < len(rrt_waypoints):
            current_target = rrt_waypoints[current_rrt_index]
            current_target_scatter.set_offsets([[current_target.x, current_target.y]])
        else:
            current_target_scatter.set_offsets(np.empty((0, 2)))
    else:
        rrt_path_line.set_data([], [])
        rrt_scatter.set_offsets(np.empty((0, 2)))
        current_target_scatter.set_offsets(np.empty((0, 2)))

    # --- 7b) Yellow guide arrow: direction to current target (red dot follows total force, so can diverge) ---
    dx = q_target[0] - q[0]
    dy = q_target[1] - q[1]
    dist = np.sqrt(dx*dx + dy*dy) + 1e-12
    arrow_len = 4.0  # fixed length for visibility
    u = (dx / dist) * arrow_len
    v = (dy / dist) * arrow_len
    guide_arrow.set_offsets(np.array([[q[0], q[1]]]))
    guide_arrow.set_UVC(np.array([[u]]), np.array([[v]]))

    # --- 8) Obstacle scatter ---
    true_scatter.set_offsets(obstacles_true)
    noisy_scatter.set_offsets(obstacles_noisy)

    # --- 9) Remove old ellipses ---
    for e in ellipse_patches:
        e.remove()
    ellipse_patches.clear()

    # --- 10) Draw ellipses ---
    for i, obs in enumerate(obstacles_true):
        vx, vy = obstacle_speeds[i]
        vmag = np.sqrt(vx**2 + vy**2)
        theta = np.degrees(np.arctan2(vy, vx))
        a, b = get_obstacle_axes(i, vmag)

        ellipse = Ellipse(
            xy=(obs[0], obs[1]),
            width=2*a,
            height=2*b,
            angle=theta,
            edgecolor='black',
            facecolor='cyan',
            alpha=0.15,
            linestyle='--',
            linewidth=1.2,
            zorder=1
        )
        ax.add_patch(ellipse)
        ellipse_patches.append(ellipse)

    return path_line, robot_dot, true_scatter, noisy_scatter, goal_dot, rrt_path_line, rrt_scatter, current_target_scatter, guide_arrow


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