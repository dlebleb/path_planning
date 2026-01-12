"""
Local Potential-Field Path Planning (elliptical obstacles) + RRT integration

Run to plot a local path from start to goal around elliptical obstacles 
using an attractive + repulsive potential field, with RRT providing
global waypoints.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from typing import List, Optional, Tuple
from global_planner_simple import Point, Obstacle, GlobalPlanner
from rrt_planner import RRTPlanner
import time





os.makedirs("figures", exist_ok=True)

q_goal = np.array([10, 10])   # goal position
q = np.array([-20.0, -20.0])  # starting position of the robot

# ---------------------------------------------------------------------------
# Human-like dynamic obstacles
# Each obstacle is one "agent": walker, walker+box, bike, scooter, etc.
# ---------------------------------------------------------------------------

# Fixed initial positions (you can keep these)
obstacles_true = np.array([
    [-5, -5],   # 0
    [-3, -3],   # 1
    [ 3,  3.5], # 2
    [ 6,  7],   # 3
    [ 9,  9],   # 4
    [ 8,  4],   # 5
    [ 5,  5],   # 6
])

sigma = 0.1  # 10 cm observation noise
obstacles_noisy = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)

# Obstacle "types" – one label per obstacle
# 0: normal walker
# 1: walker with box (bigger, slower)
# 2: bike
# 3: scooter
# 4: fast walker
# 5: normal walker
# 6: walker with box
obstacle_types = [
    "walker",
    "walker_with_box",
    "bike",
    "scooter",
    "fast_walker",
    "walker",
    "walker_with_box",
]

# Parameters per type:
# - base semi-axes (a_base, b_base) in meters
# - speed range (v_min, v_max) in m/s
TYPE_PARAMS = {
    "walker": {
        "a": 0.40, "b": 0.30,        # roughly human footprint
        "v_min": 0.8, "v_max": 1.4,  # walking speed
    },
    "walker_with_box": {
        "a": 0.60, "b": 0.40,        # bulkier footprint
        "v_min": 0.4, "v_max": 1.0,  # slower
    },
    "bike": {
        "a": 0.90, "b": 0.30,        # long and narrow
        "v_min": 2.0, "v_max": 4.0,  # faster
    },
    "scooter": {
        "a": 0.75, "b": 0.25,        # slightly shorter than bike
        "v_min": 1.5, "v_max": 3.0,
    },
    "fast_walker": {
        "a": 0.40, "b": 0.30,
        "v_min": 1.4, "v_max": 2.0,
    },
}

# Mild dynamic scaling based on speed (not huge like before)
alpha = 0.3   # how much a (major axis) grows with speed
beta  = 0.1   # how much b (minor axis) grows with speed

def init_obstacle_params(positions, types):
    """
    Initialize base ellipse sizes (a_base, b_base) and initial velocities
    for each obstacle, based on its "agent" type.
    """
    N = len(positions)
    a_base = np.zeros(N)
    b_base = np.zeros(N)
    speeds = np.zeros((N, 2))

    for i, t in enumerate(types):
        params = TYPE_PARAMS[t]
        a_base[i] = params["a"]
        b_base[i] = params["b"]

        # Sample a speed in the given range
        vmag = np.random.uniform(params["v_min"], params["v_max"])

        # Random initial direction
        theta = np.random.uniform(-np.pi, np.pi)
        vx = vmag * np.cos(theta)
        vy = vmag * np.sin(theta)
        speeds[i] = [vx, vy]

    return a_base, b_base, speeds

# Initialize obstacle geometry + velocities
a_base, b_base, obstacle_speeds = init_obstacle_params(obstacles_true, obstacle_types)

# ---------------------------------------------------------------------------
# APF parameters (same as before)
# ---------------------------------------------------------------------------
k_att, k_rep, d0, dt = 2.0, 40.0, 2.0, 0.01
max_rep_force = 12.0
path_data = [q.copy()]
initial_obstacles = obstacles_true.copy()

# (a_base, b_base, alpha, beta, obstacle_speeds are now human-like)
print("Initialized human-like obstacles:")
for i, t in enumerate(obstacle_types):
    vmag = np.linalg.norm(obstacle_speeds[i])
    print(f"  Obstacle {i}: type={t}, a_base={a_base[i]:.2f}, b_base={b_base[i]:.2f}, |v|={vmag:.2f} m/s")

"""
Dynamic obstacle modeling:
1) Velocity is aligned with the major axis of the ellipse. Frep is higher.
2) Larger obstacles have stronger repulsion (a, b larger).
3) Faster obstacles have stronger repulsion.
4) True obstacle positions are noisy.
"""

def attractive_force(q, q_goal):
    F_att = -k_att * (q - q_goal)
    return F_att

def repulsive_force(q, obstacles_noisy, obstacle_speeds):
    F_rep_total = np.array([0.0, 0.0])
    for i, obs in enumerate(obstacles_noisy):
        vx, vy = obstacle_speeds[i]
        vmag = np.hypot(vx, vy)
        theta = np.arctan2(vy, vx + 1e-12)

        a = a_base[i] + alpha * vmag
        b = b_base[i] + beta  * vmag

        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, s], [-s, c]])
        Q = R @ np.diag([1/a**2, 1/b**2]) @ R.T

        dE = np.sqrt((q - obs).T @ Q @ (q - obs) + 1e-12)
        d0_i = 1.2 * max(a, b)

        if dE < d0_i:
            F_mag = k_rep * (1/dE - 1/d0_i) * (1/dE**2)
            grad_Dq = Q @ (q - obs) / (dE + 1e-12)
            F_rep = F_mag * grad_Dq
            # Clip to max repulsive force
            if np.linalg.norm(F_rep) > max_rep_force:
                F_rep = F_rep / np.linalg.norm(F_rep) * max_rep_force
        else:
            F_rep = np.zeros(2)  # ignore distant obstacles
        F_rep_total += F_rep
    return F_rep_total


def potential(q, q_goal, obstacles_noisy, obstacle_speeds):
    U_rep_total = 0
    U_att = 0.5 * k_att * np.linalg.norm(q - q_goal)**2
    
    for i, obs in enumerate(obstacles_noisy):
        vx, vy = obstacle_speeds[i]
        vmag = np.sqrt(vx**2 + vy**2) 
        theta = np.arctan2(vy, vx + 1e-12)
        a = a_base[i] + alpha * vmag 
        b = b_base[i] + beta  * vmag 
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, s],
                      [-s,  c]])
        Q0 = np.diag([1/a**2, 1/b**2])
        Q = R @ Q0 @ R.T
        v = q - obs
        dE = np.sqrt(float(v.T @ Q @ v) + 1e-12)
        
        # dynamic avoidance boundary
        d0_i = 1.2 * max(a, b)

        if dE < 1e-6:  # avoid division by zero
            dE = 1e-6
        U_rep = 0.5 * k_rep * (1/dE - 1/d0_i)**2 if dE < d0_i else 0
        U_rep_total += U_rep
    return U_att + U_rep_total

def total_force(q, q_goal, obstacles_noisy, obstacle_speeds):
    """
    Calculate the total force on the robot
    """
    F_att = attractive_force(q, q_goal)
    F_rep = repulsive_force(q, obstacles_noisy, obstacle_speeds)
    return F_att + F_rep

def resolve_obstacle_collisions(obstacles_true, obstacle_speeds):
    N = len(obstacles_true)

    # --- compute dynamic radii ---
    radii = np.zeros(N)
    for i in range(N):
        vx, vy = obstacle_speeds[i]
        vmag = np.sqrt(vx**2 + vy**2)

        a_i = a_base[i] + alpha * vmag
        b_i = b_base[i] + beta  * vmag

        # realistic effective radius = max(axis lengths)
        radii[i] = max(a_i, b_i)

    # --- pairwise collision check ---
    for i in range(N):
        for j in range(i+1, N):

            difference = obstacles_true[i] - obstacles_true[j]
            distance = np.linalg.norm(difference)
            allowed_distance = radii[i] + radii[j]

            if distance < allowed_distance and distance > 1e-9:  # collision
                penetration = allowed_distance - distance
                direction = difference / distance  # normalized
                obstacles_true[i] += direction * (penetration / 2)
                obstacles_true[j] -= direction * (penetration / 2)

def apply_stochastic_maneuver(obstacle_speeds, maneuver_prob=0.25,
                              magnitude_sigma=0.05, turn_sigma=0.02):
    """
    Modify obstacle velocities by adding stochastic maneuvers.
    """
    new_speeds = obstacle_speeds.copy()

    for i in range(len(new_speeds)):
        vx, vy = new_speeds[i]
        vmag = np.sqrt(vx**2 + vy**2)
        
        # 1) Small continuous jitter
        vmag *= np.random.normal(1, magnitude_sigma)
        
        # 2) Slight random drift in direction
        theta = np.arctan2(vy, vx)
        theta += np.random.normal(0, turn_sigma)
        
        # reconstruct velocity
        vx = vmag * np.cos(theta)
        vy = vmag * np.sin(theta)

        # 3) Occasional large maneuver
        if np.random.rand() < maneuver_prob:
            big_turn = np.random.uniform(-np.pi/2, np.pi/2)
            theta += big_turn
            vx = vmag * np.cos(theta)
            vy = vmag * np.sin(theta)

        new_speeds[i] = [vx, vy]

    return new_speeds

def is_collision_check(q, obstacles_noisy, obstacle_speeds):
    collided_indices = []
    min_dE = float("inf")

    for i, obs in enumerate(obstacles_noisy):
        vx, vy = obstacle_speeds[i]
        vmag = np.hypot(vx, vy)
        theta = np.arctan2(vy, vx + 1e-12)

        a = a_base[i] + alpha * vmag
        b = b_base[i] + beta  * vmag

        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, s], [-s, c]])
        Q = R @ np.diag([1/a**2, 1/b**2]) @ R.T

        dE = np.sqrt((q - obs).T @ Q @ (q - obs) + 1e-12)
        min_dE = min(min_dE, dE)

        COLLISION_THRESHOLD = 1.0
        if dE < COLLISION_THRESHOLD:
            collided_indices.append(i)

    return len(collided_indices), collided_indices, min_dE




# ---------------------------------------------------------------------------
# Helper Function: Convert Dynamic Obstacles for RRT
# ---------------------------------------------------------------------------

def convert_obstacles_for_rrt(obstacles_true, obstacle_speeds, safety_margin: float = 0.5):
    """
    Convert dynamic elliptical obstacles to circular obstacles for RRT collision checking.
    Called each step to update obstacle positions and sizes based on current speeds.
    Uses max(a, b) as radius for conservative collision checking.
    """
    obstacles_rrt = []
    for i, obs_pos in enumerate(obstacles_true):
        vx, vy = obstacle_speeds[i] if i < len(obstacle_speeds) else [0, 0]
        vmag = np.sqrt(vx**2 + vy**2)
        
        # Calculate dynamic ellipse dimensions (same as potential field uses)
        a = a_base[i] + alpha * vmag
        b = b_base[i] + beta * vmag
        
        # Use maximum axis as radius (conservative approach)
        radius = max(a, b) + safety_margin
        
        # NOTE: Point & Obstacle must come from your RRT module
        obstacles_rrt.append(Obstacle(Point(obs_pos[0], obs_pos[1]), radius))
    
    return obstacles_rrt


# ---------------------------------------------------------------------------
# Integrated System: TSP + RRT + Potential Field
# ---------------------------------------------------------------------------

class RRT_PotentialSystem:
    """
    Integrated system using TSP for waypoint ordering, RRT for global path planning,
    and potential fields for local navigation with dynamic obstacles.
    
    Workflow:
    1. TSP optimizes waypoint order (replans every 5 seconds)
    2. RRT plans path to current waypoint
    3. Potential field navigates along RRT path
    4. Obstacles move dynamically each step
    5. RRT replans periodically as obstacles move
    """
    
    def __init__(self, rrt_step_size: float = 1.0, rrt_max_iter: int = 1000,
                 replan_interval: int = 200, tsp_replan_interval: float = 5.0):
        """
        Initialize the integrated system
        
        Args:
            rrt_step_size: Step size for RRT expansion
            rrt_max_iter: Maximum RRT iterations
            replan_interval: Replan RRT every N steps (0 = never replan)
            tsp_replan_interval: Replan TSP every N seconds (default: 5.0)
        """
        # Initialize RRT planner (handles collision-free pathfinding)
        self.rrt = RRTPlanner(step_size=rrt_step_size, max_iterations=rrt_max_iter)
        # Initialize TSP planner (optimizes waypoint order)
        self.global_planner = GlobalPlanner(replan_interval=tsp_replan_interval)
        
        self.rrt_waypoints: Optional[List["Point"]] = None
        self.tsp_waypoint_order: Optional[List["Point"]] = None
        self.current_waypoint_index = 0  # TSP waypoint index
        self.current_rrt_index = 0  # RRT waypoint index
        self.waypoint_reached_threshold = 0.3
        
        # Dynamic replanning parameters
        self.replan_interval = replan_interval  # RRT replans every N simulation steps
        self.last_replan_step = 0
        self.start_time = time.time()  # For TSP time-based replanning
    
    def _generate_simple_waypoints(self, start: "Point", goal: "Point", 
                                    obstacles_rrt: List["Obstacle"],
                                    num_waypoints: int = 3) -> List["Point"]:
        """
        Generate simple intermediate waypoints between start and goal.
        This is a simplified version that can be expanded later.
        
        Args:
            start: Starting position
            goal: Goal position
            obstacles_rrt: Obstacles for collision checking
            num_waypoints: Number of intermediate waypoints to generate
        
        Returns:
            List of waypoint Points
        """
        waypoints = []
        dx = goal.x - start.x
        dy = goal.y - start.y
        
        # Generate evenly spaced waypoints along the line
        for i in range(1, num_waypoints + 1):
            t = i / (num_waypoints + 1)
            wp = Point(start.x + dx * t, start.y + dy * t)
            
            # Simple collision check - skip if too close to obstacle
            too_close = False
            for obs in obstacles_rrt:
                if wp.distance_to(obs.position) < obs.radius + 1.0:
                    too_close = True
                    break
            
            if not too_close:
                waypoints.append(wp)
        
        return waypoints
    
    def plan_global_path(self, start: "Point", goal: "Point", 
                         obstacles_rrt: List["Obstacle"],
                         bounds: Optional[Tuple[float, float, float, float]] = None,
                         waypoints: Optional[List["Point"]] = None) -> bool:
        """
        Plan global path using TSP for waypoint ordering, then RRT for pathfinding.
        
        Now uses TANGENT WAYPOINTS around obstacles (matching the image approach).
        Returns True if path found, False otherwise.
        """
        # STEP 1: Generate tangent waypoints around obstacles (if not provided)
        # This replaces the old simple waypoint generation with true tangent points
        current_time = time.time() - self.start_time
        if self.global_planner.should_replan(current_time) or self.tsp_waypoint_order is None:
            if waypoints is None:
                # Use tangent waypoint generation (NEW - matches the image!)
                # This generates waypoints on obstacle boundaries where lines from start/goal are tangent
                # plan_path now automatically generates tangent waypoints and optimizes order with TSP
                self.tsp_waypoint_order = self.global_planner.plan_path(start, goal, obstacles_rrt)
            else:
                # Use provided waypoints (for custom waypoint sets)
                # Need to use optimize_waypoint_order directly since we have custom waypoints
                self.tsp_waypoint_order = self.global_planner.optimize_waypoint_order(start, waypoints, goal)
            self.current_waypoint_index = 0  # Start at first TSP waypoint
            self.current_rrt_index = 0
        
        # STEP 2: Get the current target waypoint from TSP order
        # Skip start position (index 0) - we're already there
        if not self.tsp_waypoint_order or self.current_waypoint_index >= len(self.tsp_waypoint_order):
            target = goal  # All TSP waypoints visited, go directly to goal
        elif self.current_waypoint_index == 0:
            # Skip start position, go to first actual waypoint
            if len(self.tsp_waypoint_order) > 1:
                self.current_waypoint_index = 1
                target = self.tsp_waypoint_order[self.current_waypoint_index]
            else:
                target = goal
        else:
            target = self.tsp_waypoint_order[self.current_waypoint_index]  # Current TSP waypoint
        
        # STEP 3: Use RRT to plan collision-free path from current position to target
        self.rrt_waypoints = self.rrt.plan_path(start, target, obstacles_rrt, bounds)
        
        if self.rrt_waypoints:
            self.current_rrt_index = 0  # Reset to start of new RRT path
            return True
        else:
            self.current_rrt_index = 0
            return False
    
    def get_current_target(self) -> Optional["Point"]:
        """Get the current waypoint we're navigating to"""
        if self.rrt_waypoints and self.current_rrt_index < len(self.rrt_waypoints):
            return self.rrt_waypoints[self.current_rrt_index]
        return None
    
    def update_waypoint(self, current_pos: np.ndarray):
        """Check if current waypoint reached and advance if needed"""
        if not self.rrt_waypoints:
            return
        
        current_point = Point(current_pos[0], current_pos[1])
        current_target = self.get_current_target()
        
        # Check if robot has reached the current RRT waypoint
        if current_target and current_point.distance_to(current_target) < self.waypoint_reached_threshold:
            self.current_rrt_index += 1  # Move to next RRT waypoint
            
            # Check if we've completed the entire RRT path to the current TSP waypoint
            if self.current_rrt_index >= len(self.rrt_waypoints):
                # Verify we actually reached the TSP waypoint
                if self.tsp_waypoint_order and self.current_waypoint_index < len(self.tsp_waypoint_order):
                    tsp_wp = self.tsp_waypoint_order[self.current_waypoint_index]
                    if current_point.distance_to(tsp_wp) < self.waypoint_reached_threshold:
                        # Successfully reached TSP waypoint! Move to next one
                        self.current_waypoint_index += 1
                        self.rrt_waypoints = None  # Force RRT to replan to next TSP waypoint
                        self.current_rrt_index = 0
                else:
                    # Completed all RRT waypoints, need new path
                    self.rrt_waypoints = None
                    self.current_rrt_index = 0
    
    def should_replan(self, current_step: int) -> bool:
        """Check if it's time to replan RRT path (every N simulation steps)"""
        if self.replan_interval == 0:
            return False
        
        return (current_step - self.last_replan_step) >= self.replan_interval
    
    def should_replan_tsp(self) -> bool:
        """Check if it's time to replan TSP waypoint order (every N seconds)"""
        current_time = time.time() - self.start_time
        return self.global_planner.should_replan(current_time)
    
    def is_goal_reached(self, current_pos: np.ndarray, goal: np.ndarray, 
                        tolerance: float = 0.1) -> bool:
        """Check if final goal is reached"""
        return np.linalg.norm(current_pos - goal) < tolerance


# ---------------------------------------------------------------------------
# Grid for visualization
# ---------------------------------------------------------------------------

x_range = np.linspace(-40, 40, 50)
y_range = np.linspace(-40, 40, 50)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)
U = np.zeros_like(X)
V = np.zeros_like(Y)

# ---------------------------------------------------------------------------
# RRT + Potential Field Integration Setup
# ---------------------------------------------------------------------------

# NOTE: You must have real implementations of Point, Obstacle, RRTPlanner.
# Initialize integrated system: TSP (waypoint optimization) + RRT (path planning) + Potential Fields (local navigation)
rrt_system = RRT_PotentialSystem(rrt_step_size=1.0,
                                 rrt_max_iter=1000,
                                 replan_interval=200,  # RRT replans every 200 steps
                                 tsp_replan_interval=5.0)  # TSP replans every 5 seconds

start_point = Point(q[0], q[1])
goal_point  = Point(q_goal[0], q_goal[1])

# Convert dynamic elliptical obstacles to circular obstacles for RRT collision checking
obstacles_rrt = convert_obstacles_for_rrt(obstacles_true, obstacle_speeds)

# Plan initial global path: TSP optimizes waypoint order, then RRT plans path to first waypoint
rrt_system.plan_global_path(
    start_point,
    goal_point,
    obstacles_rrt,
    bounds=(-40, 40, -40, 40)
)

# ---------------------------------------------------------------------------
# simulate the path
# ---------------------------------------------------------------------------

max_steps = 2000
tolerance = 0.1

for step in range(max_steps):
    
    # 0) random maneuver (new speeds)
    obstacle_speeds = apply_stochastic_maneuver(obstacle_speeds)

    # 1) move the real obstacles
    obstacles_true[:,0] += obstacle_speeds[:,0]*dt
    obstacles_true[:,1] += obstacle_speeds[:,1]*dt

    # 1.1) resolve collisions between obstacles (prevent overlap)
    resolve_obstacle_collisions(obstacles_true, obstacle_speeds)

    # 2) robot observes obstacles (noisy)
    obstacles_noisy = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)

    # 3) RRT global planning + APF local force calculation -------------------
    obstacles_rrt = convert_obstacles_for_rrt(obstacles_true, obstacle_speeds)

    # Check if TSP needs replanning (every 5 seconds)
    # TSP optimizes the order of waypoints to minimize total travel distance
    if rrt_system.should_replan_tsp():
        rrt_system.plan_global_path(
            Point(q[0], q[1]),  # current position as start
            goal_point,
            obstacles_rrt,
            bounds=(-40, 40, -40, 40)
        )
        rrt_system.last_replan_step = step
    
    # Check if RRT needs replanning (every 200 steps or if no path exists)
    # RRT plans collision-free path from current position to current TSP waypoint
    if rrt_system.should_replan(step) or rrt_system.rrt_waypoints is None:
        rrt_system.plan_global_path(
            Point(q[0], q[1]),  # current position as start
            goal_point,
            obstacles_rrt,
            bounds=(-40, 40, -40, 40)
        )
        rrt_system.last_replan_step = step

    # Get the current target: either next RRT waypoint or fallback to goal
    current_target_point = rrt_system.get_current_target()
    if current_target_point is not None:
        q_target = np.array([current_target_point.x, current_target_point.y])
    else:
        q_target = q_goal  # fallback to global goal

    # Potential field calculates force toward target (attractive) and away from obstacles (repulsive)
    F = total_force(q, q_target, obstacles_noisy, obstacle_speeds)

    # 4) robot moves based on potential field force
    q = q + F * dt

    # 4.1) Update waypoint tracking: check if we reached current RRT waypoint or TSP waypoint
    rrt_system.update_waypoint(q)

    # 5) collision check
    count, hits, min_dE = is_collision_check(q, obstacles_noisy, obstacle_speeds)
    if count > 0:
        print("Collision detected with obstacle:", hits)

    # 6) path
    path_data.append(q.copy())

    # 7) stop condition: close enough to goal (RRT + APF)
    if rrt_system.is_goal_reached(q, q_goal, tolerance):
        print(f"Reached goal (RRT + TSP + APF) in {step} steps!")
        break

path = np.array(path_data)

# ==========================================================
# Compute field AFTER full motion simulation
# ==========================================================

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pos = np.array([X[i,j], Y[i,j]])
        
        # potential uses noisy obstacles
        Z[i,j] = potential(pos, q_goal, obstacles_noisy, obstacle_speeds)

        # force field also uses noisy obstacles
        F = total_force(pos, q_goal, obstacles_noisy, obstacle_speeds)
        U[i,j], V[i,j] = F[0], F[1]

# ==========================================================
#                     PLOTTING SECTION
# ==========================================================

# ---- 1️⃣ 3D Potential Surface ----
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Surface + colorbar
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label("Potential Energy", fontsize=14)
cbar.ax.tick_params(labelsize=12)

# Path with correct potential computation
ax.plot(path[:,0], path[:,1],
        [potential(p, q_goal, obstacles_noisy, obstacle_speeds) for p in path],
        color='red', linewidth=2, label='Path')

# Start position
ax.scatter(path[0,0], path[0,1],
           potential(path[0], q_goal, obstacles_noisy, obstacle_speeds),
           color='cyan', marker='x', s=120, linewidths=3, label='Start')

# REAL OBSTACLES (truth)
ax.scatter(obstacles_true[:,0], obstacles_true[:,1],
           np.max(Z)*0.8, color='black', s=80, label='True Obstacle Centers')

# NOISY OBSTACLES (sensor-detected) 
ax.scatter(obstacles_noisy[:,0], obstacles_noisy[:,1],
           np.max(Z)*0.8, color='red', s=80, label='Noisy Detected Centers')

# Goal position
ax.scatter(q_goal[0], q_goal[1], np.min(Z),
           color='orange', s=80, marker='x', linewidths=3, label='Goal')

ax.set_xlabel('X', fontsize=16)
ax.set_ylabel('Y', fontsize=16)
ax.set_zlabel('Potential Energy', fontsize=16)
ax.set_title('3D Potential Field Surface', fontsize=18)
ax.tick_params(axis='both', labelsize=12)
ax.legend(fontsize=14)

plt.savefig("figures/fig_3d_potentialsurface.png", dpi=300, bbox_inches='tight')
plt.show()


# ---- 2️⃣ 2D Contour + Force Field ----
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
contour = axs[0].contourf(X, Y, Z, levels=100, cmap='viridis')

# Start
start_x, start_y = path[0]
axs[0].scatter(start_x, start_y, marker='x', s=120, color='cyan', linewidths=3, label='Start')

axs[0].plot(path[:,0], path[:,1], 'w-', label='Path')

# true vs noisy obstacles
axs[0].plot(obstacles_true[:,0], obstacles_true[:,1], 'ko', label='True Centers')
axs[0].plot(obstacles_noisy[:,0], obstacles_noisy[:,1], 'ro', label='Noisy Centers')

# Goal
axs[0].scatter(q_goal[0], q_goal[1],
               marker='x', s=120, color='orange', linewidths=3, label='Goal')

# draw ellipses for TRUE obstacles
for i, obs in enumerate(obstacles_true):
    vx, vy = obstacle_speeds[i]
    vmag = np.sqrt(vx**2 + vy**2)
    theta = np.degrees(np.arctan2(vy, vx + 1e-12))

    a = a_base[i] + alpha * vmag
    b = b_base[i] + beta  * vmag
    ellipse = Ellipse(
        xy=(obs[0], obs[1]),
        width=2*a, height=2*b,
        angle=theta,
        edgecolor='white', facecolor='none',
        linestyle='--', linewidth=1.5
    )
    axs[0].add_patch(ellipse)

axs[0].set_title("Potential Energy Map with Elliptical Obstacles")
axs[0].legend()
plt.colorbar(contour, ax=axs[0])

# FORCE FIELD PLOT
axs[1].quiver(X, Y, U, V, color='black', alpha=0.6)
axs[1].plot(path[:,0], path[:,1], 'r-', linewidth=2)
axs[1].set_title("Force Field (Gradient of Potential)")
axs[1].set_xlabel("X")
axs[1].set_ylabel("Y")

plt.tight_layout()
plt.savefig("figures/fig_contour_force.png", dpi=300, bbox_inches='tight')
plt.show()
