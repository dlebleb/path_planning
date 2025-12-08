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

q_goal = np.array([10, 10])  # goal position
q = np.array([-20.0, -20.0])  # starting position of the robot

# obstacle coordinates 
obstacles_true = np.array([[-5,-5], [-3,-3], [3, 3.5], [6,7], [9,9], [8,4], [5,5]])
sigma = 0.1  # 10 cm uncertainty
obstacles_noisy = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)

# obstacle speeds
obstacle_speeds = np.array([[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1]])
obstacle_speeds = obstacle_speeds * 5

# APF parameters
k_att, k_rep, d0, dt = 2.0, 40.0, 2.0, 0.01
max_rep_force = 14.0
path_data = [q.copy()]
initial_obstacles = obstacles_true.copy()

# elliptical obstacles
a0 = 2.0  # major axis (along velocity direction)
b0 = 1.0  # minor axis (perpendicular to velocity direction)
alpha = 1.2   # major scaling (large)
beta  = 0.3   # minor scaling (small)

# each obstacle has a size factor: 1 = normal, >1 = large, <1 = small
sizes = np.array([1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1])
# incorporate static size
a_base = a0 * sizes
b_base = b0 * sizes


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
        vmag = np.sqrt(vx**2 + vy**2) 
        theta = np.arctan2(vy, vx + 1e-12)  # obstacle motion direction

        # dynamic scaling with speed
        a = a_base[i] + alpha * vmag  # major axis (velocity direction)
        b = b_base[i] + beta  * vmag  # minor axis (perpendicular)

        # rotate into obstacle frame and calculate the Q matrix
        c, s = np.cos(theta), np.sin(theta)
        
        # rotation matrix that aligns major axis with velocity
        R = np.array([[c, s],
                      [-s,  c]])
        
        # Q0 (major axis = a, minor axis = b)
        Q0 = np.diag([1/a**2, 1/b**2])
        Q  = R @ Q0 @ R.T

        # elliptical distance
        dE = np.sqrt(float((q - obs).T @ Q @ (q - obs)) + 1e-12)

        # dynamic avoidance boundary
        d0_i = 1.2 * max(a, b)

        if dE < d0_i:
            F_mag = k_rep * (1/dE - 1/d0_i) * (1/dE**2)
            grad_Dq = Q @ (q - obs) / (dE + 1e-12)
            F_rep = F_mag * grad_Dq

            if np.linalg.norm(F_rep) > max_rep_force: 
                F_rep = F_rep/(np.linalg.norm(F_rep) + 1e-12)* max_rep_force
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
    counter = 0

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
        Q  = R @ Q0 @ R.T

        # elliptical distance
        dE = np.sqrt(float((q - obs).T @ Q @ (q - obs)) + 1e-12)
        
        # COLLISION condition: robot is inside the ellipse
        if dE < 1.0:
            counter += 1
            collided_indices.append(i)
    
    return counter, collided_indices


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
        self.rrt = RRTPlanner(step_size=rrt_step_size, max_iterations=rrt_max_iter)
        self.global_planner = GlobalPlanner(replan_interval=tsp_replan_interval)
        
        self.rrt_waypoints: Optional[List["Point"]] = None
        self.tsp_waypoint_order: Optional[List["Point"]] = None
        self.current_waypoint_index = 0  # TSP waypoint index
        self.current_rrt_index = 0  # RRT waypoint index
        self.waypoint_reached_threshold = 0.3
        
        # Dynamic replanning parameters
        self.replan_interval = replan_interval
        self.last_replan_step = 0
        self.start_time = time.time()
    
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
        Returns True if path found, False otherwise.
        """
        # Generate waypoints if not provided
        if waypoints is None:
            waypoints = self._generate_simple_waypoints(start, goal, obstacles_rrt)
        
        # Use TSP to optimize waypoint order
        current_time = time.time() - self.start_time
        if self.global_planner.should_replan(current_time) or self.tsp_waypoint_order is None:
            self.tsp_waypoint_order = self.global_planner.plan_path(start, goal, waypoints)
            self.current_waypoint_index = 0
            self.current_rrt_index = 0
        
        # Get current target waypoint from TSP order
        if not self.tsp_waypoint_order or self.current_waypoint_index >= len(self.tsp_waypoint_order):
            target = goal
        else:
            target = self.tsp_waypoint_order[self.current_waypoint_index]
        
        # Plan RRT path to current target
        self.rrt_waypoints = self.rrt.plan_path(start, target, obstacles_rrt, bounds)
        
        if self.rrt_waypoints:
            self.current_rrt_index = 0  # Reset RRT index
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
        
        # Check if we've reached the current RRT waypoint
        if current_target and current_point.distance_to(current_target) < self.waypoint_reached_threshold:
            self.current_rrt_index += 1
            
            # Check if we've completed the RRT path to current TSP waypoint
            if self.current_rrt_index >= len(self.rrt_waypoints):
                # Check if we've reached the TSP waypoint
                if self.tsp_waypoint_order and self.current_waypoint_index < len(self.tsp_waypoint_order):
                    tsp_wp = self.tsp_waypoint_order[self.current_waypoint_index]
                    if current_point.distance_to(tsp_wp) < self.waypoint_reached_threshold:
                        self.current_waypoint_index += 1
                        self.rrt_waypoints = None  # Force RRT replan to next waypoint
                        self.current_rrt_index = 0
                else:
                    # Completed all RRT waypoints, need new path
                    self.rrt_waypoints = None
                    self.current_rrt_index = 0
    
    def should_replan(self, current_step: int) -> bool:
        """Check if it's time to replan RRT path"""
        if self.replan_interval == 0:
            return False
        
        return (current_step - self.last_replan_step) >= self.replan_interval
    
    def should_replan_tsp(self) -> bool:
        """Check if it's time to replan TSP waypoint order"""
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
rrt_system = RRT_PotentialSystem(rrt_step_size=1.0,
                                 rrt_max_iter=1000,
                                 replan_interval=200,
                                 tsp_replan_interval=5.0)

start_point = Point(q[0], q[1])
goal_point  = Point(q_goal[0], q_goal[1])

# initial RRT obstacles snapshot
obstacles_rrt = convert_obstacles_for_rrt(obstacles_true, obstacle_speeds)

# plan initial global path
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

    # Check if TSP needs replanning
    if rrt_system.should_replan_tsp():
        rrt_system.plan_global_path(
            Point(q[0], q[1]),  # current position as start
            goal_point,
            obstacles_rrt,
            bounds=(-40, 40, -40, 40)
        )
        rrt_system.last_replan_step = step
    
    # Check if RRT needs replanning
    if rrt_system.should_replan(step) or rrt_system.rrt_waypoints is None:
        rrt_system.plan_global_path(
            Point(q[0], q[1]),  # current position as start
            goal_point,
            obstacles_rrt,
            bounds=(-40, 40, -40, 40)
        )
        rrt_system.last_replan_step = step

    current_target_point = rrt_system.get_current_target()
    if current_target_point is not None:
        q_target = np.array([current_target_point.x, current_target_point.y])
    else:
        q_target = q_goal  # fallback to global goal

    F = total_force(q, q_target, obstacles_noisy, obstacle_speeds)

    # 4) robot moves
    q = q + F * dt

    # 4.1) update RRT waypoint tracking
    rrt_system.update_waypoint(q)

    # 5) collision check
    count, hits = is_collision_check(q, obstacles_noisy, obstacle_speeds)
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
