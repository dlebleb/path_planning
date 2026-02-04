"""
Combined System: Potential Field (partner's code) + RRT (your code)

This combines:
- Partner's perfect potential field code from local_potential_field_demo_dynamic.py
- Your RRT path planning system

The robot uses:
1. RRT to plan collision-free path from current position to goal
2. Potential fields to navigate along RRT waypoints with dynamic obstacles
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import time

# Import your RRT scripts from simulation folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'simulation'))
from TSP import Point, Obstacle
from rrt_planner import RRTPlanner
from typing import List, Optional, Tuple

os.makedirs("figures", exist_ok=True)

# ============================================================================
# SETUP: Start, Goal, Obstacles (from Damala's code)
# ============================================================================

q_goal = np.array([40, 40])  # goal position
q = np.array([-40.0, -40.0])  # starting position of the robot

# obstacle coordinates 
obstacles_true = np.array([[-18.0,-10.0], [18,-20], [18, 8], [22,26], [23,15], [-23,15], [5,5]])

sigma = 0.1  # 10 cm uncertainty
obstacles_noisy = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)

# obstacle speeds
obstacle_speeds = np.array([[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1]])
obstacle_speeds = obstacle_speeds * 8

# Potential field: attraction to goal (4.0), repulsion from obstacles (10.0), repulsion range (10.0), time step (0.01)
k_att, k_rep, d0, dt = 4.0, 10.0, 10.0, 0.01
max_rep_force = np.inf
path_data = [q.copy()]
initial_obstacles = obstacles_true.copy()

# elliptical obstacles parameters (from Damala's code)
a0 = 2.0  # major axis (along velocity direction)
b0 = 1.0  # minor axis (perpendicular to velocity direction)
alpha = 1.2   # major scaling (large)
beta  = 0.3   # minor scaling (small)

# each obstacle has a size factor: 1 = normal, >1 = large, <1 = small
sizes = np.array([1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1]) * 1
# incorporate static size
a_base = a0 * sizes 
b_base = b0 * sizes 

# ============================================================================
# POTENTIAL FIELD FUNCTIONS (Damala's code)
# ============================================================================

def attractive_force(q, q_goal):
    F_att = -k_att * (q - q_goal)
    return F_att

def repulsive_force(q, obstacles_noisy, obstacle_speeds):
    F_rep_total = np.array([0.0, 0.0])
    for i, obs in enumerate(obstacles_noisy):
        vx, vy = obstacle_speeds[i]
        vmag = np.sqrt(vx**2 + vy**2) 

        # dynamic scaling with speed
        a = a_base[i] + alpha * vmag  # major axis (velocity direction)
        b = b_base[i] + beta  * vmag  # minor axis (perpendicular)
        
        obs_x, obs_y = obs[0], obs[1]
        eps = 1e-12
        q_x, q_y = (q[0]-obs_x)/(a+eps), (q[1]-obs_y)/(b+eps)

        # distance of the robot to origin -1
        dE = np.sqrt(q_x**2 + q_y**2) - 1

        if dE < d0:
            F_mag = k_rep * (1/dE - 1/d0) * (1/dE**2)
            grad_Dq = (q - obs) / (dE + 1e-12)
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
        
        obs_x, obs_y = obs[0], obs[1]
        eps = 1e-12
        q_x, q_y = (q[0]-obs_x)/(a+eps), (q[1]-obs_y)/(b+eps)
        
        dE = np.sqrt(q_x**2 + q_y**2) - 1

        if dE < 1e-6:  # avoid division by zero
            dE = 1e-6
        U_rep = 0.5 * k_rep * (1/dE - 1/d0)**2 if dE < d0 else 0
        U_rep_total += U_rep
    return U_att + U_rep_total

def total_force(q, q_goal, obstacles_noisy, obstacle_speeds):
    """Calculate the total force on the robot"""
    F_att = attractive_force(q, q_goal)
    F_rep = repulsive_force(q, obstacles_noisy, obstacle_speeds)
    return F_att + F_rep

def apply_stochastic_maneuver(obstacle_speeds, maneuver_prob=0.25,
                              magnitude_sigma=0.05, turn_sigma=0.02):
    """Modify obstacle velocities by adding stochastic maneuvers."""
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
        
        obs_x, obs_y = obs[0], obs[1]
        eps = 1e-12
        q_x, q_y = (q[0]-obs_x)/(a+eps), (q[1]-obs_y)/(b+eps)
        
        dE = np.sqrt(q_x**2 + q_y**2) - 1
        
        # COLLISION condition: robot is inside the ellipse
        if dE < 0:
            counter += 1
            collided_indices.append(i)
    
    return counter, collided_indices

# ============================================================================
# RRT INTEGRATION (Rachel's code)
# ============================================================================

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
        
        obstacles_rrt.append(Obstacle(Point(obs_pos[0], obs_pos[1]), radius))
    
    return obstacles_rrt


class RRT_PotentialSystem:
    """
    Integrated system using RRT for global path planning
    and potential fields for local navigation with dynamic obstacles.
    
    Workflow:
    1. RRT plans collision-free path from current position to goal
    2. Potential field navigates along RRT path toward next waypoint
    3. Obstacles move dynamically each step
    4. RRT replans periodically as obstacles move
    """
    
    def __init__(self, rrt_step_size: float = 1.0, rrt_max_iter: int = 1000,
                 replan_interval: int = 200):
        """Initialize the integrated system"""
        # Initialize RRT planner (handles collision-free pathfinding)
        self.rrt = RRTPlanner(step_size=rrt_step_size, max_iterations=rrt_max_iter)
        
        self.rrt_waypoints: Optional[List[Point]] = None
        self.current_rrt_index = 0  # RRT waypoint index
        self.waypoint_reached_threshold = 0.3
        
        # Dynamic replanning parameters
        self.replan_interval = replan_interval  # RRT replans every N simulation steps
        self.last_replan_step = 0
    
    def plan_global_path(self, start: Point, goal: Point, 
                         obstacles_rrt: List[Obstacle],
                         bounds: Optional[Tuple[float, float, float, float]] = None) -> bool:
        """
        Plan global path using RRT from current position directly to goal.
        
        Returns True if path found, False otherwise.
        """
        # Use RRT to plan collision-free path from current position to goal
        self.rrt_waypoints = self.rrt.plan_path(start, goal, obstacles_rrt, bounds)
        
        if self.rrt_waypoints:
            self.current_rrt_index = 0  # Reset to start of new RRT path
            return True
        else:
            self.current_rrt_index = 0
            return False
    
    def get_current_target(self) -> Optional[Point]:
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
            
            # Check if we've completed the entire RRT path
            if self.current_rrt_index >= len(self.rrt_waypoints):
                # Completed all RRT waypoints, need new path
                self.rrt_waypoints = None
                self.current_rrt_index = 0
    
    def should_replan(self, current_step: int) -> bool:
        """Check if it's time to replan RRT path (every N simulation steps)"""
        if self.replan_interval == 0:
            return False
        return (current_step - self.last_replan_step) >= self.replan_interval
    
    def is_goal_reached(self, current_pos: np.ndarray, goal: np.ndarray, 
                        tolerance: float = 0.1) -> bool:
        """Check if final goal is reached"""
        return np.linalg.norm(current_pos - goal) < tolerance


# ============================================================================
# GRID FOR VISUALIZATION (Damala's code)
# ============================================================================

x_range = np.linspace(-100, 100, 50)
y_range = np.linspace(-100, 100, 50)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)
U = np.zeros_like(X)
V = np.zeros_like(Y)

# ============================================================================
# RRT + POTENTIAL FIELD SYSTEM SETUP (Rachel's code)
# ============================================================================

# Initialize integrated system: RRT (path planning) + Potential Fields (local navigation)
rrt_system = RRT_PotentialSystem(
    rrt_step_size=1.0,
    rrt_max_iter=1000,
    replan_interval=200  # RRT replans every 200 steps
)

start_point = Point(q[0], q[1])
goal_point = Point(q_goal[0], q_goal[1])

# Convert dynamic elliptical obstacles to circular obstacles for RRT collision checking
obstacles_rrt = convert_obstacles_for_rrt(obstacles_true, obstacle_speeds)

# Plan initial global path: RRT plans collision-free path from start to goal
rrt_system.plan_global_path(
    start_point,
    goal_point,
    obstacles_rrt,
    bounds=(-100, 100, -100, 100)
)

# ============================================================================
# SIMULATION LOOP (Damala's base + RRT integration)
# ============================================================================

max_steps = 5000
tolerance = 0.5
goal_reached = False
goal_reached_step = None

for step in range(max_steps):
    # 0) random maneuver (new speeds) - partner's code
    obstacle_speeds = apply_stochastic_maneuver(obstacle_speeds)

    # 1) move the real obstacles - partner's code
    obstacles_true[:,0] += obstacle_speeds[:,0]*dt
    obstacles_true[:,1] += obstacle_speeds[:,1]*dt

    # 2) robot observes obstacles (noisy) - partner's code
    obstacles_noisy = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)

    # 3) RRT global planning + APF local force calculation (YOUR INTEGRATION)
    obstacles_rrt = convert_obstacles_for_rrt(obstacles_true, obstacle_speeds)

    # Check if RRT needs replanning (every 200 steps or if no path exists)
    # RRT plans collision-free path from current position to goal
    if rrt_system.should_replan(step) or rrt_system.rrt_waypoints is None:
        rrt_system.plan_global_path(
            Point(q[0], q[1]),  # current position as start
            goal_point,
            obstacles_rrt,
            bounds=(-100, 100, -100, 100)
        )
        rrt_system.last_replan_step = step

    # Get the current target: either next RRT waypoint or fallback to goal
    current_target_point = rrt_system.get_current_target()
    if current_target_point is not None:
        q_target = np.array([current_target_point.x, current_target_point.y])
    else:
        q_target = q_goal  # fallback to global goal

    # Potential field calculates force toward target (attractive) and away from obstacles (repulsive)
    # Partner's code - navigates toward RRT waypoint instead of direct goal
    F = total_force(q, q_target, obstacles_noisy, obstacle_speeds)

    # 4) robot moves based on potential field force - partner's code
    q = q + F * dt

    # 4.1) Update waypoint tracking: check if we reached current RRT waypoint
    rrt_system.update_waypoint(q)

    # 5) collision check - partner's code
    count, hits = is_collision_check(q, obstacles_noisy, obstacle_speeds)
    if count > 0:
        print("Collision detected with obstacle:", hits)

    # 6) path - partner's code
    path_data.append(q.copy())

    # 7) stop condition: close enough to goal (RRT + APF)
    if rrt_system.is_goal_reached(q, q_goal, tolerance):
        goal_reached = True
        goal_reached_step = step
        print(f"\n{'='*60}")
        print(f"✓ SUCCESS: Reached goal (RRT + APF) in {step} steps!")
        print(f"{'='*60}\n")
        break

path = np.array(path_data)

# ============================================================================
# COMPUTE FIELD AFTER FULL MOTION SIMULATION (Damala's code)
# ============================================================================

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pos = np.array([X[i,j], Y[i,j]])
        
        # potential uses noisy obstacles
        Z[i,j] = potential(pos, q_goal, obstacles_noisy, obstacle_speeds)

        # force field also uses noisy obstacles
        F = total_force(pos, q_goal, obstacles_noisy, obstacle_speeds)
        U[i,j], V[i,j] = F[0], F[1]

# ============================================================================
# PLOTTING SECTION (Damala's code - enhanced to show RRT waypoints)
# ============================================================================

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
ax.set_xlim(-75, 75)
ax.set_ylim(-75, 75)
ax.set_title('3D Potential Field Surface (RRT + APF)', fontsize=18)
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

axs[0].plot(path[:,0], path[:,1], 'w-', label='Path', linewidth=2)

# true vs noisy obstacles
axs[0].plot(obstacles_true[:,0], obstacles_true[:,1], 'ko', label='True Centers')
axs[0].plot(obstacles_noisy[:,0], obstacles_noisy[:,1], 'ro', label='Noisy Centers')

# Goal
axs[0].scatter(q_goal[0], q_goal[1],
               marker='x', s=120, color='orange', linewidths=3, label='Goal')

# Draw RRT waypoints (if available)
if rrt_system.rrt_waypoints:
    rrt_x = [wp.x for wp in rrt_system.rrt_waypoints]
    rrt_y = [wp.y for wp in rrt_system.rrt_waypoints]
    axs[0].plot(rrt_x, rrt_y, 'bo', markersize=6, 
               label='RRT Waypoints', zorder=5, alpha=0.7)
    axs[0].plot(rrt_x, rrt_y, 'b--', linewidth=1.5, 
               alpha=0.5, label='RRT Path', zorder=4)

# draw ellipses for noisy obstacles
for i, obs in enumerate(obstacles_noisy):
    vx, vy = obstacle_speeds[i]
    vmag = np.sqrt(vx**2 + vy**2)
    theta = np.degrees(np.arctan2(vy, vx + 1e-12))

    # 1) Physical ellipse (dE = 1 boundary)
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

    # Add obstacle ID label on ellipse
    axs[0].text(
        obs[0], obs[1],
        f"{i}",
        color="yellow",
        fontsize=10,
        ha="center", va="center",
        weight="bold"
    )

axs[0].set_title("Potential Energy Map with Elliptical Obstacles (RRT + APF)")
axs[0].set_xlim(-75, 75)
axs[0].set_ylim(-75, 75)
axs[0].legend()
plt.colorbar(contour, ax=axs[0])

# FORCE FIELD PLOT
axs[1].quiver(X, Y, U, V, color='black', alpha=0.6)
axs[1].plot(path[:,0], path[:,1], 'r-', linewidth=2)
axs[1].set_title("Force Field (Gradient of Potential)")
axs[1].set_xlabel("X")
axs[1].set_ylabel("Y")
axs[1].set_xlim(-75, 75)
axs[1].set_ylim(-75, 75)

plt.tight_layout()
plt.savefig("figures/fig_contour_force.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("Combined System Summary:")
print("="*60)

if goal_reached:
    print(f"✓ GOAL REACHED in {goal_reached_step} steps!")
else:
    print(f"✗ Goal NOT reached (stopped at {len(path)} steps, max: {max_steps})")
print(f"✓ Path length: {len(path)} steps")
if rrt_system.rrt_waypoints:
    print(f"✓ RRT path has {len(rrt_system.rrt_waypoints)} waypoints")
print("="*60)

