"""
Local Potential-Field Path Planning (elliptical obstacles)

Run to plot a local path from start to goal around elliptical obstacles 
using an attractive + repulsive potential field.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
from APF3_Astar import extract_polyline, init_environment
from run_simulation_Astar import run_simulation_Astar
from damla_Astar1 import *

# ===============================
# INITIAL SETUP
# ===============================
q_goal_final        = np.array([20, 10]) #[10, 10]
#q_goal_final        = np.array([0, 20]) #[10, 10]
q                   = np.array([-40.0, -40.0])
v_robot             = 30 #25
# obstacle coordinates 
obstacles_true      = np.array([[-18.0,-10.0], [18,-20], [18, 8], [22,26], [23,15], [-23,15], [5,5], [-40, -30], [15, -10], [10, 5], [-30, 0], [-20, -20], [0, 0], [-35, 3], [-26, -27], [2, -10]])
# kalabalik platform
obstacles_true      = np.array([[-18.0,-10.0], [18,-20], [18, 8], [22,26], [23,15], [-23,15], [5,5], [-40, -30], [15, -10], [10, 5], [-30, 0], [-20, -20], [0, 0], [-35, 3], [-26, -27], [2, -10], [11, -12], [-3, -3], [-45, 0], [-25, -25], [8, 10], [-35, 13], [-21, -17], [21, -10]])
sigma               = 0.1  # 10 cm uncertainity
obstacles_noisy     = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)
# obstacle speeds
obstacle_speeds     = np.array([[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1], [-0.1, 0.1], [-0.2, 0.1], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [0.2, 0.1], [-0.1, 0.1], [-0.2, 0.1]])
# kalabalik platform
obstacle_speeds     = np.array([[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1], [-0.1, 0.1], [-0.2, 0.1], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [0.2, 0.1], [-0.1, 0.1], [-0.2, 0.1],[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1], [-0.1, 0.1]])

obstacle_speeds     = obstacle_speeds * 80
vmag_max            = 24.5
vmag_min            = 5
# APF parameters
max_rep_force       = np.inf
path_data           = [q.copy()]
initial_obstacles   = obstacles_true.copy()

rect_obstacles, expanded_rects, eps_expanded_rects, waypoints = init_environment()
# rectangles are moving with the same speed as the elliptical obstacles
rectangle_speeds = np.array([[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1]])
rectangle_speeds = rectangle_speeds * 0

#rrt = RRTPlanner(step_size=0.3, max_iterations=2000)

# first iteration of A*
full_geometric_path = astar(start_state=tuple(q), goal_state = tuple(q_goal_final), heuristic_func=euclidean_distance, successors_func=successors, obstacles_noisy = obstacles_noisy, obstacle_speeds = obstacle_speeds, rect_obstacles = rect_obstacles, waypoints = waypoints)
full_geometric_polyline = extract_polyline(full_geometric_path, waypoints)
q_goal = full_geometric_polyline[1]


N_MC = 10
distances = []
collisions = []
costs = []

for i in range(N_MC):
    
    print(f"\n=== RUN {i+1} ===")
    d, c, cost = run_simulation_Astar(
    q.copy(),
    q_goal.copy(),
    q_goal_final.copy(),
    obstacle_speeds.copy(),
    rectangle_speeds.copy(),
    waypoints.copy(),
    obstacles_true.copy(),
    rect_obstacles.copy(),
    obstacles_noisy.copy(),
    expanded_rects.copy(),
    eps_expanded_rects.copy(),
    sigma,
    v_robot,
    [q.copy()],   # fresh path_data
    full_geometric_path.copy(),
    full_geometric_polyline.copy(),
    enable_animation=True
)
    distances.append(d)
    collisions.append(c)
    costs.append(cost)
    

print("\n=== MONTE CARLO RESULTS ===")
print("Average Distance:", np.mean(distances))
print("Average Collisions:", np.mean(collisions))
print("Average Cost:", np.mean(costs))

print("All Distances:", distances)
print("All Costs:", costs)
