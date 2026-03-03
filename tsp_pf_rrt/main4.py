"""
Local Potential-Field Path Planning (elliptical obstacles)

Run to plot a local path from start to goal around elliptical obstacles 
using an attractive + repulsive potential field.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
import math
import random
from TSP_main import *
from APF5 import *
from rrt_planner_main import RRTPlanner

# ===============================
# INITIAL SETUP
# ===============================
q_goal_final        = np.array([20, 10]) #[10, 10]
#q_goal_final        = np.array([0, 20]) #[10, 10]
q                   = np.array([-40.0, -40.0])
v_robot             = 30 #25
# obstacle coordinates 
obstacles_true      = np.array([[-18.0,-10.0], [18,-20], [18, 8], [22,26], [23,15], [-23,15], [5,5], [-40, -30], [15, -10], [10, 5], [-30, 0], [-20, -20], [0, 0], [-35, 3], [-26, -27], [2, -10]])
sigma               = 0.1  # 10 cm uncertainity
obstacles_noisy     = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)
# obstacle speeds
obstacle_speeds     = np.array([[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1], [-0.1, 0.1], [-0.2, 0.1], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [0.2, 0.1], [-0.1, 0.1], [-0.2, 0.1]])
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

rrt = RRTPlanner(step_size=0.3, max_iterations=2000) #step_size was 0.15 when L was chosen as 0.3.

# ===============================
# ANIMATION SETUP
# ===============================

fig, ax = plt.subplots(figsize=(8, 8), dpi = 50)
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_title("Dynamic Obstacle Avoidance Animation")
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.grid(color='gray', linestyle='--', linewidth=0.4, alpha=0.7)

path_line, = ax.plot([], [], 'r-', linewidth=2)
robot_dot, = ax.plot([], [], 'ro', markersize=6)
true_scatter = ax.scatter([], [], c='black', s=40)
noisy_scatter = ax.scatter([], [], c='red', s=40)
goal_dot, = ax.plot(q_goal_final[0], q_goal_final[1], 'go', markersize=8, label="Goal")

ellipse_patches         = []
rect_patches            = []
expanded_rect_patches   = []
tolerance               = 1
time                    = int(0)



# first iteration
traveller_sm, route_sm              = bestPath(waypoints, q, q_goal_final, rect_obstacles, expanded_rects, eps_expanded_rects)
N, START, END, WAYPOINTS, nodes     = build_tsp_indices(q, q_goal_final, waypoints)

best_path = None
best_cost = float("inf")

for run in range(1):
    path, cost = PSO_TSP(
        traveller_sm,
        START=START,
        END=END,
        WAYPOINTS=WAYPOINTS,
        n_particles=250,
        n_iter=5000
    )

    print(f"Run {run+1} cost = {cost}")   # 👈 istediğin şey bu

    if cost < best_cost:
        best_cost = cost
        best_path = path[:]

print("Best distance found by PSO (multi-run):", best_cost)
print("Best path:", best_path)
#best_path, best_cost                = PSO_TSP(traveller_sm, START=START, END=END, WAYPOINTS=WAYPOINTS, n_particles=250, n_iter=5000)

full_geometric_path                 = build_full_geometric_path(best_path,route_sm)
full_geometric_polyline             = extract_polyline(full_geometric_path, waypoints)
q_goal                              = full_geometric_polyline[2]


# ===============================
# RUN ANIMATION
# ===============================
stop_counter    = 0
ani             = None
goals_achieved_so_far = []
total_distance = 0
collision_counter = 0
done  = False #added for MCsim
mystates = {
    "q": q,
    "obstacle_speeds": obstacle_speeds,
    "ani": ani,
    "time": time,
    "q_goal_final": q_goal_final,
    "q_goal": q_goal,
    "waypoints": waypoints,
    "obstacles_true": obstacles_true,
    "rectangle_speeds": rectangle_speeds,
    "rect_obstacles": rect_obstacles,
    "obstacles_noisy": obstacles_noisy,
    "sigma": sigma,
    "v_robot": v_robot,
    "path_data": path_data,
    "path_line": path_line,
    "robot_dot": robot_dot,
    "true_scatter": true_scatter,
    "noisy_scatter": noisy_scatter,
    "goal_dot": goal_dot,
    "ellipse_patches": ellipse_patches,
    "rect_patches": rect_patches,
    "expanded_rect_patches": expanded_rect_patches,
    "ax": ax,
    "expanded_rects": expanded_rects,
    "eps_expanded_rects": eps_expanded_rects,
    "full_geometric_path": full_geometric_path,
    "stop_counter": stop_counter,
    "goals_achieved_so_far": goals_achieved_so_far,
    "rrt": rrt,
    "full_geometric_polyline": full_geometric_polyline,
    "total_distance": total_distance,
    "collision_counter": collision_counter,
    "done": done}

def init_anim():
    return init(path_line, robot_dot, true_scatter, noisy_scatter, goal_dot)

ani = animation.FuncAnimation(
    fig,
    update,   # <-- PARANTEZ YOK
    frames=400,
    fargs=(mystates,),
    init_func=init_anim,
    interval=40,
    blit=False
)

mystates["ani"] = ani
plt.show()

distance = mystates["total_distance"] 
collisions = mystates["collision_counter"]
cost = distance + 100 * collisions

print("FINAL DISTANCE:", mystates["total_distance"])
print("FINAL COLLISIONS:", mystates["collision_counter"])
print("FINAL COST:", cost)
