from run_simulation import run_simulation
import numpy as np
from run_simulation import run_simulation
from APF_RRT_Astar import init_environment
from rrt_planner_main import RRTPlanner

# ===============================
# INITIAL SETUP
# ===============================
q_goal_final        = np.array([20, 10]) #[10, 10]
#q_goal_final        = np.array([0, 20]) #[10, 10]
q_rrt               = np.array([-40.0, -40.0])
q_astar             = np.array([-40.0, -40.0])
v_robot             = 30 #25
# obstacle coordinates 
obstacles_true      = np.array([[-18.0,-10.0], [18,-20], [18, 8], [22,26], [23,15], [-23,15], [5,5], [-40, -30], [15, -10], [10, 5], [-30, 0], [-20, -20], [0, 0], [-35, 3], [-26, -27], [2, -10]])
# kalabalik platform
obstacles_true      = np.array([[-18.0,-10.0], [18,-20], [18, 8], [22,26], [23,15], [-23,15], [5,5], [-40, -30], [15, -10], [10, 5], [-30, 0], [-20, -20], [0, 0], [-35, 3], [-26, -27], [2, -10], [11, -12], [-3, -3], [-45, 0], [-25, -25], [8, 10], [-35, 13], [-21, -17], [21, -10]])
# extra kalabalik platform
# extra_obstacles = np.array([
# [-10, -5], [-12, 4], [-8, 8], [-6, -12], [-4, 15],
# [2, 12], [4, -8], [6, 18], [8, -15], [12, 6],
# [-14, -14], [-16, 10], [-20, 5], [-22, -8], [-24, 12],
# [14, 14], [16, -6], [20, 2], [24, -12], [26, 8]
# ])
# cluster1 = np.array([
# [-22, -22],
# [-21, -19],
# [-19, -23],
# [-18, -21],
# [-23, -18],
# [-20, -24]
# ])
# cluster2 = np.array([
# [18, -42],
# [21, -38],
# [23, -41],
# [19, -37],
# [24, -39],
# [17, -40]
# ])
# obstacles_true = np.vstack((obstacles_true, extra_obstacles,cluster1,cluster2))
sigma               = 0.1  # 10 cm uncertainity
obstacles_noisy     = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)
# obstacle speeds
obstacle_speeds     = np.array([[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1], [-0.1, 0.1], [-0.2, 0.1], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [0.2, 0.1], [-0.1, 0.1], [-0.2, 0.1]])
# kalabalik platform
obstacle_speeds     = np.array([[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1], [-0.1, 0.1], [-0.2, 0.1], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [0.2, 0.1], [-0.1, 0.1], [-0.2, 0.1],[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1], [-0.1, 0.1]])
# extra kalabalik platform
# extra_speeds = np.array([
# [-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1],
# [-0.1, 0.1], [0.2, 0.1], [-0.1, 0.1], [-0.2, 0.1], [-0.2, 0.2],
# [0.1, -0.2], [-0.2, 0.1], [0.2, -0.1], [-0.1, -0.2], [0.1, 0.1],
# [-0.2, 0.2], [0.1, -0.1], [-0.1, 0.2], [0.2, -0.2], [-0.2, 0.1]
# ])
# extra_speeds_cluster = np.array([
# [-0.1, 0.1],
# [-0.2, 0.2],
# [0.1, 0.2],
# [-0.2, -0.1],
# [0.1, -0.1],
# [-0.1, 0.1],

# [0.2, 0.1],
# [-0.1, 0.1],
# [-0.2, 0.1],
# [-0.2, 0.2],
# [0.1, -0.2],
# [-0.2, 0.1]
# ])
# obstacle_speeds = np.vstack((obstacle_speeds, extra_speeds,extra_speeds_cluster))

obstacle_speeds     = obstacle_speeds * 80
vmag_max            = 24.5
vmag_min            = 5
# APF parameters
max_rep_force       = np.inf
path_data_rrt       = [q_rrt.copy()]
path_data_astar     = [q_astar.copy()]
initial_obstacles   = obstacles_true.copy()

rect_obstacles, expanded_rects, eps_expanded_rects, waypoints = init_environment()
# rectangles are moving with the same speed as the elliptical obstacles
rectangle_speeds = np.array([[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1]])
rectangle_speeds = rectangle_speeds * 0


rrt = RRTPlanner(step_size=0.3, max_iterations=2000) #step_size was 0.15 when L was chosen as 0.3.


# ===============================
# ANIMATION SETUP
# ===============================
full_geometric_path_rrt             = np.load("tsp_path.npy")
full_geometric_polyline_rrt         = np.load("tsp_polyline.npy")
q_goal_rrt                          = full_geometric_polyline_rrt[2]


full_geometric_path_astar      = full_geometric_path_rrt
full_geometric_polyline_astar  = full_geometric_polyline_rrt 
q_goal_astar                   = full_geometric_polyline_astar[2]



N_MC = 10
distances_rrt  = []
collisions_rrt = []
costs_rrt = []

distances_astar  = []
collisions_astar = []
costs_astar = []

for i in range(N_MC):
    
    print(f"\n=== RUN {i+1} ===")
    d_rrt, c_rrt, cost_rrt, d_astar, c_astar, cost_astar = run_simulation(
    q_rrt.copy(),
    q_astar.copy(),
    q_goal_rrt.copy(),
    q_goal_astar.copy(),
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
    [q_rrt.copy()],   # fresh path_data
    [q_astar.copy()], # fresh path_data
    full_geometric_path_rrt.copy(),
    full_geometric_polyline_rrt.copy(),
    full_geometric_path_astar.copy(),
    full_geometric_polyline_astar.copy(),
    RRTPlanner(step_size=0.3, max_iterations=2000),  # fresh planner
    enable_animation=True
)
    distances_rrt.append(d_rrt)
    collisions_rrt.append(c_rrt)
    costs_rrt.append(cost_rrt)

    distances_astar.append(d_astar)
    collisions_astar.append(c_astar)
    costs_astar.append(cost_astar)
    

print("\n=== MONTE CARLO RESULTS ===")
print("Average Distance RRT:", np.mean(distances_rrt))
print("Average Collisions: RRT", np.mean(collisions_rrt))
print("Average Cost: RRT", np.mean(costs_rrt))

print("All Distances RRT:", distances_rrt)
print("All Costs RRT:", costs_rrt)

print("Average Distance Astar:", np.mean(distances_astar))
print("Average Collisions: Astar", np.mean(collisions_astar))
print("Average Cost: Astar", np.mean(costs_astar))

print("All Distances Astar:", distances_astar)
print("All Costs Astar:", costs_astar)