from run_simulation import run_simulation
import numpy as np
from TSP_main import bestPath, build_tsp_indices, PSO_TSP, build_full_geometric_path
from run_simulation import run_simulation
from APF_Astar import extract_polyline, init_environment
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
# extra kalabalik platform
extra_obstacles = np.array([
[-10, -5], [-12, 4], [-8, 8], [-6, -12], [-4, 15],
[2, 12], [4, -8], [6, 18], [8, -15], [12, 6],
[-14, -14], [-16, 10], [-20, 5], [-22, -8], [-24, 12],
[14, 14], [16, -6], [20, 2], [24, -12], [26, 8]
])
cluster1 = np.array([
[-22, -22],
[-21, -19],
[-19, -23],
[-18, -21],
[-23, -18],
[-20, -24]
])
cluster2 = np.array([
[18, -42],
[21, -38],
[23, -41],
[19, -37],
[24, -39],
[17, -40]
])
obstacles_true = np.vstack((obstacles_true, extra_obstacles,cluster1,cluster2))
sigma               = 0.1  # 10 cm uncertainity
obstacles_noisy     = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)
# obstacle speeds
obstacle_speeds     = np.array([[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1], [-0.1, 0.1], [-0.2, 0.1], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [0.2, 0.1], [-0.1, 0.1], [-0.2, 0.1]])
# kalabalik platform
obstacle_speeds     = np.array([[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1], [-0.1, 0.1], [-0.2, 0.1], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [0.2, 0.1], [-0.1, 0.1], [-0.2, 0.1],[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1], [-0.1, 0.1]])
# extra kalabalik platform
extra_speeds = np.array([
[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1],
[-0.1, 0.1], [0.2, 0.1], [-0.1, 0.1], [-0.2, 0.1], [-0.2, 0.2],
[0.1, -0.2], [-0.2, 0.1], [0.2, -0.1], [-0.1, -0.2], [0.1, 0.1],
[-0.2, 0.2], [0.1, -0.1], [-0.1, 0.2], [0.2, -0.2], [-0.2, 0.1]
])
extra_speeds_cluster = np.array([
[-0.1, 0.1],
[-0.2, 0.2],
[0.1, 0.2],
[-0.2, -0.1],
[0.1, -0.1],
[-0.1, 0.1],

[0.2, 0.1],
[-0.1, 0.1],
[-0.2, 0.1],
[-0.2, 0.2],
[0.1, -0.2],
[-0.2, 0.1]
])
obstacle_speeds = np.vstack((obstacle_speeds, extra_speeds,extra_speeds_cluster))

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


# ===============================
# ANIMATION SETUP
# ===============================

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

N_MC = 1
distances = []
collisions = []
costs = []

for i in range(N_MC):
    
    print(f"\n=== RUN {i+1} ===")
    #d, c, cost = run_simulation(q,q_goal,q_goal_final,obstacle_speeds,rectangle_speeds,waypoints,obstacles_true,rect_obstacles,obstacles_noisy,expanded_rects,eps_expanded_rects,sigma,v_robot,path_data,full_geometric_path,full_geometric_polyline,rrt,enable_animation=True)
    d, c, cost = run_simulation(
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