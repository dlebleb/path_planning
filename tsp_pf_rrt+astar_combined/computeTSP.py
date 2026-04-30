import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from TSP_main import bestPath, build_tsp_indices, PSO_TSP, build_full_geometric_path
from APF_RRT_Astar import extract_polyline, init_environment


# ===============================
# INITIAL SETUP
# ===============================

q_start = np.array([-40.0, -40.0])
q_goal_final = np.array([20, 10])

rect_obstacles, expanded_rects, eps_expanded_rects, waypoints = init_environment()

#11-waypoint station
waypoints = np.array([
    [-30, -35],
    [25, -40],
    [-48, 20],
    [7, 30],
    [15, -2],
    [0, 5],
    [-10, 0],
    [-20, -20],
    # [0, 40],
    # [40, -20],
    # [-5, 20]
])

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
sigma = 0.1
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

# elliptical obstacle dimensions
a0 = 2.0 # major axis (along velocity direction)
b0 = 1.0 # minor axis (perpendicular to velocity direction)
alpha = 0.2   # major scaling (large)
beta  = 0.1   # minor scaling (small)
# extra kalabalik platform
sizes = np.array([
1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1, 1.0,
1.2, 1.8, 2.0, 1.8, 1.9, 1.4, 1.5, 1.8,
1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1, 1.0,

# extra 20
1.3, 1.7, 1.1, 1.4, 0.9,
1.6, 1.2, 1.0, 1.5, 1.8,
2.0, 1.7, 1.3, 1.6, 1.2,
0.9, 1.4, 1.1, 1.9, 1.5
])
extra_sizes_cluster = np.array([
1.6, 1.4, 1.8, 1.5, 1.7, 1.3,
1.9, 1.5, 1.6, 1.8, 1.4, 1.7
])
sizes = np.concatenate((sizes, extra_sizes_cluster))
# incorporate static size                          
a_base = a0 * sizes 
b_base = b0 * sizes
a_max = 3
b_max = 1.5

a = np.zeros(obstacles_true.shape[0])
b = np.zeros(obstacles_true.shape[0])
theta = np.zeros(obstacles_true.shape[0])

for i, obs in enumerate(obstacle_speeds):
    vx, vy = obs
    vmag = np.sqrt(vx**2 + vy**2) 
    theta[i] = np.degrees(np.arctan2(vy, vx))
    a[i] = a_base[i] + alpha * vmag # major axis (velocity direction)
    b[i] = b_base[i] + beta  * vmag # minor axis (perpendicular)
    # clamp to max
    a[i] = min(a[i], a_max)
    b[i] = min(b[i], b_max)


# ===============================
# COMPUTE TSP PATH
# ===============================

traveller_sm, route_sm = bestPath(
    waypoints,
    q_start,
    q_goal_final,
    rect_obstacles,
    expanded_rects,
    eps_expanded_rects
)

N, START, END, WAYPOINTS, nodes = build_tsp_indices(
    q_start,
    q_goal_final,
    waypoints
)

best_path = None
best_cost = float("inf")

for run in range(10):

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

print("Best TSP cost:", best_cost)

full_geometric_path = build_full_geometric_path(best_path, route_sm)
full_geometric_polyline = extract_polyline(full_geometric_path, waypoints)

# ===============================
# SAVE PATH
# ===============================

np.save("tsp_path.npy", full_geometric_path)
np.save("tsp_polyline.npy", full_geometric_polyline)

print("TSP path saved!")

# ===============================
# PLOT
# ===============================

fig, ax = plt.subplots(figsize=(6,6))

ax.set_xlim(-50,50)
ax.set_ylim(-50,50)
ax.set_title("TSP")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_aspect('equal')
ax.grid(True)

# start & goal
ax.plot(q_start[0], q_start[1], 'ro', label="Start")
ax.plot(q_goal_final[0], q_goal_final[1], 'go', label="Goal")

# stations
ax.scatter(
    waypoints[:,0],
    waypoints[:,1],
    color='purple',
    marker='*',
    s=120,
    label="Stations"
)

#ax.scatter(obstacles_true[:,0], obstacles_true[:,1], marker='x')
#ax.scatter(obstacles_noisy[:,0], obstacles_noisy[:,1], marker='o', alpha=0.5)
ax.scatter(obstacles_true[:,0], obstacles_true[:,1], marker='o', alpha=0.5)

# True obstacles as ellipses
for i in range(obstacles_true.shape[0]):
    x0, y0 = obstacles_true[i]
    ell = Ellipse(
        (x0, y0),               
        width=2*a[i],
        height=2*b[i],
        angle=theta[i],
        edgecolor='black',
        facecolor='cyan',
        alpha=0.15,       # şeffaflık
        linestyle='--',
        linewidth=1.2
    )
    ax.add_patch(ell)


# Dikdörtgen engeller and the circle
for x, y, w, h in rect_obstacles:
    rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=2)
    ax.add_patch(rect)

# Genişletilmiş rectangle'lar (buffered)
for x, y, w, h in expanded_rects:
    rect2 = plt.Rectangle((x, y), w, h, fill=False, linewidth=1.5,
                        edgecolor='orange', linestyle='--')
    ax.add_patch(rect2)

# TSP path
path_arr = np.array(full_geometric_path)

ax.plot(
    path_arr[:,0],
    path_arr[:,1],
    color='pink',
    linewidth=3,
    label="TSP path"
)

ax.legend()

plt.savefig("TSP_path.pdf")
plt.show()