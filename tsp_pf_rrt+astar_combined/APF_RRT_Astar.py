"""
Local Potential-Field Path Planning (elliptical obstacles)

Run to plot a local path from start to goal around elliptical obstacles 
using an attractive + repulsive potential field.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
from TSP_main import *
from datetime import datetime
from damla_Astar import *
from TSP_Rachel import Point
from rrt_planner_main import RRTPlanner

# ===============================
# TSP FUNCTIONS and SETUP
# ===============================
def init(path_line_rrt,path_line_astar,robot_dot_rrt,robot_dot_astar,true_scatter,noisy_scatter, goal_dot):
    path_line_rrt.set_data([], [])
    robot_dot_rrt.set_data([], [])
    path_line_astar.set_data([], [])
    robot_dot_astar.set_data([], [])
    true_scatter.set_offsets(np.empty((0, 2)))
    noisy_scatter.set_offsets(np.empty((0, 2)))
    return path_line_rrt, path_line_astar,robot_dot_rrt, robot_dot_astar,true_scatter, noisy_scatter, goal_dot

def init_environment():
    """
    Initializes rectangular obstacles, expanded buffers, and waypoints.
    """

    # Rectangular large obstacles
    # x_min, y_min, width, height
    rect_obstacles = [
        [-40, -30, 12, 3],   # truck: narrow and long
        [20, -35, 6, 3],     # van / minibus
        [-45, 15, 10, 4],    # large vehicle
        [5, 25, 3, 2],       # bicycle / small vehicle
        [-6, -6.5, 12, 3],   # truck
        [-13.5, -23.0, 12, 6]
    ] # not an np.array

    # Expanded rectangles (safety buffer)
    expanded_rects = []
    for x, y, w, h in rect_obstacles:
        expanded_rects.append([
            x - 1,
            y - 1,
            w + 2,
            h + 2
        ])

    # Slightly shrunken expanded rectangles
    eps = 0.001
    eps_expanded_rects = []
    for x, y, w, h in rect_obstacles:
        eps_expanded_rects.append([
            x - 1 + eps,
            y - 1 + eps,
            w + 2 - 2*eps,
            h + 2 - 2*eps
        ])

    # Waypoints (stations)
    waypoints = np.array([
        [-30, -35],
        [25, -40],
        [-48, 20],
        [7, 30],
        [15, -2],
        [0, 5],
        [-10, 0],
        [-20, -20],
    ])

    return rect_obstacles, expanded_rects, eps_expanded_rects, waypoints

def init_environment_waypoints():
    """
    Initializes waypoints.
    """
    # Waypoints (stations)
    waypoints = np.array([
        [-30, -35],
        [25, -40],
        [-48, 20],
        [7, 30],
        [15, -2],
        [0, 5],
        [-10, 0],
        [-20, -20],
    ])

    return waypoints

# function for updating expanded rects from rect_obstacles
def compute_expanded_rects(rect_obstacles, buffer=1.0):
    """
    rect_obstacles: (N,4) array [x, y, w, h]
    buffer: safety margin (meters)
    """
    expanded = np.zeros_like(rect_obstacles)
    rect_obstacles = np.array(rect_obstacles)
    expanded[:, 0] = rect_obstacles[:, 0] - buffer
    expanded[:, 1] = rect_obstacles[:, 1] - buffer
    expanded[:, 2] = rect_obstacles[:, 2] + 2*buffer
    expanded[:, 3] = rect_obstacles[:, 3] + 2*buffer
    return expanded

def remove_visited_waypoints(waypoints, visited_goals, tol=1.0):
    """
    waypoints: (N,2) np.array
    visited_goals: list of np.array shape (2,)  OR  np.array shape (M,2)
    tol: distance threshold to consider "same point"
    returns: filtered waypoints (K,2)
    """
    if visited_goals is None or len(visited_goals) == 0:
        return waypoints

    visited = np.array(visited_goals)  # (M,2)

    keep_mask = np.ones(len(waypoints), dtype=bool)
    for i, wp in enumerate(waypoints):
        dists = np.linalg.norm(visited - wp, axis=1)
        if np.min(dists) < tol:
            keep_mask[i] = False

    return waypoints[keep_mask]

# removing rectangle corners from the full_geometric_path
def extract_polyline(full_geometric_path, waypoints):

    polyline = []

    # 1️⃣ başlangıcı iki kez ekle
    start = full_geometric_path[0]
    polyline.append(start)
    polyline.append(start)

    # 2️⃣ path sırasına göre waypointleri ekle
    for p in full_geometric_path:
        for wp in waypoints:
            if np.array_equal(p, wp):
                polyline.append(p)

    # 3️⃣ final goal ekle
    final_goal = full_geometric_path[-1]
    polyline.append(final_goal)

    return np.array(polyline)


# new function for calculating the projection of q on the closest line segment coming from the global path
# t = AQ nun AB uzerindeki projection'in uzunlugu / AB nin uzunlugu 

def closest_point_and_tangent_on_polyline(q, full_geometric_path, q_goal, flag = 1):
    """
    q: robot position
    path_pts: full_geometric_path (P)
    q_goal_final: optional, tangent yönünü final goal'a göre düzeltebilmek için

    Returns:
      p_proj: q'ya polyline üzerinde en yakin nokta
      t_hat:  o segmentin unit tangent yönü (direction of the vector AB)
      seg_idx: projeksiyonun düştüğü segment index'i
      t_clamped: float, AQ projeksiyon AB segment'inde nereye denk geliyor? (between 0 and 1)
    """
    P = np.array(full_geometric_path)

    best_d = float("inf")
    best_p = None
    best_t_hat = None

    # her segment için tara
    for i in range(len(P) - 1):
        A = P[i]
        B= P[i+1]
        AB = B - A
        AQ = q - A

        AB2 = np.linalg.norm(AB)**2  # ||ab||^2
        if AB2 < 1e-12:
            continue

        # 1) projection parameter (infinite line)
        # t = AQ.AB/norm(AB)^2
        t = float(np.dot(AQ, AB) / AB2)

        # 2) clamp to segment
        t_clamped = max(0.0, min(1.0, t))

        # 3) find projected point and calculate the distance to projected point
        p = A+ t_clamped * AB
        d = calculateDistance(q,p)

        # 5) keep best
        if d < best_d:
            best_d = d
            best_p = p

            # tangent (unit)
            norm_AB = np.linalg.norm(AB)
            best_t_hat = AB / norm_AB

    # fallback: path çok bozuksa
    if best_p is None:
        best_p = P[0].copy()
        best_t_hat = np.array([1.0, 0.0])

    current_goal = q_goal   # bu aktif station olabilir veya final olabilir
    
    # İSTEĞE BAĞLI: tangent yönünü final goal'a doğru seç
    if current_goal is not None and flag == 1:
        to_goal = current_goal - q
        if np.dot(best_t_hat, to_goal) < 0:
            best_t_hat = -best_t_hat

    return best_p, best_t_hat

def pull_tangent_force(q, p, t_hat):
    k_pull, k_tan = 15.0, 60.0 
    F_pull = -k_pull * (q - p)
    F_tan = k_tan * t_hat
    return F_pull + F_tan


# ===============================
# FORCE FUNCTIONS
# ===============================
def attractive_force(q, q_goal):
    k_att, k_rep, d0, dt = 100.0, 10.0, 10.0, 0.01 #default k_att is 60.
    F_att = -k_att * (q - q_goal)/np.linalg.norm(q-q_goal)
    return F_att

def repulsive_force(q, obstacles_noisy, obstacle_speeds):
    a0 = 2.0 # major axis (along velocity direction)
    b0 = 1.0 # minor axis (perpendicular to velocity direction)
    alpha = 0.2   # major scaling (large)
    beta  = 0.1   # minor scaling (small)
    # each obstacle has a size factor: 1 = normal, >1 = large, <1 = small
    sizes = np.array([1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1, 1.0, 1.2, 1.8, 2.0, 1.8, 1.9, 1.4, 1.5, 1.8])
    #kalabalik platform
    sizes = np.array([1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1, 1.0, 1.2, 1.8, 2.0, 1.8, 1.9, 1.4, 1.5, 1.8, 1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1, 1.0])
    # extra kalabalik platform
    # sizes = np.array([
    # 1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1, 1.0,
    # 1.2, 1.8, 2.0, 1.8, 1.9, 1.4, 1.5, 1.8,
    # 1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1, 1.0,

    # # extra 20
    # 1.3, 1.7, 1.1, 1.4, 0.9,
    # 1.6, 1.2, 1.0, 1.5, 1.8,
    # 2.0, 1.7, 1.3, 1.6, 1.2,
    # 0.9, 1.4, 1.1, 1.9, 1.5
    # ])
    # extra_sizes_cluster = np.array([
    # 1.6, 1.4, 1.8, 1.5, 1.7, 1.3,
    # 1.9, 1.5, 1.6, 1.8, 1.4, 1.7
    # ])
    # sizes = np.concatenate((sizes, extra_sizes_cluster))
    # incorporate static size                          
    a_base = a0 * sizes 
    b_base = b0 * sizes
    a_max = 3
    b_max = 1.5
    k_att, k_rep, d0, dt = 10.0, 10.0, 1.0, 0.01 #d0 was 10.0

    F_rep_total = np.array([0.0, 0.0])
    for i, obs in enumerate(obstacles_noisy):
        vx, vy = obstacle_speeds[i]
        vmag = np.sqrt(vx**2 + vy**2) 
        theta = np.arctan2(vy, vx + 1e-12)  # obstacle motion direction

        # dynamic scaling with speed
        a = a_base[i] + alpha * vmag # major axis (velocity direction)
        b = b_base[i] + beta  * vmag# minor axis (perpendicular)

        # clamp to max
        a = min(a, a_max)
        b = min(b, b_max)

        # move obstacle center to origin and transform
        obs_x, obs_y = obs[0], obs[1]
        # x2, y2 = -obs_x/a, -obs_y/b

        eps = 1e-12
        # move robot center to origin and transform
        q_x, q_y = (q[0]-obs_x)/(a+eps), (q[1]-obs_y)/(b+eps)

        # distance of the robot to origin -1
        dE = np.sqrt(q_x**2 + q_y**2) - 1

        # dynamic avoidance boundary
        # d0_i = d0 * max(a, b)

        if dE < d0:
            if dE >= 0.5: 
                F_mag = 0.04 #k_rep * (1/4 - 1/d0) * (1/4**2) #0.04
                #F_mag = k_rep * (1/dE - 1/d0) * (1/dE**2)
                # yön vektörü (normalize edilmiş fark)
                grad_Dq = (q - obs)/np.linalg.norm(q - obs)
                # toplam kuvvet
                F_rep = F_mag * grad_Dq
            else:
                F_mag = 400 #k_rep * (1/0.1 - 1/d0) * (1/0.1**2) #F_mag is decreased tp 100 from 1000.
                grad_Dq = (q - obs)/np.linalg.norm(q - obs)
                F_rep = F_mag * grad_Dq

        else:
            F_rep = np.array([0.0, 0.0])
        F_rep_total += F_rep
    return F_rep_total

def repulsive_force_rectangles(q, rect_obstacles):
    k_rep_rectangle, d0 = 50.0, 20.0 #k_rep is changed to 50 to avoid collisions with rectangles, d0 20.

    F_rep_total_rectangle = np.array([0.0, 0.0])
    for obs in rect_obstacles:
        center, radius = computeRectangleCircumcircle(obs)
        
        obs_x, obs_y = center[0], center[1]
        q_x, q_y = (q[0]-obs_x), (q[1]-obs_y)

        # distance of the robot to the center of rectangle
        dE = np.sqrt(q_x**2 + q_y**2) - radius

        if dE < d0:
            F_mag = k_rep_rectangle * (1/dE - 1/d0) * (1/dE**2)
            # yön vektörü (normalize edilmiş fark)
            grad_Dq = (q - center) / (dE + 1e-12)
            # toplam kuvvet
            F_rep = F_mag * grad_Dq
        else:
            F_rep = np.array([0.0, 0.0])
        F_rep_total_rectangle += F_rep
    return F_rep_total_rectangle

def potential(q, q_goal, obstacles_noisy, obstacle_speeds):
    k_att, k_rep, d0, dt = 10.0, 10.0, 10.0, 0.01
    a0 = 2.0 # major axis (along velocity direction)
    b0 = 1.0 # minor axis (perpendicular to velocity direction)
    alpha = 0.2   # major scaling (large)
    beta  = 0.1   # minor scaling (small)
    # each obstacle has a size factor: 1 = normal, >1 = large, <1 = small
    sizes = np.array([1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1, 1.0, 1.2, 1.8, 2.0, 1.8, 1.9, 1.4, 1.5, 1.8])
    #kalabalik platform
    sizes = np.array([1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1, 1.0, 1.2, 1.8, 2.0, 1.8, 1.9, 1.4, 1.5, 1.8, 1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1, 1.0])
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

    U_rep_total = 0
    U_att = 0.5 * k_att * np.linalg.norm(q - q_goal)**2
    
    for i, obs in enumerate(obstacles_noisy):
        vx, vy = obstacle_speeds[i]
        vmag = np.sqrt(vx**2 + vy**2) 
       
        a = a_base[i] + alpha * vmag 
        b = b_base[i] + beta  * vmag 
        a = min(a, a_max)
        b = min(b, b_max)
       
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

def total_force(q, q_goal, obstacles_noisy, obstacle_speeds, rect_obstacles):
    """
    Calculate the total force on the robot
    """
    F_att = attractive_force(q, q_goal)
    F_rep = repulsive_force(q, obstacles_noisy, obstacle_speeds)
    F_rep_rect = repulsive_force_rectangles(q, rect_obstacles)
    return F_att + F_rep + F_rep_rect

def apply_stochastic_maneuver(obstacle_speeds, maneuver_prob=0.25,
                              magnitude_sigma=0.05, turn_sigma=0.02):
    """
    Modify obstacle velocities by adding stochastic maneuvers.
    """
    vmag_max = 24.5
    vmag_min = 5
    new_speeds = obstacle_speeds.copy()

    for i in range(len(new_speeds)):
        vx, vy = new_speeds[i]
        vmag = np.sqrt(vx**2 + vy**2)
        
        # 1) Small continuous jitter (small speed increase & decrease, 5%) (1 ± 0.15)
        # [1 ± 0.15] contains 99.7% 
        # vmag *= np.random.normal(1, magnitude_sigma)
        scale = np.random.normal(1, magnitude_sigma)
        scale = np.clip(scale, 0.85, 1.15)   # 3σ limit
        vmag *= scale
        vmag = np.clip(vmag, vmag_min, vmag_max)
       
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

def apply_stochastic_maneuver_rectangles(rectangle_speeds, maneuver_prob=0.5,
                              magnitude_sigma=0.05, turn_sigma=0.02):
    """
    Modify obstacle velocities by adding stochastic maneuvers.
    """
    vmag_max = 24.5
    vmag_min = 5
    new_speeds_rects = rectangle_speeds.copy()

    for i in range(len(new_speeds_rects)):
        vx, vy = new_speeds_rects[i]
        vmag = np.sqrt(vx**2 + vy**2)
        
        # 1) Small continuous jitter (small speed increase & decrease, 5%) (1 ± 0.15)
        # [1 ± 0.15] contains 99.7% 
        # vmag *= np.random.normal(1, magnitude_sigma)
        scale = np.random.normal(1, magnitude_sigma)
        scale = np.clip(scale, 0.85, 1.15)   # 3σ limit
        vmag *= scale
        vmag = np.clip(vmag, vmag_min, vmag_max)
       
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

        new_speeds_rects[i] = [vx, vy]

    return new_speeds_rects

def is_collision_check(q, obstacles_noisy, obstacle_speeds):
    collided_indices = []
    counter = 0
    a0 = 2.0 # major axis (along velocity direction)
    b0 = 1.0 # minor axis (perpendicular to velocity direction)
    alpha = 0.2   # major scaling (large)
    beta  = 0.1   # minor scaling (small)
    # each obstacle has a size factor: 1 = normal, >1 = large, <1 = small
    sizes = np.array([1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1, 1.0, 1.2, 1.8, 2.0, 1.8, 1.9, 1.4, 1.5, 1.8])
    #kalabalik platform
    sizes = np.array([1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1, 1.0, 1.2, 1.8, 2.0, 1.8, 1.9, 1.4, 1.5, 1.8, 1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1, 1.0])
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

    for i, obs in enumerate(obstacles_noisy):
        vx, vy = obstacle_speeds[i]
        vmag = np.sqrt(vx**2 + vy**2) 
        theta = np.arctan2(vy, vx + 1e-12)
               
        a = a_base[i] + alpha * vmag 
        b = b_base[i] + beta  * vmag
        a = min(a, a_max)
        b = min(b, b_max)
        
       # move obstacle center to origin and transform
        obs_x, obs_y = obs[0], obs[1]
        # x2, y2 = -obs_x/a, -obs_y/b

        eps = 1e-12
        # move robot center to origin and transform
        q_x, q_y = (q[0]-obs_x)/(a+eps), (q[1]-obs_y)/(b+eps)

        # distance of the robot to origin -1
        dE = np.sqrt(q_x**2 + q_y**2) - 1
        
        # COLLISION condition: robot is inside the ellipse
        if dE < 0:
            counter += 1
            collided_indices.append(i)
            print(dE)
    
    return counter, collided_indices

# ===============================
# UPDATE FUNCTION
# ===============================
def update(frame,mystates):
# def update(frame,q,obstacle_speeds,ani,time,q_goal_final,q_goal,waypoints,obstacles_true,rectangle_speeds,rect_obstacles,obstacles_noisy,sigma,
#            v_robot,path_data,path_line, robot_dot, true_scatter, noisy_scatter, goal_dot, ellipse_patches, rect_patches, expanded_rect_patches,
#            ax,expanded_rects, eps_expanded_rects, full_geometric_path,stop_counter):
    #from TSP import bestPath,build_tsp_indices,PSO_TSP,build_full_geometric_path
    q_rrt                 = mystates["q_rrt"]
    q_astar               = mystates["q_astar"]
    obstacle_speeds       = mystates["obstacle_speeds"]
    ani                   = mystates["ani"]
    time                  = mystates["time"]
    q_goal_final          = mystates["q_goal_final"]
    q_goal_rrt            = mystates["q_goal_rrt"]
    q_goal_astar          = mystates["q_goal_astar"]
    waypoints             = mystates["waypoints"]
    obstacles_true        = mystates["obstacles_true"]
    rectangle_speeds      = mystates["rectangle_speeds"]
    rect_obstacles        = mystates["rect_obstacles"]
    obstacles_noisy       = mystates["obstacles_noisy"]
    sigma                 = mystates["sigma"]
    v_robot               = mystates["v_robot"]
    path_data_rrt         = mystates["path_data_rrt"]
    path_data_astar       = mystates["path_data_astar"]
    path_line_rrt         = mystates["path_line_rrt"]
    path_line_astar       = mystates["path_line_astar"]
    robot_dot_rrt         = mystates["robot_dot_rrt"]
    robot_dot_astar       = mystates["robot_dot_astar"]
    true_scatter          = mystates["true_scatter"]
    noisy_scatter         = mystates["noisy_scatter"]
    goal_dot              = mystates["goal_dot"]
    ellipse_patches       = mystates["ellipse_patches"]
    rect_patches          = mystates["rect_patches"]
    expanded_rect_patches = mystates["expanded_rect_patches"]
    ax                    = mystates["ax"]
    expanded_rects        = mystates["expanded_rects"]
    eps_expanded_rects    = mystates["eps_expanded_rects"]
    full_geometric_path_rrt       = mystates["full_geometric_path_rrt"]
    full_geometric_path_astar     = mystates["full_geometric_path_astar"]
    stop_counter_rrt              = mystates["stop_counter_rrt"]
    stop_counter_astar            = mystates["stop_counter_astar"]
    goals_achieved_so_far_rrt     = mystates["goals_achieved_so_far_rrt"]
    goals_achieved_so_far_astar   = mystates["goals_achieved_so_far_astar"]
    rrt                           = mystates["rrt"]
    full_geometric_polyline_rrt   = mystates["full_geometric_polyline_rrt"]
    full_geometric_polyline_astar = mystates["full_geometric_polyline_astar"]
    total_distance_rrt            = mystates["total_distance_rrt"]
    total_distance_astar          = mystates["total_distance_astar"]
    collision_counter_rrt         = mystates["collision_counter_rrt"]
    collision_counter_astar       = mystates["collision_counter_astar"]
    done                        = mystates["done"]
    done_rrt                    = mystates["done_rrt"]
    done_astar                  = mystates["done_astar"]
    finish_counter_rrt        = mystates["finish_counter_rrt"]
    finish_counter_astar      = mystates["finish_counter_astar"]


    #path_line,robot_dot,true_scatter,noisy_scatter, goal_dot = init(path_line,robot_dot,true_scatter,noisy_scatter, goal_dot)
    tolerance = 1
    dt      = 0.01
    a0 = 2.0 # major axis (along velocity direction)
    b0 = 1.0 # minor axis (perpendicular to velocity direction)
    alpha = 0.2   # major scaling (large)
    beta  = 0.1   # minor scaling (small)
    # each obstacle has a size factor: 1 = normal, >1 = large, <1 = small
    sizes = np.array([1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1, 1.0, 1.2, 1.8, 2.0, 1.8, 1.9, 1.4, 1.5, 1.8])
    #kalabalik platform
    sizes = np.array([1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1, 1.0, 1.2, 1.8, 2.0, 1.8, 1.9, 1.4, 1.5, 1.8, 1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1, 1.0])
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


    time = time + 1
    if (time % 1000 == 0): ## path is updated every 10s.
        # initialize the waypoints again 
        waypoints = init_environment_waypoints()
        
        # remove visited waypoints from the waypoint list
        waypoints_rrt   = remove_visited_waypoints(waypoints,goals_achieved_so_far_rrt, tol = tolerance)
        waypoints_astar = remove_visited_waypoints(waypoints,goals_achieved_so_far_astar, tol = tolerance)

        traveller_sm, route_sm = bestPath(waypoints_rrt, q_rrt, q_goal_final, rect_obstacles, expanded_rects, eps_expanded_rects)
        N, START, END, WAYPOINTS, nodes = build_tsp_indices(q_rrt, q_goal_final, waypoints_rrt)
        best_path, best_cost = PSO_TSP(traveller_sm, START=START, END=END, WAYPOINTS=WAYPOINTS, n_particles=500, n_iter=300)
        full_geometric_path_rrt = build_full_geometric_path(best_path,route_sm)
        full_geometric_polyline_rrt = extract_polyline(full_geometric_path_rrt, waypoints_rrt)

        traveller_sm, route_sm = bestPath(waypoints_astar, q_astar, q_goal_final, rect_obstacles, expanded_rects, eps_expanded_rects)
        N, START, END, WAYPOINTS, nodes = build_tsp_indices(q_astar, q_goal_final, waypoints_astar)
        best_path, best_cost = PSO_TSP(traveller_sm, START=START, END=END, WAYPOINTS=WAYPOINTS, n_particles=500, n_iter=300)
        full_geometric_path_astar = build_full_geometric_path(best_path,route_sm)
        full_geometric_polyline_astar = extract_polyline(full_geometric_path_astar, waypoints_astar)

        # RESET HERE
        stop_counter_rrt   = 0
        stop_counter_astar = 0

        q_goal_rrt   = full_geometric_polyline_rrt[2+stop_counter_rrt]
        q_goal_astar = full_geometric_polyline_astar[2+stop_counter_astar]

    # 0) random maneuver (new speeds)
    obstacle_speeds = apply_stochastic_maneuver(obstacle_speeds)
    #rectangle_speeds = apply_stochastic_maneuver_rectangles(rectangle_speeds)
    # --- 1) Move obstacles ---
    obstacles_true[:,0] += obstacle_speeds[:,0] * dt
    obstacles_true[:,1] += obstacle_speeds[:,1] * dt

   # Move rectangles (ONLY x, y)
    for i in range(len(rect_obstacles)):
        rect_obstacles[i][0] += rectangle_speeds[i][0] * dt
        rect_obstacles[i][1] += rectangle_speeds[i][1] * dt

    # --- Recompute expanded rectangles ---
    expanded_rects = compute_expanded_rects(rect_obstacles, buffer=1.0)

    # --- 2) Noise sample ---
    obstacles_noisy[:] = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)

    # --- 3) Robot force ---
    F_rrt     = total_force(q_rrt, q_goal_rrt, obstacles_noisy, obstacle_speeds, rect_obstacles) # vektor toplami zaten sana yonu verir.
    F1 = repulsive_force(q_rrt,obstacles_noisy,obstacle_speeds) + repulsive_force_rectangles(q_rrt,rect_obstacles)
   
    F_astar     = total_force(q_astar, q_goal_astar, obstacles_noisy, obstacle_speeds, rect_obstacles) # vektor toplami zaten sana yonu verir.

    # new addition for the global path
    p_rrt, t_hat_rrt = closest_point_and_tangent_on_polyline(q_rrt, full_geometric_path_rrt, q_goal_rrt, flag = 0)
    p_astar, t_hat_astar = closest_point_and_tangent_on_polyline(q_astar, full_geometric_path_astar, q_goal_astar, flag = 0)
    F_rrt = F_rrt + pull_tangent_force(q_rrt, p_rrt, t_hat_rrt)
    F_astar = F_astar + pull_tangent_force(q_astar, p_astar, t_hat_astar)

    # --- 4) Robot motion ---
    # ================= RRT MOTION =================
    if not done_rrt:

        F_rrt_norm = np.linalg.norm(F_rrt)

        if F_rrt_norm < 1e-8:
            direction_rrt = np.zeros_like(F_rrt)
        else:
            direction_rrt = F_rrt / F_rrt_norm

        q_old_rrt = q_rrt.copy()

        L_rrt = 1.0
        q_target_local_rrt = q_rrt + L_rrt * direction_rrt * 0.5

        rrt_path = rrt.plan_path(
            start=Point(q_rrt[0], q_rrt[1]),
            goal=Point(q_target_local_rrt[0], q_target_local_rrt[1]),
            obstacles_noisy=obstacles_noisy,
            rect_obstacles=rect_obstacles,
            obstacle_speeds=obstacle_speeds,
            F=F1,
            expanded_rects=expanded_rects
        )

        if rrt_path is not None and len(rrt_path) > 1:

            rrt_path = rrt.smooth_moving_average(rrt_path)

            target_rrt = np.array([rrt_path[1].x, rrt_path[1].y])
            vector_to_target_rrt = target_rrt - q_rrt
            dist_rrt = np.linalg.norm(vector_to_target_rrt)

            q_new = q_rrt + v_robot * dt * vector_to_target_rrt / dist_rrt

            p1 = Point(q_rrt[0], q_rrt[1])
            p2 = Point(q_new[0], q_new[1])

            if not rrt._is_path_clear(p1, p2, obstacles_noisy, obstacle_speeds):
                q_new = np.array([rrt_path[1].x, rrt_path[1].y])

            q_rrt = q_new

        else:
            q_rrt = q_rrt + min(v_robot * dt, rrt.step_size) * direction_rrt

        path_data_rrt.append(q_rrt.copy())

        step_distance_rrt = np.linalg.norm(q_rrt - q_old_rrt)
        total_distance_rrt += step_distance_rrt


   # ================= ASTAR MOTION =================
    if not done_astar:

        F_astar_norm = np.linalg.norm(F_astar)

        if F_astar_norm < 1e-8:
            direction_astar = np.zeros_like(F_astar)
        else:
            direction_astar = F_astar / F_astar_norm

        q_old_astar = q_astar.copy()

        L_astar = 3.0
        q_target_local_astar = q_astar + L_astar * direction_astar

        xmin, xmax = q_astar[0]-5, q_astar[0]+5
        ymin, ymax = q_astar[1]-5, q_astar[1]+5

        q_target_local_astar = np.clip(q_target_local_astar, [xmin, ymin], [xmax, ymax])

        astar_path = astar_local(
            start_state=tuple(q_astar),
            goal_state=tuple(q_target_local_astar),
            heuristic_func=euclidean_distance,
            successors_func=successors,
            obstacles_noisy=obstacles_noisy,
            obstacle_speeds=obstacle_speeds,
            rect_obstacles=rect_obstacles,
            bounds=(xmin, xmax, ymin, ymax),
            step=0.5
        )

        if astar_path is not None and len(astar_path) > 1:

            target_astar = np.array([astar_path[1][0], astar_path[1][1]])
            vector_to_target_astar = target_astar - q_astar
            dist_astar = np.linalg.norm(vector_to_target_astar)

            if dist_astar > 1e-8:
                q_astar = q_astar + v_robot * dt * vector_to_target_astar / dist_astar
            else:
                q_astar = q_astar

        else:
            print("A* failed to find path")
            q_astar = q_astar + v_robot * dt * direction_astar

        path_data_astar.append(q_astar.copy())

        step_distance_astar = np.linalg.norm(q_astar - q_old_astar)
        total_distance_astar += step_distance_astar


    # --- STOP CONDITION: close enough to goal ---

    # INTERMEDIATE GOALS ONLY
    if not done_rrt:
        if np.linalg.norm(q_rrt - q_goal_rrt) < tolerance: # if arrives at at a stop
            print(f"I've arrived at stop with RRT {stop_counter_rrt}")
            goals_achieved_so_far_rrt.append(q_goal_rrt)
            stop_counter_rrt = stop_counter_rrt + 1 
            if np.all(q_goal_rrt == q_goal_final):
                pass
            else:
                q_goal_rrt = full_geometric_polyline_rrt[2+stop_counter_rrt]

    if not done_astar:
        if np.linalg.norm(q_astar - q_goal_astar) < tolerance: # if arrives at at a stop
            print(f"I've arrived at stop with Astar {stop_counter_astar}")
            goals_achieved_so_far_astar.append(q_goal_astar)
            stop_counter_astar = stop_counter_astar + 1 
            if np.all(q_goal_astar == q_goal_final):
                pass
            else:
                q_goal_astar = full_geometric_polyline_astar[2+stop_counter_astar]

    
    # ===== RRT FINAL GOAL =====
    if (not done_rrt and
        np.linalg.norm(q_rrt - q_goal_final) < tolerance and
        np.allclose(q_goal_rrt, q_goal_final)):

        if finish_counter_rrt == 0:
            print(f"Reached goal with RRT at time {time/100}")
            print(f"Total traveled distance (RRT): {total_distance_rrt:.3f}")
            done_rrt = True
            mystates["distance_rrt"] = total_distance_rrt
            mystates["collision_rrt"] = collision_counter_rrt
        finish_counter_rrt += 1

    # ===== ASTAR FINAL GOAL =====
    if (not done_astar and
        np.linalg.norm(q_astar - q_goal_final) < tolerance and
        np.allclose(q_goal_astar, q_goal_final)):
        
        if finish_counter_astar == 0:
            print(f"Reached goal with A* at time {time/100}")
            print(f"Total traveled distance (A*): {total_distance_astar:.3f}")
            done_astar = True
            mystates["distance_astar"] = total_distance_astar
            mystates["collision_astar"] = collision_counter_astar
        finish_counter_astar += 1

    # ===== FINAL GOAL =====
    if done_rrt and done_astar:

        cost_rrt = total_distance_rrt + 100 * collision_counter_rrt
        cost_astar = total_distance_astar + 100 * collision_counter_astar

        mystates["cost_rrt"] = cost_rrt
        mystates["cost_astar"] = cost_astar

        mystates["done"] = True

        if ani is not None:
            ani.event_source.stop()

            mytime = datetime.now().strftime("%Y%m%d_%H%M%S")

            plt.savefig(
                f"sim_result_rrt_{cost_rrt:.2f}_astar_{cost_astar:.2f}_{mytime}.png",
                dpi=600,
                bbox_inches="tight"
            )

        return mystates
    
    #. check the collision for the RRT path
    if not done_rrt:

        count, hits = is_collision_check(q_rrt, obstacles_noisy, obstacle_speeds)

        if count > 0:
            print("Collision detected with obstacle RRT:", hits)
            collision_counter_rrt += count

        for rect in rect_obstacles:
            if point_in_rect(q_rrt, rect):
                print("Collision detected with rectangle RRT:", rect)
                collision_counter_rrt += 1
    
    #. check the collision for the Astar path
    if not done_astar:

        count, hits = is_collision_check(q_astar, obstacles_noisy, obstacle_speeds)

        if count > 0:
            print("Collision detected with obstacle Astar:", hits)
            collision_counter_astar += count

        for rect in rect_obstacles:
            if point_in_rect(q_astar, rect):
                print("Collision detected with rectangle Astar:", rect)
                collision_counter_astar += 1
    
    # --- 5) Update path ---
    arr_rrt = np.array(path_data_rrt)
    path_line_rrt.set_data(arr_rrt[:,0], arr_rrt[:,1])
    robot_dot_rrt.set_data([q_rrt[0]], [q_rrt[1]])

    arr_astar = np.array(path_data_astar)
    path_line_astar.set_data(arr_astar[:,0], arr_astar[:,1])
    robot_dot_astar.set_data([q_astar[0]], [q_astar[1]])

    # --- 6) Obstacle scatter ---
    true_scatter.set_offsets(obstacles_true)
    noisy_scatter.set_offsets(obstacles_noisy)

    # --- 7) Remove old ellipses ---
    for e in ellipse_patches:
        e.remove()
    ellipse_patches.clear()

    # --- 8) Draw ellipses (WITHOUT d0 scaling!) ---
    for i, obs in enumerate(obstacles_true):
        vx, vy = obstacle_speeds[i]
        vmag = np.sqrt(vx**2 + vy**2)
        theta = np.degrees(np.arctan2(vy, vx))

        a = a_base[i] + alpha * vmag 
        b = b_base[i] + beta  * vmag
        a = min(a, a_max)
        b = min(b, b_max)

        ellipse = Ellipse(
            xy=(obs[0], obs[1]),
            width=2*a,
            height=2*b,
            angle=theta,
            edgecolor='black',
            facecolor='cyan',
            alpha=0.15,       # şeffaflık
            linestyle='--',
            linewidth=1.2
        )
        ax.add_patch(ellipse)
        ellipse_patches.append(ellipse)

    # --- Remove old rectangles ---
    for r in rect_patches:
        r.remove()
    rect_patches.clear()

    # Dikdörtgen engeller and the circle
    for x, y, w, h in rect_obstacles:
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=2)
        ax.add_patch(rect)
        rect_patches.append(rect)

    # --- Remove old expanded rectangles ---
    for er in expanded_rect_patches:
        er.remove()
    expanded_rect_patches.clear()

    # Genişletilmiş rectangle'lar (buffered)
    for x, y, w, h in expanded_rects:
        rect2 = plt.Rectangle((x, y), w, h, fill=False, linewidth=1.5,
                            edgecolor='orange', linestyle='--')
        ax.add_patch(rect2)
        expanded_rect_patches.append(rect2)

    print(f"Time: {time/100} s.")
    mystates["q_rrt"]                 = q_rrt
    mystates["q_astar"]               = q_astar
    mystates["obstacle_speeds"]       = obstacle_speeds
    mystates["ani"]                   = ani
    mystates["time"]                  = time
    mystates["q_goal_final"]          = q_goal_final
    mystates["q_goal_rrt"]            = q_goal_rrt
    mystates["q_goal_astar"]          = q_goal_astar
    mystates["waypoints"]             = waypoints
    mystates["obstacles_true"]        = obstacles_true
    mystates["rectangle_speeds"]      = rectangle_speeds
    mystates["rect_obstacles"]        = rect_obstacles
    mystates["obstacles_noisy"]       = obstacles_noisy
    mystates["sigma"]                 = sigma
    mystates["v_robot"]               = v_robot
    mystates["path_data_rrt"]         = path_data_rrt
    mystates["path_data_astar"]       = path_data_astar
    mystates["path_line_rrt"]         = path_line_rrt
    mystates["path_line_astar"]       = path_line_astar
    mystates["robot_dot_rrt"]         = robot_dot_rrt
    mystates["robot_dot_astar"]       = robot_dot_astar
    mystates["true_scatter"]          = true_scatter
    mystates["noisy_scatter"]         = noisy_scatter
    mystates["goal_dot"]              = goal_dot
    mystates["ellipse_patches"]       = ellipse_patches
    mystates["rect_patches"]          = rect_patches
    mystates["expanded_rect_patches"] = expanded_rect_patches
    mystates["ax"]                    = ax
    mystates["expanded_rects"]        = expanded_rects
    mystates["eps_expanded_rects"]    = eps_expanded_rects
    mystates["full_geometric_path_rrt"]     = full_geometric_path_rrt
    mystates["full_geometric_path_astar"]   = full_geometric_path_astar
    mystates["stop_counter_rrt"]            = stop_counter_rrt
    mystates["stop_counter_astar"]          = stop_counter_astar
    mystates["goals_achieved_so_far_rrt"]   = goals_achieved_so_far_rrt
    mystates["goals_achieved_so_far_astar"] = goals_achieved_so_far_astar
    mystates["rrt"]                   = rrt
    mystates["full_geometric_polyline_rrt"]   = full_geometric_polyline_rrt
    mystates["full_geometric_polyline_astar"] = full_geometric_polyline_astar
    mystates["total_distance_rrt"]         = total_distance_rrt
    mystates["total_distance_astar"]       = total_distance_astar
    mystates["collision_counter_rrt"]      = collision_counter_rrt
    mystates["collision_counter_astar"]    = collision_counter_astar
    mystates["done"]                   = done
    mystates["done_rrt"]               = done_rrt
    mystates["done_astar"]             = done_astar
    mystates["finish_counter_rrt"]     = finish_counter_rrt
    mystates["finish_counter_astar"]   = finish_counter_astar

    return mystates
