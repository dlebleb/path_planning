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
from TSP import *
from rrt_planner import *
# ===============================
# TSP FUNCTIONS and SETUP
# ===============================
def init(path_line,robot_dot,true_scatter,noisy_scatter, goal_dot):
    path_line.set_data([], [])
    robot_dot.set_data([], [])
    true_scatter.set_offsets(np.empty((0, 2)))
    noisy_scatter.set_offsets(np.empty((0, 2)))
    return path_line, robot_dot, true_scatter, noisy_scatter, goal_dot

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
    P = full_geometric_path

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
    k_att, k_rep, d0, dt = 60.0, 10.0, 10.0, 0.01 #k_att was previously 10.0, 20.0 (last)
    F_att = -k_att * (q - q_goal)
    return F_att

def repulsive_force(q, obstacles_noisy, obstacle_speeds):
    a0 = 2.0 # major axis (along velocity direction)
    b0 = 1.0 # minor axis (perpendicular to velocity direction)
    alpha = 0.2   # major scaling (large)
    beta  = 0.1   # minor scaling (small)
    # each obstacle has a size factor: 1 = normal, >1 = large, <1 = small
    sizes = np.array([1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1, 1.0, 1.2, 1.8, 2.0, 1.8, 1.9, 1.4, 1.5, 1.8])
    # incorporate static size                          
    a_base = a0 * sizes 
    b_base = b0 * sizes
    a_max = 7
    b_max = 5
    k_att, k_rep, d0, dt = 10.0, 10.0, 1.0, 0.01 #d0 was 10.0, 5.0

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
                F_mag = 0.02 #k_rep * (1/4 - 1/d0) * (1/4**2)
                #F_mag = k_rep * (1/dE - 1/d0) * (1/dE**2)
                # yön vektörü (normalize edilmiş fark)
                grad_Dq = (q - obs) / (dE + 1e-12)
                # toplam kuvvet
                F_rep = F_mag * grad_Dq
            else:
                F_mag = 100 #k_rep * (1/0.1 - 1/d0) * (1/0.1**2)
                grad_Dq = (q - obs) / (dE + 1e-12)
                F_rep = F_mag * grad_Dq

        else:
            F_rep = np.array([0.0, 0.0])
        F_rep_total += F_rep
    return F_rep_total

def repulsive_force_rectangles(q, rect_obstacles):
    k_rep_rectangle, d0 = 50.0, 20.0

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
    sizes = np.array([1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1, 1.0, 1.2, 1.8, 2.0, 1.8, 1.9])
    # incorporate static size                          
    a_base = a0 * sizes 
    b_base = b0 * sizes
    a_max = 7
    b_max = 5

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
    # incorporate static size                          
    a_base = a0 * sizes 
    b_base = b0 * sizes
    a_max = 7
    b_max = 5

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
    
    return counter, collided_indices

# ===============================
# UPDATE FUNCTION
# ===============================
def update(frame,mystates):
# def update(frame,q,obstacle_speeds,ani,time,q_goal_final,q_goal,waypoints,obstacles_true,rectangle_speeds,rect_obstacles,obstacles_noisy,sigma,
#            v_robot,path_data,path_line, robot_dot, true_scatter, noisy_scatter, goal_dot, ellipse_patches, rect_patches, expanded_rect_patches,
#            ax,expanded_rects, eps_expanded_rects, full_geometric_path,stop_counter):
    #from TSP import bestPath,build_tsp_indices,PSO_TSP,build_full_geometric_path
    q                     = mystates["q"]
    obstacle_speeds       = mystates["obstacle_speeds"]
    ani                   = mystates["ani"]
    time                  = mystates["time"]
    q_goal_final          = mystates["q_goal_final"]
    q_goal                = mystates["q_goal"]
    waypoints             = mystates["waypoints"]
    obstacles_true        = mystates["obstacles_true"]
    rectangle_speeds      = mystates["rectangle_speeds"]
    rect_obstacles        = mystates["rect_obstacles"]
    obstacles_noisy       = mystates["obstacles_noisy"]
    sigma                 = mystates["sigma"]
    v_robot               = mystates["v_robot"]
    path_data             = mystates["path_data"]
    path_line             = mystates["path_line"]
    robot_dot             = mystates["robot_dot"]
    true_scatter          = mystates["true_scatter"]
    noisy_scatter         = mystates["noisy_scatter"]
    goal_dot              = mystates["goal_dot"]
    ellipse_patches       = mystates["ellipse_patches"]
    rect_patches          = mystates["rect_patches"]
    expanded_rect_patches = mystates["expanded_rect_patches"]
    ax                    = mystates["ax"]
    expanded_rects        = mystates["expanded_rects"]
    eps_expanded_rects    = mystates["eps_expanded_rects"]
    full_geometric_path   = mystates["full_geometric_path"]
    stop_counter          = mystates["stop_counter"]
    goals_achieved_so_far = mystates["goals_achieved_so_far"]
    rrt                   = mystates["rrt"]
    full_geometric_polyline = mystates["full_geometric_polyline"]
    total_distance        = mystates["total_distance"]

    #path_line,robot_dot,true_scatter,noisy_scatter, goal_dot = init(path_line,robot_dot,true_scatter,noisy_scatter, goal_dot)
    tolerance = 1
    dt      = 0.01
    a0 = 2.0 # major axis (along velocity direction)
    b0 = 1.0 # minor axis (perpendicular to velocity direction)
    alpha = 0.2   # major scaling (large)
    beta  = 0.1   # minor scaling (small)
    # each obstacle has a size factor: 1 = normal, >1 = large, <1 = small
    sizes = np.array([1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1, 1.0, 1.2, 1.8, 2.0, 1.8, 1.9, 1.4, 1.5, 1.8])
    # incorporate static size                          
    a_base = a0 * sizes 
    b_base = b0 * sizes
    a_max = 7
    b_max = 5


    time = time + 1
    if (time % 1000 == 0): ## path is updated every 1s.
        #q_goal = tsp()
        # initialize the waypoints again 
        #rect_obstacles, expanded_rects, eps_expanded_rects, waypoints = init_environment()
        waypoints = init_environment_waypoints()
        
        # remove visited waypoints from the waypoint list
        waypoints = remove_visited_waypoints(waypoints,goals_achieved_so_far, tol = tolerance)

        traveller_sm, route_sm = bestPath(waypoints, q, q_goal_final, rect_obstacles, expanded_rects, eps_expanded_rects)
        N, START, END, WAYPOINTS, nodes = build_tsp_indices(q, q_goal_final, waypoints)
        best_path, best_cost = PSO_TSP(traveller_sm, START=START, END=END, WAYPOINTS=WAYPOINTS, n_particles=500, n_iter=300)
        full_geometric_path = build_full_geometric_path(best_path,route_sm)
        full_geometric_polyline = extract_polyline(full_geometric_path, waypoints)
        
        # RESET HERE
        stop_counter = 0

        #q_goal = full_geometric_path[3+stop_counter]
        q_goal = full_geometric_polyline[2+stop_counter]

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
    F = total_force(q, q_goal, obstacles_noisy, obstacle_speeds, rect_obstacles) # vektor toplami zaten sana yonu verir.

    # new addition for the global path
    p, t_hat = closest_point_and_tangent_on_polyline(q, full_geometric_path, q_goal, flag = 0)
    F = F + pull_tangent_force(q, p, t_hat)

    # --- 4) Robot motion ---
    F_norm = np.linalg.norm(F)

    if F_norm < 1e-8:
        direction = np.zeros_like(F)
    else:
        direction = F / F_norm

    q_old = q.copy()
    L = 1.0   # local horizon
    q_target_local = q + L * direction * 0.5 # added 0.3 recently and it seems to be better

    rrt_path = rrt.plan_path(
        start = Point(q[0],q[1]),
        goal = Point(q_target_local[0], q_target_local[1]),
        obstacles_noisy=obstacles_noisy,
        rect_obstacles=rect_obstacles,
        obstacle_speeds=obstacle_speeds,
        F=F,
        expanded_rects=expanded_rects
    )

    if rrt_path is not None and len(rrt_path) > 1:
       
        rrt_path = rrt.smooth_moving_average(rrt_path)
        
        target = np.array([rrt_path[1].x, rrt_path[1].y])
        vector_to_target = target - q
        dist = np.linalg.norm(vector_to_target)
        q = q + v_robot*dt * vector_to_target/dist
        #q = q + min(v_robot*dt,dist) * vector_to_target/dist
        
        #q = np.array([rrt_path[1].x,rrt_path[1].y])

    else:
        # fallback APF micro step
        #q = q + rrt.step_size * direction
        q = q + min(v_robot*dt, rrt.step_size) * direction
        #q[:] = q + direction * v_robot * dt # rrt varken 1.2 ve 0.8 yap. 

    path_data.append(q.copy())
    step_distance = np.linalg.norm(q - q_old)
    total_distance += step_distance

    # --- STOP CONDITION: close enough to goal ---

    # INTERMEDIATE GOALS ONLY
    if np.linalg.norm(q - q_goal) < tolerance: # if arrives at at a stop
        print(f"I've arrived at stop {stop_counter}")
        goals_achieved_so_far.append(q_goal)
        stop_counter = stop_counter + 1 
        #q_goal = full_geometric_path[3+stop_counter]
        if np.all(q_goal == q_goal_final):
            pass
        else:
            q_goal = full_geometric_polyline[2+stop_counter]
        # next_idx = 2 + stop_counter
        # if next_idx < len(full_geometric_path):
        #     q_goal = full_geometric_path[next_idx]
        # else:
        #     q_goal = q_goal_final.copy()

    # FINAL GOAL
    if np.linalg.norm(q - q_goal_final) < tolerance:

        print(f"Reached goal at time {time/100}")
        print(f"Total traveled distance: {total_distance:.3f}")
        #print(f"Reached goal at frame {frame}")
        ani.event_source.stop()
        return mystates
        #return []
    
    #. check the collision
    count, hits = is_collision_check(q, obstacles_noisy, obstacle_speeds)
    if count > 0:
        print("Collision detected with obstacle:", hits)

    #. check the collision with rectangles
    for rect in rect_obstacles:
        if point_in_rect(q, rect):
            print("Collision detected with rectangle:", rect)

    # --- 5) Update path ---
    arr = np.array(path_data)
    path_line.set_data(arr[:,0], arr[:,1])
    robot_dot.set_data([q[0]], [q[1]])

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

    # Waypointleri mor yıldız olarak çiz
    ax.scatter(waypoints[:,0], waypoints[:,1],
    color='purple', marker='*', s=120, label="Stations")
    
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
    mystates["q"]                     = q
    mystates["obstacle_speeds"]       = obstacle_speeds
    mystates["ani"]                   = ani
    mystates["time"]                  = time
    mystates["q_goal_final"]          = q_goal_final
    mystates["q_goal"]                = q_goal
    mystates["waypoints"]             = waypoints
    mystates["obstacles_true"]        = obstacles_true
    mystates["rectangle_speeds"]      = rectangle_speeds
    mystates["rect_obstacles"]        = rect_obstacles
    mystates["obstacles_noisy"]       = obstacles_noisy
    mystates["sigma"]                 = sigma
    mystates["v_robot"]               = v_robot
    mystates["path_data"]             = path_data
    mystates["path_line"]             = path_line
    mystates["robot_dot"]             = robot_dot
    mystates["true_scatter"]          = true_scatter
    mystates["noisy_scatter"]         = noisy_scatter
    mystates["goal_dot"]              = goal_dot
    mystates["ellipse_patches"]       = ellipse_patches
    mystates["rect_patches"]          = rect_patches
    mystates["expanded_rect_patches"] = expanded_rect_patches
    mystates["ax"]                    = ax
    mystates["expanded_rects"]        = expanded_rects
    mystates["eps_expanded_rects"]    = eps_expanded_rects
    mystates["full_geometric_path"]   = full_geometric_path
    mystates["stop_counter"]          = stop_counter
    mystates["goals_achieved_so_far"] = goals_achieved_so_far
    mystates["rrt"]                   = rrt
    mystates["full_geometric_polyline"] = full_geometric_polyline 
    mystates["total_distance"]         = total_distance
    
    #return []
    return mystates
    #return path_line, robot_dot, true_scatter, noisy_scatter, goal_dot ,q, obstacle_speeds,ani,time,q_goal_final,q_goal,waypoints,obstacles_true,rectangle_speeds,rect_obstacles,obstacles_noisy,sigma,v_robot,path_data,path_line, robot_dot, true_scatter, noisy_scatter, goal_dot, ellipse_patches, rect_patches, expanded_rect_patches,ax, full_geometric_path, stop_counter