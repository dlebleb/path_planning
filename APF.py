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

# ===============================
# FORCE FUNCTIONS
# ===============================

def attractive_force(q, q_goal):
    k_att, k_rep, d0, dt = 10.0, 10.0, 10.0, 0.01
    F_att = -k_att * (q - q_goal)
    return F_att

def repulsive_force(q, obstacles_noisy, obstacle_speeds):
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
    k_att, k_rep, d0, dt = 10.0, 10.0, 10.0, 0.01

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
            F_mag = k_rep * (1/dE - 1/d0) * (1/dE**2)
            # yön vektörü (normalize edilmiş fark)
            grad_Dq = (q - obs) / (dE + 1e-12)
            # toplam kuvvet
            F_rep = F_mag * grad_Dq

        else:
            F_rep = np.array([0.0, 0.0])
        F_rep_total += F_rep
    return F_rep_total

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

def total_force(q, q_goal, obstacles_noisy, obstacle_speeds):
    """
    Calculate the total force on the robot
    """
    F_att = attractive_force(q, q_goal)
    F_rep = repulsive_force(q, obstacles_noisy, obstacle_speeds)
    return F_att + F_rep

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

def is_collision_check(q, obstacles_noisy, obstacle_speeds):
    collided_indices = []
    counter = 0
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
    from TSP import bestPath,build_tsp_indices,PSO_TSP,build_full_geometric_path
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


    path_line,robot_dot,true_scatter,noisy_scatter, goal_dot = init(path_line,robot_dot,true_scatter,noisy_scatter, goal_dot)
    tolerance = 1
    dt      = 0.01
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


    time = time + 1
    if (time % 1000 == 0): ## path is updated every 1s.
        #q_goal = tsp()
        # initialize the waypoints again 
        rect_obstacles, expanded_rects, eps_expanded_rects, waypoints = init_environment()
        
        # remove visited waypoints from the waypoint list
        waypoints = remove_visited_waypoints(waypoints,goals_achieved_so_far, tol = tolerance)

        traveller_sm, route_sm = bestPath(waypoints, q, q_goal, rect_obstacles, expanded_rects, eps_expanded_rects)
        N, START, END, WAYPOINTS, nodes = build_tsp_indices(q, q_goal, waypoints)
        best_path, best_cost = PSO_TSP(traveller_sm, START=START, END=END, WAYPOINTS=WAYPOINTS, n_particles=500, n_iter=300)
        full_geometric_path = build_full_geometric_path(best_path,route_sm)

        # RESET HERE
        stop_counter = 0

        #q_goal = full_geometric_path[3+stop_counter]
        q_goal = full_geometric_path[2+stop_counter]

    # 0) random maneuver (new speeds)
    obstacle_speeds = apply_stochastic_maneuver(obstacle_speeds)

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
    F = total_force(q, q_goal, obstacles_noisy, obstacle_speeds)

    # --- 4) Robot motion ---
    F_norm = np.linalg.norm(F)
    direction = F/F_norm
    q[:] = q + direction * v_robot * dt
    path_data.append(q.copy())

    # --- STOP CONDITION: close enough to goal ---
    if np.linalg.norm(q - q_goal) < tolerance: # if arrives at at a stop
        print(f"I've arrived at stop {stop_counter}")
        goals_achieved_so_far.append(q_goal)
        stop_counter = stop_counter + 1 
        #q_goal = full_geometric_path[3+stop_counter]
        if np.all(q_goal == q_goal_final):
            pass
        else:
            q_goal = full_geometric_path[2+stop_counter]

    if np.linalg.norm(q - q_goal_final) < tolerance:
        print(f"Reached goal at time {time/100}")
        #print(f"Reached goal at frame {frame}")
        ani.event_source.stop()
        return mystates

    #. check the collision
    count, hits = is_collision_check(q, obstacles_noisy, obstacle_speeds)
    if count > 0:
        print("Collision detected with obstacle:", hits)

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


    return mystates
    #return path_line, robot_dot, true_scatter, noisy_scatter, goal_dot ,q, obstacle_speeds,ani,time,q_goal_final,q_goal,waypoints,obstacles_true,rectangle_speeds,rect_obstacles,obstacles_noisy,sigma,v_robot,path_data,path_line, robot_dot, true_scatter, noisy_scatter, goal_dot, ellipse_patches, rect_patches, expanded_rect_patches,ax, full_geometric_path, stop_counter