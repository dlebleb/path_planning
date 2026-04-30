# Function for MC simulations
def run_simulation(
    q_rrt,
    q_astar,
    q_goal_rrt,
    q_goal_astar,
    q_goal_final,
    obstacle_speeds,
    rectangle_speeds,
    waypoints,
    obstacles_true,
    rect_obstacles,
    obstacles_noisy,
    expanded_rects,
    eps_expanded_rects,
    sigma,
    v_robot,
    path_data_rrt,
    path_data_astar,
    full_geometric_path_rrt,
    full_geometric_polyline_rrt,
    full_geometric_path_astar,
    full_geometric_polyline_astar,
    rrt,
    enable_animation=True
):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from APF_RRT_Astar import init, update
    plt.close("all")

    fig, ax = plt.subplots(figsize=(8, 8), dpi = 50)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_title("Dynamic Obstacle Avoidance Animation")
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.grid(color='gray', linestyle='--', linewidth=0.4, alpha=0.7)

    path_line_rrt, = ax.plot([], [], 'r-', linewidth=2, label="TSP + PF + RRT (ours)")
    robot_dot_rrt, = ax.plot([], [], 'ro', markersize=6)
    path_line_astar, = ax.plot([], [], 'b-', linewidth=2, label="TSP + PF + A* (ours)")
    robot_dot_astar, = ax.plot([], [], 'bo', markersize=6)
    
    true_scatter = ax.scatter([], [], c='black', s=40)
    noisy_scatter = ax.scatter([], [], c='red', s=40)
    goal_dot, = ax.plot(q_goal_final[0], q_goal_final[1], 'go', markersize=8, label="Goal")

    # Waypointleri mor yıldız olarak çiz
    ax.scatter(waypoints[:,0], waypoints[:,1],
    color='purple', marker='*', s=120, label="Stations")

    ax.legend(loc="upper right")

    ellipse_patches         = []
    rect_patches            = []
    expanded_rect_patches   = []
    time                    = int(0)

    # ===============================
    # RUN ANIMATION
    # ===============================
    stop_counter_rrt      = 0
    stop_counter_astar    = 0

    ani             = None

    goals_achieved_so_far_rrt   = []
    goals_achieved_so_far_astar = []

    total_distance_rrt = 0
    collision_counter_rrt = 0
    total_distance_astar = 0
    collision_counter_astar = 0

    done = False
    done_rrt    = False
    done_astar  = False
    finish_counter_rrt   = 0
    finish_counter_astar = 0

    mystates = {
        "q_rrt": q_rrt,
        "q_astar": q_astar,
        "obstacle_speeds": obstacle_speeds,
        "ani": ani,
        "time": time,
        "q_goal_final": q_goal_final,
        "q_goal_rrt": q_goal_rrt,
        "q_goal_astar": q_goal_astar,
        "waypoints": waypoints,
        "obstacles_true": obstacles_true,
        "rectangle_speeds": rectangle_speeds,
        "rect_obstacles": rect_obstacles,
        "obstacles_noisy": obstacles_noisy,
        "sigma": sigma,
        "v_robot": v_robot,
        "path_data_rrt": path_data_rrt,
        "path_data_astar": path_data_astar,
        "path_line_rrt": path_line_rrt,
        "path_line_astar": path_line_astar,
        "robot_dot_rrt": robot_dot_rrt,
        "robot_dot_astar": robot_dot_astar,
        "true_scatter": true_scatter,
        "noisy_scatter": noisy_scatter,
        "goal_dot": goal_dot,
        "ellipse_patches": ellipse_patches,
        "rect_patches": rect_patches,
        "expanded_rect_patches": expanded_rect_patches,
        "ax": ax,
        "expanded_rects": expanded_rects,
        "eps_expanded_rects": eps_expanded_rects,
        "full_geometric_path_rrt": full_geometric_path_rrt,
        "full_geometric_path_astar": full_geometric_path_astar,
        "stop_counter_rrt": stop_counter_rrt,
        "stop_counter_astar": stop_counter_astar,
        "goals_achieved_so_far_rrt": goals_achieved_so_far_rrt,
        "goals_achieved_so_far_astar": goals_achieved_so_far_astar,
        "rrt": rrt,
        "full_geometric_polyline_rrt": full_geometric_polyline_rrt,
        "full_geometric_polyline_astar": full_geometric_polyline_astar,
        "total_distance_rrt": total_distance_rrt,
        "total_distance_astar": total_distance_astar,
        "collision_counter_rrt": collision_counter_rrt,
        "collision_counter_astar": collision_counter_astar,
        "done": done,
        "done_rrt": done_rrt,
        "done_astar": done_astar,
        "finish_counter_rrt": finish_counter_rrt,
        "finish_counter_astar": finish_counter_astar}

    def init_anim():
        return init(path_line_rrt, path_line_astar, robot_dot_rrt, robot_dot_astar, true_scatter, noisy_scatter, goal_dot)

    if enable_animation:
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
        
    
    else:
        frame = 0
        while not mystates["done"]:
            update(frame, mystates)
            frame += 1
        
    distance_rrt = mystates["total_distance_rrt"] 
    collisions_rrt = mystates["collision_counter_rrt"]
    cost_rrt = distance_rrt + 100 * collisions_rrt

    distance_astar = mystates["total_distance_astar"] 
    collisions_astar = mystates["collision_counter_astar"]
    cost_astar = distance_astar + 100 * collisions_astar
    
    print("FINAL DISTANCE RRT:", mystates["total_distance_rrt"])
    print("FINAL COLLISIONS RRT:", mystates["collision_counter_rrt"])
    print("FINAL COST RRT:", cost_rrt)

    print("FINAL DISTANCE ASTAR:", mystates["total_distance_astar"])
    print("FINAL COLLISIONS ASTAR:", mystates["collision_counter_astar"])
    print("FINAL COST ASTAR:", cost_astar)
    return distance_rrt, collisions_rrt, cost_rrt, distance_astar, collisions_astar, cost_astar