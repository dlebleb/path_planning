# Function for MC simulations
def run_simulation(
    q,
    q_goal,
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
    path_data,
    full_geometric_path,
    full_geometric_polyline,
    enable_animation=True
):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from APF_Astar import init, update
    plt.close("all")

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

    # Waypointleri mor yıldız olarak çiz (in the beginning)
    ax.scatter(waypoints[:,0], waypoints[:,1],
    color='purple', marker='*', s=120, label="Stations")

    ellipse_patches         = []
    rect_patches            = []
    expanded_rect_patches   = []
    tolerance               = 1
    time                    = int(0)

    # ===============================
    # RUN ANIMATION
    # ===============================
    stop_counter    = 0
    ani             = None
    goals_achieved_so_far = []
    total_distance = 0
    collision_counter = 0
    done  = False
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
        "full_geometric_polyline": full_geometric_polyline,
        "total_distance": total_distance,
        "collision_counter": collision_counter,
        "done": done}

    def init_anim():
        return init(path_line, robot_dot, true_scatter, noisy_scatter, goal_dot)

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
        
    distance = mystates["total_distance"] 
    collisions = mystates["collision_counter"]
    cost = distance + 100 * collisions
    
    print("FINAL DISTANCE:", mystates["total_distance"])
    print("FINAL COLLISIONS:", mystates["collision_counter"])
    print("FINAL COST:", cost)
    return distance, collisions, cost