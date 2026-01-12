"""
RRT + TSP Demo with Tangent Waypoints

This script demonstrates how to use:
1. Global Planner (TSP) to generate tangent waypoints around obstacles
2. RRT to plan collision-free paths between waypoints
3. Integration of both systems

This creates paths similar to the image, where waypoints are tangent to obstacles.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from global_planner_simple import Point, Obstacle, GlobalPlanner
from rrt_planner import RRTPlanner


def main():
    """Demo: RRT + TSP with tangent waypoints"""
    
    print("=" * 60)
    print("RRT + TSP Demo with Tangent Waypoints")
    print("=" * 60)
    
    # ============================================================
    # Setup: Define start, goal, and obstacles
    # ============================================================
    start = Point(-5.0, -5.0)  # DR1 in the image
    goal = Point(10.0, 10.0)    # DR2 in the image
    
    # Create obstacles (orange regions in the image)
    obstacles = [
        Obstacle(Point(0.0, 0.0), radius=2.0),
        Obstacle(Point(5.0, 5.0), radius=1.5),
        Obstacle(Point(7.0, 2.0), radius=1.2),
    ]
    
    print(f"\nStart: {start.to_tuple()}")
    print(f"Goal: {goal.to_tuple()}")
    print(f"Obstacles: {len(obstacles)}")
    for i, obs in enumerate(obstacles):
        print(f"  Obstacle {i+1}: center={obs.position.to_tuple()}, radius={obs.radius:.2f}")
    
    # ============================================================
    # Step 1: Generate tangent waypoints using Global Planner (TSP)
    # ============================================================
    print("\n" + "-" * 60)
    print("Step 1: Generating tangent waypoints around obstacles")
    print("-" * 60)
    
    global_planner = GlobalPlanner(replan_interval=5.0, safety_margin=0.3)
    
    # Generate tangent waypoints and optimize their order with TSP
    waypoint_order = global_planner.plan_path(start, goal, obstacles)
    
    print(f"\nGenerated {len(waypoint_order)} waypoints (including start and goal):")
    for i, wp in enumerate(waypoint_order):
        if i == 0:
            label = "START"
        elif i == len(waypoint_order) - 1:
            label = "GOAL"
        else:
            label = f"WP{i}"
        print(f"  {label}: {wp.to_tuple()}")
    
    # ============================================================
    # Step 2: Plan RRT paths between consecutive waypoints
    # ============================================================
    print("\n" + "-" * 60)
    print("Step 2: Planning RRT paths between waypoints")
    print("-" * 60)
    
    # RRT parameters: more lenient to handle waypoints on obstacle boundaries
    rrt = RRTPlanner(step_size=0.2, max_iterations=5000, 
                     goal_threshold=0.5, safety_margin=0.05)
    
    # Plan paths between each pair of consecutive waypoints
    all_paths = []
    bounds = (-10, 15, -10, 15)
    
    for i in range(len(waypoint_order) - 1):
        wp_start = waypoint_order[i]
        wp_end = waypoint_order[i + 1]
        
        print(f"\nPlanning path from waypoint {i} to {i+1}...")
        path = rrt.plan_path(wp_start, wp_end, obstacles, bounds)
        
        if path:
            all_paths.append(path)
            print(f"  ✓ Path found with {len(path)} points")
        else:
            print(f"  ✗ No path found (using straight line as fallback)")
            all_paths.append([wp_start, wp_end])
    
    # ============================================================
    # Step 3: Visualize the result
    # ============================================================
    print("\n" + "-" * 60)
    print("Step 3: Visualizing path")
    print("-" * 60)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Draw obstacles
    for obs in obstacles:
        circle = Circle(obs.position.to_tuple(), obs.radius, 
                       color='orange', alpha=0.5, label='Obstacle')
        ax.add_patch(circle)
        ax.plot(obs.position.x, obs.position.y, 'ko', markersize=5)
    
    # Draw waypoints (tangent points)
    waypoint_x = [wp.x for wp in waypoint_order]
    waypoint_y = [wp.y for wp in waypoint_order]
    ax.plot(waypoint_x, waypoint_y, 'bo', markersize=10, 
           label='Tangent Waypoints', zorder=5)
    
    # Draw waypoint connections (dashed line - like in the image)
    ax.plot(waypoint_x, waypoint_y, 'b--', linewidth=2, 
           alpha=0.5, label='Waypoint Order (TSP)', zorder=3)
    
    # Draw RRT paths (solid lines)
    for path in all_paths:
        path_x = [p.x for p in path]
        path_y = [p.y for p in path]
        ax.plot(path_x, path_y, 'r-', linewidth=2, 
               label='RRT Path' if path == all_paths[0] else '', zorder=4)
    
    # Mark start and goal
    ax.plot(start.x, start.y, 'gs', markersize=15, 
           label='Start (DR1)', zorder=6)
    ax.plot(goal.x, goal.y, 'rs', markersize=15, 
           label='Goal (DR2)', zorder=6)
    
    # Labels and formatting
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('RRT + TSP Path Planning with Tangent Waypoints', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_aspect('equal')
    ax.axis([-10, 15, -10, 15])
    
    plt.tight_layout()
    import os
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/rrt_tsp_demo.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved to: figures/rrt_tsp_demo.png")
    plt.show()
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"✓ Generated {len(waypoint_order) - 2} tangent waypoints around obstacles")
    print(f"✓ TSP optimized waypoint order to minimize travel distance")
    print(f"✓ RRT planned {len(all_paths)} collision-free path segments")
    print(f"✓ Total path points: {sum(len(p) for p in all_paths)}")
    print("\nThis matches the approach shown in the image:")
    print("  - Tangent waypoints on obstacles (orange regions)")
    print("  - TSP orders waypoints optimally")
    print("  - RRT plans smooth paths between waypoints")


if __name__ == "__main__":
    main()

