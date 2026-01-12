"""
Quick test to verify tangent waypoint computation works correctly.
This tests if waypoints are truly tangent to obstacles like in the image.
"""

import math
from global_planner_simple import Point, Obstacle, GlobalPlanner


def test_tangent_waypoints():
    """Test that tangent waypoints are computed correctly"""
    
    print("Testing Tangent Waypoint Computation")
    print("=" * 60)
    
    # Setup similar to the image: DR1 (start), DR2 (goal), obstacles in between
    start = Point(-5.0, -5.0)  # DR1
    goal = Point(10.0, 10.0)   # DR2
    
    # Two obstacles between start and goal (like orange regions in image)
    obstacles = [
        Obstacle(Point(0.0, 0.0), radius=2.0),
        Obstacle(Point(5.0, 5.0), radius=1.5),
    ]
    
    planner = GlobalPlanner(safety_margin=0.3)
    
    # Generate tangent waypoints
    waypoints = planner.compute_tangent_waypoints(start, goal, obstacles)
    
    print(f"\nStart (DR1): {start.to_tuple()}")
    print(f"Goal (DR2): {goal.to_tuple()}")
    print(f"\nObstacles: {len(obstacles)}")
    for i, obs in enumerate(obstacles):
        print(f"  Obstacle {i+1}: center={obs.position.to_tuple()}, radius={obs.radius}")
    
    print(f"\nGenerated {len(waypoints)} tangent waypoints:")
    for i, wp in enumerate(waypoints):
        print(f"  Waypoint {i+1}: {wp.to_tuple()}")
        
        # Verify waypoint is on obstacle boundary (or close to it)
        for j, obs in enumerate(obstacles):
            dist_to_center = wp.distance_to(obs.position)
            expected_dist = obs.radius
            error = abs(dist_to_center - expected_dist)
            if error < 0.5:  # Allow small tolerance
                print(f"    → On boundary of Obstacle {j+1} (distance: {dist_to_center:.3f}, expected: {expected_dist:.3f}, error: {error:.3f})")
    
    # Test TSP ordering
    print("\n" + "-" * 60)
    print("Testing TSP waypoint ordering:")
    print("-" * 60)
    
    ordered = planner.plan_path(start, goal, obstacles)
    
    print(f"\nOrdered waypoints (TSP optimized):")
    for i, wp in enumerate(ordered):
        if i == 0:
            label = "START (DR1)"
        elif i == len(ordered) - 1:
            label = "GOAL (DR2)"
        else:
            label = f"WP{i}"
        print(f"  {i+1}. {label}: {wp.to_tuple()}")
    
    # Verify the path makes sense
    print("\n" + "-" * 60)
    print("Verification:")
    print("-" * 60)
    
    if len(waypoints) > 0:
        print("✓ Tangent waypoints generated")
    else:
        print("✗ No waypoints generated")
    
    if len(ordered) >= 3:  # At least start, one waypoint, goal
        print("✓ TSP ordering includes intermediate waypoints")
    else:
        print("✗ TSP ordering may be incomplete")
    
    # Check if waypoints are actually on obstacle boundaries
    all_on_boundary = True
    for wp in waypoints:
        on_boundary = False
        for obs in obstacles:
            dist = wp.distance_to(obs.position)
            if abs(dist - obs.radius) < 0.5:  # Within tolerance
                on_boundary = True
                break
        if not on_boundary:
            all_on_boundary = False
    
    if all_on_boundary:
        print("✓ All waypoints are on obstacle boundaries (tangent points)")
    else:
        print("⚠ Some waypoints may not be exactly on boundaries")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_tangent_waypoints()

