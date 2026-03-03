"""
TSP (Traveling Salesman Problem) Path Planner with Tangent Waypoints.

This planner:
1. Generates tangent waypoints around obstacles (relative to obstacle centers)
2. Uses TSP to optimize the order of visiting these waypoints
3. Works with RRT for path planning between waypoints

The robot uses:
- TSP to generate tangent waypoints around obstacles
- TSP to optimize waypoint order (minimize total travel distance)
- Replans every 5 seconds to adapt to robot's current position
"""

import math
from typing import List, Optional, Tuple


class Point:
    """
    Simple 2D point representation.
    Used for waypoints, start, and goal positions.
    """
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def distance_to(self, other: 'Point') -> float:
        """Calculate Euclidean distance to another point"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def to_tuple(self):
        """Convert to tuple for easy printing"""
        return (self.x, self.y)


class Obstacle:
    """
    Circular obstacle representation for collision checking.
    
    Used by RRT planner to check if paths are collision-free.
    """
    def __init__(self, position: Point, radius: float):
        """
        Initialize obstacle.
        
        Args:
            position: Center point of the obstacle
            radius: Radius of the circular obstacle
        """
        self.position = position
        self.radius = radius
    
    def collides_with(self, point: Point, safety_margin: float = 0.0) -> bool:
        """
        Check if a point collides with this obstacle.
        
        Args:
            point: Point to check for collision
            safety_margin: Additional safety margin around obstacle
        
        Returns:
            True if point is inside obstacle (including safety margin), False otherwise
        """
        distance = self.position.distance_to(point)
        return distance < (self.radius + safety_margin)


class GlobalPlanner:
    """
    Global planner that generates tangent waypoints around obstacles and optimizes their order.
    
    This planner implements the approach shown in the image:
    - Computes tangent waypoints on obstacles (like the orange regions in the image)
    - Uses TSP to find the optimal order to visit these waypoints
    - Returns an ordered sequence: [start, tangent_wp1, tangent_wp2, ..., goal]
    """
    
    def __init__(self, replan_interval: float = 5.0, safety_margin: float = 0.3):
        """
        Initialize global planner.
        
        Args:
            replan_interval: Time in seconds between global replanning (default: 5.0)
            safety_margin: Safety margin for tangent waypoints (default: 0.3)
        """
        self.replan_interval = replan_interval
        self.last_replan_time = 0.0
        self.current_waypoint_order: List[Point] = []
        self.safety_margin = safety_margin
    
    def compute_tangent_waypoints(self, start: Point, goal: Point, 
                                  obstacles: List[Obstacle]) -> List[Point]:
        """
        Compute tangent waypoints around obstacles between start and goal.
        
        For each obstacle that blocks the path from start to goal, this function
        computes two tangent points on the obstacle boundary. These points are
        where lines from start/goal would be tangent to the obstacle circle.
        
        This creates waypoints similar to the image, where the path navigates
        around obstacles using tangent points.
        
        Args:
            start: Starting position
            goal: Goal position
            obstacles: List of obstacles to navigate around
        
        Returns:
            List of tangent waypoint Points
        """
        waypoints = []
        
        # Check each obstacle to see if it's between start and goal
        for obs in obstacles:
            # Check if obstacle is roughly between start and goal
            if self._obstacle_blocks_path(start, goal, obs):
                # Compute tangent points on this obstacle
                tangent_points = self._compute_tangent_points(start, goal, obs)
                waypoints.extend(tangent_points)
        
        return waypoints
    
    def _obstacle_blocks_path(self, start: Point, goal: Point, obs: Obstacle) -> bool:
        """
        Check if an obstacle blocks the direct path from start to goal.
        
        Args:
            start: Starting point
            goal: Goal point
            obs: Obstacle to check
        
        Returns:
            True if obstacle is between start and goal, False otherwise
        """
        # Vector from start to goal
        dx = goal.x - start.x
        dy = goal.y - start.y
        path_length = math.sqrt(dx*dx + dy*dy)
        
        if path_length < 1e-6:
            return False
        
        # Vector from start to obstacle center
        obs_dx = obs.position.x - start.x
        obs_dy = obs.position.y - start.y
        
        # Project obstacle center onto the start-goal line
        t = (obs_dx * dx + obs_dy * dy) / (path_length * path_length)
        
        # Check if obstacle is between start and goal (0 < t < 1)
        if t < 0 or t > 1:
            return False
        
        # Find closest point on line to obstacle center
        closest_x = start.x + t * dx
        closest_y = start.y + t * dy
        closest_point = Point(closest_x, closest_y)
        
        # Check if obstacle is close enough to the path to matter
        distance_to_path = obs.position.distance_to(closest_point)
        
        # Obstacle blocks path if it's within (radius + safety_margin) of the path
        return distance_to_path < (obs.radius + self.safety_margin + 0.5)
    
    def _compute_tangent_points(self, start: Point, goal: Point, 
                                obs: Obstacle) -> List[Point]:
        """
        Compute waypoints relative to obstacle center (black center points).
        
        Instead of computing geometric tangents on the boundary, this places waypoints
        at specific angles relative to the obstacle center. The waypoints are positioned
        to allow smooth navigation around the obstacle, relative to the center point.
        
        Args:
            start: Starting point (DR1 in the image)
            goal: Goal point (DR2 in the image)
            obs: Obstacle to compute waypoints for (center is the black dot)
        
        Returns:
            List of 1-2 waypoint Points positioned relative to obstacle center
        """
        # Vector from obstacle center to start
        dx_start = start.x - obs.position.x
        dy_start = start.y - obs.position.y
        dist_start = math.sqrt(dx_start*dx_start + dy_start*dy_start)
        
        # Vector from obstacle center to goal
        dx_goal = goal.x - obs.position.x
        dy_goal = goal.y - obs.position.y
        dist_goal = math.sqrt(dx_goal*dx_goal + dy_goal*dy_goal)
        
        # If too close to obstacle center, skip
        if dist_start < 0.1 or dist_goal < 0.1:
            return []
        
        # Compute angles from obstacle center to start and goal
        angle_to_start = math.atan2(dy_start, dx_start)
        angle_to_goal = math.atan2(dy_goal, dx_goal)
        
        # Compute the angle of the direct start-goal line from obstacle center
        # This is the direction from center toward the path
        angle_to_path = math.atan2(goal.y - obs.position.y, goal.x - obs.position.x)
        
        # Place waypoints relative to obstacle center at perpendicular angles
        # This creates waypoints on either side of the direct path, relative to center
        waypoint_distance = obs.radius + self.safety_margin + 0.2  # Distance from center
        
        # Perpendicular angles to the start-goal line (90 degrees on each side)
        perp_angle_left = angle_to_path + math.pi / 2   # Left side (counterclockwise)
        perp_angle_right = angle_to_path - math.pi / 2  # Right side (clockwise)
        
        # Create waypoints relative to obstacle center
        waypoints = []
        
        # Left side waypoint (relative to center)
        wp_left = Point(
            obs.position.x + waypoint_distance * math.cos(perp_angle_left),
            obs.position.y + waypoint_distance * math.sin(perp_angle_left)
        )
        
        # Right side waypoint (relative to center)
        wp_right = Point(
            obs.position.x + waypoint_distance * math.cos(perp_angle_right),
            obs.position.y + waypoint_distance * math.sin(perp_angle_right)
        )
        
        # Select the waypoint that's on the "outside" of the direct path
        # This ensures we go around the obstacle, not through it
        # Use cross product to determine which side of start-goal line each waypoint is on
        def cross_product_sign(candidate):
            v1_x = goal.x - start.x
            v1_y = goal.y - start.y
            v2_x = candidate.x - start.x
            v2_y = candidate.y - start.y
            return v1_x * v2_y - v1_y * v2_x
        
        # Determine which waypoint is better (further from direct path)
        left_sign = cross_product_sign(wp_left)
        right_sign = cross_product_sign(wp_right)
        
        # If waypoints are on opposite sides, return both
        if left_sign > 0 and right_sign < 0:
            # Both sides available - return the one further from path
            path_mid = Point((start.x + goal.x) / 2, (start.y + goal.y) / 2)
            if wp_left.distance_to(path_mid) > wp_right.distance_to(path_mid):
                return [wp_left]
            else:
                return [wp_right]
        elif left_sign > 0:
            return [wp_left]
        elif right_sign < 0:
            return [wp_right]
        else:
            # Fallback: return the one further from direct path
            path_mid = Point((start.x + goal.x) / 2, (start.y + goal.y) / 2)
            if wp_left.distance_to(path_mid) > wp_right.distance_to(path_mid):
                return [wp_left]
            else:
                return [wp_right]
    
    def optimize_waypoint_order(self, start: Point, waypoints: List[Point], 
                                goal: Point) -> List[Point]:
        """
        Optimize waypoint order using nearest-neighbor TSP heuristic.
        
        This is a greedy algorithm that always picks the nearest unvisited waypoint.
        While not guaranteed to be optimal, it's fast and works well in practice.
        
        Args:
            start: Starting position (current robot position)
            waypoints: List of waypoints to visit
            goal: Final destination/goal position
        
        Returns:
            Optimized order: [start, wp1, wp2, ..., goal]
        """
        if not waypoints:
            return [start, goal]
        
        # Start with current position
        optimized = [start]
        remaining = waypoints.copy()  # Unvisited waypoints
        current = start
        
        # Greedy selection: always pick nearest unvisited waypoint
        while remaining:
            # Find nearest waypoint to current position
            nearest = min(remaining, key=lambda p: current.distance_to(p))
            optimized.append(nearest)
            remaining.remove(nearest)
            current = nearest  # Move to selected waypoint
        
        # Always end at goal
        optimized.append(goal)
        return optimized
    
    def plan_path(self, current_pos: Point, goal: Point, 
                  obstacles: List[Obstacle], 
                  waypoints: Optional[List[Point]] = None) -> List[Point]:
        """
        Plan global path with tangent waypoints around obstacles.
        
        This function:
        1. Generates tangent waypoints around obstacles (if waypoints not provided)
        2. Uses TSP to optimize the order of visiting waypoints
        3. Returns ordered sequence: [start, wp1, wp2, ..., goal]
        
        This is called periodically (every 5 seconds) to ensure the robot
        follows the shortest route as it moves through the environment.
        
        Args:
            current_pos: Current robot position
            goal: Final destination
            obstacles: List of obstacles to navigate around
            waypoints: Optional pre-computed waypoints. If None, generates tangent waypoints.
        
        Returns:
            Optimized waypoint sequence
        """
        # Generate tangent waypoints if not provided
        if waypoints is None:
            waypoints = self.compute_tangent_waypoints(current_pos, goal, obstacles)
        
        # Use TSP to optimize waypoint order
        optimized = self.optimize_waypoint_order(current_pos, waypoints, goal)
        self.current_waypoint_order = optimized
        return optimized
    
    def should_replan(self, current_time: float) -> bool:
        """
        Check if it's time to replan (every 5 seconds by default).
        
        Global replanning ensures the robot adapts to its current position
        and always follows the shortest route, even if it has deviated from
        the original plan.
        
        Args:
            current_time: Current timestamp
        
        Returns:
            True if replanning is needed, False otherwise
        """
        if current_time - self.last_replan_time >= self.replan_interval:
            self.last_replan_time = current_time
            return True
        return False
    
    def get_current_waypoint_order(self) -> List[Point]:
        """
        Get current optimized waypoint order.
        
        Returns:
            Current waypoint sequence from global planner
        """
        return self.current_waypoint_order


# -------------------------------------g--------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

def main():
    """Simple example demonstrating tangent waypoint generation and TSP ordering"""
    planner = GlobalPlanner(replan_interval=5.0)
    
    start = Point(0.0, 0.0)
    goal = Point(10.0, 10.0)
    
    # Create some obstacles between start and goal
    obstacles = [
        Obstacle(Point(3.0, 3.0), radius=1.5),
        Obstacle(Point(6.0, 6.0), radius=1.2),
        Obstacle(Point(7.0, 4.0), radius=1.0),
    ]
    
    # Plan path - this will generate tangent waypoints and optimize their order
    optimized = planner.plan_path(start, goal, obstacles)
    
    print("Optimized waypoint order (with tangent waypoints):")
    for i, wp in enumerate(optimized):
        print(f"  {i+1}. {wp.to_tuple()}")
    
    print(f"\nTotal waypoints: {len(optimized)}")
    print(f"  Start: {optimized[0].to_tuple()}")
    print(f"  Goal: {optimized[-1].to_tuple()}")
    print(f"  Intermediate waypoints: {len(optimized) - 2}")


if __name__ == "__main__":
    main()
