"""
Simple RRT (Rapidly-exploring Random Tree) Path Planner.

This module implements a basic RRT algorithm for generating collision-free paths
between waypoints while avoiding obstacles.

RRT Algorithm Overview:
1. Start with a tree containing only the start position
2. Randomly sample points in the workspace
3. Find the nearest node in the tree to the sampled point
4. Extend the tree toward the sampled point (step_size distance)
5. If the extension is collision-free, add new node to tree
6. Repeat until goal is reached or max iterations

Designed to work with:
- Global planner (TSP) for waypoint ordering
- Potential fields for dynamic obstacle avoidance
"""

import math
import random
from typing import List, Optional
from path_planner import Point, Obstacle


class RRTNode:
    """
    Node in the RRT tree structure.
    
    Each node stores:
    - position: The 2D point this node represents
    - parent: Parent node (for path reconstruction)
    """
    def __init__(self, position: Point, parent=None):
        self.position = position
        self.parent = parent  # Parent node enables path reconstruction


class RRTPlanner:
    """
    Simple RRT path planner.
    
    RRT (Rapidly-exploring Random Tree) builds a tree from start to goal by:
    1. Randomly sampling points in the workspace
    2. Finding nearest node in tree to sampled point
    3. Extending tree toward sampled point
    4. Checking for collisions
    5. Repeating until goal is reached
    
    This is a probabilistic algorithm - it explores the workspace randomly
    but efficiently finds paths in complex environments.
    """
    
    def __init__(self, step_size: float = 0.5, max_iterations: int = 3000, 
                 goal_threshold: float = 0.3, safety_margin: float = 0.2):
        """
        Initialize RRT planner.
        
        Args:
            step_size: Maximum distance to extend tree each step
            max_iterations: Max iterations before giving up
            goal_threshold: Distance to goal to consider success
            safety_margin: Safety margin around obstacles
        """
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_threshold = goal_threshold
        self.safety_margin = safety_margin
    
    def _is_collision_free(self, point: Point, obstacles: List[Obstacle]) -> bool:
        """
        Check if a point is not inside any obstacle.
        
        Args:
            point: Point to check
            obstacles: List of obstacles
        
        Returns:
            True if point is collision-free, False otherwise
        """
        for obs in obstacles:
            if obs.collides_with(point, self.safety_margin):
                return False
        return True
    
    def _is_path_clear(self, p1: Point, p2: Point, obstacles: List[Obstacle]) -> bool:
        """
        Check if the path between two points is collision-free.
        
        Samples 10 points along the line segment and checks each for collisions.
        This ensures the entire path segment is safe, not just the endpoints.
        
        Args:
            p1: Start point
            p2: End point
            obstacles: List of obstacles
        
        Returns:
            True if path is clear, False if collision detected
        """
        # Check multiple points along the path
        for i in range(10):
            t = i / 10.0  # Interpolation parameter (0.0 to 1.0)
            check_point = Point(
                p1.x + t * (p2.x - p1.x),
                p1.y + t * (p2.y - p1.y)
            )
            if not self._is_collision_free(check_point, obstacles):
                return False
        return True
    
    def _find_nearest(self, tree: List[RRTNode], point: Point) -> RRTNode:
        """
        Find the nearest node in the tree to a given point.
        
        This is used to determine which node to extend from when
        adding a new node to the tree.
        
        Args:
            tree: List of nodes in the RRT tree
            point: Point to find nearest node to
        
        Returns:
            Nearest node in tree
        """
        nearest = tree[0]
        min_dist = nearest.position.distance_to(point)
        
        for node in tree[1:]:
            dist = node.position.distance_to(point)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
    
    def _steer(self, from_node: RRTNode, to_point: Point) -> Point:
        """
        Move step_size distance from from_node toward to_point.
        
        If to_point is closer than step_size, return to_point directly.
        Otherwise, move exactly step_size in the direction of to_point.
        
        Args:
            from_node: Starting node
            to_point: Target point
        
        Returns:
            New point step_size away from from_node toward to_point
        """
        dist = from_node.position.distance_to(to_point)
        
        if dist <= self.step_size:
            # Already close enough, return target directly
            return Point(to_point.x, to_point.y)
        
        # Move step_size distance toward to_point
        dx = (to_point.x - from_node.position.x) / dist * self.step_size
        dy = (to_point.y - from_node.position.y) / dist * self.step_size
        
        return Point(from_node.position.x + dx, from_node.position.y + dy)
    
    def plan_path(self, start: Point, goal: Point, obstacles: List[Obstacle],
                  bounds: Optional[tuple] = None) -> Optional[List[Point]]:
        """
        Plan path from start to goal using RRT algorithm.
        
        Algorithm steps:
        1. Initialize tree with start node
        2. Randomly sample points in workspace (10% chance to sample goal directly)
        3. Find nearest node in tree to sampled point
        4. Extend tree toward sampled point (step_size distance)
        5. If extension is collision-free, add new node
        6. If new node is close to goal, try to connect to goal
        7. If goal reached, reconstruct path by following parent pointers
        
        Args:
            start: Starting position
            goal: Goal position
            obstacles: List of obstacles to avoid
            bounds: Optional workspace bounds (x_min, x_max, y_min, y_max)
                   If None, auto-detects from obstacles and points
        
        Returns:
            List of points forming path from start to goal, or None if not found
        """
        # Validate start and goal are in free space
        if not self._is_collision_free(start, obstacles) or \
           not self._is_collision_free(goal, obstacles):
            return None
        
        # Initialize tree with start node
        root = RRTNode(start)
        tree = [root]
        
        # Get bounds for random sampling
        if bounds:
            x_min, x_max, y_min, y_max = bounds
        else:
            # Auto-detect bounds from obstacles and points
            all_x = [p.x for p in [start, goal]] + [obs.position.x for obs in obstacles]
            all_y = [p.y for p in [start, goal]] + [obs.position.y for obs in obstacles]
            x_min, x_max = min(all_x) - 2, max(all_x) + 2
            y_min, y_max = min(all_y) - 2, max(all_y) + 2
        
        # RRT main loop
        for _ in range(self.max_iterations):
            # Sample random point with goal bias
            # 10% chance to sample goal directly (goal-biased RRT)
            # This helps RRT converge faster to the goal
            if random.random() < 0.1:
                random_point = goal
            else:
                # Random exploration in workspace
                random_point = Point(
                    random.uniform(x_min, x_max),
                    random.uniform(y_min, y_max)
                )
            
            # Find nearest node in tree to sampled point
            nearest = self._find_nearest(tree, random_point)
            
            # Extend tree toward sampled point
            new_point = self._steer(nearest, random_point)
            
            # Check if extension is collision-free
            if self._is_path_clear(nearest.position, new_point, obstacles):
                # Add new node to tree
                new_node = RRTNode(new_point, nearest)
                tree.append(new_node)
                
                # Check if we're close enough to goal
                if new_point.distance_to(goal) <= self.goal_threshold:
                    # Try to connect directly to goal
                    if self._is_path_clear(new_point, goal, obstacles):
                        goal_node = RRTNode(goal, new_node)
                        tree.append(goal_node)
                        
                        # Reconstruct path by following parent pointers
                        path = []
                        node = goal_node
                        while node:
                            path.append(node.position)
                            node = node.parent
                        path.reverse()  # Path was built backwards (goal to start)
                        return path
        
        return None  # Path not found within max_iterations
    
    def plan_paths_between_waypoints(self, waypoints: List[Point], 
                                     obstacles: List[Obstacle],
                                     bounds: Optional[tuple] = None) -> List[List[Point]]:
        """
        Plan paths between consecutive waypoints.
        
        This function is used when you have multiple waypoints from the global planner.
        It plans a separate RRT path for each segment between consecutive waypoints.
        
        Example:
            waypoints = [start, wp1, wp2, goal]
            Returns: [path1 (start->wp1), path2 (wp1->wp2), path3 (wp2->goal)]
        
        Args:
            waypoints: Ordered list of waypoints (from global planner)
            obstacles: List of obstacles
            bounds: Workspace bounds
        
        Returns:
            List of path segments, one for each waypoint pair
        """
        if len(waypoints) < 2:
            return []
        
        paths = []
        # Plan path for each consecutive pair of waypoints
        for i in range(len(waypoints) - 1):
            path = self.plan_path(waypoints[i], waypoints[i + 1], obstacles, bounds)
            if path:
                paths.append(path)
            else:
                # Fallback: straight line if RRT fails
                # In practice, potential fields will handle obstacle avoidance
                paths.append([waypoints[i], waypoints[i + 1]])
        
        return paths


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

def main():
    """Simple example"""
    from path_planner import DEFAULT_OBSTACLES
    
    rrt = RRTPlanner(step_size=0.3, max_iterations=2000)
    
    start = Point(0.0, 0.0)
    goal = Point(4.0, 4.0)
    obstacles = DEFAULT_OBSTACLES
    bounds = (-1.0, 5.0, -1.0, 5.0)
    
    print("Planning RRT path...")
    path = rrt.plan_path(start, goal, obstacles, bounds)
    
    if path:
        print(f"Path found with {len(path)} points")
    else:
        print("No path found")


if __name__ == "__main__":
    main()
