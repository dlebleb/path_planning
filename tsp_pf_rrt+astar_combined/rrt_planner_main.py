"""
RRT (Rapidly-exploring Random Tree) Path Planner.

This planner generates collision-free paths between waypoints while avoiding obstacles.

RRT Algorithm:
1. Start with a tree containing only the start position
2. Randomly sample points in the workspace
3. Find the nearest node in the tree to the sampled point
4. Extend the tree toward the sampled point (step_size distance)
5. If the extension is collision-free, add new node to tree
6. Repeat until goal is reached or max iterations

Designed to work with:
- TSP planner for waypoint ordering
- Potential fields for dynamic obstacle avoidance
"""

import random
import numpy as np
from typing import List, Optional
from TSP_Rachel import Point
from APF_RRT_Astar import *
from TSP_main import *

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
    
    def __init__(self, step_size: float = 0.15, max_iterations: int = 3000, 
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
    
    def _is_collision_free(self, point: Point, obstacles_noisy: np.ndarray, obstacle_speeds: np.ndarray) -> bool:
        """
        Check if a point is not inside any of the elliptical obstacles.
        
        Args:
            point: Point to check
            obstacles: List of elliptical obstacles
        
        Returns:
            True if point is collision-free according to humans, False otherwise
        """
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
                
            a = a_base[i] + alpha * vmag 
            b = b_base[i] + beta  * vmag
            a = min(a, a_max)
            b = min(b, b_max)
            
            # move obstacle center to origin and transform
            obs_x, obs_y = obs[0], obs[1]

            eps = 1e-12
            # move robot center to origin and transform
            q_x, q_y = (point.x-obs_x)/(a+eps), (point.y-obs_y)/(b+eps)

            # distance of the robot to origin -1
            dE = np.sqrt(q_x**2 + q_y**2) - 1
            
            # COLLISION condition: robot is inside the ellipse
            if dE < 0: 
               return False  #collision detected with the ellipse
    
        return True

    # returns false or true if a point is inside the rectangular region
    def _is_collision_free_rect(self, point: Point, rect_obstacles: List):
        X, Y = point.x, point.y
        for rect in rect_obstacles:
            x, y, w, h = rect
            if (x <= X <= x + w) and (y <= Y <= y + h):
                return False  # point is inside the rectangular region
        return True

    def _is_path_clear(self, p1: Point, p2: Point, obstacles_noisy: np.ndarray, obstacle_speeds: np.ndarray) -> bool:
        """
        Check if the path between two points is collision-free from ellipses.
        
        Samples 50 points along the line segment and checks each for collisions.
        This ensures the entire path segment is safe, not just the endpoints.
        
        Args:
            p1: Start point
            p2: End point
            obstacles: List of obstacles
        
        Returns:
            True if path is clear, False if collision detected
        """
        # Check multiple points along the path from p1 to p2 (exclude points)
        # removed [1:-1], for t in np.linspace(0, 1, 50, endpoint = False):
        
        dist = p1.distance_to(p2)
        num_samples = max(10, int(dist / 0.05))
    
        for t in np.linspace(0, 1, num_samples):
            check_point = Point(
                p1.x + t * (p2.x - p1.x),
                p1.y + t * (p2.y - p1.y)
            )
            if not self._is_collision_free(check_point, obstacles_noisy, obstacle_speeds):
                return False
        return True
    
    def _is_path_clear_rect(self, p1: Point, p2: Point, rect_obstacles: List, expanded_rects: List) -> bool:
            """
            Check if the path between two points is collision-free from rectangles.
            
            Samples 50 points along the line segment and checks each for collisions.
            This ensures the entire path segment is safe, not just the endpoints.
            
            Args:
                p1: Start point
                p2: End point
                obstacles: List of obstacles
            
            Returns:
                True if path is clear, False if collision detected
            """
            p1_coord = np.array([p1.x, p1.y])
            p2_coord = np.array([p2.x, p2.y])
            eq = line_mb(p1_coord,p2_coord)
            # Check multiple points along the path for rectangular obstacles
            for rect, erect in zip(rect_obstacles, expanded_rects):
                eps = 0.015 # error threshold
                # 1) radius approximation (coarse filter)
                center, radius = computeRectangleCircumcircle(rect)
                d = point_to_line_distance(eq, center)
                if d >= radius + eps:
                    continue

                # 2) actual check on the SAME expanded rect
                # for t in np.linspace(0, 1, 50, endpoint = False)[1:-1]:
                for t in np.linspace(0, 1, 50):
                    check_point = Point(
                        p1.x + t * (p2.x - p1.x),
                        p1.y + t * (p2.y - p1.y)
                    )
                    if not self._is_collision_free_rect(check_point, [rect]): # [erect] aslinda ama degistirdim. 
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
    
    def _push_out_of_collision(self, q: Point, rect_obstacles: List, obstacles_noisy: np.ndarray, obstacle_speeds: np.ndarray, F: np.ndarray, max_iter = 20) -> Point:
        """
        Move step_size distance from goal or start if there is collision between elliptical obstacles and rectangles.
        
        If q (start or goal of the tree) is colliding with obstacles,
        move exactly step_size in the direction of the force coming from the global path.
        
        Args:
            q: Start/Goal
            F: Force coming from APF
        
        Returns:
            New point step_size away from start/goal toward APF
        """
        normF = np.linalg.norm(F)
        if normF < 1e-8:
            direction = np.zeros_like(F)
        else:
            direction = F / normF
        #direction = F / normF 

        for _ in range(max_iter):

            free = (
                self._is_collision_free(q, obstacles_noisy, obstacle_speeds) and
                self._is_collision_free_rect(q, rect_obstacles)
            )

            if free:
                return q

            q = Point(
                q.x + self.step_size * direction[0],
                q.y + self.step_size * direction[1])

        # check again after max_iterations
        final_free = (
                self._is_collision_free(q, obstacles_noisy, obstacle_speeds) and
                self._is_collision_free_rect(q, rect_obstacles)
            )
        
        if not final_free:
            print("Warning: Could not fully escape obstacle after max_iter.")
        return q

    def plan_path(self, start: Point, goal: Point, obstacles_noisy:np.ndarray, rect_obstacles: List, obstacle_speeds: np.ndarray, F: np.ndarray, expanded_rects: List,
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
        # Validate start and goal are in free space, if not push out of collision
        start = self._push_out_of_collision(start, rect_obstacles, obstacles_noisy, obstacle_speeds, F)
        goal  = self._push_out_of_collision(goal, rect_obstacles, obstacles_noisy, obstacle_speeds, F)

    
        # Initialize tree with start node
        root = RRTNode(start)
        tree = [root]
        
        # Get bounds for random sampling
        if bounds:
            x_min, x_max, y_min, y_max = bounds
        else:
            # Auto-detect bounds from start and goal points initially
            all_x = [p.x for p in [start, goal]]
            all_y = [p.y for p in [start, goal]]
            margin  = 10
            x_min, x_max = min(all_x) - margin, max(all_x) + margin
            y_min, y_max = min(all_y) - margin, max(all_y) + margin
        
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
            
            # Check if extension is collision-free from elliptical onbstacles and rectangles
            if self._is_path_clear(nearest.position, new_point, obstacles_noisy, obstacle_speeds) and self._is_path_clear_rect(nearest.position, new_point, rect_obstacles, expanded_rects):
                # Add new node to tree
                new_node = RRTNode(new_point, nearest)
                tree.append(new_node)
                
                # Check if we're close enough to goal
                if new_point.distance_to(goal) <= self.goal_threshold:
                    # Try to connect directly to goal
                    if self._is_path_clear(new_point, goal, obstacles_noisy, obstacle_speeds) and self._is_path_clear_rect(new_point, goal, rect_obstacles, expanded_rects):
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
    
    # def plan_paths_between_waypoints(self, waypoints: List[Point], 
    #                                  obstacles: List[Obstacle],
    #                                  bounds: Optional[tuple] = None) -> List[List[Point]]:
    #     """
    #     Plan paths between consecutive waypoints.
        
    #     This function is used when you have multiple waypoints from the global planner.
    #     It plans a separate RRT path for each segment between consecutive waypoints.
        
    #     Example:
    #         waypoints = [start, wp1, wp2, goal]
    #         Returns: [path1 (start->wp1), path2 (wp1->wp2), path3 (wp2->goal)]
        
    #     Args:
    #         waypoints: Ordered list of waypoints (from global planner)
    #         obstacles: List of obstacles
    #         bounds: Workspace bounds
        
    #     Returns:
    #         List of path segments, one for each waypoint pair
    #     """
    #     if len(waypoints) < 2:
    #         return []
        
    #     paths = []
    #     # Plan path for each consecutive pair of waypoints
    #     for i in range(len(waypoints) - 1):
    #         path = self.plan_path(waypoints[i], waypoints[i + 1], obstacles, bounds)
    #         if path:
    #             paths.append(path)
    #         else:
    #             # Fallback: straight line if RRT fails
    #             # In practice, potential fields will handle obstacle avoidance
    #             paths.append([waypoints[i], waypoints[i + 1]])
        
    #     return paths


    # Smmothing the rrt path

    def smooth_moving_average(self, path):
        
        # Point objelerini numpy array'e çevir
        pts = np.array([[p.x, p.y] for p in path])
        smoothed = pts.copy()

        for i in range(1, len(pts)-1):
            smoothed[i] = (
                0.4*pts[i-1] +
                0.2*pts[i] +
                0.4*pts[i+1]
            )

        # tekrar Point objesine çevir
        smoothed_points = [Point(x,y) for x,y in smoothed]

        return smoothed_points

# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

def main():
    """Simple example"""

    #from path_planner import DEFAULT_OBSTACLES
    rect_obstacles, expanded_rects, eps_expanded_rects, waypoints = init_environment()
    rectangle_speeds = np.array([[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1]])
    rectangle_speeds = rectangle_speeds * 80
    obstacles_true      = np.array([[-18.0,-10.0], [18,-20], [18, 8], [22,26], [23,15], [-23,15], [5,5], [-40, -30], [15, -10], [10, 5], [-30, 0], [-20, -20], [0, 0]])
    sigma               = 0.1  # 10 cm uncertainity
    obstacles_noisy     = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)
    obstacle_speeds     = np.array([[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1], [-0.1, 0.1], [-0.2, 0.1], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2]])
    obstacle_speeds     = obstacle_speeds * 80

    start = Point(0.0, 0.0)
    goal = Point(4.0, 4.0)
    rrt = RRTPlanner(step_size=0.3, max_iterations=2000)
    #obstacles = DEFAULT_OBSTACLES
    #bounds = (-1.0, 5.0, -1.0, 5.0)
    F = np.array([-2, 3])

    print("Planning RRT path...")
    path = rrt.plan_path(start, goal, obstacles_noisy, rect_obstacles, obstacle_speeds, F, expanded_rects)
    
    if path:
        print(f"Path found with {len(path)} points")
    else:
        print("No path found")


if __name__ == "__main__":
    main()
