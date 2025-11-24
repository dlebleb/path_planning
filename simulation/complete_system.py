"""
Complete Path Planning System

Combines:
- Global Planner (TSP): Optimizes waypoint order every 5 seconds
- Local Planner (RRT + Potential Fields): Navigates between waypoints

Workflow:
1. Global planner provides optimized waypoint order
2. Local planner navigates to next waypoint using RRT + Potential Fields
3. When waypoint reached, move to next waypoint
4. Global planner replans every 5 seconds
"""

import numpy as np
import math
import random
import time
from typing import List, Optional, Tuple

# Import potential field functions from existing code
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from local_potential_field_demo_dynamic import (
    total_force, dt  # total_force calculates attractive + repulsive forces, dt is time step
)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class Point:
    """2D point representation"""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([self.x, self.y])
    
    def distance_to(self, other: 'Point') -> float:
        """Calculate Euclidean distance"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> 'Point':
        """Create Point from numpy array"""
        return cls(float(arr[0]), float(arr[1]))


class Obstacle:
    """Circular obstacle for collision checking"""
    def __init__(self, position: Point, radius: float):
        self.position = position
        self.radius = radius
    
    def collides_with(self, point: Point, safety_margin: float = 0.0) -> bool:
        """Check if point is within obstacle radius w/ a safety margin"""
        distance = self.position.distance_to(point)
        return distance <= (self.radius + safety_margin)


class RRTNode:
    """Node in RRT tree (stores position and parent for path reconstruction)"""
    def __init__(self, position: Point, parent=None):
        self.position = position
        self.parent = parent


# ---------------------------------------------------------------------------
# Global Planner (TSP)
# ---------------------------------------------------------------------------

class GlobalPlanner:
    """
    Global planner using TSP nearest-neighbor heuristic.
    Optimizes waypoint order to minimize travel distance. Replans every 5 seconds.
    """
    
    def __init__(self, replan_interval: float = 5.0):
        """Initialize global planner"""
        self.replan_interval = replan_interval
        self.last_replan_time = 0.0
        self.current_waypoint_order: List[Point] = []
    
    def optimize_waypoint_order(self, start: Point, waypoints: List[Point], 
                                goal: Point) -> List[Point]:
        """
        Optimize waypoint order using nearest-neighbor TSP.
        Greedy algorithm: always pick nearest unvisited waypoint.
        takes in the start point, the waypoints to visit, and the goal point & returns the optimized waypoint order
        """
        if not waypoints:
            return [start, goal]
        
        optimized = [start]
        remaining = waypoints.copy()
        current = start
        
        while remaining:
            nearest = min(remaining, key=lambda p: current.distance_to(p))
            optimized.append(nearest)
            remaining.remove(nearest)
            current = nearest
        
        optimized.append(goal)
        return optimized
    
    def plan_path(self, current_pos: Point, goal: Point, 
                  waypoints: List[Point]) -> List[Point]:
        """
        Calculates a new waypoint order and stores it internally.
        
        This CALCULATES a new waypoint order and STORES it internally.
        Use this when you want to create/update the plan.
        
        Returns: The optimized waypoint sequence
        """
        optimized = self.optimize_waypoint_order(current_pos, waypoints, goal)
        self.current_waypoint_order = optimized  # Store it
        return optimized
    
    def should_replan(self, current_time: float) -> bool:
        """
        Check if it's time to replan (every 5 seconds).
        
        "Replan" means recalculating the waypoint order from the robot's
        current position. This adapts the plan as the robot moves.
        """
        if current_time - self.last_replan_time >= self.replan_interval:
            self.last_replan_time = current_time
            return True
        return False
    
    def get_current_waypoint_order(self) -> List[Point]:
        """
        Get the currently stored waypoint order (without recalculating).
        
        Difference from plan_path():
        - plan_path(): CALCULATES new order and stores it
        - get_current_waypoint_order(): Just RETRIEVES the stored order
        
        Use this when you just need to read the existing plan.
        """
        return self.current_waypoint_order


# ---------------------------------------------------------------------------
# Local Planner (RRT + Potential Fields)
# ---------------------------------------------------------------------------

class RRTPlanner:
    """
    RRT path planner for generating collision-free paths.
    Builds tree by randomly sampling and extending toward goal.
    Potential field force biases sampling .
    """
    
    def __init__(self, step_size: float = 0.5, max_iterations: int = 500,
                 goal_threshold: float = 0.3, safety_margin: float = 0.2):
        """Initialize RRT planner"""
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_threshold = goal_threshold
        self.safety_margin = safety_margin
    
    def _is_collision_free(self, point: Point, obstacles: List[Obstacle]) -> bool:
        """Check if a point is not inside any obstacle"""
        for obs in obstacles:
            if obs.collides_with(point, self.safety_margin):
                return False
        return True
    
    def _is_path_clear(self, p1: Point, p2: Point, obstacles: List[Obstacle]) -> bool:
        """Check if path between two points is collision-free"""
        for i in range(10):
            t = i / 10.0
            check_point = Point(
                p1.x + t * (p2.x - p1.x),
                p1.y + t * (p2.y - p1.y)
            )
            if not self._is_collision_free(check_point, obstacles):
                return False
        return True
    
    def _find_nearest(self, tree: List[RRTNode], point: Point) -> RRTNode:
        """Find the nearest node in the tree to a given point"""
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
        """
        dist = from_node.position.distance_to(to_point)
        if dist <= self.step_size:
            return Point(to_point.x, to_point.y)
        
        # Move step_size distance in the direction of to_point
        dx = (to_point.x - from_node.position.x) / dist * self.step_size
        dy = (to_point.y - from_node.position.y) / dist * self.step_size
        return Point(from_node.position.x + dx, from_node.position.y + dy)
    
    def plan_path(self, start: Point, goal: Point, obstacles: List[Obstacle], 
                  force_direction: Optional[np.ndarray] = None,
                  bounds: Optional[Tuple[float, float, float, float]] = None) -> Optional[List[Point]]:
        """
        Plan path using RRT, biased by potential field force.
        Sampling: 10% goal, 20% force direction, 70% random.
        """
        if not self._is_collision_free(start, obstacles) or \
           not self._is_collision_free(goal, obstacles):
            return None
        
        root = RRTNode(start)
        tree = [root]
        
        if bounds:
            x_min, x_max, y_min, y_max = bounds
        else:
            all_x = [p.x for p in [start, goal]] + [obs.position.x for obs in obstacles]
            all_y = [p.y for p in [start, goal]] + [obs.position.y for obs in obstacles]
            x_min, x_max = min(all_x) - 2, max(all_x) + 2
            y_min, y_max = min(all_y) - 2, max(all_y) + 2
        
        for _ in range(self.max_iterations):
            rand = random.random()
            if rand < 0.1:
                random_point = goal
            elif rand < 0.3 and force_direction is not None:
                start_np = start.to_numpy()
                direction = force_direction / (np.linalg.norm(force_direction) + 1e-6)
                biased_point = start_np + direction * self.step_size * 3
                random_point = Point(biased_point[0], biased_point[1])
            else:
                random_point = Point(
                    random.uniform(x_min, x_max),
                    random.uniform(y_min, y_max)
                )
            
            nearest = self._find_nearest(tree, random_point)
            new_point = self._steer(nearest, random_point)
            
            if self._is_path_clear(nearest.position, new_point, obstacles):
                new_node = RRTNode(new_point, nearest)
                tree.append(new_node)
                
                if new_point.distance_to(goal) <= self.goal_threshold:
                    if self._is_path_clear(new_point, goal, obstacles):
                        goal_node = RRTNode(goal, new_node)
                        tree.append(goal_node)
                        
                        path = []
                        node = goal_node
                        while node:
                            path.append(node.position)
                            node = node.parent
                        path.reverse()
                        return path
        
        return None


# ---------------------------------------------------------------------------
# Complete Integrated System
# ---------------------------------------------------------------------------

class CompleteSystem:
    """
    Complete system combining global (TSP) and local (RRT + Potential Fields) planners.
    Global planner optimizes waypoint order every 5 seconds.
    Local planner navigates between waypoints.
    """
    
    def __init__(self, replan_interval: float = 5.0, 
                 rrt_step_size: float = 0.5, rrt_max_iter: int = 500):
        """Initialize complete system"""
        self.global_planner = GlobalPlanner(replan_interval)
        self.rrt = RRTPlanner(step_size=rrt_step_size, max_iterations=rrt_max_iter)
        
        self.current_waypoint_index = 0
        self.current_rrt_path: Optional[List[Point]] = None
        self.rrt_path_index = 0
    
    def get_next_position(self, q: np.ndarray, q_goal: np.ndarray,
                         waypoints: List[Point],
                         obstacles_noisy: np.ndarray, obstacle_speeds: np.ndarray,
                         obstacles_rrt: List[Obstacle],
                         bounds: Optional[Tuple[float, float, float, float]] = None) -> Optional[np.ndarray]:
        """
        Main function called each time step to get the next robot position.
        
        This is the core of the complete system - it coordinates:
        - Global planner (TSP) for waypoint ordering
        - Local planner (RRT) for pathfinding
        - Potential fields for dynamic obstacle avoidance
        
        Args:
            q: Current robot position (numpy array [x, y])
            q_goal: Final goal position (numpy array [x, y])
            waypoints: List of waypoints to visit (Point objects)
            obstacles_noisy: Noisy obstacle positions (for potential fields)
            obstacle_speeds: Obstacle speeds (for potential fields)
            obstacles_rrt: Obstacles for RRT collision checking (Obstacle objects)
            bounds: Workspace bounds (x_min, x_max, y_min, y_max)
        
        Returns:
            Next position to move to (numpy array), or None if goal reached
        """
        # Convert numpy arrays to Point objects for easier manipulation
        current_point = Point.from_numpy(q)
        goal_point = Point.from_numpy(q_goal)
        
        # Check if robot has reached the final goal
        if current_point.distance_to(goal_point) < 0.1:
            return None  # Mission complete!
        
        current_time = time.time()
        
        # ============================================================
        # STEP 1: GLOBAL REPLANNING (every 5 seconds)
        # ============================================================
        
        if self.global_planner.should_replan(current_time):
            # Recalculate waypoint order from current position
            waypoint_order = self.global_planner.plan_path(
                current_point, goal_point, waypoints
            )
            # Reset to start of new plan
            self.current_waypoint_index = 0
            self.current_rrt_path = None  # Need new RRT path for new plan
            print(f"[Global Replan] New waypoint order: {len(waypoint_order)} waypoints")
        
        # Get the current waypoint order (either from replanning or existing)
        waypoint_order = self.global_planner.get_current_waypoint_order()
        if not waypoint_order:
            # First time - do initial planning
            waypoint_order = self.global_planner.plan_path(
                current_point, goal_point, waypoints
            )
        
        # ============================================================
        # STEP 2: DETERMINE CURRENT TARGET WAYPOINT
        # ============================================================
        # Check if we've completed all waypoints
        if self.current_waypoint_index >= len(waypoint_order):
            return None  # All waypoints visited, should be at goal
        
        # Get the waypoint we're currently navigating to
        current_target = waypoint_order[self.current_waypoint_index]
        
        # Check if we've reached the current waypoint (within 0.2m)
        if current_point.distance_to(current_target) < 0.2:
            # Move to next waypoint in the sequence
            self.current_waypoint_index += 1
            self.current_rrt_path = None  # Need new RRT path to next waypoint
            if self.current_waypoint_index >= len(waypoint_order):
                return None  # Reached final goal
            current_target = waypoint_order[self.current_waypoint_index]
        
        # ============================================================
        # STEP 3: LOCAL PLANNING (RRT + Potential Fields)
        # ============================================================
        # Calculate potential field force toward current target waypoint
        # This uses total_force() from local_potential_field_demo_dynamic.py
        # It combines:
        # - Attractive force: pulls robot toward target waypoint
        # - Repulsive force: pushes robot away from obstacles
        target_np = current_target.to_numpy()
        force = total_force(q, target_np, obstacles_noisy, obstacle_speeds)
        
        # Check if we need a new RRT path
        # RRT path is a list of points from current position to target waypoint
        # Each "point" in the path is a waypoint along the collision-free route
        # Each "step" is moving from one point to the next in the path
        if self.current_rrt_path is None or self.rrt_path_index >= len(self.current_rrt_path):
            # Plan new RRT path from current position to target waypoint
            # RRT.plan_path() is NOT from local_potential_field_demo_dynamic.py
            # It's from the RRTPlanner class in this file - it generates a
            # collision-free path by building a tree and sampling points.
            # The potential field force biases the RRT sampling (20% of samples
            # follow the force direction, guiding RRT toward preferred paths)
            self.current_rrt_path = self.rrt.plan_path(
                current_point, current_target, obstacles_rrt, force, bounds
            )
            self.rrt_path_index = 0  # Start at beginning of new path
            
            if self.current_rrt_path is None:
                # RRT failed to find path (maybe obstacles block everything)
                # Fallback: use potential field direction directly
                direction = force / (np.linalg.norm(force) + 1e-6)
                return q + direction * dt
        
        # ============================================================
        # STEP 4: GET NEXT POSITION FROM RRT PATH
        # ============================================================
        # The RRT path is a list of points: [p1, p2, p3, ..., target]
        # We move through this path step by step:
        # - Step 1: Move from current position to p1
        # - Step 2: Move from p1 to p2
        # - Step 3: Move from p2 to p3
        # - ... and so on until we reach the target waypoint
        if self.current_rrt_path and self.rrt_path_index < len(self.current_rrt_path):
            # Get the next point in the RRT path
            next_point = self.current_rrt_path[self.rrt_path_index]
            self.rrt_path_index += 1  # Move to next point for next call
            return next_point.to_numpy()  # Convert back to numpy array
        
        return None


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

def main():
    """Complete system example"""
    from local_potential_field_demo_dynamic import (
        obstacles_true, obstacle_speeds, sigma
    )
    
    # Setup
    system = CompleteSystem(replan_interval=5.0, rrt_step_size=0.3, rrt_max_iter=300)
    
    q = np.array([0.0, 0.0])
    q_goal = np.array([10.0, 10.0])
    
    # Define waypoints
    waypoints = [
        Point(2.0, 3.0),
        Point(5.0, 5.0),
        Point(8.0, 2.0),
        Point(3.0, 8.0),
    ]
    
    # Convert obstacles for RRT
    obstacles_rrt = []
    for obs_pos in obstacles_true:
        obstacles_rrt.append(Obstacle(Point(obs_pos[0], obs_pos[1]), 0.5))
    
    bounds = (-1.0, 11.0, -1.0, 11.0)
    
    # Simulate
    path_data = [q.copy()]
    max_steps = 2000
    tolerance = 0.1
    
    print("Running complete system (Global TSP + Local RRT + Potential Fields)...")
    print(f"Waypoints to visit: {len(waypoints)}")
    
    for step in range(max_steps):
        # Update obstacles
        obstacles_true[:, 0] += obstacle_speeds[:, 0] * dt
        obstacles_true[:, 1] += obstacle_speeds[:, 1] * dt
        
        # Robot observes obstacles (noisy)
        obstacles_noisy = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)
        
        # Get next position
        next_pos = system.get_next_position(
            q, q_goal, waypoints, obstacles_noisy, obstacle_speeds, 
            obstacles_rrt, bounds
        )
        
        if next_pos is None:
            print(f"Reached goal in {step} steps!")
            break
        
        # Move robot
        q = next_pos
        path_data.append(q.copy())
        
        if step % 100 == 0:
            print(f"Step {step}: Position = ({q[0]:.2f}, {q[1]:.2f})")
        
        # Check if reached goal
        if np.linalg.norm(q - q_goal) < tolerance:
            print(f"Reached goal in {step} steps!")
            break
    
    print(f"Final path has {len(path_data)} points")


if __name__ == "__main__":
    main()

