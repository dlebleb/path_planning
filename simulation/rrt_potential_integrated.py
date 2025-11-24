"""
Integrated RRT + Potential Field System

This system combines RRT path planning with potential field navigation.
It uses your existing potential field code without modification.

How it works:
1. Calculate potential field force vector (attractive + repulsive forces)
2. Run RRT path planning, biased by the force direction
3. Get next position from RRT path
4. Move robot to that position
5. Repeat until goal is reached

Key Integration:
- Potential field provides preferred direction (force vector)
- RRT uses this direction to bias its random sampling (20% of samples)
- This combines reactive behavior (potential fields) with planning (RRT)
- Result: RRT finds collision-free paths that align with potential field guidance
"""

import numpy as np
import math
import random
from typing import List, Optional, Tuple

# Import potential field functions from existing code
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from local_potential_field_demo_dynamic import (
    attractive_force, repulsive_force, total_force,  # Force calculation functions
    k_att, k_rep, d0, dt  # Potential field parameters and time step
)


# ---------------------------------------------------------------------------
# Simple Point class for RRT
# ---------------------------------------------------------------------------

class Point:
    """Simple 2D point"""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y])
    
    def distance_to(self, other: 'Point') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> 'Point':
        return cls(float(arr[0]), float(arr[1]))


class Obstacle:
    """Simple circular obstacle"""
    def __init__(self, position: Point, radius: float):
        self.position = position
        self.radius = radius
    
    def collides_with(self, point: Point, safety_margin: float = 0.0) -> bool:
        distance = self.position.distance_to(point)
        return distance <= (self.radius + safety_margin)


class RRTNode:
    """Node in RRT tree"""
    def __init__(self, position: Point, parent=None):
        self.position = position
        self.parent = parent


# ---------------------------------------------------------------------------
# Simple RRT Planner
# ---------------------------------------------------------------------------

class RRTPlanner:
    """Simple RRT planner"""
    
    def __init__(self, step_size: float = 0.5, max_iterations: int = 500,
                 goal_threshold: float = 0.3, safety_margin: float = 0.2):
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_threshold = goal_threshold
        self.safety_margin = safety_margin
    
    def _is_collision_free(self, point: Point, obstacles: List[Obstacle]) -> bool:
        for obs in obstacles:
            if obs.collides_with(point, self.safety_margin):
                return False
        return True
    
    def _is_path_clear(self, p1: Point, p2: Point, obstacles: List[Obstacle]) -> bool:
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
        nearest = tree[0]
        min_dist = nearest.position.distance_to(point)
        for node in tree[1:]:
            dist = node.position.distance_to(point)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        return nearest
    
    def _steer(self, from_node: RRTNode, to_point: Point) -> Point:
        dist = from_node.position.distance_to(to_point)
        if dist <= self.step_size:
            return Point(to_point.x, to_point.y)
        dx = (to_point.x - from_node.position.x) / dist * self.step_size
        dy = (to_point.y - from_node.position.y) / dist * self.step_size
        return Point(from_node.position.x + dx, from_node.position.y + dy)
    
    def plan_path(self, start: Point, goal: Point, obstacles: List[Obstacle],
                  force_direction: Optional[np.ndarray] = None,
                  bounds: Optional[Tuple[float, float, float, float]] = None) -> Optional[List[Point]]:
        """Plan path using RRT, optionally biased by potential field force"""
        
        if not self._is_collision_free(start, obstacles) or \
           not self._is_collision_free(goal, obstacles):
            return None
        
        root = RRTNode(start)
        tree = [root]
        
        # Get bounds
        if bounds:
            x_min, x_max, y_min, y_max = bounds
        else:
            all_x = [p.x for p in [start, goal]] + [obs.position.x for obs in obstacles]
            all_y = [p.y for p in [start, goal]] + [obs.position.y for obs in obstacles]
            x_min, x_max = min(all_x) - 2, max(all_x) + 2
            y_min, y_max = min(all_y) - 2, max(all_y) + 2
        
        # RRT loop
        for _ in range(self.max_iterations):
            # Sample: 10% goal, 20% force direction, 70% random
            rand = random.random()
            if rand < 0.1:
                random_point = goal
            elif rand < 0.3 and force_direction is not None:
                # Bias toward potential field direction
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
                        
                        # Reconstruct path
                        path = []
                        node = goal_node
                        while node:
                            path.append(node.position)
                            node = node.parent
                        path.reverse()
                        return path
        
        return None


# ---------------------------------------------------------------------------
# Integrated System
# ---------------------------------------------------------------------------

class IntegratedPlanner:
    """
    Integrates RRT path planning with potential field navigation.
    
    This class combines two approaches:
    - RRT: Plans collision-free paths (planning approach)
    - Potential Fields: Provides reactive navigation (reactive approach)
    
    Workflow:
    1. Calculate potential field force vector (using existing functions)
       - Attractive force: pulls robot toward goal
       - Repulsive force: pushes robot away from obstacles
    2. Run RRT path planning, biased by force direction
       - 20% of RRT samples follow potential field direction
       - This guides RRT toward preferred paths
    3. Get next position from RRT path
    4. Move robot to that position
    5. Repeat until goal is reached
    
    Benefits:
    - RRT ensures collision-free paths
    - Potential fields guide RRT toward efficient routes
    - Handles dynamic obstacles through potential fields
    """
    
    def __init__(self, rrt_step_size: float = 0.5, rrt_max_iter: int = 500):
        """
        Initialize integrated planner.
        
        Args:
            rrt_step_size: Maximum distance RRT extends tree each iteration
            rrt_max_iter: Maximum RRT iterations before giving up
        """
        self.rrt = RRTPlanner(step_size=rrt_step_size, max_iterations=rrt_max_iter)
        self.current_path: Optional[List[Point]] = None  # Current RRT path being followed
        self.path_index = 0  # Current position in RRT path
    
    def get_next_position(self, q: np.ndarray, q_goal: np.ndarray,
                         obstacles_noisy: np.ndarray, obstacle_speeds: np.ndarray,
                         obstacles_rrt: List[Obstacle],
                         bounds: Optional[Tuple[float, float, float, float]] = None) -> Optional[np.ndarray]:
        """
        Get next position using RRT + Potential Field integration.
        
        This is the main function called each time step. It:
        1. Calculates potential field force toward goal
        2. Plans/uses RRT path (guided by force direction)
        3. Returns next position along the path
        
        Args:
            q: Current robot position (numpy array)
            q_goal: Goal position (numpy array)
            obstacles_noisy: Noisy obstacle positions (for potential fields)
            obstacle_speeds: Obstacle speeds (for potential fields)
            obstacles_rrt: Obstacles for RRT collision checking
            bounds: Workspace bounds
        
        Returns:
            Next position to move to, or None if at goal
        """
        current_point = Point.from_numpy(q)
        goal_point = Point.from_numpy(q_goal)
        
        # Check if robot has reached goal
        if current_point.distance_to(goal_point) < 0.1:
            return None
        
        # Step 1: Calculate potential field force
        # This uses your existing total_force function which combines:
        # - Attractive force: toward goal
        # - Repulsive force: away from obstacles
        force = total_force(q, q_goal, obstacles_noisy, obstacle_speeds)
        
        # Step 2: Plan or use existing RRT path
        # If no path exists or current path is exhausted, plan a new one
        if self.current_path is None or self.path_index >= len(self.current_path):
            # Plan new RRT path from current position to goal
            # RRT sampling is biased by potential field force direction (20% of samples)
            self.current_path = self.rrt.plan_path(
                current_point, goal_point, obstacles_rrt, force, bounds
            )
            self.path_index = 0
            
            if self.current_path is None:
                # RRT failed to find path, fallback to potential field direction
                # This ensures robot can still move even if RRT fails
                direction = force / (np.linalg.norm(force) + 1e-6)
                next_pos = q + direction * dt
                return next_pos
        
        # Step 3: Get next position from RRT path
        # Follow the planned RRT path step by step
        if self.current_path and self.path_index < len(self.current_path):
            next_point = self.current_path[self.path_index]
            self.path_index += 1
            return next_point.to_numpy()
        
        return None


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

def main():
    """Example using integrated system with your potential field code"""
    from local_potential_field_demo_dynamic import (
        obstacles_true, obstacle_speeds, sigma
    )
    
    # Setup
    planner = IntegratedPlanner(rrt_step_size=0.3, rrt_max_iter=300)
    
    q = np.array([0.0, 0.0])
    q_goal = np.array([10.0, 10.0])
    
    # Convert obstacles for RRT
    obstacles_rrt = []
    for obs_pos in obstacles_true:
        obstacles_rrt.append(Obstacle(Point(obs_pos[0], obs_pos[1]), 0.5))
    
    bounds = (-1.0, 11.0, -1.0, 11.0)
    
    # Simulate
    path_data = [q.copy()]
    max_steps = 1000
    tolerance = 0.1
    
    print("Running integrated RRT + Potential Field system...")
    
    for step in range(max_steps):
        # Update obstacles (from your code)
        obstacles_true[:, 0] += obstacle_speeds[:, 0] * dt
        obstacles_true[:, 1] += obstacle_speeds[:, 1] * dt
        
        # Robot observes obstacles (noisy)
        obstacles_noisy = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)
        
        # Get next position using integrated system
        next_pos = planner.get_next_position(
            q, q_goal, obstacles_noisy, obstacle_speeds, obstacles_rrt, bounds
        )
        
        if next_pos is None:
            print(f"Reached goal in {step} steps!")
            break
        
        # Move robot
        q = next_pos
        path_data.append(q.copy())
        
        if step % 50 == 0:
            print(f"Step {step}: Position = ({q[0]:.2f}, {q[1]:.2f})")
        
        # Check if reached goal
        if np.linalg.norm(q - q_goal) < tolerance:
            print(f"Reached goal in {step} steps!")
            break
    
    print(f"Final path has {len(path_data)} points")


if __name__ == "__main__":
    main()
