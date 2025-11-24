"""
Simple Global Path Planner using TSP (Traveling Salesman Problem).

This is a simplified global planner that only handles waypoint ordering.
The local planner (RRT + Potential Fields) handles actual pathfinding and obstacle avoidance.

Purpose:
- Optimize the order in which waypoints are visited to minimize total travel distance
- Replan every 5 seconds to adapt to robot's current position

Algorithm (Nearest-Neighbor TSP):
1. Start at the current robot position
2. Find the nearest unvisited waypoint
3. Go to that waypoint
4. Repeat until all waypoints are visited
5. End at the goal

This is a greedy heuristic - not optimal but fast and works well for real-time planning.
"""

import math
from typing import List, Optional


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


class GlobalPlanner:

    
    def __init__(self, replan_interval: float = 5.0):
        """
        Initialize global planner.
        
        Args:
            replan_interval: Time in seconds between global replanning (default: 5.0)
        """
        self.replan_interval = replan_interval
        self.last_replan_time = 0.0  # Timestamp of last replanning
        self.current_waypoint_order: List[Point] = []  # Current optimized sequence
    
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
                  waypoints: List[Point]) -> List[Point]:
        """
        Plan global path - optimizes waypoint order from current position.
        
        This is called periodically (every 5 seconds) to ensure the robot
        follows the shortest route as it moves through the environment.
        
        Args:
            current_pos: Current robot position
            goal: Final destination
            waypoints: Waypoints to visit
        
        Returns:
            Optimized waypoint sequence
        """
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


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

def main():
    """Simple example"""
    planner = GlobalPlanner(replan_interval=5.0)
    
    start = Point(0.0, 0.0)
    goal = Point(10.0, 10.0)
    waypoints = [
        Point(2.0, 3.0),
        Point(5.0, 5.0),
        Point(8.0, 2.0),
        Point(3.0, 8.0),
    ]
    
    # Plan path
    optimized = planner.plan_path(start, goal, waypoints)
    
    print("Optimized waypoint order:")
    for i, wp in enumerate(optimized):
        print(f"  {i+1}. {wp.to_tuple()}")


if __name__ == "__main__":
    main()

