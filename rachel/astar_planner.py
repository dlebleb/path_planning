"""
A* Path Planner for comparison with the TSP + RRT + Potential Field system.

This planner finds optimal collision-free paths from start to goal on a 2D grid
using A* search with an 8-connected neighborhood. 

Uses the same obstacle layout as the integrated system (elliptical obstacles are
approximated as circles with max axis length plus safety margin).

Run directly to see a plotted path:
    python astar_planner.py
"""

import heapq
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Setup: Start, Goal, Obstacles
START = np.array([-40.0, -40.0])
GOAL = np.array([20.0, 10.0])  # Moved away from obstacles to ensure it's collision-free

# Obstacle centers
obstacles_true = np.array(
    [[-5, -5], [-3, -3], [3, 3.5], [6, 7], [9, 9], [8, 4], [5, 5]]
)

# Obstacle velocities (used only to scale ellipse size; motion ignored for A*)
obstacle_speeds = np.array(
    [
        [-0.1, 0.1],
        [-0.2, 0.2],
        [0.1, 0.2],
        [-0.2, -0.1],
        [0.1, -0.1],
        [-0.1, 0.1],
        [0.2, 0.1],
    ]
) * 5

# Ellipse parameters
a0 = 2.0
b0 = 1.0
alpha = 1.2
beta = 0.3
sizes = np.array([1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1])
a_base = a0 * sizes
b_base = b0 * sizes
SAFETY_MARGIN = 0.5

def elliptical_to_radius(i: int) -> float:
    """Convert elliptical obstacle to circular radius (max axis + margin)."""
    vx, vy = obstacle_speeds[i]
    vmag = math.hypot(vx, vy)
    a = a_base[i] + alpha * vmag
    b = b_base[i] + beta * vmag
    return max(a, b) + SAFETY_MARGIN

def build_circular_obstacles() -> List[Tuple[float, float, float]]:
    """Build circular obstacles from elliptical obstacles for collision checking."""
    obstacles = []
    for i in range(len(obstacles_true)):
        x, y = obstacles_true[i]
        r = elliptical_to_radius(i)
        obstacles.append((x, y, r))
    return obstacles


@dataclass
class Node:
    """Node in A* search tree."""
    x: float
    y: float
    g: float  # Cost from start to this node
    h: float  # Heuristic (estimated cost to goal)
    parent: Optional['Node']

    @property
    def f(self) -> float:
        """Total estimated cost: f = g + h"""
        return self.g + self.h


def heuristic(pos: Tuple[float, float], goal: Tuple[float, float]) -> float:
    """Euclidean distance heuristic for A*."""
    return math.hypot(goal[0] - pos[0], goal[1] - pos[1])


def is_free(x: float, y: float, obstacles: List[Tuple[float, float, float]]) -> bool:
    """Check if point (x, y) is collision-free."""
    for ox, oy, r in obstacles:
        if math.hypot(x - ox, y - oy) < r:
            return False
    return True


def astar(
    start: Tuple[float, float],
    goal: Tuple[float, float],
    obstacles: List[Tuple[float, float, float]],
    bounds: Tuple[float, float, float, float] = (-40, 40, -40, 40),
    step: float = 0.5,
) -> Optional[List[Tuple[float, float]]]:
    """
    A* search algorithm for finding optimal path from start to goal.
    
    Uses an 8-connected grid (step size 0.5) and Euclidean heuristic.
    Returns the path as a list of (x, y) tuples, or None if no path exists.
    """
    x_min, x_max, y_min, y_max = bounds

    # Snap start and goal to grid (ensures we work with discrete grid points)
    start_n = (round(start[0] / step) * step, round(start[1] / step) * step)
    goal_n = (round(goal[0] / step) * step, round(goal[1] / step) * step)

    # Initialize priority queue (open set) - nodes to explore, sorted by f-cost
    open_heap: List[Tuple[float, int, Node]] = []
    counter = 0  # Used to break ties in heap (ensures stable ordering)
    
    # Create start node: g=0 (no distance traveled), f = g + heuristic
    start_node = Node(start_n[0], start_n[1], 0.0, heuristic(start_n, goal_n), None)
    heapq.heappush(open_heap, (start_node.f, counter, start_node))

    # Data structures to track search progress
    came_from: Dict[Tuple[float, float], Node] = {}  # For path reconstruction
    g_score: Dict[Tuple[float, float], float] = {start_n: 0.0}  # Best known g-cost for each position
    visited = set()  # Positions we've already fully explored

    # 8-connected neighborhood: 4 cardinal directions + 4 diagonals
    directions = [
        (-1, 0),  # left
        (1, 0),   # right
        (0, -1),  # down
        (0, 1),   # up
        (-1, -1), # down-left
        (-1, 1),  # up-left
        (1, -1),  # down-right
        (1, 1),   # up-right
    ]

    # Main A* search loop: explore nodes until we reach the goal
    while open_heap:
        # Get node with lowest f-cost from priority queue
        _, _, current = heapq.heappop(open_heap)
        current_pos = (current.x, current.y)

        # Skip if we've already explored this position with a better path
        if current_pos in visited:
            continue

        # Mark as visited
        visited.add(current_pos)

        # Check if we reached the goal
        if heuristic(current_pos, goal_n) < step * 0.5:
            # Reconstruct path by following parent pointers
            path = []
            node = current
            while node:
                path.append((node.x, node.y))
                node = node.parent
            path.reverse()
            return path

        # Explore neighbors
        for dx, dy in directions:
            neighbor_pos = (
                round((current.x + dx * step) / step) * step,
                round((current.y + dy * step) / step) * step,
            )

            # Check bounds
            if not (x_min <= neighbor_pos[0] <= x_max and y_min <= neighbor_pos[1] <= y_max):
                continue

            # Check collision
            if not is_free(neighbor_pos[0], neighbor_pos[1], obstacles):
                continue

            # Skip if already visited
            if neighbor_pos in visited:
                continue

            # Calculate cost to reach neighbor (diagonal moves cost more)
            move_cost = step if abs(dx) + abs(dy) == 1 else step * math.sqrt(2)
            tentative_g = current.g + move_cost

            # If we found a better path to this neighbor, update it
            if neighbor_pos not in g_score or tentative_g < g_score[neighbor_pos]:
                g_score[neighbor_pos] = tentative_g
                h = heuristic(neighbor_pos, goal_n)
                neighbor_node = Node(neighbor_pos[0], neighbor_pos[1], tentative_g, h, current)
                counter += 1
                heapq.heappush(open_heap, (neighbor_node.f, counter, neighbor_node))
                came_from[neighbor_pos] = current

    # No path found
    return None


def plot_path(
    path: List[Tuple[float, float]],
    obstacles: List[Tuple[float, float, float]],
    bounds: Tuple[float, float, float, float],
    outfile: str = "figures/fig_astar_path.png",
):
    """Plot the A* path with obstacles."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Draw obstacles
    for ox, oy, r in obstacles:
        circle = plt.Circle((ox, oy), r, color="orange", alpha=0.5)
        ax.add_patch(circle)

    # Draw path
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, "r-", linewidth=2, label="A* Path")
        ax.plot(path_x[0], path_y[0], "gs", markersize=12, label="Start")
        ax.plot(path_x[-1], path_y[-1], "rs", markersize=12, label="Goal")

    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("A* Path Planning")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"Saved A* path figure to {outfile}")
    plt.close()


def main():
    """Run A* path planning and generate visualization."""
    obstacles = build_circular_obstacles()
    bounds = (-40, 40, -40, 40)
    path = astar(tuple(START), tuple(GOAL), obstacles, bounds=bounds, step=0.5)
    if path is None:
        print("A* failed to find a path.")
        return
    print(f"A* path found with {len(path)} points. Total length ≈ {path_length(path):.2f}")
    plot_path(path, obstacles, bounds)


def path_length(path: List[Tuple[float, float]]) -> float:
    """Calculate total Euclidean distance along the path."""
    total = 0.0
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


if __name__ == "__main__":
    main()
