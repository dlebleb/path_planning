# RRT + TSP with Tangent Waypoints

This system implements path planning with tangent waypoints around obstacles, matching the approach shown in Figure 4.4.

## Overview

The system has three main components:

1. **Global Planner (TSP)**: Generates tangent waypoints around obstacles and optimizes their order
2. **RRT Planner**: Plans collision-free paths between waypoints
3. **Integration**: Combines both to create smooth paths around obstacles

## How It Works

### Step 1: Generate Tangent Waypoints

For each obstacle that blocks the path from start (DR1) to goal (DR2):

1. **Detect obstacles** between start and goal
2. **Compute tangent points** where lines from start/goal are tangent to the obstacle circle
3. **Select waypoints** on the obstacle boundary that allow smooth navigation around it

**Key Function**: `compute_tangent_waypoints(start, goal, obstacles)`

This matches the image where waypoints are positioned at tangent points on obstacle boundaries (orange regions).

### Step 2: Optimize Waypoint Order (TSP)

Uses a nearest-neighbor TSP heuristic to find the optimal order to visit waypoints:

1. Start at current position
2. Always pick the nearest unvisited waypoint
3. Continue until all waypoints visited
4. End at goal

**Key Function**: `optimize_waypoint_order(start, waypoints, goal)`

### Step 3: Plan RRT Paths Between Waypoints

For each consecutive pair of waypoints, RRT plans a collision-free path:

1. Builds a tree from start waypoint toward end waypoint
2. Randomly explores the workspace
3. Finds collision-free path avoiding obstacles
4. Returns smooth path between waypoints

**Key Function**: `plan_path(start, goal, obstacles)`

## Usage Example

```python
from global_planner_simple import Point, Obstacle, GlobalPlanner
from rrt_planner import RRTPlanner

# Setup
start = Point(-5.0, -5.0)  # DR1
goal = Point(10.0, 10.0)   # DR2
obstacles = [
    Obstacle(Point(0.0, 0.0), radius=2.0),
    Obstacle(Point(5.0, 5.0), radius=1.5),
]

# Step 1: Generate tangent waypoints and optimize order
global_planner = GlobalPlanner()
waypoint_order = global_planner.plan_path(start, goal, obstacles)
# Returns: [start, tangent_wp1, tangent_wp2, ..., goal]

# Step 2: Plan RRT paths between waypoints
rrt = RRTPlanner()
all_paths = []
for i in range(len(waypoint_order) - 1):
    path = rrt.plan_path(waypoint_order[i], waypoint_order[i+1], obstacles)
    all_paths.append(path)
```

## Files

- `global_planner_simple.py`: TSP planner with tangent waypoint generation
- `rrt_planner.py`: RRT path planner
- `rrt_tsp_demo.py`: Complete demo showing the system in action
- `test_tangent_waypoints.py`: Test script to verify tangent computation

## Key Features

✅ **True Tangent Points**: Computes actual geometric tangent points on obstacle boundaries  
✅ **TSP Optimization**: Orders waypoints to minimize total travel distance  
✅ **RRT Path Planning**: Creates collision-free paths between waypoints  
✅ **Simple & Clean**: Easy to understand and integrate with other systems  

## Matching the Image

The system now matches Figure 4.4:
- **DR1** = start position
- **DR2** = goal position  
- **Orange obstacles** = obstacles between start and goal
- **Tangent waypoints** = points on obstacle boundaries where path is tangent
- **TSP ordering** = optimal order to visit waypoints (dashed line in image)
- **RRT paths** = smooth collision-free paths between waypoints (solid line in image)

