# Robot Path Planning System

A Python-based integrated path planning system for robot navigation combining global planning (TSP) with local planning (RRT + Potential Fields) for obstacle avoidance.

## Overview

The system uses a two-level planning approach:
- **Global Planner (TSP)**: Optimizes waypoint order every 5 seconds to minimize travel distance
- **Local Planner (RRT + Potential Fields)**: Handles real-time navigation and collision-free pathfinding between waypoints

## Features

- TSP-based waypoint optimization with obstacle awareness
- RRT path planning for collision-free navigation
- Potential field integration for dynamic obstacle avoidance
- Dynamic waypoint generation (no hardcoded waypoints)
- Periodic replanning every 5 seconds
- Visualization of potential fields and robot paths

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from complete_system import CompleteSystem
from local_potential_field_demo_dynamic import obstacles_true, obstacle_speeds, sigma
import numpy as np

# Initialize system
system = CompleteSystem(replan_interval=5.0, rrt_step_size=0.3, rrt_max_iter=300)

# Set start and goal positions
q = np.array([0.0, 0.0])
q_goal = np.array([10.0, 10.0])

# Run simulation
# (See complete_system.py main() for full example)
```

## File Structure

```
simulation/
├── complete_system.py          # Main integrated system (TSP + RRT + Potential Fields)
├── global_planner_simple.py    # Standalone global planner
├── rrt_planner.py              # Standalone RRT planner
├── rrt_potential_integrated.py # RRT + Potential Fields integration
├── figures/                    # Template visualization figures
├── generated_figures/          # Generated figures from each run
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## System Components

- **GlobalPlanner**: TSP-based waypoint optimization with obstacle-aware path checking
- **RRTPlanner**: Rapidly-exploring Random Tree for collision-free path generation
- **CompleteSystem**: Orchestrates global and local planning

## Visualization

The system generates visualization figures showing:
- 3D potential field surface
- 2D contour map with force field
- Robot path and obstacle positions

Figures are saved to `generated_figures/` with timestamps for each run.
