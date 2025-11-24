# Robot Path Planning - Work in Progress

A Python-based global path planning system for robot navigation using a Traveling Salesman Problem (TSP) approach for waypoint optimization and A* algorithm for obstacle avoidance. Designed for future integration with Isaac Sim.

## Current Status

**Work in Progress** - Basic structure and core components implemented.

## Features (Planned)

- Global path planning with A* algorithm
- TSP-based waypoint optimization  
- Dynamic obstacle avoidance (2 moving obstacles)
- Real-time path replanning
- Isaac Sim integration

## Installation

```bash
pip install -r requirements.txt
```

## Basic Usage

```python
from path_planner import PathPlanner, Obstacle, Point
from simulation import Simulation

# Create simulation
sim = Simulation()

# Setup scenario
start = Point(0.0, 0.0)
goal = Point(4.0, 4.0)

obstacles = [
    Obstacle(Point(1.5, 1.5), 0.4, Point(0.3, 0.2), id=1),
    Obstacle(Point(2.5, 2.5), 0.3, Point(-0.2, 0.3), id=2)
]

sim.setup_scenario(start, goal, obstacles)

# Plan path
path = sim.planner.plan_global_path(start, goal)
```

## File Structure

```
simulation/
├── path_planner.py      # Core path planning algorithms (A*, TSP)
├── simulation.py        # Basic simulation class
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Next Steps

- Complete path planning implementation
- Add robot movement simulation
- Implement dynamic obstacle handling
- Integrate with Isaac Sim

