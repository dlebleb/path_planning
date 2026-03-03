import heapq #h[0] her zaman minimumdur. h[1], h[2], h[3] sıralı değildir.
import numpy as np
from APF3 import *
from matplotlib.patches import Ellipse


class Node:
    def __init__(self, state, parent=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.g = g  # Cost from start node to current node
        self.h = h  # Heuristic estimate of cost from current node to goal node

    def f(self):
        return self.g + self.h

def astar(start_state, goal_state, heuristic_func, successors_func, obstacles_noisy, obstacle_speeds, rect_obstacles):
    tolerance = 1
    open_list = []
    closed_set = set()
    g_score = {}

    start_node = Node(state=start_state, g=0, h=heuristic_func(start_state, goal_state))
    heapq.heappush(open_list, (start_node.f(), id(start_node), start_node))

    g_score[start_state] = start_node.g

    while open_list:
        _, _, current_node = heapq.heappop(open_list)

        if heuristic_func(current_node.state, goal_state) < tolerance:
            path = []
            while current_node:
                path.append(current_node.state)
                current_node = current_node.parent
            return path[::-1]

        closed_set.add(current_node.state)

        for successor_state, cost in successors_func(current_node.state, bounds = (-50, 50, -50, 50), step = 1.0):
            if successor_state in closed_set:
                continue

            # check collision
            if not is_collision_free(successor_state, obstacles_noisy, obstacle_speeds) or not is_collision_free_rect(successor_state, rect_obstacles):
                continue

            # tentative g_score
            tentative_g = g_score[current_node.state] + cost
            if (successor_state not in g_score) or (tentative_g < g_score[successor_state]):
                #g = current_node.g + cost
                g_score[successor_state] = tentative_g
                h = heuristic_func(successor_state, goal_state)
                successor_node = Node(state=successor_state, parent=current_node, g=tentative_g, h=h)
                heapq.heappush(open_list, (successor_node.f(), id(successor_node), successor_node))

    return None  # No path found

# calculate h(n)
def euclidean_distance(state, goal_state):
    return np.linalg.norm(np.array(goal_state) - np.array(state))

def successors(state, bounds, step = 1.0):
    x, y = state
    # grid resolution = step size
    # Assuming movements are allowed in 8 directions.
    # left, right. down, up, down-left, up-left, down-right, up-right
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)] 
    x_min, x_max, y_min, y_max = bounds
    result = []
    # Explore neighbors
    for dx, dy in moves:
        # step scaling
        dx_s = dx * step
        dy_s = dy * step
        new_x, new_y = x + dx_s, y + dy_s
        if x_min <= new_x < x_max and y_min <= new_y < y_max:  # Adjust boundaries according to your problem
            # Adjust the cost if it is diagonal
            if dx_s != 0 and dy_s != 0:
                c = math.sqrt(dx_s**2+ dy_s**2)
            else:
                c = 1 * step
            result.append(((new_x, new_y), c)) # Assuming each step has a cost of 1
    return result

# returns false or true if a point is inside the elliptical obstacle
def is_collision_free(neighbor: tuple, obstacles_noisy: np.ndarray, obstacle_speeds: np.ndarray):
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
    sizes = np.array([1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1, 1.0, 1.2, 1.8, 2.0, 1.8, 1.9])
    # incorporate static size                          
    a_base = a0 * sizes 
    b_base = b0 * sizes
    a_max = 7
    b_max = 5

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
        q_x, q_y = (neighbor[0]-obs_x)/(a+eps), (neighbor[1]-obs_y)/(b+eps)

        # distance of the robot to origin -1
        dE = np.sqrt(q_x**2 + q_y**2) - 1
        
        # COLLISION condition: robot is inside the ellipse
        if dE < 0:
            return False  #collision detected with the ellipse

    return True

# returns false or true if a point is inside the rectangular region
def is_collision_free_rect(neighbor: tuple, rect_obstacles: list):
    X, Y = neighbor[0], neighbor[1]
    for rect in rect_obstacles:
        x, y, w, h = rect
        if (x <= X <= x + w) and (y <= Y <= y + h):
            return False  # point is inside the rectangular region
    return True


# Running the script
def main():
    #from APF3 import init_environment
    rect_obstacles, expanded_rects, eps_expanded_rects, waypoints = init_environment()
    rectangle_speeds = np.array([[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1]])
    rectangle_speeds = rectangle_speeds * 80
    obstacles_true      = np.array([[-18.0,-10.0], [18,-20], [18, 8], [22,26], [23,15], [-23,15], [5,5], [-40, -30], [15, -10], [10, 5], [-30, 0], [-20, -20], [0, 0]])
    sigma               = 0.1  # 10 cm uncertainity
    obstacles_noisy     = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)
    obstacle_speeds     = np.array([[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1], [-0.1, 0.1], [-0.2, 0.1], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2]])
    obstacle_speeds     = obstacle_speeds * 80

    start_state = (-40, -40)
    goal_state = (10, 10)
    path = astar(start_state, goal_state, euclidean_distance, successors, obstacles_noisy, obstacle_speeds, rect_obstacles)
    print("Path:", path)

    
if __name__ == "__main__":
    main()
