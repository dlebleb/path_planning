import heapq #h[0] her zaman minimumdur. h[1], h[2], h[3] sıralı değildir.
import numpy as np
import math
from APF_Astar import init_environment
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

class Node:
    def __init__(self, state, parent=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.g = g  # Cost from start node to current node
        self.h = h  # Heuristic estimate of cost from current node to goal node

    def f(self):
        return self.g + self.h

def astar(start_state, goal_state, heuristic_func, successors_func, obstacles_noisy, obstacle_speeds, rect_obstacles, waypoints):
    tolerance = 1
    open_list = []
    closed_set = set()
    g_score = {}

    n_wp = len(waypoints)
    start_mask = 0 # which duties are completed?
    start_key = (start_state, start_mask)

    start_node = Node(state=start_key, g=0, h=heuristic_func(start_state, goal_state))
    heapq.heappush(open_list, (start_node.f(), id(start_node), start_node))

    g_score[start_key] = start_node.g

    while open_list:
        _, _, current_node = heapq.heappop(open_list)

        state, mask = current_node.state
        all_visited = (mask == (1 << n_wp) - 1)
        at_goal = heuristic_func(state, goal_state) < tolerance

        if all_visited and at_goal:
            path = []
            while current_node:
                path.append(current_node.state[0])
                current_node = current_node.parent
            return path[::-1]

        closed_set.add(current_node.state)

        for successor_state, cost in successors_func(state, bounds = (-50, 50, -50, 50), step = 1.0):
            neighbor_state = tuple(successor_state)
            new_mask = mask

            for i, wp in enumerate(waypoints):
                if heuristic_func(neighbor_state, wp) < tolerance:
                    new_mask |= (1 << i)

            new_key = (neighbor_state, new_mask)
            if new_key in closed_set:
                continue

            # check collision
            #if not is_collision_free(neighbor_state, obstacles_noisy, obstacle_speeds) or not is_collision_free_rect(neighbor_state, rect_obstacles):
            if not is_collision_free_rect(neighbor_state, rect_obstacles):
                continue

            # tentative g_score
            tentative_g = g_score[current_node.state] + cost
            
            if (new_key not in g_score) or (tentative_g < g_score[new_key]):
                #g = current_node.g + cost
                g_score[new_key] = tentative_g
                full_mask = (1 << n_wp) - 1 # full_mask = 11111 = 31
                
                if new_mask != full_mask:
                    remaining = [
                        heuristic_func(neighbor_state, wp)
                        for i, wp in enumerate(waypoints)
                        if not (new_mask & (1 << i))
                    ]
                    h = min(remaining) if remaining else 0
                else:
                    h = heuristic_func(neighbor_state, goal_state)

                #h = heuristic_func(neighbor_state, goal_state)
                successor_node = Node(state=new_key, parent=current_node, g=tentative_g, h=h)
                heapq.heappush(open_list, (successor_node.f(), id(successor_node), successor_node))

    return None  # No path found

def astar_local(start_state, goal_state, heuristic_func, successors_func,
                obstacles_noisy, obstacle_speeds, rect_obstacles, bounds, step):

    tolerance = 0.5
    open_list = []
    closed_set = set()
    g_score = {}

    start_node = Node(state=start_state, g=0,
                      h=heuristic_func(start_state, goal_state))

    heapq.heappush(open_list, (start_node.f(), id(start_node), start_node))
    g_score[start_state] = 0

    while open_list:

        _, _, current_node = heapq.heappop(open_list)
        state = current_node.state

        if heuristic_func(state, goal_state) < tolerance:
            path = []
            while current_node:
                path.append(current_node.state)
                current_node = current_node.parent
            return path[::-1]

        closed_set.add(state)

        for successor_state, cost in successors_func(
                state, bounds, step):

            neighbor_state = tuple(successor_state)

            if neighbor_state in closed_set:
                continue

            #f not is_collision_free(neighbor_state, obstacles_noisy, obstacle_speeds) or not is_collision_free_rect(neighbor_state, rect_obstacles):
            if not is_collision_free_rect(neighbor_state, rect_obstacles):
                continue

            tentative_g = g_score[state] + cost

            if (neighbor_state not in g_score) or (tentative_g < g_score[neighbor_state]):

                g_score[neighbor_state] = tentative_g
                h = heuristic_func(neighbor_state, goal_state)

                successor_node = Node(
                    state=neighbor_state,
                    parent=current_node,
                    g=tentative_g,
                    h=h
                )

                heapq.heappush(open_list,
                               (successor_node.f(), id(successor_node), successor_node))

    return None

# calculate h(n)
def euclidean_distance(state, goal_state):
    return np.linalg.norm(np.array(goal_state) - np.array(state))

def successors(state, bounds, step = 0.5):
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
    sizes = np.array([1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1, 1.0, 1.2, 1.8, 2.0, 1.8, 1.9, 1.4, 1.5, 1.8])
    #kalabalik platform
    sizes = np.array([1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1, 1.0, 1.2, 1.8, 2.0, 1.8, 1.9, 1.4, 1.5, 1.8, 1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1, 1.0])
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
    #from APF3_Astar import init_environment
    rect_obstacles, expanded_rects, eps_expanded_rects, waypoints = init_environment()
    # Waypoints (stations)
    # waypoints = np.array([
    #     [-30, -35],
    #     [25, -40],
    #     [-48, 20],
    #     [7, 30],
    #     [15, -2],
    #     [0, 5],
    #     [-10, 0],
    #     [-20, -20],
    #     [0, 40],
    #     [40, -20],
    #     [-5, 20]
    # ])

    rectangle_speeds = np.array([[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1]])
    rectangle_speeds = rectangle_speeds * 80

    # obstacle coordinates 
    obstacles_true      = np.array([[-18.0,-10.0], [18,-20], [18, 8], [22,26], [23,15], [-23,15], [5,5], [-40, -30], [15, -10], [10, 5], [-30, 0], [-20, -20], [0, 0], [-35, 3], [-26, -27], [2, -10]])
    # kalabalik platform
    obstacles_true      = np.array([[-18.0,-10.0], [18,-20], [18, 8], [22,26], [23,15], [-23,15], [5,5], [-40, -30], [15, -10], [10, 5], [-30, 0], [-20, -20], [0, 0], [-35, 3], [-26, -27], [2, -10], [11, -12], [-3, -3], [-45, 0], [-25, -25], [8, 10], [-35, 13], [-21, -17], [21, -10]])
    
    # extra kalabalik platform
    extra_obstacles = np.array([
    [-10, -5], [-12, 4], [-8, 8], [-6, -12], [-4, 15],
    [2, 12], [4, -8], [6, 18], [8, -15], [12, 6],
    [-14, -14], [-16, 10], [-20, 5], [-22, -8], [-24, 12],
    [14, 14], [16, -6], [20, 2], [24, -12], [26, 8]
    ])
    cluster1 = np.array([
    [-22, -22],
    [-21, -19],
    [-19, -23],
    [-18, -21],
    [-23, -18],
    [-20, -24]
    ])
    cluster2 = np.array([
    [18, -42],
    [21, -38],
    [23, -41],
    [19, -37],
    [24, -39],
    [17, -40]
    ])
    obstacles_true = np.vstack((obstacles_true, extra_obstacles,cluster1,cluster2))

    sigma               = 0.1  # 10 cm uncertainity
    obstacles_noisy     = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)
    # obstacle speeds
    obstacle_speeds     = np.array([[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1], [-0.1, 0.1], [-0.2, 0.1], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [0.2, 0.1], [-0.1, 0.1], [-0.2, 0.1]])
    # kalabalik platform
    obstacle_speeds     = np.array([[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1], [-0.1, 0.1], [-0.2, 0.1], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [0.2, 0.1], [-0.1, 0.1], [-0.2, 0.1],[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1], [-0.1, 0.1]])
    # extra kalabalik platform
    extra_speeds = np.array([
    [-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1],
    [-0.1, 0.1], [0.2, 0.1], [-0.1, 0.1], [-0.2, 0.1], [-0.2, 0.2],
    [0.1, -0.2], [-0.2, 0.1], [0.2, -0.1], [-0.1, -0.2], [0.1, 0.1],
    [-0.2, 0.2], [0.1, -0.1], [-0.1, 0.2], [0.2, -0.2], [-0.2, 0.1]
    ])
    extra_speeds_cluster = np.array([
    [-0.1, 0.1],
    [-0.2, 0.2],
    [0.1, 0.2],
    [-0.2, -0.1],
    [0.1, -0.1],
    [-0.1, 0.1],

    [0.2, 0.1],
    [-0.1, 0.1],
    [-0.2, 0.1],
    [-0.2, 0.2],
    [0.1, -0.2],
    [-0.2, 0.1]
    ])
    obstacle_speeds = np.vstack((obstacle_speeds, extra_speeds,extra_speeds_cluster))
    
    obstacle_speeds     = obstacle_speeds * 80

    start_state = (-40, -40)
    goal_state = (20, 10) #(10,12)
    path = astar(start_state, goal_state, euclidean_distance, successors, obstacles_noisy, obstacle_speeds, rect_obstacles, waypoints)
    print("Path:", path)

    fig, ax = plt.subplots(figsize=(8, 8), dpi = 50)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_title("Dynamic Obstacle Avoidance Animation")
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.grid(color='gray', linestyle='--', linewidth=0.4, alpha=0.7)

    true_scatter = ax.scatter([], [], c='black', s=40)
    noisy_scatter = ax.scatter([], [], c='red', s=40)
    goal_dot = ax.plot(goal_state[0], goal_state[1], 'go', markersize=8, label="Goal")

    if path is not None:
        path = np.array(path)
        ax.plot(path[:,0], path[:,1], 'r-', linewidth=2, label="A* Path")
        ax.scatter(path[:,0], path[:,1], c='red', s=10)

    ellipse_patches         = []
    rect_patches            = []
    expanded_rect_patches   = []

    a0 = 2.0 # major axis (along velocity direction)
    b0 = 1.0 # minor axis (perpendicular to velocity direction)
    alpha = 0.2   # major scaling (large)
    beta  = 0.1   # minor scaling (small)
    # each obstacle has a size factor: 1 = normal, >1 = large, <1 = small
    sizes = np.array([1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1, 1.0, 1.2, 1.8, 2.0, 1.8, 1.9, 1.4, 1.5, 1.8])
    #sizes = np.array([1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1, 1.0, 1.2, 1.8, 2.0, 1.8, 1.9])
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

    # Waypointleri mor yıldız olarak çiz
    ax.scatter(waypoints[:,0], waypoints[:,1],
    color='purple', marker='*', s=120, label="Stations")

    # --- 6) Obstacle scatter ---
    true_scatter.set_offsets(obstacles_true)
    noisy_scatter.set_offsets(obstacles_noisy)

    # --- 8) Draw ellipses (WITHOUT d0 scaling!) ---
    for i, obs in enumerate(obstacles_true):
        vx, vy = obstacle_speeds[i]
        vmag = np.sqrt(vx**2 + vy**2)
        theta = np.degrees(np.arctan2(vy, vx))

        a = a_base[i] + alpha * vmag 
        b = b_base[i] + beta  * vmag
        a = min(a, a_max)
        b = min(b, b_max)

        ellipse = Ellipse(
            xy=(obs[0], obs[1]),
            width=2*a,
            height=2*b,
            angle=theta,
            edgecolor='black',
            facecolor='cyan',
            alpha=0.15,       # şeffaflık
            linestyle='--',
            linewidth=1.2
        )
        ax.add_patch(ellipse)
        ellipse_patches.append(ellipse)

    # Dikdörtgen engeller and the circle
    for x, y, w, h in rect_obstacles:
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=2)
        ax.add_patch(rect)
        rect_patches.append(rect)

    # Genişletilmiş rectangle'lar (buffered)
    for x, y, w, h in expanded_rects:
        rect2 = plt.Rectangle((x, y), w, h, fill=False, linewidth=1.5,
                            edgecolor='orange', linestyle='--')
        ax.add_patch(rect2)
        expanded_rect_patches.append(rect2)

    plt.show()

if __name__ == "__main__":
    main()

#ben sadece rectangle lardan kac dedim ona, rectangle'a gelmiyorsa yildiz sorun yok.

#kendime not: Astar is designed for handling static obstacles, o yuzden ellipse'lere gittigi corner/node giriyor mu diye bakmiyor. 