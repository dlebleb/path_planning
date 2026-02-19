import heapq #h[0] her zaman minimumdur. h[1], h[2], h[3] sıralı değildir.
import numpy as np

start_state = (0, 0)
goal_state = (4, 6)


class Node:
    def __init__(self, state, parent=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.g = g  # Cost from start node to current node
        self.h = h  # Heuristic estimate of cost from current node to goal node

    def f(self):
        return self.g + self.h

def astar(start_state, goal_state, heuristic_func, successors_func):
    open_list = []
    closed_set = set()

    start_node = Node(state=start_state, g=0, h=heuristic_func(start_state, goal_state))
    heapq.heappush(open_list, (start_node.f(), id(start_node), start_node))

    while open_list:
        _, _, current_node = heapq.heappop(open_list)

        if current_node.state == goal_state:
            path = []
            while current_node:
                path.append(current_node.state)
                current_node = current_node.parent
            return path[::-1]

        closed_set.add(current_node.state)

        for successor_state, cost in successors_func(current_node.state, bounds = (-40, 40, -40, 40)):
            if successor_state in closed_set:
                continue

            g = current_node.g + cost
            h = heuristic_func(successor_state, goal_state)
            successor_node = Node(state=successor_state, parent=current_node, g=g, h=h)
            heapq.heappush(open_list, (successor_node.f(), id(successor_node), successor_node))

    return None  # No path found

# calculate h(n)
def euclidean_distance(state, goal_state):
    return np.linalg.norm(np.array(goal_state) - np.array(state))

def successors(state, bounds):
    x, y = state
    # Assuming movements are allowed in 8 directions.
    # left, right. down, up, down-left, up-left, down-right, up-right
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    x_min, x_max, y_min, y_max = bounds
    result = []
    for dx, dy in moves:
        new_x, new_y = x + dx, y + dy
        if x_min <= new_x < x_max and y_min <= new_y < y_max:  # Adjust boundaries according to your problem
            result.append(((new_x, new_y), 1)) # Assuming each step has a cost of 1
    return result


start_state = (0, 0)
goal_state = (4, 6)
path = astar(start_state, goal_state, euclidean_distance, successors)
print("Path:", path)