"""""
Global planning with TSP
functions:
"""""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math
import random

def rectangle_corners_center(rect):
    '''function definition:
    inputs-->
    rect: sdfkljhadsfkjsdhflj
    outputs -->
    dalkjshdaksjhdkajs'''
    x, y, w, h = rect
    return [
        (x, y),         # left-bottom
        (x+w, y),       # right-bottom
        (x+w, y+h),     # right-top
        (x, y+h),       # left-top
        (x+(w/2),y+(h/2)) # center 
    ]

def line_mb(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    if x2 == x1:
        return ("vertical", float(round(x1, 2)))   # x = x1
    if y2 == y1:
         return ("horizontal", float(round(y1, 2)))   # y = y1
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m*x1
    return ("normal", float(round(m, 2)), float(round(b, 2)))      # y = m x + b

# draw the circle around the rectangle
def computeRectangleCircumcircle(rect):
    c0, c1, c2, c3, c4 = rectangle_corners_center(rect)
    center = c4
    
    w = rect[2]
    h = rect[3]
    
    radius = math.sqrt((w/2)**2 + (h/2)**2)
    
    return center, radius

def eq_to_abc(eq):
    # eq: ('normal', m, b)  or ('vertical', x0) or ('horizontal', y0)
    typ = eq[0]

    if typ == "normal":
        m, b = eq[1], eq[2]
        # y = m x + b  ->  m x - 1*y = -b
        return (m, -1.0, -b)

    if typ == "vertical":
        x0 = eq[1]
        # x = x0 -> 1*x + 0*y = x0
        return (1.0, 0.0, x0)

    if typ == "horizontal":
        y0 = eq[1]
        # y = y0 -> 0*x + 1*y = y0
        return (0.0, 1.0, y0)

    raise ValueError(f"Unknown equation type: {eq}")

def point_to_line_distance(eq, p):
    # eq_to_abc(eq) -> (a, b, c) such that ax + by + c = 0
    a, b, c = eq_to_abc(eq)
    c = -c
    x0, y0 = p
    d = abs(a*x0 + b*y0 + c) / math.sqrt(a*a + b*b)
    return d

# returns false or true if a point is inside the rectangular region
def point_in_rect(pt, rect):
    X, Y = pt
    x, y, w, h = rect
    return (x <= X <= x + w) and (y <= Y <= y + h)

def check_rectHit(line_store,rect_obstacles, expanded_rects): # checks whether a line intersects with rectangle, and store intersected lines in line_store2


    # p1 : previous point
    # p2 : current point
    # p2_goal : goal point
    # p1_before: history points
    # hit_rects: rectangles that we hit
    

    line_store_hit = {}
    p1          = np.array(line_store["p1"])
    p2          = np.array(line_store["p2"])
    p2_goal     = np.array(line_store["p2_goal"])
    eq          = line_store["equation"]

    true_hit_rects = []   # <-- reset for each line

    for rect, erect in zip(rect_obstacles, expanded_rects):
        eps = 0.015 # error threshold
        # 1) radius approximation (coarse filter)
        center, radius = computeRectangleCircumcircle(rect)
        d = point_to_line_distance(eq, center)
        if d >= radius + eps:
            continue

        # 2) actual check on the SAME expanded rect
        hit_this_rect = False
        for t in np.linspace(0, 1, 50, endpoint = False)[1:]:
            p = p2 + t * (p2_goal - p2)
            if point_in_rect(p, erect):
                hit_this_rect = True
                break

        if hit_this_rect:
            true_hit_rects.append(rect)

    # <-- rect loop is over, now save it inside the line_store2
    if len(true_hit_rects) > 0:
        line_store_hit = {
            "p1_before": line_store["p1_before"],
            "p1": p1,
            "p2": p2,
            "p2_goal": p2_goal,
            "hit_rects": true_hit_rects,
            "weight": line_store["weight"],
            "equation": eq
        }

    if not line_store_hit:
        line_store_hit = {
            "p1_before": line_store["p1_before"],
            "p1": p1, 
            "p2": p2,  
            "p2_goal": p2_goal,
            "hit_rects": 0,
            "weight": line_store["weight"],
            "equation": eq
        } ## condition if there is no obstacle along the way
    return line_store_hit

def gotoCorner3(visited, myblue_line, goal, rect_obstacles, expanded_rects, eps_expanded_rects):

    # we will move 1 step forward.

    # p1 : previous point
    # p2 : current point
    # p2_goal : goal point
    # p1_before: history points
    # hit_rects: rectangles that we hit
    # 

    myblue_line = check_rectHit(myblue_line,rect_obstacles, expanded_rects) # check for intersection
    valid_lines2= []
    p1          = myblue_line["p1"]  # previous point
    p2          = myblue_line["p2"]  # current point
    p2_goal     = myblue_line["p2_goal"]  # goal
    p1_before   = myblue_line["p1_before"] # actual start of green lines
    hit_rects   = myblue_line["hit_rects"] # intersected rectangles


    if hit_rects == 0: # if there is no intersection, we don't take further action.
        valid_lines2.append({
        "p1_before": np.vstack([p1_before,p2,p2_goal]), ### last change
        "p1": p2,
        "p2": p2_goal,
        "p2_goal": p2_goal,
        "hit_rects": hit_rects,
        "equation": line_mb(p2,p2_goal),
        "weight": myblue_line["weight"] + calculateDistance(p2, p2_goal)
        })

        goal.append({
            "p1_before": np.vstack([p1_before,p2,p2_goal]), ### last change
            "p1": p2,
            "p2": p2_goal,
            "p2_goal": p2_goal,
            "hit_rects": hit_rects,
            "equation": line_mb(p2,p2_goal),
            "weight": myblue_line["weight"] + calculateDistance(p2, p2_goal)
        })

        return valid_lines2, goal  # this vline ends here, we don't need to move forward.
        
    if len(hit_rects) == 1:
        chosen_rect = hit_rects[0]
    else:
        dmin = float("inf")
        chosen_rect = None
        for rect in hit_rects:
            c = np.array(rectangle_corners_center(rect)[4])
            d = np.linalg.norm(c - p2)   # distance between two points
            if d < dmin:
                dmin = d
                chosen_rect = rect       # this is the rectangle closest to p2 (current point)

    idx             = rect_obstacles.index(chosen_rect)
    chosen_expanded = expanded_rects[idx]
    chosen_expanded2 = eps_expanded_rects[idx]
    corners         = rectangle_corners_center(chosen_expanded)[:-1]  # remove the center
    candidate_lines = []
    for corner in corners:
        candidate_lines.append({
            "p1": np.array(p2),
            "p2": np.array(corner), # update the corner
            "p2_goal": p2_goal,#np.array(p2),
            "hit_rects": hit_rects,
            "weight": myblue_line["weight"] + calculateDistance(p2,corner),
            "p1_before": np.vstack([np.array(myblue_line["p1_before"]),np.array(p2)]),
            "equation": line_mb(corner,p2_goal)
        })

    for line in candidate_lines:
        p1_line = np.array(line["p1"])
        p2_line = np.array(line["p2"])

        enters_expanded = False
        for t in np.linspace(0, 1, 50, endpoint = False)[1:]:
            p = p1_line + t * (p2_line - p1_line)
            if point_in_rect(p, chosen_expanded2):
                enters_expanded = True
                break

        # if it does not enter the expanded rectangle
        if not enters_expanded:
            valid_lines2.append(line)

    valid_lines2 = [line for line in valid_lines2 if not any(np.allclose(line["p2"], v) for v in visited)] # visited olanlari valid_lines2den cikariyor.
    visited_brother = visited.copy()
    valid_lines3 = []
    for line in valid_lines2:
        visited_brother.append(line["p2"])
        dummy_validlines3 = gotoCorner3(visited_brother,line, goal, rect_obstacles, expanded_rects, eps_expanded_rects)
        if dummy_validlines3: # if bos kume -- do not append. this is for avoiding visited ends.
            valid_lines3.append(dummy_validlines3[0])

    return valid_lines3, goal
  
def calculateDistance(a,b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)   

def bestPath(waypoints,q,q_goal, rect_obstacles, expanded_rects, eps_expanded_rects):

    nodes = []

    nodes.append(tuple(q))              # start
    nodes.extend([tuple(w) for w in waypoints])
    nodes.append(tuple(q_goal))          # goal
    N = len(nodes)

    line_store = {}
    traveller_sm = np.empty((N,N))
    route_sm = np.empty((N,N), dtype=object)


    for i, p1 in enumerate(nodes):
        for j, p2 in enumerate(nodes):
            if i == j:
                continue

            eq = line_mb(p1, p2)
            weight = 0

            line_store = {
                "p1": p1, # iteration's starting point
                "p2": p1, # iteration's current point
                "equation": eq,
                "p1_before" : p1,
                "weight": weight,
                "p2_goal": p2,
                "hit_rects": 0
            }
            

            visited = []
            goal = []
            valid_lines2 = []
            for vline in line_store:
                try:
                    visited = [vline["p2"]]
                    valid_lines2.append(gotoCorner3(visited, vline, goal, rect_obstacles, expanded_rects, eps_expanded_rects)[0])
                except Exception as e:
                    vline = line_store
                    visited = [vline["p2"]]
                    valid_lines2.append(gotoCorner3(visited, vline, goal, rect_obstacles, expanded_rects, eps_expanded_rects)[0])
                    break

            weight_list = []
            for itemm in goal:
                weight_list.append(itemm["weight"])

            min_weight = min(weight_list) # there is a bug here: ValueError: min() arg is an empty sequence
            traveller_sm[i,j] = min_weight
            
            # min weight
            best_item = None
            for item in goal:
                if item["weight"] == min_weight:
                    best_item = item
                    break

            # route'u kaydet
            route_sm[i, j] = best_item["p1_before"]
    return traveller_sm, route_sm

# ===== TSP solved with PSO =====

def build_tsp_indices(q, q_goal, waypoints):
    nodes = [tuple(q)]
    nodes.extend([tuple(w) for w in waypoints])
    nodes.append(tuple(q_goal))
    N = len(nodes)
    START = 0
    END = N - 1
    WAYPOINTS = [i for i in range(N) if i not in (START, END)] # visit edilmesi gereken waypoint indexleri
    return N, START, END, WAYPOINTS, nodes

def random_path(START, END, WAYPOINTS):
    mid = WAYPOINTS[:]
    random.shuffle(mid)
    return [START] + mid + [END]

def fitness(path, traveller_sm):
    return sum(traveller_sm[a, b] for a, b in zip(path[:-1], path[1:]))

def get_swaps(p1, p2):
    swaps = []
    temp = p1[:]
    for i in range(1, len(p1)-1):
        if temp[i] != p2[i]:
            j = temp.index(p2[i])
            swaps.append((i, j))
            temp[i], temp[j] = temp[j], temp[i]
    return swaps

def apply_swaps(path, swaps, prob=0.5):
    new = path[:]
    for i, j in swaps:
        if random.random() < prob:
            new[i], new[j] = new[j], new[i]
    return new

def PSO_TSP(traveller_sm, START, END, WAYPOINTS,
            n_particles=30, n_iter=200,
            w=0.4, c1=1.5, c2=1.5):

    # 1️⃣ Başlangıç popülasyonu
    particles = [random_path(START, END, WAYPOINTS) for _ in range(n_particles)]

    # personal best
    pbest = particles[:]
    pbest_cost = [fitness(p, traveller_sm) for p in particles]

    # global best
    gbest_idx = np.argmin(pbest_cost)
    gbest = pbest[gbest_idx][:]
    gbest_cost = pbest_cost[gbest_idx]

    # 2️⃣ Iterasyonlar
    for it in range(n_iter):
        for i in range(n_particles):

            # --- velocity (swap farkları) ---
            v_pbest = get_swaps(particles[i], pbest[i])
            v_gbest = get_swaps(particles[i], gbest)

            # --- yeni pozisyon ---
            new_path = apply_swaps(particles[i], v_pbest, c1)
            new_path = apply_swaps(new_path, v_gbest, c2)

            particles[i] = new_path

            # --- fitness ---
            cost = fitness(new_path, traveller_sm)

            # personal best güncelle
            if cost < pbest_cost[i]:
                pbest[i] = new_path
                pbest_cost[i] = cost

                # global best güncelle
                if cost < gbest_cost:
                    gbest = new_path
                    gbest_cost = cost

        #if it % 20 == 0:
            #print(f"Iter {it:3d} | Best cost: {gbest_cost:.3f}")

    return gbest, gbest_cost

# ===== ASIL YOL: route_sm'den gerçek geometrik path =====

def build_full_geometric_path(best_path, route_sm):
    """
    Reconstructs the full geometric path from PSO output indices.
    """

    full_geometric_path = []

    for a1, b1 in zip(best_path[:-1], best_path[1:]):
        segment = route_sm[a1, b1]

        if segment is None or len(segment) == 0:
            # sessizce geç veya istersen warning ver
            continue
        # ilk segmentte tümünü ekle
        if len(full_geometric_path) == 0:
            full_geometric_path.extend(segment)
        else:
            # tekrar eden noktayı ekleme
            #full_geometric_path.extend(segment[1:])
            full_geometric_path.extend(segment[2:])

    return np.array(full_geometric_path)


