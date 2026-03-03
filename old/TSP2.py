"""""
Global planning with TSP
"""""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math

# Start & Goal
q = np.array([-40.0, -40.0])
q_goal = np.array([10.0, 12.0])

# humans modeled as elliptical obstacles
obstacles_true = np.array([
    [-18.0,-10.0], [18,-20], [18, 8], [22,26], [25,15], [-23,15], [5,5]
])
sigma = 0.1
obstacles_noisy = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)

# elliptical obstacle speeds
obstacle_speeds = np.array([[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1]])
obstacle_speeds = obstacle_speeds * 80

# elliptical obstacle dimensions
a0 = 2.0 # major axis (along velocity direction)
b0 = 1.0 # minor axis (perpendicular to velocity direction)
alpha = 0.2   # major scaling (large)
beta  = 0.1   # minor scaling (small)

# each obstacle has a size factor: 1 = normal, >1 = large, <1 = small
sizes = np.array([1.2, 1.5, 1.0, 1.3, 0.8, 1.6, 1.1])
# incorporate static size
a_base = a0 + sizes 
b_base = b0 + sizes

a = np.zeros(obstacles_true.shape[0])
b = np.zeros(obstacles_true.shape[0])
theta = np.zeros(obstacles_true.shape[0])

for i, obs in enumerate(obstacle_speeds):
    vx, vy = obs
    vmag = np.sqrt(vx**2 + vy**2) 
    theta[i] = np.degrees(np.arctan2(vy, vx))
    a[i] = a_base[i] + alpha * vmag # major axis (velocity direction)
    b[i] = b_base[i] + beta  * vmag # minor axis (perpendicular)

# rectangular large obstacles
# x_min, y_min, width (m), height (m)
rect_obstacles = [
    [-40, -30, 12, 3],   # truck: narrow and long
    [20, -35, 6, 3],     # van / minibus
    [-45, 15, 10, 4],    # large vehicle
    [5, 25, 3, 2],       # bicycle / small vehicle
    [-6, -6.5, 12, 3],   # truck
    [-13.5, -23.0, 12, 6] 
]

expanded_rects = []
for x, y, w, h in rect_obstacles:
    new_x = x - 1
    new_y = y - 1
    new_w = w + 2
    new_h = h + 2
    expanded_rects.append([new_x, new_y, new_w, new_h])

eps_expanded_rects = []
for x, y, w, h in rect_obstacles:
    new_x = x - 1 + 0.001
    new_y = y - 1 + 0.001
    new_w = w + 2 - 2*0.001
    new_h = h + 2 - 2*0.001
    eps_expanded_rects.append([new_x, new_y, new_w, new_h])



# Stations to stop at (post office, hospital, grocery store, home, cafe)
waypoints = np.array([
    [-30, -35],  
    [25, -40],   
    [-48, 20],   
    [7, 30],      
    [15, -2],     
])

def rectangle_corners_center(rect):
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

# store lines between waypoints, start and goal
line_store = {}  
all_points = np.vstack([q, q_goal, waypoints])
n = all_points.shape[0]
for i in range(n):
    for j in range(i+1, n):
        p1 = (float(all_points[i,0]), float(all_points[i,1]))
        p2 = (float(all_points[j,0]), float(all_points[j,1]))
        eq = line_mb(p1, p2)

        # store p1, p2 and the equation
        line_store[(i, j)] = {
            "p1": p1,
            "p2": p2,
            "equation": eq
        }

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

line_store2 = {}
for (i, j), data in line_store.items():
    p1 = np.array(data["p1"])
    p2 = np.array(data["p2"])
    eq = data["equation"]

    true_hit_rects = []   # <-- reset for each line

    for rect, erect in zip(rect_obstacles, expanded_rects):

        # 1) radius approximation (coarse filter)
        center, radius = computeRectangleCircumcircle(rect)
        d = point_to_line_distance(eq, center)
        if d >= radius:
            continue

        # 2) actual check on the SAME expanded rect
        hit_this_rect = False
        for t in np.linspace(0, 1, 100):
            p = p1 + t * (p2 - p1)
            if point_in_rect(p, erect):
                hit_this_rect = True
                break

        if hit_this_rect:
            true_hit_rects.append(rect)

    # <-- rect loop is over, now save it inside the line_store2
    if len(true_hit_rects) > 0:
        line_store2[(i, j)] = {
            "p1": p1,
            "p2": p2,
            "equation": eq,
            "hit_rects": true_hit_rects
        }


# recursive search / iterative deepening obstacle-avoiding path search
valid_lines = []
for key,data in line_store2.items():
    p1 = data["p1"]
    p2 = data["p2"]
    hit_rects = data["hit_rects"]

    if len(hit_rects) == 1:
        chosen_rect = hit_rects[0]
    else:
        dmin = float("inf")
        chosen_rect = None
        for rect in hit_rects:
            c = np.array(rectangle_corners_center(rect)[4])
            d = np.linalg.norm(c - p1)   # distance between two points
            if d < dmin:
                dmin = d
                chosen_rect = rect       # this is the rectangle closest to p1

    idx = rect_obstacles.index(chosen_rect)
    chosen_expanded = expanded_rects[idx]
    corners = rectangle_corners_center(chosen_expanded)[:-1]  # remove the center
    candidate_lines = []
    for corner in corners:
        eq_new = line_mb(tuple(p1), corner)
        candidate_lines.append({
            "p1": np.array(p1),
            "p2": np.array(corner),
            "p2_goal": np.array(p2),
            "hit_rects": hit_rects,
            "equation": eq_new
        })

    for line in candidate_lines:
        p1_line = np.array(line["p1"])
        p2_line = np.array(line["p2"])

        enters_expanded = False
        for t in np.linspace(0, 1, 50, endpoint = False)[1:]:
            p = p1_line + t * (p2_line - p1_line)
            if point_in_rect(p, chosen_expanded):
                enters_expanded = True
                break

        # if it does not enter the expanded rectangle
        if not enters_expanded:
            valid_lines.append(line)

valid_lines2 = []
for vline in valid_lines:

    p_start = vline["p2"]       # corner
    p_goal  = vline["p2_goal"]  # goal
    hit_rects = vline["hit_rects"] # intersected rectangles
    p1_before = vline["p1"] # actual start of green lines

    # --- 1) corner -> goal test ---
    enters_any = False

    for rect in rect_obstacles:
        idx = rect_obstacles.index(rect)
        expanded = expanded_rects[idx]

        for t in np.linspace(0, 1, 50, endpoint=False)[1:]:
            p = p_start + t * (p_goal - p_start)
            if point_in_rect(p, expanded):
                enters_any = True
                break

        if enters_any:
            break

    # --- 2) We go to goal directly ---
    if not enters_any:
        eq_new = line_mb(p_start,p_goal)

        valid_lines2.append({
            "p1_before": p1_before.copy(),
            "p1": p_start.copy(),
            "p2": p_goal.copy(),
            "p2_goal": p_goal.copy(),
            "hit_rects": hit_rects,
            "equation": eq_new
        })
        continue   # this vline ends here, we don't need to move forward.

    # --- 3) Yoksa: tekrar corner'lara kır --- and select from the corner's of hit_rects
    for rect in hit_rects:
        idx = rect_obstacles.index(rect)
        expanded = expanded_rects[idx]
        corners = rectangle_corners_center(expanded)[:-1]

        for corner in corners:
            corner = np.array(corner, dtype=float)

            # corner -> corner ignore
            if np.allclose(corner, p_start):
                continue
            
            intersects_any = False

            for rect2 in rect_obstacles:
                idx2 = rect_obstacles.index(rect2)
                expanded2 = eps_expanded_rects[idx2]

                for t in np.linspace(0, 1, 50, endpoint=False)[1:]:
                    p = p_start + t * (corner - p_start)
                    if point_in_rect(p, expanded2):
                        intersects_any = True
                        break

                if intersects_any:
                    break

            if intersects_any:
                continue

            eq_new = line_mb(p_start, corner)

            valid_lines2.append({
                "p1_before": p1_before.copy(),
                "p1": p_start.copy(),
                "p2": corner.copy(),
                "p2_goal": p_goal.copy(),
                "hit_rects": hit_rects,
                "equation": eq_new
            })


# FINDING THE KEY PAIR IN LINE_STORE2
# target_p1 = np.array([25, -40.])
# target_p2 = np.array([7., 30.])

# found = False

# for key, data in line_store2.items():
#     if np.allclose(data["p1"], target_p1) and np.allclose(data["p2"], target_p2):
#         print("FOUND line_store2 entry for (-40,-40) -> (15,-2)")
#         print("key:", key)
#         print("p1:", data["p1"])
#         print("p2:", data["p2"])
#         print("hit_rects:", data["hit_rects"])
#         found = True
#         break

# if not found:
#     print("This line is NOT in line_store2")



target_p1 = np.array([25., -40.])
target_p2_goal = np.array([7., 30.])
matches = []

for idx, line in enumerate(valid_lines):
    if np.allclose(line["p1"], target_p1) and np.allclose(line["p2_goal"], target_p2_goal):
        matches.append((idx, line))

if matches:
    print(f"FOUND {len(matches)} matching entries\n")
    for idx, line in matches:
        print("index:", idx)
        print("p1:", line["p1"])
        print("p2:", line["p2"])
        print("p2_goal:", line["p2_goal"])
        print("hit_rects:", line["hit_rects"])
        print("-" * 40)
else:
    print("This (p1, p2_goal) pair is NOT in valid_lines")



target_p1_before = np.array([25., -40.])
target_p2_goal   = np.array([7., 30.])

matches2 = []

for idx, line in enumerate(valid_lines2):
    if np.allclose(line["p1_before"], target_p1_before) and \
       np.allclose(line["p2_goal"], target_p2_goal):
        matches2.append((idx, line))

if matches2:
    print(f"FOUND {len(matches2)} matching entries in valid_lines2\n")
    for idx, line in matches2:
        print("index:", idx)
        print("p1_before:", line["p1_before"])
        print("p1:", line["p1"])
        print("p2:", line["p2"])
        print("p2_goal:", line["p2_goal"])
        print("hit_rects:", line["hit_rects"])
        print("-" * 40)
else:
    print("This (p1_before, p2_goal) pair is NOT in valid_lines2")










# ===============================
# PLOTTING
# ===============================

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_aspect('equal')
ax.set_title("TSP")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.grid(True)

# Start & Goal
ax.plot(q[0], q[1], 'go', markersize=8, label="Start")
ax.plot(q_goal[0], q_goal[1], 'ro', markersize=8, label="Goal")

# Nokta engeller
ax.scatter(obstacles_true[:,0], obstacles_true[:,1], marker='x', label="True obstacles")
ax.scatter(obstacles_noisy[:,0], obstacles_noisy[:,1], marker='o', alpha=0.5, label="Noisy obstacles")

# True obstacles as ellipses
for i in range(obstacles_true.shape[0]):
    x0, y0 = obstacles_true[i]
    ell = Ellipse(
        (x0, y0),               
        width=2*a[i],
        height=2*b[i],
        angle=theta[i],
        edgecolor='black',
        facecolor='cyan',
        alpha=0.15,       # şeffaflık
        linestyle='--',
        linewidth=1.2
    )
    ax.add_patch(ell)


# Dikdörtgen engeller and the circle
for x, y, w, h in rect_obstacles:
    rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=2)
    ax.add_patch(rect)

    # --- Circumcircle ---
    cx = x + w/2
    cy = y + h/2
    r = np.sqrt((w/2)**2 + (h/2)**2)

    circle = plt.Circle((cx, cy), r, fill=False, linestyle=':', linewidth=1.5)
    ax.add_patch(circle)

# Genişletilmiş rectangle'lar (buffered)
for x, y, w, h in expanded_rects:
    rect2 = plt.Rectangle((x, y), w, h, fill=False, linewidth=1.5,
                           edgecolor='orange', linestyle='--')
    ax.add_patch(rect2)

# Waypointleri mor yıldız olarak çiz
ax.scatter(waypoints[:,0], waypoints[:,1],
           color='purple', marker='*', s=120, label="Uğranacak noktalar")

# Tüm noktaları birbirine bağla
n = all_points.shape[0]
for i in range(n):
    for j in range(i+1, n):
        x_vals = [all_points[i,0], all_points[j,0]]
        y_vals = [all_points[i,1], all_points[j,1]]

        if (i, j) in line_store2:
            ax.plot(x_vals, y_vals, color='green', linewidth=2.0, alpha=0.9)
        else:
            ax.plot(x_vals, y_vals, color='#ff1493', linewidth=1.2, alpha=0.9)  # hot pink

 # engeli dolanan iyi adaylar
# for line in valid_lines:
#     x_vals = [line["p1"][0], line["p2"][0]]
#     y_vals = [line["p1"][1], line["p2"][1]]
#     ax.plot(x_vals, y_vals, color='blue', linewidth=3.0, alpha=1.0)

# ikinci adımda üretilen line'lar
# for line in valid_lines2:
#     x_vals = [line["p1"][0], line["p2"][0]]
#     y_vals = [line["p1"][1], line["p2"][1]]

#     ax.plot(
#         x_vals,
#         y_vals,
#         color="orange",
#         linewidth=3.0,
#         alpha=1.0
#     )


for idx, line in matches:
    x_vals = [line["p1"][0], line["p2"][0]]
    y_vals = [line["p1"][1], line["p2"][1]]

    ax.plot(x_vals, y_vals, color="blue", linewidth=3)


for idx, line in matches2:
    ax.plot(
        [line["p1"][0], line["p2"][0]],
        [line["p1"][1], line["p2"][1]],
        color="purple",
        linewidth=3
    )

    xm = (line["p1"][0] + line["p2"][0]) / 2
    ym = (line["p1"][1] + line["p2"][1]) / 2
    ax.text(xm, ym, str(idx), color="purple", fontsize=9)


ax.legend()
plt.show()


