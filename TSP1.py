"""""
Global planning with TSP
"""""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Start & Goal
q = np.array([-40.0, -40.0])
q_goal = np.array([10.0, 12.0])

# Nokta engeller (insan gibi)
obstacles_true = np.array([
    [-18.0,-10.0], [18,-20], [18, 8], [22,26], [25,15], [-23,15], [5,5]
])
sigma = 0.1
obstacles_noisy = obstacles_true + np.random.normal(0, sigma, obstacles_true.shape)

# obstacle speeds
obstacle_speeds = np.array([[-0.1, 0.1], [-0.2, 0.2], [0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1]])
obstacle_speeds = obstacle_speeds * 80

# elliptical obstacles
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

# x_min, y_min, width (m), height (m)
rect_obstacles = [
    [-40, -30, 12, 3],   # kamyon gibi uzun ve dar
    [20, -35, 6, 3],     # van / minibüs
    [-45, 15, 10, 4],    # büyük araç
    [5, 25, 3, 2],       # bisikletli / küçük araç alanı
    [-6, -6.5, 12, 3],   # yeni kamyon: merkezi (0, -5)
    [-13.5, -23.0, 12, 6]
]

# Uğranacak noktalar (postane, hastane, market, ev, kafe gibi düşün)
waypoints = np.array([
    [-30, -35],   # kamyonun arkasında
    [25, -40],    # alttaki dikdörtgenin arkasında
    [-48, 20],    # soldaki üst engelin arkasında
    [7, 30],      # küçük dikdörtgenin arkasında
    [15, -2],     # orta bölgede ama engellerle kesilen hat üzerinde
])

def rectangle_corners(rect):
    x, y, w, h = rect
    return [
        (x, y),         # sol-alt
        (x+w, y),       # sağ-alt
        (x+w, y+h),     # sağ-üst
        (x, y+h)        # sol-üst
    ]

def line_mb(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if x2 == x1:
        return ("vertical", round(x1, 2))   # x = x1
    if y2 == y1:
         return ("horizontal", round(y1, 2))   # y = y1
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m*x1
    return ("normal", round(m, 2), round(b, 2))      # y = m x + b

def rectangle_edge_lines(rect):
    c0, c1, c2, c3 = rectangle_corners(rect)

    edges = {
        "bottom": (c0, c1),
        "right":  (c1, c2),
        "top":    (c2, c3),
        "left":   (c3, c0),
    }

    edge_lines = {}
    for name, (p1, p2) in edges.items():
        edge_lines[name] = line_mb(p1, p2)

    return (edge_lines)

def rectangle_edge_lines_with_points(rect):
    c0, c1, c2, c3 = rectangle_corners(rect)

    edges = {
        "bottom": (c0, c1),
        "right":  (c1, c2),
        "top":    (c2, c3),
        "left":   (c3, c0),
    }

    edge_data = {}
    for name, (p1, p2) in edges.items():
        eq = line_mb(p1, p2)
        edge_data[name] = {
            "p1": p1,
            "p2": p2,
            "equation": eq
        }

    return edge_data

line_store = {}   # ana yapı
all_points = np.vstack([q, q_goal, waypoints])
n = all_points.shape[0]
for i in range(n):
    for j in range(i+1, n):
        p1 = (float(all_points[i,0]), float(all_points[i,1]))
        p2 = (float(all_points[j,0]), float(all_points[j,1]))
        eq = line_mb(p1, p2)

        # anahtar olarak nokta çiftini kullanalım
        line_store[(i, j)] = {
            "p1": p1,
            "p2": p2,
            "equation": eq
        }

all_rect_edge_lines = {}
for i, rect in enumerate(rect_obstacles):
    edge_data = rectangle_edge_lines_with_points(rect)  # p1,p2,eq var
    only_eq = {}
    for name, info in edge_data.items():
        only_eq[name] = info["equation"]   # sadece denklemi al
    all_rect_edge_lines[i] = only_eq

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

def intersect_abc(line1, line2):
    a1, b1, c1 = line1
    a2, b2, c2 = line2

    det = a1*b2 - a2*b1

    if det == 0:
        return None   # paralel veya çakışık

    x = (c1*b2 - c2*b1) / det
    y = (a1*c2 - a2*c1) / det
    return (x, y)

def point_on_segment(p, a, b):
    x, y = p
    x1, y1 = a
    x2, y2 = b

    return (min(x1, x2) <= x <= max(x1, x2) and
            min(y1, y2) <= y <= max(y1, y2))

collisions = []
for (i,j), info in line_store.items():
    lp1 = info["p1"]
    lp2 = info["p2"]
    leq = info["equation"]
    L1 = eq_to_abc(leq)

    for rid, rect in enumerate(rect_obstacles):
        edges = rectangle_edge_lines_with_points(rect)   # artık dict of dict
        # veya sen bunu all_rect_edge_lines'ten alıyorsan:
        # edges = all_rect_edge_lines_with_points[rid]

        for edge_name, info_edge in edges.items():
            ep1 = info_edge["p1"]
            ep2 = info_edge["p2"]
            eq2 = info_edge["equation"]

            L2 = eq_to_abc(eq2)
            P = intersect_abc(L1, L2)

            if P is not None:
                P = (round(P[0], 2), round(P[1], 2))   # <-- burada
                if point_on_segment(P, lp1, lp2) and point_on_segment(P, ep1, ep2):
                    collisions.append({
                        "line_index": (i,j),
                        "line_points": (lp1, lp2),
                        "line_eq": leq,
                        "rect_id": rid,
                        "edge_name": edge_name,
                        "edge_points": (ep1, ep2),
                        "edge_eq": eq2,
                        "intersection": P
                    })

if len(collisions) > 0:
    inter_points = np.array([c["intersection"] for c in collisions])


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

if len(collisions) > 0:
    ax.scatter(inter_points[:,0], inter_points[:,1],
               color='navy', s=60, marker='o', label="Intersection points")

     
# Dikdörtgen engeller
for x, y, w, h in rect_obstacles:
    rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=2)
    ax.add_patch(rect)

# Waypointleri mor yıldız olarak çiz
ax.scatter(waypoints[:,0], waypoints[:,1],
           color='purple', marker='*', s=120, label="Uğranacak noktalar")

# Tüm noktaları birbirine bağla
n = all_points.shape[0]
for i in range(n):
    for j in range(i+1, n):
        x_vals = [all_points[i,0], all_points[j,0]]
        y_vals = [all_points[i,1], all_points[j,1]]
        ax.plot(x_vals, y_vals, color='#ff1493', linewidth=1.2, alpha=0.9)  # hot pink

ax.legend()
plt.show()


