"""
AI-Based Crop-Safe Path Planning for an Autonomous Potato-Field Robot
=====================================================================
Group Members:
  1. Nakul Bhadade    (BT24CSA047) - Task 1 & 2: Field Environment & Formulation
  2. Pranjal Baghel   (BT24CSA052) - Task 3 (Part A): BFS & DFS Search
  3. Ness Dubey       (BT24CSA053) - Task 3 (Part B): Greedy Best-First Search
  4. Siddhi Dhoke     (BT24CSA050) - Task 3 (Part C) & 4: A* & Crop-Safe Cost
  5. Lakshuki Hatwar  (BT24CSA063) - Task 5 & 6: Replanning & Performance Evaluation

Standalone script that runs all tasks and saves screenshots.
Run this from the project folder: python generate_screenshots.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import collections
import heapq
import time
import os

os.makedirs('screenshots', exist_ok=True)

# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────
FREE        = 0
POTATO_ROW  = 1
CROP_ZONE   = 2
OBSTACLE    = 3
START_CELL  = 4
GOAL_CELL   = 5

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# ─────────────────────────────────────────────────────────────
#  TASK 1 – Field Environment
# ─────────────────────────────────────────────────────────────
FIELD = np.array([
    [4, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 2, 0, 1, 1, 0, 3, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 3, 0, 2, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 2, 0],
    [0, 0, 2, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 3, 0, 1, 1, 0, 0],
    [0, 3, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
], dtype=int)

START = (0, 0)
GOAL  = (9, 9)
ROWS, COLS = FIELD.shape

print(f"Field: {ROWS}×{COLS}  Start={START}  Goal={GOAL}")

# ─────────────────────────────────────────────────────────────
#  Shared Colour Map
# ─────────────────────────────────────────────────────────────
COLORS  = ['#FFFFFF', '#228B22', '#90EE90', '#CC0000', '#1565C0', '#FFD700']
CMAP    = ListedColormap(COLORS)
BOUNDS  = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
NORM    = BoundaryNorm(BOUNDS, CMAP.N)
LABELS  = {FREE: '', POTATO_ROW: 'P', CROP_ZONE: 'C',
           OBSTACLE: 'X', START_CELL: 'S', GOAL_CELL: 'G'}

def _draw_grid(ax, grid, path=None):
    nr, nc = grid.shape
    ax.imshow(grid, cmap=CMAP, norm=NORM, origin='upper')
    ax.set_xticks(np.arange(-0.5, nc, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nr, 1), minor=True)
    ax.grid(which='minor', color='gray', linewidth=0.5)
    ax.set_xticks(np.arange(0, nc))
    ax.set_yticks(np.arange(0, nr))
    for r in range(nr):
        for c in range(nc):
            lbl = LABELS.get(grid[r, c], '')
            if lbl:
                ax.text(c, r, lbl, ha='center', va='center', fontsize=9,
                        fontweight='bold',
                        color='white' if grid[r, c] in [POTATO_ROW, OBSTACLE, START_CELL] else 'black')
    if path and len(path) > 1:
        pr = [p[0] for p in path]
        pc = [p[1] for p in path]
        ax.plot(pc, pr, 'o-', color='orange', linewidth=2.5, markersize=6, zorder=5)

def save_fig(fname):
    plt.savefig(f'screenshots/{fname}.png', dpi=120, bbox_inches='tight')
    print(f"  💾 Saved: screenshots/{fname}.png")
    plt.close()

# ─────────────────────────────────────────────────────────────
#  TASK 2 – Problem Formulation Helpers
# ─────────────────────────────────────────────────────────────
def is_valid(row, col, grid):
    nr, nc = grid.shape
    if row < 0 or row >= nr or col < 0 or col >= nc:
        return False
    return grid[row, col] not in [POTATO_ROW, OBSTACLE]

def step_cost(row, col, grid):
    return 3 if grid[row, col] == CROP_ZONE else 1

def get_neighbors(row, col, grid):
    nbrs = []
    for dr, dc in DIRECTIONS:
        nr, nc = row + dr, col + dc
        if is_valid(nr, nc, grid):
            nbrs.append((nr, nc, step_cost(nr, nc, grid)))
    return nbrs

def path_cost(path, grid):
    if path is None:
        return float('inf')
    return sum(step_cost(r, c, grid) for r, c in path[1:])

# ─────────────────────────────────────────────────────────────
#  TASK 3 – Search Algorithms
# ─────────────────────────────────────────────────────────────
def bfs(grid, start, goal):
    t0 = time.perf_counter()
    queue   = collections.deque([(start, [start])])
    visited = {start}
    exp     = 0
    while queue:
        (r, c), path = queue.popleft()
        exp += 1
        if (r, c) == goal:
            return path, exp, (time.perf_counter()-t0)*1000
        for nr, nc, _ in get_neighbors(r, c, grid):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))
    return None, exp, (time.perf_counter()-t0)*1000

def dfs(grid, start, goal):
    t0 = time.perf_counter()
    stack   = [(start, [start])]
    visited = {start}
    exp     = 0
    while stack:
        (r, c), path = stack.pop()
        exp += 1
        if (r, c) == goal:
            return path, exp, (time.perf_counter()-t0)*1000
        for nr, nc, _ in get_neighbors(r, c, grid):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                stack.append(((nr, nc), path + [(nr, nc)]))
    return None, exp, (time.perf_counter()-t0)*1000

def manhattan(pos, goal):
    return abs(pos[0]-goal[0]) + abs(pos[1]-goal[1])

def greedy(grid, start, goal):
    t0  = time.perf_counter()
    pq  = [(manhattan(start, goal), start, [start])]
    vis = set()
    exp = 0
    while pq:
        h, (r, c), path = heapq.heappop(pq)
        if (r, c) in vis:
            continue
        vis.add((r, c));  exp += 1
        if (r, c) == goal:
            return path, exp, (time.perf_counter()-t0)*1000
        for nr, nc, _ in get_neighbors(r, c, grid):
            if (nr, nc) not in vis:
                heapq.heappush(pq, (manhattan((nr,nc), goal), (nr, nc), path+[(nr,nc)]))
    return None, exp, (time.perf_counter()-t0)*1000

def astar(grid, start, goal):
    t0      = time.perf_counter()
    ctr     = 0
    pq      = [(manhattan(start,goal), ctr, 0, start, [start])]
    g_cost  = {start: 0}
    vis     = set()
    exp     = 0
    while pq:
        f, _, g, (r, c), path = heapq.heappop(pq)
        if (r, c) in vis:
            continue
        vis.add((r, c));  exp += 1
        if (r, c) == goal:
            return path, exp, (time.perf_counter()-t0)*1000
        for nr, nc, cost in get_neighbors(r, c, grid):
            if (nr, nc) not in vis:
                ng = g + cost
                if ng < g_cost.get((nr,nc), float('inf')):
                    g_cost[(nr,nc)] = ng
                    ctr += 1
                    heapq.heappush(pq, (ng+manhattan((nr,nc),goal), ctr, ng, (nr,nc), path+[(nr,nc)]))
    return None, exp, (time.perf_counter()-t0)*1000

# ─────────────────────────────────────────────────────────────
#  Screenshot 1 – Task 1: Base Field
# ─────────────────────────────────────────────────────────────
print("\n[Task 1] Generating field grid screenshot...")
fig, ax = plt.subplots(figsize=(8, 8))
_draw_grid(ax, FIELD)
legend_els = [
    mpatches.Patch(facecolor='#FFFFFF', edgecolor='gray', label='Free Path (0)'),
    mpatches.Patch(facecolor='#228B22', label='Potato Row – blocked (1)'),
    mpatches.Patch(facecolor='#90EE90', label='Crop-Protection Zone – penalty (2)'),
    mpatches.Patch(facecolor='#CC0000', label='Obstacle – blocked (3)'),
    mpatches.Patch(facecolor='#1565C0', label='Start (4)'),
    mpatches.Patch(facecolor='#FFD700', label='Goal (5)'),
]
ax.legend(handles=legend_els, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9)
ax.set_title('Task 1 – Potato Field Grid (10×10)', fontsize=14, fontweight='bold')
ax.set_xlabel('Column →');  ax.set_ylabel('Row ↓')
plt.tight_layout()
save_fig('task1_field_grid')

# ─────────────────────────────────────────────────────────────
#  Screenshot 2 – Task 3: All 4 algorithms on main field
# ─────────────────────────────────────────────────────────────
print("[Task 3] Running all algorithms...")
bfs_path,  bfs_exp,  bfs_t  = bfs(FIELD,   START, GOAL)
dfs_path,  dfs_exp,  dfs_t  = dfs(FIELD,   START, GOAL)
grdy_path, grdy_exp, grdy_t = greedy(FIELD, START, GOAL)
astr_path, astr_exp, astr_t = astar(FIELD,  START, GOAL)

algo_results = [
    ('BFS',    bfs_path,  bfs_exp,  bfs_t),
    ('DFS',    dfs_path,  dfs_exp,  dfs_t),
    ('Greedy', grdy_path, grdy_exp, grdy_t),
    ('A*',     astr_path, astr_exp, astr_t),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()
for ax, (name, path, exp, t) in zip(axes, algo_results):
    _draw_grid(ax, FIELD, path)
    if path:
        plen  = len(path)
        pcost = path_cost(path, FIELD)
        ax.set_title(f'{name}  |  Steps: {plen}  Cost: {pcost}\n'
                     f'Nodes Expanded: {exp}  |  Time: {t:.3f} ms',
                     fontsize=11, fontweight='bold')
    else:
        ax.set_title(f'{name}  |  ❌ No path found', fontsize=11)
    ax.set_xlabel('Column →');  ax.set_ylabel('Row ↓')

plt.suptitle('Task 3 – All Search Algorithm Paths on 10×10 Potato Field',
             fontsize=14, fontweight='bold')
plt.tight_layout()
save_fig('task3_all_algorithms')

# ─────────────────────────────────────────────────────────────
#  Screenshot 3 – Task 4: Crop-Safe Planning (BFS vs A*)
# ─────────────────────────────────────────────────────────────
print("[Task 4] Crop-safe planning demo...")
CROP_DEMO = np.array([
    [4, 0, 0, 2, 0, 0, 0],
    [0, 1, 0, 2, 0, 1, 0],
    [0, 1, 0, 2, 0, 1, 0],
    [0, 0, 0, 2, 0, 0, 0],
    [0, 1, 0, 2, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 5],
], dtype=int)
DS, DG = (0, 0), (6, 6)

bfs_dp, _, _  = bfs(CROP_DEMO,  DS, DG)
astr_dp, _, _ = astar(CROP_DEMO, DS, DG)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (title, path) in zip(axes, [
    ('BFS Path  (Step-Optimal, ignores cost)', bfs_dp),
    ('A* Path   (Cost-Optimal, crop-safe)',    astr_dp),
]):
    _draw_grid(ax, CROP_DEMO, path)
    if path:
        cost_v = path_cost(path, CROP_DEMO)
        ax.set_title(f'{title}\nSteps: {len(path)}   Total Cost: {cost_v}', fontsize=11)
    else:
        ax.set_title(f'{title}\n❌ No path')
    ax.set_xlabel('Column →');  ax.set_ylabel('Row ↓')

plt.suptitle('Task 4 – Crop-Safe Planning: BFS vs A*  (Crop-Zone penalty = 3 per cell)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
save_fig('task4_crop_safe_planning')

bfs_dp_cost  = path_cost(bfs_dp,  CROP_DEMO)
astr_dp_cost = path_cost(astr_dp, CROP_DEMO)
print(f"   BFS  steps={len(bfs_dp) if bfs_dp else 'N/A'}  cost={bfs_dp_cost}")
print(f"   A*   steps={len(astr_dp) if astr_dp else 'N/A'}  cost={astr_dp_cost}")
print("   → A* prefers the lower-cost path, avoiding crop-zone cells.")

# ─────────────────────────────────────────────────────────────
#  Screenshot 4 – Task 5: Obstacle Detection & Replanning
# ─────────────────────────────────────────────────────────────
print("[Task 5] Obstacle detection & replanning...")
TRUE_FIELD      = FIELD.copy()
# Hidden obstacle placed ON the initial A* path (row 9, col 5)
# so the robot will always detect it and trigger replanning
HIDDEN_OBS      = (9, 5)
TRUE_FIELD[HIDDEN_OBS] = OBSTACLE

KNOWN_MAP       = FIELD.copy()
initial_path, _, _ = astar(KNOWN_MAP, START, GOAL)

replanning_count = 0
current_path     = initial_path
current_pos      = START
detection_pos    = None

step_idx = 0
while current_pos != GOAL:
    if step_idx + 1 >= len(current_path):
        print("  ⚠️  No next step — stopping.")
        break
    next_cell = current_path[step_idx + 1]
    if TRUE_FIELD[next_cell] == OBSTACLE and KNOWN_MAP[next_cell] != OBSTACLE:
        print(f"  🚨 Step {step_idx+1}: Obstacle detected at {next_cell}! "
              f"Robot at {current_pos}")
        detection_pos = current_pos
        KNOWN_MAP[next_cell] = OBSTACLE
        new_path, _, _ = astar(KNOWN_MAP, current_pos, GOAL)
        replanning_count += 1
        if new_path:
            current_path = new_path
            step_idx     = 0
            print(f"     🔄 Replan #{replanning_count}: new path len={len(new_path)}")
        else:
            print("     ❌ No alternative path found!")
            break
    else:
        current_pos = next_cell
        step_idx   += 1

print(f"  ✅ Journey done.  Replanning ops: {replanning_count}")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left – initial plan on original map, hidden obstacle marked
ax = axes[0]
_draw_grid(ax, FIELD, initial_path)
ax.add_patch(plt.Rectangle((HIDDEN_OBS[1]-0.5, HIDDEN_OBS[0]-0.5),
                             1, 1, facecolor='purple', alpha=0.6, zorder=3))
ax.text(HIDDEN_OBS[1], HIDDEN_OBS[0], '?', ha='center', va='center',
        fontsize=14, fontweight='bold', color='white', zorder=4)
ax.set_title(f'Initial A* Plan\n(Hidden obstacle "?" at {HIDDEN_OBS} — robot unaware)',
             fontsize=11)
ax.set_xlabel('Column →');  ax.set_ylabel('Row ↓')

# Right – updated map + replanned path
ax = axes[1]
_draw_grid(ax, KNOWN_MAP)
replan_path, _, _ = astar(KNOWN_MAP, detection_pos if detection_pos else START, GOAL)
if replan_path:
    rr = [p[0] for p in replan_path]
    rc = [p[1] for p in replan_path]
    ax.plot(rc, rr, 'o-', color='limegreen', linewidth=2.5, markersize=6,
            zorder=5, label='Replanned Path')
if detection_pos:
    ax.scatter([detection_pos[1]], [detection_pos[0]], s=200, c='red',
               zorder=6, label=f'Detection at {detection_pos}')
ax.legend(loc='upper right', fontsize=9)
ax.set_title(f'After Detection & Replanning\n'
             f'(Obstacle now at {HIDDEN_OBS}, replanning ops = {replanning_count})',
             fontsize=11)
ax.set_xlabel('Column →');  ax.set_ylabel('Row ↓')

plt.suptitle('Task 5 – Dynamic Obstacle Detection and A* Replanning',
             fontsize=13, fontweight='bold')
plt.tight_layout()
save_fig('task5_obstacle_replanning')

# ─────────────────────────────────────────────────────────────
#  Screenshot 5 – Task 6: Performance Table & Bar Charts
# ─────────────────────────────────────────────────────────────
print("[Task 6] Performance evaluation...")

print('\n' + '='*90)
print('  TASK 6 – PERFORMANCE COMPARISON TABLE')
print('='*90)
print(f'{"Algorithm":<12} {"Path Len":>10} {"Path Cost":>12} {"Nodes Exp":>12} {"Time (ms)":>12} {"Reached?":>10}')
print('-'*90)

table_rows = []
for name, path, exp, t in algo_results:
    plen    = len(path) if path else 'N/A'
    pcost   = path_cost(path, FIELD) if path else 'N/A'
    reached = 'YES' if path else 'NO'
    print(f'{name:<12} {str(plen):>10} {str(pcost):>12} {exp:>12} {t:>12.4f} {reached:>10}')
    table_rows.append((name, plen, pcost, exp, round(t, 4), reached))

print('='*90)
print(f'\nObstacle Detection Experiment:')
print(f'  Hidden obstacle: {HIDDEN_OBS}')
print(f'  Replanning operations: {replanning_count}')

# Bar charts
reached_rows = [r for r in table_rows if r[5]=='YES']
names   = [r[0] for r in reached_rows]
plens   = [r[1] for r in reached_rows]
nexps   = [r[3] for r in reached_rows]
times   = [r[4] for r in reached_rows]
bar_col = ['#4472C4', '#ED7D31', '#A9D18E', '#FF4444']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].bar(names, plens,  color=bar_col[:len(names)], edgecolor='black')
axes[0].set_title('Path Length (steps)', fontweight='bold')
axes[0].set_ylabel('Steps')
for i, v in enumerate(plens):
    axes[0].text(i, v+0.2, str(v), ha='center', fontweight='bold')

axes[1].bar(names, nexps,  color=bar_col[:len(names)], edgecolor='black')
axes[1].set_title('Nodes Expanded', fontweight='bold')
axes[1].set_ylabel('Count')
for i, v in enumerate(nexps):
    axes[1].text(i, v+0.2, str(v), ha='center', fontweight='bold')

axes[2].bar(names, times,  color=bar_col[:len(names)], edgecolor='black')
axes[2].set_title('Execution Time (ms)', fontweight='bold')
axes[2].set_ylabel('Milliseconds')
for i, v in enumerate(times):
    axes[2].text(i, v+0.00005, f'{v:.4f}', ha='center', fontweight='bold', fontsize=9)

plt.suptitle('Task 6 – Algorithm Performance Comparison (10×10 Potato Field)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
save_fig('task6_performance_comparison')

print('\n✅ All screenshots generated in ./screenshots/')
print('Screenshots:')
for f in sorted(os.listdir('screenshots')):
    print(f'  📸 {f}')
