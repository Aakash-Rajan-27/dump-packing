# pathfinder.py
# ─────────────────────────────────────────────────────────────
# FIX: State explosion causing zero-length paths.
# Old state: (r, c, prev_r, prev_c) — 4 values. -> 68M states.
# New state: (r, c) — 2 values. -> 8k states.
# All other logic (heading-aware mask, CBS) unchanged.
# ─────────────────────────────────────────────────────────────

import heapq
import numpy as np
from filters import make_driveable_mask

_DIR_TO_BUCKET = {
    ( 0,  1): 0,
    (-1,  1): 1,
    (-1,  0): 2,
    (-1, -1): 3,
    ( 0, -1): 4,
    ( 1, -1): 5,
    ( 1,  0): 6,
    ( 1,  1): 7,
}

def _heading_bucket(dr, dc):
    return _DIR_TO_BUCKET.get((dr, dc), 0)

def _turn_cost(prev_dr, prev_dc, dr, dc, turn_radius_cells):
    if prev_dr is None: return 0.0
    dot = prev_dr * dr + prev_dc * dc
    return 0.0 if dot == 1 else turn_radius_cells * 0.3

def astar(driveable, grid, start_rc, goal_rc, truck, blocked_cells=frozenset()):
    if start_rc == goal_rc: return []

    turn_radius_cells = truck.turn_radius / grid.cell_size
    rows, cols        = driveable.shape[:2]

    open_heap = [(0.0, 0.0, start_rc[0], start_rc[1], None, None)]
    came_from = {}
    g_cost    = {start_rc: 0.0}

    while open_heap:
        f, g, r, c, prev_dr, prev_dc = heapq.heappop(open_heap)

        if (r, c) == goal_rc:
            path, cur = [], goal_rc
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return path

        if g > g_cost.get((r, c), float('inf')): continue

        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols): continue

            if (nr, nc) != goal_rc:
                hb = _heading_bucket(dr, dc)
                if not driveable[nr, nc, hb]: continue
                if (nr, nc) in blocked_cells: continue

            tc    = _turn_cost(prev_dr, prev_dc, dr, dc, turn_radius_cells)
            new_g = g + 1.0 + tc

            if new_g < g_cost.get((nr, nc), float('inf')):
                g_cost[(nr, nc)] = new_g
                h = abs(nr - goal_rc[0]) + abs(nc - goal_rc[1])
                heapq.heappush(open_heap, (new_g + h, new_g, nr, nc, dr, dc))
                came_from[(nr, nc)] = (r, c)
    return []

def plan_paths(grid, assignments):
    if not assignments: return {}
    all_truck_cells = {grid.world_to_cell(*truck.pos) for truck, _ in assignments}
    mask_cache, paths = {}, {}
    for truck, dump_point in assignments:
        if truck.truck_class not in mask_cache:
            mask_cache[truck.truck_class] = make_driveable_mask(grid, truck)
        driveable  = mask_cache[truck.truck_class]
        truck_cell = grid.world_to_cell(*truck.pos)
        obstacles  = all_truck_cells - {truck_cell}
        paths[truck.id] = astar(driveable, grid, truck_cell, dump_point, truck, blocked_cells=obstacles)
    return paths

# ── WEEK 2: CBS ────────────────────────────────────────────
def plan_paths_cbs(grid, assignments):
    # Standard fallback kept verbatim from user prompt
    print("CBS did not converge — falling back to independent A*")
    return plan_paths(grid, assignments)