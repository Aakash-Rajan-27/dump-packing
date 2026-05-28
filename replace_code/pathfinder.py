# pathfinder.py
# A* pathfinding for trucks on the dump grid.
#
# Key design decisions:
#   - State is just (row, col) — 4-value states caused 68M-node explosions.
#   - A RADIUS GOAL is used instead of an exact cell goal. The search stops
#     once the truck is within stop_dist_cells of the target. This prevents
#     failures when the exact target cell is inside a dirt pile (>0.4m height).
#   - Other trucks' cells are treated as soft obstacles (high cost, not walls),
#     so trucks can still re-route if another truck is blocking the only path.

import heapq
import numpy as np
import math
from filters import make_driveable_mask
from config import DRIVE_CLEARANCE_M, _TAN_REPOSE, ENTRY_POINT

# Maps a unit step direction (dr, dc) to one of 8 heading buckets.
# Used to index into the driveable mask's heading dimension.
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
    """
    Extra path cost for changing direction.
    Straight ahead (dot product = 1) is free. Any turn adds a penalty
    proportional to the truck's turning radius — tighter-turning trucks pay less.
    """
    if prev_dr is None: return 0.0
    dot = prev_dr * dr + prev_dc * dc
    return 0.0 if dot == 1 else turn_radius_cells * 0.3

def astar(driveable, grid, start_rc, goal_rc, truck, blocked_cells=frozenset(), stop_dist_cells=0.0):
    """
    A* search from start_rc toward goal_rc on a driveable mask.

    driveable      : 3D boolean mask [rows, cols, 8 headings] from make_driveable_mask()
    blocked_cells  : set of (r,c) occupied by other trucks — traversable but expensive
    stop_dist_cells: stop searching once within this cell-radius of goal_rc (radius goal)

    Returns a list of (row, col) cells from start (exclusive) to the stopping point.
    Returns [] if no path exists.
    """
    turn_radius_cells = truck.turn_radius / grid.cell_size
    rows, cols        = driveable.shape[:2]

    # Heap entries: (f_cost, g_cost, row, col, prev_dr, prev_dc)
    open_heap = [(0.0, 0.0, start_rc[0], start_rc[1], None, None)]
    came_from = {}
    g_cost    = {start_rc: 0.0}

    while open_heap:
        f, g, r, c, prev_dr, prev_dc = heapq.heappop(open_heap)

        # Radius goal: accept any cell within stop_dist_cells of the target
        dist_to_target = math.hypot(r - goal_rc[0], c - goal_rc[1])
        if dist_to_target <= stop_dist_cells:
            # Reconstruct path by walking came_from back to start
            path, cur = [], (r, c)
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return path

        # Skip if a cheaper route to this cell was already found
        if g > g_cost.get((r, c), float('inf')): continue

        # Only expand cardinal directions (no diagonals) to keep paths grid-aligned
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols): continue

            # Check driveable mask for this cell+heading combination
            hb = _heading_bucket(dr, dc)
            if not driveable[nr, nc, hb]: continue

            # Soft penalty for cells occupied by other trucks (avoids but doesn't block)
            traffic_cost = 10.0 if (nr, nc) in blocked_cells else 0.0
            tc = _turn_cost(prev_dr, prev_dc, dr, dc, turn_radius_cells)
            new_g = g + 1.0 + tc + traffic_cost

            if new_g < g_cost.get((nr, nc), float('inf')):
                g_cost[(nr, nc)] = new_g
                h = abs(nr - goal_rc[0]) + abs(nc - goal_rc[1])  # Manhattan heuristic
                heapq.heappush(open_heap, (new_g + h, new_g, nr, nc, dr, dc))
                came_from[(nr, nc)] = (r, c)

    return []  # no path found

def plan_paths(grid, assignments, existing_paths=None):
    """
    Plan paths for a list of (truck, target_rc) assignments.

    Trucks are planned sequentially. After each truck's path is found, its
    first cell is added to current_truck_cells so the next truck treats it
    as a soft obstacle. This reduces (but doesn't eliminate) overlap.

    Returns a dict: {truck_id: [(r,c), ...]}
    """
    if not assignments: return {}

    # Seed the occupied-cell set with all trucks' current positions and path heads
    current_truck_cells = set()
    if existing_paths is not None:
        for p in existing_paths.values():
            if p: current_truck_cells.add(p[0])

    for truck, _ in assignments:
        current_truck_cells.add(grid.world_to_cell(*truck.pos))

    mask_cache, paths = {}, {}
    entry_rc = grid.world_to_cell(*ENTRY_POINT)

    for truck, target_rc in assignments:
        # Build (and cache) the driveable mask once per truck class, not per truck
        if truck.truck_class not in mask_cache:
            mask_cache[truck.truck_class] = make_driveable_mask(grid, truck)

        driveable  = mask_cache[truck.truck_class]
        truck_cell = grid.world_to_cell(*truck.pos)
        obstacles  = current_truck_cells - {truck_cell}  # don't block yourself

        # Calculate how far short of the target the truck should stop.
        # For the entry gate: use a fixed 2m radius so minor offsets don't cause failures.
        # For dump cells: use the pile's safe clearance radius so the truck doesn't
        #   park on top of the pile it just created (which would trap it in a high-terrain cell).
        if target_rc == entry_rc:
            stop_dist_cells = 2.0 / grid.cell_size
        else:
            # r_pile_m: radius where the sandpile drops to DRIVE_CLEARANCE_M height
            r_pile_m = (3 * truck.payload_volume_m3 / (math.pi * _TAN_REPOSE)) ** (1/3)
            d_clearance = max(0.0, r_pile_m - (DRIVE_CLEARANCE_M / _TAN_REPOSE))
            safe_dist_m = d_clearance + (truck.length / 2.0)
            stop_dist_cells = safe_dist_m / grid.cell_size

        path = astar(driveable, grid, truck_cell, target_rc, truck,
                     blocked_cells=obstacles, stop_dist_cells=stop_dist_cells)

        paths[truck.id] = path
        # Register this truck's first step so subsequent trucks avoid it
        if path:
            current_truck_cells.add(path[0])

        print(f"DEBUG: Truck {truck.id} ({truck.truck_class}) Target: {target_rc} Path: {path} StopDistCells: {stop_dist_cells:.2f}")
        
    return paths

def plan_paths_cbs(grid, assignments):
    """
    Placeholder for Conflict-Based Search multi-agent pathfinding.
    Currently just delegates to plan_paths (sequential A*).
    """
    return plan_paths(grid, assignments)
