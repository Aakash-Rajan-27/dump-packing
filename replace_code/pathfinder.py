# pathfinder.py
# ─────────────────────────────────────────────────────────────
# FIX: State explosion causing zero-length paths.
# Old state: (r, c, prev_r, prev_c) — 4 values. -> 68M states.
# New state: (r, c) — 2 values. -> 8k states.
# All other logic (heading-aware mask, CBS) unchanged.
# ─────────────────────────────────────────────────────────────

# pathfinder.py
# ─────────────────────────────────────────────────────────────
# PURE A* PATHFINDING: Reverted to pure distance/turn cost search.
# With exact clearance math handled in truck.py, the pathfinder
# does not need soft obstacle penalties.
# ─────────────────────────────────────────────────────────────

# pathfinder.py
# ─────────────────────────────────────────────────────────────
# RADIUS GOAL A* FIX: Stops the search within a safe radius of 
# the goal. This prevents search failures at dirt peaks and 
# ensures trucks can "reach" the entry gate without coordinate 
# perfection.
# ─────────────────────────────────────────────────────────────
#Agent Obstacle & Reassignment Resolution
#Technical Note: Previously, the simulation relied on instantaneous teleportation 
# to ENTRY_POINT. Upon switching to physical exit navigation, trucks often stopped 
# reassigning because they would arrive at the gate at slight coordinate offsets or incorrect headings. 
# The A* pathfinder would also fail (the "Inside Obstacle" bug) because it was attempting to route to
# the exact center of a dirt pile. The following code implements Radius Goals and Heading Validation 
# to ensure continuous task flow.
# ─────────────────────────────────────────────────────────────

# pathfinder.py
# ─────────────────────────────────────────────────────────────
# RADIUS GOAL A* & AGENT OBSTACLE RESOLUTION
# 
# TECHNICAL NOTE: 
# Previously, the simulation relied on instantaneous teleportation to ENTRY_POINT. 
# In physical navigation, agents often failed reassignment due to slight coordinate 
# offsets. The A* also failed (the "Inside Obstacle" bug) when routing to the 
# exact center of a dirt pile.
# 
# RESOLUTION:
# 1. RADIUS GOAL: astar() terminates when within stop_dist_cells of the goal.
# 2. HEADING VALIDATION: Although handled in truck.py, pathfinder ensures 
#    the spatial path is clear to the entry radius.
# ─────────────────────────────────────────────────────────────

import heapq
import numpy as np
import math
from filters import make_driveable_mask
from config import DRIVE_CLEARANCE_M, _TAN_REPOSE, ENTRY_POINT

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

def astar(driveable, grid, start_rc, goal_rc, truck, blocked_cells=frozenset(), stop_dist_cells=0.0):
    turn_radius_cells = truck.turn_radius / grid.cell_size
    rows, cols        = driveable.shape[:2]

    open_heap = [(0.0, 0.0, start_rc[0], start_rc[1], None, None)]
    came_from = {}
    g_cost    = {start_rc: 0.0}

    while open_heap:
        f, g, r, c, prev_dr, prev_dc = heapq.heappop(open_heap)

        # ─── RADIUS GOAL CHECK ───
        # Success if we are within the stopping radius of the goal
        dist_to_target = math.hypot(r - goal_rc[0], c - goal_rc[1])
        if dist_to_target <= stop_dist_cells:
            path, cur = [], (r, c)
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return path

        if g > g_cost.get((r, c), float('inf')): continue

        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols): continue

            hb = _heading_bucket(dr, dc)
            if not driveable[nr, nc, hb]: continue
            
            traffic_cost = 10.0 if (nr, nc) in blocked_cells else 0.0
            tc = _turn_cost(prev_dr, prev_dc, dr, dc, turn_radius_cells)
            new_g = g + 1.0 + tc + traffic_cost

            if new_g < g_cost.get((nr, nc), float('inf')):
                g_cost[(nr, nc)] = new_g
                h = abs(nr - goal_rc[0]) + abs(nc - goal_rc[1])
                heapq.heappush(open_heap, (new_g + h, new_g, nr, nc, dr, dc))
                came_from[(nr, nc)] = (r, c)
                
    return []

def plan_paths(grid, assignments, existing_paths=None):
    if not assignments: return {}
    
    current_truck_cells = set()
    if existing_paths is not None:
         for p in existing_paths.values():
             if p: current_truck_cells.add(p[0]) 

    for truck, _ in assignments:
         current_truck_cells.add(grid.world_to_cell(*truck.pos))

    mask_cache, paths = {}, {}
    entry_rc = grid.world_to_cell(*ENTRY_POINT)
    
    for truck, target_rc in assignments:
        if truck.truck_class not in mask_cache:
            mask_cache[truck.truck_class] = make_driveable_mask(grid, truck)
            
        driveable  = mask_cache[truck.truck_class]
        truck_cell = grid.world_to_cell(*truck.pos)
        obstacles  = current_truck_cells - {truck_cell}
        
        # ─── CALCULATE RADIUS STOPPING DISTANCE ───
        if target_rc == entry_rc:
            # Arrival zone radius for the entrance gate (2.0 meters)
            stop_dist_cells = 2.0 / grid.cell_size
        else:
            # Safe distance for dumping to avoid "Inside Obstacle" failures
            r_pile_m = (3 * truck.payload_volume_m3 / (math.pi * _TAN_REPOSE)) ** (1/3)
            d_clearance = max(0.0, r_pile_m - (DRIVE_CLEARANCE_M / _TAN_REPOSE))
            safe_dist_m = d_clearance + (truck.length / 2.0)
            stop_dist_cells = safe_dist_m / grid.cell_size
        
        path = astar(driveable, grid, truck_cell, target_rc, truck, 
                     blocked_cells=obstacles, stop_dist_cells=stop_dist_cells)
        
        paths[truck.id] = path
        if path:
             current_truck_cells.add(path[0]) 

        # --- DEBUG PRINT (Placed inside the loop for per-truck visibility) ---
        print(f"DEBUG: Truck {truck.id} ({truck.truck_class}) Target: {target_rc} Path: {path} StopDistCells: {stop_dist_cells:.2f}")
             
    return paths

def plan_paths_cbs(grid, assignments):
    return plan_paths(grid, assignments)