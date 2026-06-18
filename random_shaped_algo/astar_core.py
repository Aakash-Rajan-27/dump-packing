# astar_core.py
# ─────────────────────────────────────────────────────────────
# Core A* planners:
#   • astar()    — spatial A* for single-agent dump-path planning
#   • astar_st() — space-time A* used by CBS multi-agent planning
#   • plan_paths() — single-tick wrapper that builds smoothed paths
# ─────────────────────────────────────────────────────────────

import heapq
import math
import numpy as np
import grid_map
from path_utils import (_bucket_from_heading, _BUCKET_TO_DIR, _BUCKET_TO_HEADING,
                        _heading_for_bucket, _truck_front_cell, _state_cell)
from bicycle_model import interpolate_path_to_truck_states
from filters import make_driveable_mask
from config import (DRIVE_CLEARANCE_M, _TAN_REPOSE, ENTRY_POINT,
                    ASTAR_WAIT_COST, ASTAR_MAX_TIME, ENTRY_CORRIDOR_CELLS)


def _turn_cost(prev_dr, prev_dc, dr, dc, turn_radius_cells):
    # Penalise direction changes to discourage zig-zag paths.
    # No penalty for the first step (prev is None) or for going straight (dot product == 1).
    # A turn costs turn_radius * 0.3 to approximate the real-world arc length penalty.
    if prev_dr is None: return 0.0
    dot = prev_dr * dr + prev_dc * dc  # 1 = straight ahead, 0 = 90° turn, -1 = U-turn
    return 0.0 if dot == 1 else turn_radius_cells * 0.3


def astar(driveable, grid, start_rc, goal_rc, truck, blocked_cells=frozenset(), stop_dist_cells=0.0):
    turn_radius_cells = truck.turn_radius / grid.cell_size
    rows, cols        = driveable.shape[:2]

    start_hb    = _bucket_from_heading(truck.heading)
    start_state = (start_rc[0], start_rc[1], start_hb)

    open_heap = [(0.0, 0.0, start_rc[0], start_rc[1], start_hb)]
    came_from = {}
    g_cost    = {start_state: 0.0}

    closest_state = start_state
    min_dist_to_target = math.hypot(start_rc[0] - goal_rc[0], start_rc[1] - goal_rc[1])

    while open_heap:
        f, g, r, c, hb = heapq.heappop(open_heap)
        state = (r, c, hb)

        dist_to_target = math.hypot(r - goal_rc[0], c - goal_rc[1])
        if dist_to_target < min_dist_to_target:
            min_dist_to_target = dist_to_target
            closest_state = state

        if dist_to_target <= stop_dist_cells:
            closest_state = state
            break

        if g > g_cost.get(state, float('inf')):
            continue

        next_states = []
        for turn in (-1, 0, 1):
            next_hb = (hb + turn) % 8
            dr, dc = _BUCKET_TO_DIR[next_hb]
            turn_cost = 0.0 if turn == 0 else 0.35 * turn_radius_cells
            next_states.append((r + dr, c + dc, next_hb, math.hypot(dr, dc) + turn_cost))

        for nr, nc, nhb, action_cost in next_states:
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            if not driveable[nr, nc, nhb]:
                continue

            if (nr, nc) in blocked_cells:  # hard block — another truck is here, never enter
                continue
            new_g = g + action_cost
            next_state = (nr, nc, nhb)

            if new_g < g_cost.get(next_state, float('inf')):
                g_cost[next_state] = new_g
                h = abs(nr - goal_rc[0]) + abs(nc - goal_rc[1])
                heapq.heappush(open_heap, (new_g + h, new_g, nr, nc, nhb))
                came_from[next_state] = state

    path, cur = [], closest_state
    while cur in came_from:
        path.append((cur[0], cur[1], _heading_for_bucket(cur[2])))
        cur = came_from[cur]
    path.reverse()

    return path


def plan_paths(grid, assignments, existing_paths=None):
    if not assignments: return {}  # nothing to plan

    current_truck_cells = set()  # cells occupied by trucks — hard blocks for subsequent planners
    if existing_paths is not None:
        for p in existing_paths.values():
            from path_utils import _path_cells
            cells = _path_cells(grid, p)
            if cells: current_truck_cells.add(cells[0])

    for truck, _ in assignments:
        current_truck_cells.add(_truck_front_cell(grid, truck))

    mask_cache, paths = {}, {}  # cache driveable masks per truck class (expensive to recompute)
    entry_rc = grid.world_to_cell(*ENTRY_POINT)  # precompute the entry gate cell once

    for truck, target_rc in assignments:
        if truck.truck_class not in mask_cache:
            mask_cache[truck.truck_class] = make_driveable_mask(grid, truck)  # build (and cache) terrain mask for this truck type

        driveable  = mask_cache[truck.truck_class]        # the [row, col, bucket] boolean mask for this truck
        truck_cell = _truck_front_cell(grid, truck)       # current front-centre cell of the truck
        obstacles  = current_truck_cells - {truck_cell}   # other trucks' cells (this truck excluded)

        # ─── BULLDOZER MODE: FORCE START CELL TO BE VALID ───
        # If the truck is standing in a restricted zone, this forces the pathfinder
        # to let it drive out, rather than permanently freezing it.
        if 0 <= truck_cell[0] < grid.rows and 0 <= truck_cell[1] < grid.cols:
            driveable[truck_cell[0], truck_cell[1], :] = True  # override all directions at start cell to driveable
        # ────────────────────────────────────────────────────

        # ─── CALCULATE RADIUS STOPPING DISTANCE ───
        if target_rc == entry_rc:
            stop_dist_cells = 0.0  # plan all the way to the entry cell — no early stop
        else:
            # Safe distance for dumping to avoid "Inside Obstacle" failures
            r_pile_m = (1.2* truck.payload_volume_m3 / (math.pi * _TAN_REPOSE)) ** (1/3)  # radius where sandpile height equals the repose limit
            d_clearance = max(0.0, r_pile_m - (DRIVE_CLEARANCE_M / _TAN_REPOSE))           # extra clearance so the pile never buries the truck
            safe_dist_m = d_clearance + truck.length/2                                       # front centre is one full truck length ahead of the rear
            stop_dist_cells = safe_dist_m / grid.cell_size                                  # convert metres to cells

        coarse_path = astar(driveable, grid, truck_cell, target_rc, truck,
                            blocked_cells=frozenset(obstacles), stop_dist_cells=stop_dist_cells)
        path = interpolate_path_to_truck_states(grid, truck, coarse_path)
        paths[truck.id] = path

        if coarse_path:
            current_truck_cells.add(_state_cell(coarse_path[0]))

    return paths


def astar_st(driveable, grid, start_rc, goal_rc, truck, constraints,
             stop_dist_cells=0.0, max_time=ASTAR_MAX_TIME):
    """
    Space-time A* for CBS.
    State: (r, c, hb, t) — arrival heading bucket included so turn costs are
    always computed against the actual arrival direction, eliminating zigzag paths.
    8-connected movement via heading bucket transitions (±1 bucket per step),
    matching the same scheme used by astar().
    constraints: set of (r, c, t) tuples — forbidden positions at specific timesteps.
    Returns list of (r, c) path nodes; consecutive duplicates mean "wait in place".
    """
    turn_radius_cells = truck.turn_radius / grid.cell_size
    rows, cols = driveable.shape[:2]

    start_hb    = _bucket_from_heading(truck.heading)
    start_state = (start_rc[0], start_rc[1], start_hb, 0)

    open_heap = [(0.0, 0.0, start_rc[0], start_rc[1], start_hb, 0)]
    came_from = {}
    g_cost    = {start_state: 0.0}

    closest_node = start_state
    min_dist = math.hypot(start_rc[0] - goal_rc[0], start_rc[1] - goal_rc[1])

    while open_heap:
        f, g, r, c, hb, t = heapq.heappop(open_heap)
        state = (r, c, hb, t)

        if t > max_time:
            continue

        dist = math.hypot(r - goal_rc[0], c - goal_rc[1])
        if dist < min_dist:
            min_dist = dist
            closest_node = state

        if dist <= stop_dist_cells:
            closest_node = state
            break

        if g > g_cost.get(state, float('inf')):
            continue

        nt = t + 1

        # ── WAIT ACTION ──────────────────────────────────────────────────────────
        if nt <= max_time and (r, c, nt) not in constraints:
            wait_state = (r, c, hb, nt)
            wait_g = g + ASTAR_WAIT_COST
            if wait_g < g_cost.get(wait_state, float('inf')):
                g_cost[wait_state] = wait_g
                abs_dr = abs(r - goal_rc[0])
                abs_dc = abs(c - goal_rc[1])
                h = max(abs_dr, abs_dc) + (math.sqrt(2) - 1) * min(abs_dr, abs_dc)
                heapq.heappush(open_heap, (wait_g + h, wait_g, r, c, hb, nt))
                came_from[wait_state] = state

        # ── MOVE ACTIONS (8-connected via heading buckets) ───────────────────────
        for turn in (-1, 0, 1):
            next_hb    = (hb + turn) % 8
            dr, dc     = _BUCKET_TO_DIR[next_hb]
            nr, nc     = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if not driveable[nr, nc, next_hb]:
                continue
            if nt > max_time or (nr, nc, nt) in constraints:
                continue
            turn_cost  = 0.0 if turn == 0 else 0.35 * turn_radius_cells
            move_cost  = math.hypot(dr, dc)   # 1.0 orthogonal, √2 diagonal
            new_g      = g + move_cost + turn_cost
            next_state = (nr, nc, next_hb, nt)
            if new_g < g_cost.get(next_state, float('inf')):
                g_cost[next_state] = new_g
                abs_dr = abs(nr - goal_rc[0])
                abs_dc = abs(nc - goal_rc[1])
                h = max(abs_dr, abs_dc) + (math.sqrt(2) - 1) * min(abs_dr, abs_dc)
                heapq.heappush(open_heap, (new_g + h, new_g, nr, nc, next_hb, nt))
                came_from[next_state] = state

    path, cur = [], closest_node
    while cur in came_from:
        path.append((cur[0], cur[1]))
        cur = came_from[cur]
    path.reverse()
    return path
