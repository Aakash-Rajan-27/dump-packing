# astar_core.py
# ─────────────────────────────────────────────────────────────
# Core A* planners:
#   • astar()    — spatial A* for single-agent dump-path planning
#   • astar_st() — space-time A* used by CBS multi-agent planning
#   • plan_paths() — single-tick wrapper that builds smoothed paths
# ─────────────────────────────────────────────────────────────

import heapq          # min-heap for A* open list
import math           # cos/sin/hypot/sqrt
import numpy as np    # not directly used here but kept for downstream compatibility
import grid_map       # access to grid state (BOUNDARY, z_height, etc.)

# Import shared helpers for heading-bucket conversions and cell lookups
from path_utils import (_bucket_from_heading, _BUCKET_TO_DIR, _BUCKET_TO_HEADING,
                        _heading_for_bucket, _truck_front_cell, _state_cell)

# Bicycle model smoother that turns coarse cell-paths into (x, y, heading) world paths
from bicycle_model import interpolate_path_to_truck_states

# Builds the per-heading-bucket driveable boolean mask for a truck class
from filters import make_driveable_mask

# Simulation constants used by the planners
from config import (DRIVE_CLEARANCE_M, _TAN_REPOSE, ENTRY_POINT,
                    ASTAR_WAIT_COST, ASTAR_MAX_TIME, ENTRY_CORRIDOR_CELLS,
                    PATH_BUFFER_M)

# Footprint-overlap check: oriented rectangle SAT test used by astar_st
from conflict_detect import _footprints_conflict


# FIX: This function is defined but NEVER called anywhere in this file (or elsewhere).
# astar() and astar_st() both compute their turn costs inline with 0.35 * turn_radius_cells,
# while this function uses a different coefficient (0.3). Dead code with an inconsistent value.
def _turn_cost(prev_dr, prev_dc, dr, dc, turn_radius_cells):
    # Penalise direction changes to discourage zig-zag paths.
    # No penalty for the first step (prev is None) or for going straight (dot product == 1).
    # A turn costs turn_radius * 0.3 to approximate the real-world arc length penalty.
    if prev_dr is None: return 0.0                  # first step — no previous direction to compare
    dot = prev_dr * dr + prev_dc * dc               # dot product: 1 = straight, 0 = 90°, -1 = U-turn
    # FIX: dot==1 only holds for cardinal (non-diagonal) straight moves.
    # For diagonal moves (dr=±1, dc=±1), same direction gives dot = 1+1 = 2, not 1,
    # so diagonal straight-ahead would incorrectly receive a turn penalty here.
    return 0.0 if dot == 1 else turn_radius_cells * 0.3


def astar(driveable, grid, start_rc, goal_rc, truck, stop_dist_cells=0.0):
    # Convert the truck's physical turn radius from metres into grid cells
    turn_radius_cells = truck.turn_radius / grid.cell_size

    # Grid dimensions, needed for bounds checks on every neighbour
    rows, cols        = driveable.shape[:2]

    # Quantise the truck's current continuous heading into one of 8 45°-wide buckets
    start_hb    = _bucket_from_heading(truck.heading)

    # Initial search state: (row, col, heading_bucket)
    start_state = (start_rc[0], start_rc[1], start_hb)

    # Min-heap entries: (f_cost, g_cost, row, col, heading_bucket)
    open_heap = [(0.0, 0.0, start_rc[0], start_rc[1], start_hb)]

    # Back-pointer map: state → parent state, used to reconstruct the path
    came_from = {}

    # Best known g-cost to each (r, c, hb) state; used to skip stale heap entries
    g_cost    = {start_state: 0.0}

    # Track the state that got closest to the goal (used as fallback when goal unreachable)
    closest_state = start_state
    min_dist_to_target = math.hypot(start_rc[0] - goal_rc[0], start_rc[1] - goal_rc[1])

    while open_heap:
        # Pop the lowest f-cost state
        f, g, r, c, hb = heapq.heappop(open_heap)
        state = (r, c, hb)

        # Euclidean distance to goal in cell space (used for early-stop and closest tracking)
        dist_to_target = math.hypot(r - goal_rc[0], c - goal_rc[1])

        # Update "closest we've ever been" for fallback path reconstruction
        if dist_to_target < min_dist_to_target:
            min_dist_to_target = dist_to_target
            closest_state = state

        #If we're within the stopping distance (e.g., near a dump pile), accept this as goal
        if dist_to_target <= stop_dist_cells:
            closest_state = state
            break

        # Skip stale heap entry (a cheaper path to this state was already found)
        if g > g_cost.get(state, float('inf')):
            continue

        # Generate three successor heading buckets: turn left (−1), go straight (0), turn right (+1)
        next_states = []
        for turn in (-1, 0, 1):
            next_hb = (hb + turn) % 8                          # wrap bucket modulo 8
            dr, dc = _BUCKET_TO_DIR[next_hb]                   # unit direction vector for this heading
            turn_cost = 0.0 if turn == 0 else 0.1 * turn_radius_cells   # straight = free, turning = arc cost
            # Step cost = Euclidean step length (1.0 cardinal, √2 diagonal) + turning penalty
            next_states.append((r + dr, c + dc, next_hb, math.hypot(dr, dc) + turn_cost))

        for nr, nc, nhb, action_cost in next_states:
            # Discard moves that leave the grid bounds
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            # Discard cells that are not driveable for this heading bucket
            if not driveable[nr, nc, nhb]:
                continue

            # Tentative cost to reach the neighbour
            new_g = g + action_cost
            next_state = (nr, nc, nhb)

            # Only push if this is strictly better than any previously known cost
            if new_g < g_cost.get(next_state, float('inf')):
                g_cost[next_state] = new_g

                # FIX: Heuristic uses Manhattan distance but this is an 8-connected grid.
                # Octile distance ( max(|dr|,|dc|) + (√2−1)·min(|dr|,|dc|) ) is admissible
                # AND tighter, yielding fewer node expansions. astar_st() already uses octile.
                h = abs(nr - goal_rc[0]) + abs(nc - goal_rc[1])   # Manhattan heuristic (admissible but loose)

                heapq.heappush(open_heap, (new_g + h, new_g, nr, nc, nhb))
                came_from[next_state] = state                       # record parent for path reconstruction

    # Reconstruct path by walking back-pointers from closest_state to start
    path, cur = [], closest_state
    while cur in came_from:
        # Convert heading bucket back to continuous radians when storing the waypoint
        path.append((cur[0], cur[1], _heading_for_bucket(cur[2])))
        cur = came_from[cur]
    path.reverse()   # back-pointers give reversed order; flip to start→goal

    return path      # list of (row, col, heading_rad) waypoints


def plan_paths(grid, assignments, existing_paths=None):
    # Nothing to plan — return empty dict immediately
    if not assignments: return {}

    # Cache driveable masks per truck class (building one mask is expensive — ~ms)
    mask_cache = {}

    # Precompute entry gate cell once (used to decide stop distance)
    entry_rc = grid.world_to_cell(*ENTRY_POINT)

    paths = {}   # output: truck_id → smoothed path

    for truck, target_rc in assignments:
        # Build (or retrieve cached) driveable mask for this truck class
        if truck.truck_class not in mask_cache:
            mask_cache[truck.truck_class] = make_driveable_mask(grid, truck)

        driveable  = mask_cache[truck.truck_class]   # [row, col, bucket] boolean array
        truck_cell = _truck_front_cell(grid, truck)  # (row, col) of truck's front centre

        # Force start cell driveable so the planner can route out of restricted zones
        if 0 <= truck_cell[0] < grid.rows and 0 <= truck_cell[1] < grid.cols:
            driveable[truck_cell[0], truck_cell[1], :] = True

        # Calculate stopping distance
        if target_rc == entry_rc:
            stop_dist_cells = 0.0
        else:
            r_pile_m        = (1.2 * truck.payload_volume_m3 / (math.pi * _TAN_REPOSE)) ** (1/3)
            d_clearance     = max(0.0, r_pile_m - (DRIVE_CLEARANCE_M / _TAN_REPOSE))
            safe_dist_m     = d_clearance + truck.length / 2
            stop_dist_cells = safe_dist_m / grid.cell_size

        # Run spatial A* to get a coarse cell-by-cell path
        coarse_path = astar(driveable, grid, truck_cell, target_rc, truck,
                            stop_dist_cells=stop_dist_cells)

        # Smooth the coarse waypoints through the bicycle model → world-space (x,y,heading) states
        path = interpolate_path_to_truck_states(grid, truck, coarse_path)
        paths[truck.id] = path

    return paths   # dict: truck_id → smoothed (x, y, heading) path


def astar_st(driveable, grid, start_rc, goal_rc, truck, fp_constraints,
             stop_dist_cells=0.0, max_time=ASTAR_MAX_TIME):
    """
    Space-time A* for CBS.
    State: (r, c, hb, t) — arrival heading bucket included so turn costs are
    always computed against the actual arrival direction, eliminating zigzag paths.
    8-connected movement via heading bucket transitions (±1 bucket per step),
    matching the same scheme used by astar().

    fp_constraints: dict {t: [(cx, cy, heading, half_length, half_width), ...]}
      Forbidden truck footprints at specific timesteps.  Every candidate pose is
      checked via SAT (_rect_overlap_2d) through _footprints_conflict — no (r,c,t)
      cell-occupancy logic exists anywhere in this function.

    Returns list of (r, c) path nodes; consecutive duplicates mean "wait in place".
    """
    # Convert turn radius to grid cells for use in turn-cost computation
    turn_radius_cells = truck.turn_radius / grid.cell_size

    # Truck half-dimensions for footprint overlap checks
    hl = truck.length / 2.0
    hw = truck.width  / 2.0

    # Grid dimensions for bounds checking
    rows, cols = driveable.shape[:2]

    # Quantise starting heading into one of 8 45°-wide buckets
    start_hb    = _bucket_from_heading(truck.heading)

    # ── EARLY CONFLICT CHECK (start of planning) ─────────────────────────────────
    constraint_ts = sorted(fp_constraints.keys()) if fp_constraints else []
    if fp_constraints:
        print(f"[ASTAR_ST] Truck {truck.id}: planning from {start_rc} → {goal_rc}, "
              f"fp_constraints at t={constraint_ts}")
        swx, swy = grid.cell_to_world(start_rc[0], start_rc[1])
        swh = _heading_for_bucket(start_hb)
        for check_t in constraint_ts[:3]:
            if _footprints_conflict(swx, swy, swh, hl, hw, check_t, fp_constraints):
                print(f"[ASTAR_ST] Truck {truck.id}: RECTANGLE CONFLICT at start pos "
                      f"({start_rc[0]},{start_rc[1]}) heading={swh:.2f} at t={check_t}")
                break
        else:
            print(f"[ASTAR_ST] Truck {truck.id}: no rectangle conflict at start position")
    else:
        print(f"[ASTAR_ST] Truck {truck.id}: planning from {start_rc} → {goal_rc}, "
              f"no fp_constraints")
    # ─────────────────────────────────────────────────────────────────────────────

    # Initial space-time state: (row, col, heading_bucket, timestep=0)
    start_state = (start_rc[0], start_rc[1], start_hb, 0)

    # Min-heap: (f_cost, g_cost, row, col, heading_bucket, timestep)
    open_heap = [(0.0, 0.0, start_rc[0], start_rc[1], start_hb, 0)]

    # Back-pointer map: space-time state → parent space-time state
    came_from = {}

    # Best known g-cost for each (r, c, hb, t) state
    g_cost    = {start_state: 0.0}

    # Best state reached so far (fallback when goal unreachable within max_time)
    closest_node = start_state
    min_dist = math.hypot(start_rc[0] - goal_rc[0], start_rc[1] - goal_rc[1])

    # Conflict counters — accumulated across all expansions; printed once after search.
    _n_wait_conflicts = 0
    _n_move_conflicts = 0

    while open_heap:
        # Pop the lowest f-cost space-time state
        f, g, r, c, hb, t = heapq.heappop(open_heap)
        state = (r, c, hb, t)

        # Hard cutoff: discard expansions beyond the maximum allowed timestep
        if t > max_time:
            continue

        # Euclidean distance to goal (spatial only, ignoring time)
        dist = math.hypot(r - goal_rc[0], c - goal_rc[1])

        # Update "closest ever" for fallback path reconstruction
        if dist < min_dist:
            min_dist = dist
            closest_node = state

        # Accept goal when within stop distance (e.g., pile-clearance radius)
        if dist <= stop_dist_cells:
            closest_node = state
            break

        # Skip stale heap entry
        if g > g_cost.get(state, float('inf')):
            continue

        nt = t + 1   # next timestep (same for both wait and move actions)

        # ── WAIT ACTION ──────────────────────────────────────────────────────────
        # Stay in the same cell for one timestep; used to let other trucks pass.
        # Collision check: current truck footprint vs all locked footprints at nt.
        if nt <= max_time:
            wx, wy = grid.cell_to_world(r, c)
            wh = _heading_for_bucket(hb)
            if not _footprints_conflict(wx, wy, wh, hl, hw, nt, fp_constraints):
                wait_state = (r, c, hb, nt)
                wait_g = g + ASTAR_WAIT_COST
                if wait_g < g_cost.get(wait_state, float('inf')):
                    g_cost[wait_state] = wait_g

                    # Octile heuristic: exact cost to goal on an 8-connected grid (admissible)
                    abs_dr = abs(r - goal_rc[0])
                    abs_dc = abs(c - goal_rc[1])
                    h = max(abs_dr, abs_dc) + (math.sqrt(2) - 1) * min(abs_dr, abs_dc)

                    heapq.heappush(open_heap, (wait_g + h, wait_g, r, c, hb, nt))
                    came_from[wait_state] = state
            else:
                _n_wait_conflicts += 1

        # ── MOVE ACTIONS (8-connected via heading buckets) ───────────────────────
        for turn in (-1, 0, 1):
            next_hb    = (hb + turn) % 8              # clamp heading change to ±1 bucket (45°)
            dr, dc     = _BUCKET_TO_DIR[next_hb]      # direction unit vector for the new heading
            nr, nc     = r + dr, c + dc               # candidate next cell

            # Out-of-bounds check
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            # Terrain / clearance check for this heading bucket at this cell
            if not driveable[nr, nc, next_hb]:
                continue

            if nt > max_time:
                continue

            # Footprint overlap check: candidate truck body at (nr, nc, next_hb) at timestep nt.
            nwx, nwy = grid.cell_to_world(nr, nc)
            nwh      = _heading_for_bucket(next_hb)
            if _footprints_conflict(nwx, nwy, nwh, hl, hw, nt, fp_constraints):
                _n_move_conflicts += 1
                continue

            # Turn cost: 0 if straight, arc-length penalty otherwise
            turn_cost  = 0.0 if turn == 0 else 0.01 * turn_radius_cells

            # Euclidean step length: 1.0 for cardinal, √2 for diagonal
            move_cost  = math.hypot(dr, dc)
            new_g      = g + move_cost + turn_cost

            next_state = (nr, nc, next_hb, nt)

            # Only push if this path to the neighbour is strictly cheaper
            if new_g < g_cost.get(next_state, float('inf')):
                g_cost[next_state] = new_g

                # Octile heuristic (consistent with the wait-action heuristic above)
                abs_dr = abs(nr - goal_rc[0])
                abs_dc = abs(nc - goal_rc[1])
                h = max(abs_dr, abs_dc) + (math.sqrt(2) - 1) * min(abs_dr, abs_dc)

                heapq.heappush(open_heap, (new_g + h, new_g, nr, nc, next_hb, nt))
                came_from[next_state] = state   # record parent

    if fp_constraints and (_n_wait_conflicts or _n_move_conflicts):
        print(f"[ASTAR_ST] T{truck.id}: search done — "
              f"{_n_wait_conflicts} wait-conflict(s), {_n_move_conflicts} move-conflict(s) rejected")

    # Reconstruct (r, c) path from back-pointers; heading bucket is dropped here
    path, cur = [], closest_node
    while cur in came_from:
        path.append((cur[0], cur[1]))   # store only cell coords; heading was internal to search
        cur = came_from[cur]
    path.reverse()   # back-pointers give reversed order
    return path      # list of (row, col) — consecutive duplicates = wait-in-place
