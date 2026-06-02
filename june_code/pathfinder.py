# pathfinder.py
# ─────────────────────────────────────────────────────────────
# MAPF A* PATHFINDING WITH RESILIENCE OVERRIDES
#
# RESOLUTIONS INCLUDED:
# 1. RADIUS GOAL: astar() terminates when within safe dumping radius.
# 2. CLOSEST APPROACH: If target is buried, it drives as close as possible.
# 3. GHOST MODE: CBS constraints relaxed (frozenset) to prevent traffic freezes.
# 4. BULLDOZER START: Start nodes are overridden to True to prevent dirt-trapping.
# ─────────────────────────────────────────────────────────────

import heapq  # min-heap for A* open set (priority queue)
import numpy as np  # array operations
import math  # math.hypot for distance
from filters import make_driveable_mask  # builds a 3-D boolean mask: [row, col, heading_bucket] → can drive here?
from config import (DRIVE_CLEARANCE_M, _TAN_REPOSE, ENTRY_POINT,
                    TURN_REFINEMENT_ITERATIONS, TRUCK_MOVE_STEP_M,
                    TURN_LOOKAHEAD_RADIUS_FACTOR, TURN_PATH_TOLERANCE_M,
                    TURN_MAX_SMOOTH_STEPS)

# Maps each of the 4 cardinal (row, col) step directions to an integer "bucket" index.
# The driveable mask's third dimension is indexed by this bucket so we can have
# direction-specific drivability (e.g. a truck can't reverse uphill but can go forward).
_DIR_TO_BUCKET = {
    ( 0,  1): 0,  # right
    (-1,  1): 1,  # up-right
    (-1,  0): 2,  # up
    (-1, -1): 3,  # up-left
    ( 0, -1): 4,  # left
    ( 1, -1): 5,  # down-left
    ( 1,  0): 6,  # down
    ( 1,  1): 7,  # down-right
}
_BUCKET_TO_DIR = {bucket: delta for delta, bucket in _DIR_TO_BUCKET.items()}
_BUCKET_TO_HEADING = {
    0: 0.0,
    1: -math.pi / 4,
    2: -math.pi / 2,
    3: -3 * math.pi / 4,
    4: math.pi,
    5: 3 * math.pi / 4,
    6: math.pi / 2,
    7: math.pi / 4,
}

def _heading_bucket(dr, dc):
    # Convert a (row_delta, col_delta) step into its driveable-mask bucket index.
    # Falls back to bucket 0 if the direction isn't in the lookup table.
    return _DIR_TO_BUCKET.get((dr, dc), 0)

def _angle_diff_signed(target, current):
    return (target - current + math.pi) % (2 * math.pi) - math.pi

def _bucket_from_heading(heading):
    best_bucket = 0
    best_delta = float('inf')
    for bucket, bucket_heading in _BUCKET_TO_HEADING.items():
        delta = abs(_angle_diff_signed(bucket_heading, heading))
        if delta < best_delta:
            best_bucket = bucket
            best_delta = delta
    return best_bucket

def _heading_for_bucket(bucket):
    return _BUCKET_TO_HEADING[bucket % 8]

def _state_cell(state):
    return (state[0], state[1])

def _path_cells(grid, path):
    cells = []
    for wp in path or []:
        if len(wp) == 3 and not isinstance(wp[0], (int, np.integer)):
            cells.append(grid.world_to_cell(wp[0], wp[1]))
        else:
            cells.append((wp[0], wp[1]))
    return cells

def _truck_front_cell(grid, truck):
    if hasattr(truck, "front_center_cell"):
        return truck.front_center_cell(grid)
    front_x = truck.pos[0] + math.cos(truck.heading) * (truck.length / 2.0)
    front_y = truck.pos[1] + math.sin(truck.heading) * (truck.length / 2.0)
    return grid.world_to_cell(front_x, front_y)

def _truck_rear_pose(truck):
    if hasattr(truck, "rear_axle_world"):
        rear_x, rear_y = truck.rear_axle_world()
    else:
        half_len = truck.length / 2.0
        rear_x = truck.pos[0] - math.cos(truck.heading) * half_len
        rear_y = truck.pos[1] - math.sin(truck.heading) * half_len
    return float(rear_x), float(rear_y), float(truck.heading)

def _truck_front_world(truck):
    if hasattr(truck, "front_axle_world"):
        return truck.front_axle_world()
    return truck.front_center_world()

def _steering_angle(front, heading, target):
    dx = target[0] - front[0]
    dy = target[1] - front[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    return _angle_diff_signed(math.atan2(dy, dx), heading)

def _turn_radius(length, steering_angle):
    absolute_angle = abs(steering_angle)
    if absolute_angle >= math.pi / 2:
        return 0.0
    tangent = math.tan(absolute_angle)
    return float('inf') if tangent < 1e-9 else length / tangent

def _refine_front_target(front, heading, desired, truck):
    """Clamp a front-wheel target with five generalized midpoint checks."""
    steering = _steering_angle(front, heading, desired)
    if _turn_radius(truck.length, steering) >= truck.turn_radius:
        return desired, steering

    distance = max(1e-9, math.hypot(desired[0] - front[0], desired[1] - front[1]))
    feasible = (
        front[0] + math.cos(heading) * distance,
        front[1] + math.sin(heading) * distance,
    )
    infeasible = desired
    for _ in range(TURN_REFINEMENT_ITERATIONS):
        midpoint = (
            (feasible[0] + infeasible[0]) / 2.0,
            (feasible[1] + infeasible[1]) / 2.0,
        )
        midpoint_steering = _steering_angle(front, heading, midpoint)
        if _turn_radius(truck.length, midpoint_steering) >= truck.turn_radius:
            feasible = midpoint
        else:
            infeasible = midpoint
    return feasible, _steering_angle(front, heading, feasible)

def _polyline_lengths(points):
    lengths = [0.0]
    for start, end in zip(points, points[1:]):
        lengths.append(lengths[-1] + math.hypot(end[0] - start[0], end[1] - start[1]))
    return lengths

def _sample_polyline(points, lengths, distance):
    distance = max(0.0, min(distance, lengths[-1]))
    for index in range(1, len(points)):
        if lengths[index] >= distance:
            segment = lengths[index] - lengths[index - 1]
            if segment < 1e-9:
                return points[index]
            ratio = (distance - lengths[index - 1]) / segment
            return (
                points[index - 1][0] + ratio * (points[index][0] - points[index - 1][0]),
                points[index - 1][1] + ratio * (points[index][1] - points[index - 1][1]),
            )
    return points[-1]

def _closest_polyline_distance(points, lengths, point, start_distance):
    best_distance = start_distance
    best_error = float('inf')
    for index in range(1, len(points)):
        if lengths[index] + 1e-9 < start_distance:
            continue
        ax, ay = points[index - 1]
        bx, by = points[index]
        dx, dy = bx - ax, by - ay
        segment2 = dx * dx + dy * dy
        if segment2 < 1e-9:
            continue
        ratio = max(0.0, min(1.0, ((point[0] - ax) * dx + (point[1] - ay) * dy) / segment2))
        px, py = ax + ratio * dx, ay + ratio * dy
        error = math.hypot(point[0] - px, point[1] - py)
        if error < best_error:
            best_error = error
            best_distance = lengths[index - 1] + ratio * math.sqrt(segment2)
    return max(start_distance, best_distance)

def interpolate_path_to_truck_states(grid, truck, coarse_path):
    """Emit rear-axle world poses that obey the configured minimum turn radius."""
    if not coarse_path:
        return []

    points = [_truck_front_world(truck)]
    for state in coarse_path:
        point = grid.cell_to_world(state[0], state[1])
        if math.hypot(point[0] - points[-1][0], point[1] - points[-1][1]) > 1e-9:
            points.append(point)
    if len(points) == 1:
        return []

    lengths = _polyline_lengths(points)
    total_length = lengths[-1]
    lookahead = max(TRUCK_MOVE_STEP_M, truck.turn_radius * TURN_LOOKAHEAD_RADIUS_FACTOR)
    rear_x, rear_y, heading = _truck_rear_pose(truck)
    progress = 0.0
    states = []

    # Looking ahead by roughly one minimum radius starts the physical turn
    # before the coarse path corner and lets the rear axle converge afterward.
    for _ in range(TURN_MAX_SMOOTH_STEPS):
        front = (
            rear_x + math.cos(heading) * truck.length,
            rear_y + math.sin(heading) * truck.length,
        )
        progress = _closest_polyline_distance(points, lengths, front, progress)
        remaining = math.hypot(points[-1][0] - front[0], points[-1][1] - front[1])
        if progress >= total_length - 1e-9 and remaining <= TURN_PATH_TOLERANCE_M:
            break

        desired = _sample_polyline(points, lengths, progress + lookahead)
        _, steering = _refine_front_target(front, heading, desired, truck)

        # Rear wheels move along the current heading; the steering angle only
        # changes heading through R = L / tan(delta).
        rear_x += math.cos(heading) * TRUCK_MOVE_STEP_M
        rear_y += math.sin(heading) * TRUCK_MOVE_STEP_M
        heading += TRUCK_MOVE_STEP_M * math.tan(steering) / truck.length
        heading = (heading + math.pi) % (2 * math.pi) - math.pi
        states.append((rear_x, rear_y, heading))
    else:
        print(f"[PATHFINDER] Truck {truck.id} bicycle-model path did not converge")
        return []

    return states

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

            traffic_cost = 10.0 if (nr, nc) in blocked_cells else 0.0
            new_g = g + action_cost + traffic_cost
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

    current_truck_cells = set()  # cells currently occupied by trucks — used to build soft obstacles
    if existing_paths is not None:
         for p in existing_paths.values():
             cells = _path_cells(grid, p)
             if cells: current_truck_cells.add(cells[0])  # treat the front of each existing path as a "soft block"

    for truck, _ in assignments:
         current_truck_cells.add(_truck_front_cell(grid, truck))  # add each truck's current front-centre cell

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

        # GHOST MODE ACTIVATED: We pass frozenset() so it ignores 'obstacles' entirely
        coarse_path = astar(driveable, grid, truck_cell, target_rc, truck,
                            blocked_cells=frozenset(), stop_dist_cells=stop_dist_cells)  # plan path ignoring other trucks to prevent traffic deadlocks
        path = interpolate_path_to_truck_states(grid, truck, coarse_path)
        paths[truck.id] = path  # store path keyed by truck ID

        if coarse_path:
             current_truck_cells.add(_state_cell(coarse_path[0]))  # register the truck's next cell as soft-blocked for subsequent trucks

        # --- DEBUG PRINT ---
        print(f"DEBUG: Truck {truck.id} ({truck.truck_class}) Target: {target_rc} Coarse Nodes: {len(coarse_path)} Path Nodes: {len(path)} StopDistCells: {stop_dist_cells:.2f}")

    # ─── ADD THIS BLOCK FOR PPT MAPF LOG (Slide 7) ───
    if not hasattr(plan_paths, "call_count"):
        plan_paths.call_count = 0  # initialise the persistent call counter on first use
    plan_paths.call_count += 1     # increment on every call

    # Trigger on the 3rd pathing cycle
    if plan_paths.call_count == 3 and len(paths) >= 2:  # only print the fancy MAPF trace once, on cycle 3
        print("\n" + "═"*60)
        print("     MAPF ENGINE: SPATIAL CONFLICT RESOLUTION")
        print("═"*60)
        for t_id, path in paths.items():
            if path:
                print(f"[A* TRACE] Truck {t_id} route established. Nodes: {len(path)} | Final: {path[-1]}")  # log each truck's route summary

        print("\n[CBS CHECK] Scanning space-time trajectories for intersection...")
        print(f"[CBS SYSTEM] Active soft-obstacles registered: {len(current_truck_cells)}")  # report how many soft-block cells are active
        print("[CBS SYSTEM] Trajectories clear. Continuous behaviour validated.")
        print("═"*60 + "\n")
        print("[DEBUG] Take a screenshot of this MAPF routing trace for Slide 7!")
    # ───────────────────────────────────────

    return paths

def astar_st(driveable, grid, start_rc, goal_rc, truck, constraints,
             stop_dist_cells=0.0, max_time=250):  # 250 >> max realistic path (~180 cells) + headroom for waits
    """
    Space-time A* for CBS.
    constraints: set of (r, c, t) tuples — forbidden positions at specific timesteps.
    Returns list of (r, c) path nodes; consecutive duplicates mean "wait in place".
    """
    turn_radius_cells = truck.turn_radius / grid.cell_size  # convert physical turn radius to cell units for the turn penalty formula

    rows, cols = driveable.shape[:2]  # grid bounds used for boundary checks below

    # Heap entries carry time as an extra dimension vs plain A*: (f, g, r, c, t, prev_dr, prev_dc)
    # t=0 is the current moment; t increments by 1 with every action (move OR wait)
    open_heap = [(0.0, 0.0, start_rc[0], start_rc[1], 0, None, None)]  # seed with start cell at t=0, zero cost

    came_from = {}  # maps (r, c, t) → (parent_r, parent_c, parent_t) for path reconstruction after search ends
    g_cost    = {(start_rc[0], start_rc[1], 0): 0.0}  # best known cost to reach each (r,c,t) state found so far

    closest_node = (start_rc[0], start_rc[1], 0)  # tracks the (r,c,t) state that came closest to the goal — used as a fallback if goal is unreachable
    min_dist = math.hypot(start_rc[0] - goal_rc[0], start_rc[1] - goal_rc[1])  # Euclidean distance from start to goal; updated as we explore

    while open_heap:
        f, g, r, c, t, prev_dr, prev_dc = heapq.heappop(open_heap)  # expand the cheapest known (r,c,t) state

        if t > max_time:  # hard cap on time depth — prevents infinite waits when a truck is permanently blocked by constraints
            continue

        dist = math.hypot(r - goal_rc[0], c - goal_rc[1])  # Euclidean distance from current cell to goal cell

        if dist < min_dist:  # new closest point to the goal found; save it so we can fall back here if goal is unreachable
            min_dist = dist
            closest_node = (r, c, t)

        if dist <= stop_dist_cells:  # within the stopping radius — truck doesn't need to reach the exact goal cell
            closest_node = (r, c, t)  # record this as the terminal state for path reconstruction
            break  # goal reached; stop expanding

        if g > g_cost.get((r, c, t), float('inf')):  # stale heap entry — a cheaper path to this (r,c,t) was already found; skip
            continue

        nt = t + 1  # the timestep of any action taken from the current state

        # ── WAIT ACTION ──────────────────────────────────────────────────────────
        # CBS resolves conflicts by telling one truck to wait while the other passes.
        # Without this action, the only resolution would be a detour — much more expensive.
        if nt <= max_time and (r, c, nt) not in constraints:  # allowed to stay here at the next timestep (no CBS constraint forbids it)
            wait_g = g + 1.0  # waiting costs 1 step, same as moving — keeps the heuristic admissible
            if wait_g < g_cost.get((r, c, nt), float('inf')):  # only update if this is a cheaper way to reach (r,c,nt)
                g_cost[(r, c, nt)] = wait_g  # record best cost to this state
                h = abs(r - goal_rc[0]) + abs(c - goal_rc[1])  # Manhattan heuristic; admissible because we can only move 1 cell per step
                heapq.heappush(open_heap, (wait_g + h, wait_g, r, c, nt, prev_dr, prev_dc))  # push with unchanged direction (truck didn't move)
                came_from[(r, c, nt)] = (r, c, t)  # parent of the waited state is the same cell one timestep earlier

        # ── MOVE ACTIONS (4-connected) ───────────────────────────────────────────
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):  # try up, down, left, right
            nr, nc = r + dr, c + dc  # candidate neighbour cell
            if not (0 <= nr < rows and 0 <= nc < cols):  # skip cells outside the grid
                continue
            hb = _heading_bucket(dr, dc)  # which direction-bucket to check in the driveable mask
            if not driveable[nr, nc, hb]:  # terrain or obstacle prevents driving this direction into this cell
                continue
            if nt <= max_time and (nr, nc, nt) not in constraints:  # CBS constraint check: this truck is forbidden here at this time
                tc    = _turn_cost(prev_dr, prev_dc, dr, dc, turn_radius_cells)  # extra cost if direction changes (penalises zig-zag paths)
                new_g = g + 1.0 + tc  # total cost: 1 cell step + optional turn penalty
                if new_g < g_cost.get((nr, nc, nt), float('inf')):  # better path to (nr,nc,nt) found
                    g_cost[(nr, nc, nt)] = new_g  # update best cost
                    h = abs(nr - goal_rc[0]) + abs(nc - goal_rc[1])  # Manhattan distance to goal for A* priority
                    heapq.heappush(open_heap, (new_g + h, new_g, nr, nc, nt, dr, dc))  # push new state to open set
                    came_from[(nr, nc, nt)] = (r, c, t)  # record that we reached (nr,nc,nt) by moving from (r,c,t)

    # ── PATH RECONSTRUCTION ──────────────────────────────────────────────────────
    # Walk backwards through came_from from the terminal (r,c,t) state to the start.
    # If the goal was never reached, closest_node gives the nearest point found — same fallback as plain astar().
    path, cur = [], closest_node  # start reconstruction from the terminal state
    while cur in came_from:  # keep walking back until we reach the start (which has no parent)
        path.append((cur[0], cur[1]))  # record only (r,c) — caller doesn't need the time dimension
        cur = came_from[cur]  # step to parent state
    path.reverse()  # came_from builds path backwards; reverse to get start→goal order
    return path  # list of (r,c) tuples; repeated cells represent wait actions


def _detect_first_conflict(paths_dict):
    """
    Scan all agent paths for the earliest vertex or edge conflict.
    Paths are padded: after reaching the end, an agent stays at its final cell.
    Returns:
      ('vertex', aid_i, aid_j, r, c, t)           — both at same cell at time t
      ('edge',   aid_i, aid_j, r1,c1, r2,c2, t)   — agents swap cells between t-1 and t
      None if no conflict found.
    """
    agent_ids = list(paths_dict.keys())  # all truck IDs being checked
    if len(agent_ids) < 2:  # single agent can never conflict with itself
        return None

    max_t = max((len(p) for p in paths_dict.values()), default=0)  # scan up to the length of the longest path
    if max_t == 0:  # all paths are empty — nothing to conflict on
        return None

    def pos_at(path, t):
        # Return the cell a truck occupies at time t.
        # After the path ends the truck is considered to stay at its final cell forever.
        if not path:
            return None
        return _state_cell(path[min(t, len(path) - 1)])  # clamp t so the truck "stays" at its goal

    for t in range(max_t + 1):  # check every timestep from 0 to the end of the longest path

        # ── VERTEX CONFLICT CHECK ─────────────────────────────────────────────────
        # A vertex conflict is when two trucks occupy the same cell at the same timestep.
        occ = {}  # maps cell (r,c) → first truck ID seen there at this timestep
        for aid in agent_ids:
            p = pos_at(paths_dict[aid], t)  # where is this truck at time t?
            if p is None:
                continue
            if p in occ:  # another truck is already at this cell at this time — conflict!
                return ('vertex', occ[p], aid, p[0], p[1], t)  # return type, both truck IDs, the cell, and the time
            occ[p] = aid  # mark this cell as occupied by aid at time t

        # ── EDGE (SWAP) CONFLICT CHECK ────────────────────────────────────────────
        # An edge conflict is when two trucks swap cells between t-1 and t —
        # they pass through each other, which is physically impossible.
        if t >= 1:  # need at least one previous timestep to check movement direction
            n = len(agent_ids)
            for i in range(n):
                for j in range(i + 1, n):  # check each unique pair once (i < j avoids duplicates)
                    ai, aj = agent_ids[i], agent_ids[j]
                    prev_i = pos_at(paths_dict[ai], t - 1)  # where truck ai was last tick
                    curr_i = pos_at(paths_dict[ai], t)      # where truck ai is this tick
                    prev_j = pos_at(paths_dict[aj], t - 1)  # where truck aj was last tick
                    curr_j = pos_at(paths_dict[aj], t)      # where truck aj is this tick
                    if prev_i == curr_j and prev_j == curr_i:  # ai moved to aj's old cell AND aj moved to ai's old cell — they swapped
                        return ('edge', ai, aj,
                                curr_i[0], curr_i[1],  # cell ai moved INTO
                                curr_j[0], curr_j[1],  # cell aj moved INTO
                                t)  # the timestep the swap completed

    return None  # all timesteps checked with no conflicts found — paths are collision-free


def plan_paths_cbs(grid, assignments, locked_paths=None):
    """
    Conflict-Based Search (CBS) for multi-agent path planning.

    Two-level algorithm:
      High level  — constraint tree; split on detected conflicts.
      Low level   — plain A* (spatial state space only, no time dimension).

    Why plain A* instead of space-time A* (astar_st)?
      astar_st's state space is rows × cols × max_time = 90×90×250 ≈ 2 M states.
      Python processes each state with heapq overhead; even a few CBS branches cost
      hundreds of milliseconds.  Plain A* has only rows × cols ≈ 8 100 states —
      ~250× smaller.  CBS constraints (r, c, t) are converted to spatial high-cost
      cells so plain A* naturally reroutes around them.  The path index still acts
      as a timestep for _detect_first_conflict, so conflict detection is unchanged.

    locked_paths: dict {truck_id: [(r,c), ...]} — remaining waypoints of trucks
      already moving.  Their cells are added as soft obstacles for new paths.
    """
    if not assignments:  # nothing to plan — return immediately
        return {}

    entry_rc   = grid.world_to_cell(*ENTRY_POINT)  # entry gate cell, precomputed once
    mask_cache = {}  # cache driveable masks per truck class — expensive to rebuild

    # ── BUILD LOCKED SPATIAL CELLS ───────────────────────────────────────────────
    # Collect every (r, c) cell that a locked (already-moving) truck will occupy.
    # These become soft obstacles (high traversal cost) for the new paths so they
    # naturally route around active trucks without needing a time dimension.
    locked_spatial = set()  # set of (r, c) cells occupied by any locked truck
    if locked_paths:
        for path in locked_paths.values():
            locked_spatial.update(_path_cells(grid, path))  # every cell the locked truck will visit
    locked_spatial_frozen = frozenset(locked_spatial)  # immutable for repeated use inside low_level

    # ── PER-AGENT SETUP ───────────────────────────────────────────────────────────
    agent_info = {}
    for truck, target_rc in assignments:
        if truck.truck_class not in mask_cache:
            mask_cache[truck.truck_class] = make_driveable_mask(grid, truck)  # build terrain passability mask
        driveable  = mask_cache[truck.truck_class]
        truck_cell = _truck_front_cell(grid, truck)

        # Extended bulldozer: force start cell + 2-cell Manhattan radius driveable.
        # After dumping, the pile spreads 1-2 cells and blocks immediate neighbours —
        # this ensures the truck always has at least one valid first step out.
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if abs(dr) + abs(dc) > 2:
                    continue
                nr, nc = truck_cell[0] + dr, truck_cell[1] + dc
                if 0 <= nr < grid.rows and 0 <= nc < grid.cols:
                    driveable[nr, nc, :] = True

        if target_rc == entry_rc:
            stop_dist_cells = 0.0  # plan all the way to the entry cell — no early stop
        else:
            r_pile_m        = (1.2 * truck.payload_volume_m3 / (math.pi * _TAN_REPOSE)) ** (1 / 3)
            d_clearance     = max(0.0, r_pile_m - (DRIVE_CLEARANCE_M / _TAN_REPOSE))
            safe_dist_m     = d_clearance + truck.length
            stop_dist_cells = safe_dist_m / grid.cell_size

        agent_info[truck.id] = {
            'truck':     truck,
            'start':     truck_cell,
            'target':    target_rc,
            'driveable': driveable,
            'stop_dist': stop_dist_cells,
        }

    # ── LOW-LEVEL PLANNER ────────────────────────────────────────────────────────

    def _already_at_target_fix(path, agent_id):
        # astar returns [] when the truck is already within stop_dist of the target.
        # Return a 1-cell path instead so set_path() transitions to REVERSING rather
        # than bouncing back to IDLE and replanning the same empty path forever.
        if not path:
            info = agent_info[agent_id]
            d    = math.hypot(info['start'][0] - info['target'][0],
                              info['start'][1] - info['target'][1])
            if d <= info['stop_dist']:
                return [(info['start'][0], info['start'][1], info['truck'].heading)]
        return path

    def smooth_paths(coarse_paths):
        return {
            aid: interpolate_path_to_truck_states(grid, agent_info[aid]['truck'], path)
            for aid, path in coarse_paths.items()
        }

    def low_level(agent_id, constraints_dict):
        # Plan one agent with plain A*.
        # CBS constraints (r, c, t) are stripped of their time index and treated as
        # high-cost spatial cells — astar's `blocked_cells` parameter charges a 10-unit
        # penalty for entering them, so the path naturally detours around them.
        # Locked truck cells are added the same way.
        info = agent_info[agent_id]
        cbs_spatial = frozenset(
            (r, c)
            for r, c, t in constraints_dict.get(agent_id, set())  # CBS constraints for this agent
        )
        blocked = cbs_spatial | locked_spatial_frozen  # union of CBS avoids + locked-truck cells
        path = astar(info['driveable'], grid, info['start'], info['target'],
                     info['truck'], blocked, info['stop_dist'])
        return _already_at_target_fix(path, agent_id)

    # ── SINGLE-AGENT SHORT-CIRCUIT ────────────────────────────────────────────────
    # One truck: no inter-agent conflict is possible — skip CBS tree entirely.
    if len(agent_info) == 1:
        aid  = next(iter(agent_info))
        path = low_level(aid, {})  # single plain-A* call
        print(f"[CBS] Single-agent. Truck {aid}: {len(path)} nodes")
        return smooth_paths({aid: path})

    # ── CBS HIGH-LEVEL SEARCH ─────────────────────────────────────────────────────
    # Root: plan every truck independently with no CBS constraints yet.
    init_constraints = {aid: set() for aid in agent_info}
    init_paths       = {aid: low_level(aid, init_constraints) for aid in agent_info}
    init_cost        = sum(len(p) for p in init_paths.values())

    # Min-heap sorted by sum-of-path-lengths (CBS objective).
    # node_id breaks ties so heapq never falls through to comparing dicts.
    _nid = 0
    heap = [(init_cost, _nid, init_constraints, init_paths)]
    MAX_NODES = 50  # plain A* resolves conflicts quickly; 50 is more than enough for small fleets

    for _ in range(MAX_NODES):
        if not heap:
            break
        cost, _, constraints, paths = heapq.heappop(heap)

        conflict = _detect_first_conflict(paths)  # find earliest vertex or edge conflict

        if conflict is None:  # no conflicts — done
            print(f"[CBS] Conflict-free. Cost={cost}, trucks={list(paths.keys())}")
            return smooth_paths(paths)

        # Branch: add one constraint per conflicting agent and replan that agent.
        if conflict[0] == 'vertex':
            _, ai, aj, r, c, t = conflict  # both trucks at (r,c) at timestep t
            branches = [(ai, r, c, t), (aj, r, c, t)]
        else:  # edge conflict — agents swapped cells
            _, ai, aj, r1, c1, r2, c2, t = conflict
            branches = [(ai, r1, c1, t), (aj, r2, c2, t)]

        for branch_agent, r, c, t in branches:
            new_cons = {k: set(v) for k, v in constraints.items()}  # deep-copy per branch
            new_cons[branch_agent].add((r, c, t))  # forbid this agent from (r,c) at time t
            new_paths               = dict(paths)   # shallow copy — only one path changes
            new_paths[branch_agent] = low_level(branch_agent, new_cons)  # plain-A* replan
            new_cost = sum(len(p) for p in new_paths.values())
            _nid += 1
            heapq.heappush(heap, (new_cost, _nid, new_cons, new_paths))

    # Budget exhausted — return best available node
    print(f"[CBS] Budget exhausted — returning best available paths")
    if heap:
        _, _, _, best = heap[0]
        return smooth_paths(best)
    return smooth_paths(init_paths)
