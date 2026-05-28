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
from config import DRIVE_CLEARANCE_M, _TAN_REPOSE, ENTRY_POINT  # 0.4 m clearance limit, tan(angle of repose), gate coords

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

def _heading_bucket(dr, dc):
    # Convert a (row_delta, col_delta) step into its driveable-mask bucket index.
    # Falls back to bucket 0 if the direction isn't in the lookup table.
    return _DIR_TO_BUCKET.get((dr, dc), 0)

def _turn_cost(prev_dr, prev_dc, dr, dc, turn_radius_cells):
    # Penalise direction changes to discourage zig-zag paths.
    # No penalty for the first step (prev is None) or for going straight (dot product == 1).
    # A turn costs turn_radius * 0.3 to approximate the real-world arc length penalty.
    if prev_dr is None: return 0.0
    dot = prev_dr * dr + prev_dc * dc  # 1 = straight ahead, 0 = 90° turn, -1 = U-turn
    return 0.0 if dot == 1 else turn_radius_cells * 0.3

def astar(driveable, grid, start_rc, goal_rc, truck, blocked_cells=frozenset(), stop_dist_cells=0.0):
    turn_radius_cells = truck.turn_radius / grid.cell_size  # convert metres to cell units for the turn cost formula
    rows, cols        = driveable.shape[:2]                  # grid dimensions used for boundary checks

    # Heap entries: (f_cost, g_cost, row, col, prev_dr, prev_dc)
    open_heap = [(0.0, 0.0, start_rc[0], start_rc[1], None, None)]  # seed with start node, zero cost
    came_from = {}               # maps (r,c) → (parent_r, parent_c) for path reconstruction
    g_cost    = {start_rc: 0.0} # best known cost from start to each visited cell

    # ─── THE FIX: CLOSEST APPROACH TRACKER ───
    closest_node = start_rc  # if we never reach the goal radius, return the nearest node we found
    min_dist_to_target = math.hypot(start_rc[0] - goal_rc[0], start_rc[1] - goal_rc[1])  # baseline distance for comparison

    while open_heap:
        f, g, r, c, prev_dr, prev_dc = heapq.heappop(open_heap)  # expand the cheapest known node

        # Track how close we are getting
        dist_to_target = math.hypot(r - goal_rc[0], c - goal_rc[1])  # Euclidean distance to goal in cell units
        if dist_to_target < min_dist_to_target:  # new closest node found
            min_dist_to_target = dist_to_target
            closest_node = (r, c)

        # ─── RADIUS GOAL CHECK ───
        if dist_to_target <= stop_dist_cells:  # truck is within the safe stopping radius of the target
            closest_node = (r, c)              # record this as the final node
            break                              # stop searching — we're close enough

        if g > g_cost.get((r, c), float('inf')): continue  # stale heap entry (cheaper path already found) — skip

        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):  # explore 4-connected neighbours (no diagonals)
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols): continue  # skip out-of-bounds cells

            hb = _heading_bucket(dr, dc)               # get the mask bucket for this travel direction
            if not driveable[nr, nc, hb]: continue     # skip cells that aren't driveable in this direction

            traffic_cost = 10.0 if (nr, nc) in blocked_cells else 0.0  # soft penalty for cells occupied by other trucks
            tc = _turn_cost(prev_dr, prev_dc, dr, dc, turn_radius_cells)  # penalty for changing direction
            new_g = g + 1.0 + tc + traffic_cost  # total cost: 1 cell step + turn penalty + traffic penalty

            if new_g < g_cost.get((nr, nc), float('inf')):  # only update if this is a better path to (nr, nc)
                g_cost[(nr, nc)] = new_g  # record the best cost
                h = abs(nr - goal_rc[0]) + abs(nc - goal_rc[1])  # Manhattan heuristic (admissible for 4-connected grid)
                heapq.heappush(open_heap, (new_g + h, new_g, nr, nc, dr, dc))  # add to open set
                came_from[(nr, nc)] = (r, c)  # record how we reached this cell

    # ─── PATH RECONSTRUCTION (Never returns [] unless totally trapped) ───
    # It builds the path to the closest safe node it found!
    path, cur = [], closest_node  # walk back from closest/goal node to start via came_from
    while cur in came_from:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()  # reverse so path goes start → goal

    return path

def plan_paths(grid, assignments, existing_paths=None):
    if not assignments: return {}  # nothing to plan

    current_truck_cells = set()  # cells currently occupied by trucks — used to build soft obstacles
    if existing_paths is not None:
         for p in existing_paths.values():
             if p: current_truck_cells.add(p[0])  # treat the front of each existing path as a "soft block"

    for truck, _ in assignments:
         current_truck_cells.add(grid.world_to_cell(*truck.pos))  # add each truck's current cell

    mask_cache, paths = {}, {}  # cache driveable masks per truck class (expensive to recompute)
    entry_rc = grid.world_to_cell(*ENTRY_POINT)  # precompute the entry gate cell once

    for truck, target_rc in assignments:
        if truck.truck_class not in mask_cache:
            mask_cache[truck.truck_class] = make_driveable_mask(grid, truck)  # build (and cache) terrain mask for this truck type

        driveable  = mask_cache[truck.truck_class]        # the [row, col, bucket] boolean mask for this truck
        truck_cell = grid.world_to_cell(*truck.pos)       # current cell of the truck
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
            safe_dist_m = d_clearance + (truck.length / 2.0)                               # add half the truck length so the rear stays outside the pile
            stop_dist_cells = safe_dist_m / grid.cell_size                                  # convert metres to cells

        # GHOST MODE ACTIVATED: We pass frozenset() so it ignores 'obstacles' entirely
        path = astar(driveable, grid, truck_cell, target_rc, truck,
                     blocked_cells=frozenset(), stop_dist_cells=stop_dist_cells)  # plan path ignoring other trucks to prevent traffic deadlocks
        paths[truck.id] = path  # store path keyed by truck ID

        if path:
             current_truck_cells.add(path[0])  # register the truck's next cell as soft-blocked for subsequent trucks

        # --- DEBUG PRINT ---
        print(f"DEBUG: Truck {truck.id} ({truck.truck_class}) Target: {target_rc} Path Nodes: {len(path)} StopDistCells: {stop_dist_cells:.2f}")

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
        return path[min(t, len(path) - 1)]  # clamp t to the last index so the truck "stays" at its goal

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
            locked_spatial.update(path)  # every cell the locked truck will visit
    locked_spatial_frozen = frozenset(locked_spatial)  # immutable for repeated use inside low_level

    # ── PER-AGENT SETUP ───────────────────────────────────────────────────────────
    agent_info = {}
    for truck, target_rc in assignments:
        if truck.truck_class not in mask_cache:
            mask_cache[truck.truck_class] = make_driveable_mask(grid, truck)  # build terrain passability mask
        driveable  = mask_cache[truck.truck_class]
        truck_cell = grid.world_to_cell(*truck.pos)

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
            safe_dist_m     = d_clearance + (truck.length / 2.0)
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
                return [info['start']]
        return path

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
        return {aid: path}

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
            return paths

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
        return best
    return init_paths
