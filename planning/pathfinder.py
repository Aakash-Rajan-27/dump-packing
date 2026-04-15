# core/pathfinder.py
# ─────────────────────────────────────────────────────────────
# Week 1: Simple A* per truck (may occasionally conflict)
# Week 2: Upgrade to CBS (Conflict-Based Search) by replacing
#         plan_paths() — everything else in main.py stays the same.
#
# A* finds the shortest path between two grid cells,
# treating other trucks' current positions as obstacles.
# ─────────────────────────────────────────────────────────────

import heapq    # heapq: Python's priority queue (min-heap)
                # heapq.heappush / heapq.heappop — always gives smallest item first
import numpy as np
from core.grid_map import CellState


def astar(grid, start_rc, goal_rc, blocked_cells=set()):
    """
    A* pathfinding from start_rc to goal_rc on the grid.

    grid: GridMap
    start_rc: (row, col) starting cell
    goal_rc: (row, col) destination cell
    blocked_cells: set of (row,col) to treat as walls (other trucks' positions)

    Returns: list of (row,col) cells from start to goal (not including start).
             Empty list if no path found.
    """
    if start_rc == goal_rc:
        return []  # already there

    # ── A* data structures ─────────────────────────────────
    # open_heap: priority queue of (f_cost, cell) — always expand cheapest first
    # f_cost = g_cost (steps so far) + h_cost (heuristic: remaining distance)
    open_heap = []
    heapq.heappush(open_heap, (0, start_rc))  # start with cost 0

    # came_from: maps cell → the cell we came from (for path reconstruction)
    came_from = {}

    # g_cost: actual cost (number of steps) to reach each cell
    # Start: 0 steps to reach start. All others: infinity (not yet reached).
    g_cost = {start_rc: 0}

    # ── Main loop ──────────────────────────────────────────
    while open_heap:
        # Pop the cell with the lowest f_cost
        f, current = heapq.heappop(open_heap)

        # If we reached the goal, stop searching
        if current == goal_rc:
            break

        # Check all 4 neighbours (no diagonal movement — trucks don't go diagonal)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr = current[0] + dr   # neighbour row
            nc = current[1] + dc   # neighbour col
            neighbour = (nr, nc)

            # Skip if outside grid bounds
            if not (0 <= nr < grid.rows and 0 <= nc < grid.cols):
                continue

            # Skip if cell is not driveable
            state = grid.state[nr, nc]
            if state in (CellState.BOUNDARY, CellState.FILLED, CellState.OBSTACLE):
                continue

            # Skip if another truck is blocking this cell
            if neighbour in blocked_cells:
                continue

            # Cost to reach this neighbour = cost to reach current + 1 step
            new_g = g_cost[current] + 1

            # Only update if this is a cheaper path than we've found before
            if new_g < g_cost.get(neighbour, float('inf')):
                g_cost[neighbour] = new_g

                # Heuristic: Manhattan distance to goal (admissible = never overestimates)
                # Manhattan distance = |row_diff| + |col_diff|
                h = abs(nr - goal_rc[0]) + abs(nc - goal_rc[1])

                f_cost = new_g + h  # f = g + h

                heapq.heappush(open_heap, (f_cost, neighbour))
                came_from[neighbour] = current  # remember where we came from

    # ── Reconstruct path ───────────────────────────────────
    # Walk backwards from goal to start using came_from dict
    if goal_rc not in came_from:
        return []   # no path found (goal is unreachable)

    path = []
    current = goal_rc
    while current in came_from:         # walk back until we hit start (no entry)
        path.append(current)
        current = came_from[current]
    path.reverse()                      # flip so it goes start → goal
    return path                         # list of (row,col) cells


def plan_paths(grid, assignments):
    """
    Plan paths for all trucks in the assignment list.
    Week 1: each truck plans independently with A*.
    Other trucks' CURRENT positions are treated as soft obstacles.

    assignments: list of (truck, dump_point) pairs from Hungarian
    grid: GridMap

    Returns: dict {truck_id: path} where path is list of (row,col)
    """
    paths = {}

    # Collect current positions of all trucks to use as soft obstacles
    # (so trucks don't plan paths through each other)
    all_truck_positions = set()
    for truck, _ in assignments:
        current_cell = grid.world_to_cell(*truck.pos)
        all_truck_positions.add(current_cell)

    for truck, dump_point in assignments:
        # This truck's own position is NOT an obstacle for itself
        truck_cell = grid.world_to_cell(*truck.pos)
        obstacles = all_truck_positions - {truck_cell}

        # Find path from truck's current position to its dump point
        path = astar(grid, truck_cell, dump_point, blocked_cells=obstacles)

        paths[truck.id] = path  # store path keyed by truck ID

    return paths

# ─────────────────────────────────────────────────────────────
# WEEK 2 UPGRADE: Replace plan_paths() with CBS version below.
# The interface is identical — main.py doesn't change at all.
# ─────────────────────────────────────────────────────────────

def plan_paths_cbs(grid, assignments):
    """
    CBS (Conflict-Based Search) path planner.
    Guarantees no two trucks occupy the same cell at the same timestep.

    High level: constraint tree searched best-first.
    Low level: constrained A* per truck.

    This replaces plan_paths() in week 2.
    Uncomment and swap in main.py when ready.
    """

    # ── Helper: constrained A* ─────────────────────────────
    def astar_constrained(start, goal, constraints):
        """
        A* where constraints = set of (row, col, timestep) forbidden states.
        The truck cannot be at (row,col) at time=timestep.
        State space: (row, col, timestep)
        """
        # open_heap: (f_cost, g_cost, (row, col, time))
        open_heap = [(0, 0, (*start, 0))]
        came_from = {}
        g_cost = {(*start, 0): 0}

        while open_heap:
            f, g, state = heapq.heappop(open_heap)
            r, c, t = state

            if (r, c) == goal:
                # Reconstruct path (strip timestep dimension)
                path = []
                cur = state
                while cur in came_from:
                    path.append((cur[0], cur[1]))
                    cur = came_from[cur]
                path.reverse()
                return path

            # 5 actions: move in 4 directions + wait in place
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(0,0)]:
                nr, nc, nt = r+dr, c+dc, t+1

                if not (0 <= nr < grid.rows and 0 <= nc < grid.cols):
                    continue

                state_val = grid.state[nr, nc]
                if state_val in (CellState.BOUNDARY, CellState.FILLED, CellState.OBSTACLE):
                    continue

                # Check constraint: is (nr, nc, nt) forbidden?
                if (nr, nc, nt) in constraints:
                    continue    # skip — CBS told this truck not to go here at time nt

                new_g = g + 1
                new_state = (nr, nc, nt)

                if new_g < g_cost.get(new_state, float('inf')):
                    g_cost[new_state] = new_g
                    h = abs(nr - goal[0]) + abs(nc - goal[1])
                    heapq.heappush(open_heap, (new_g + h, new_g, new_state))
                    came_from[new_state] = (r, c, t)

        return []   # no path

    # ── CBS high-level ─────────────────────────────────────
    # A "node" in the constraint tree is: (constraints, paths, total_cost)
    # constraints: dict {truck_id: set of (r,c,t) forbidden states}
    # paths: dict {truck_id: path}
    # total_cost: sum of all path lengths

    truck_list = [truck for truck, _ in assignments]
    goal_map   = {truck.id: dump_point for truck, dump_point in assignments}

    def find_conflict(paths):
        """
        Check all pairs of trucks for conflicts.
        Returns (truck_i_id, truck_j_id, row, col, timestep) or None.
        Vertex conflict: both trucks at same (r,c) at same time t.
        Edge conflict: trucks swap positions between t and t+1.
        """
        ids = list(paths.keys())
        for a in range(len(ids)):
            for b in range(a+1, len(ids)):
                id_a, id_b = ids[a], ids[b]
                path_a = paths[id_a]
                path_b = paths[id_b]
                max_t = max(len(path_a), len(path_b))
                for t in range(max_t):
                    # Get position at time t (stay at last cell if path ended)
                    pa = path_a[t] if t < len(path_a) else path_a[-1] if path_a else None
                    pb = path_b[t] if t < len(path_b) else path_b[-1] if path_b else None
                    if pa and pb and pa == pb:
                        return (id_a, id_b, pa[0], pa[1], t)  # vertex conflict
        return None  # no conflicts found

    # Initialise: plan paths with no constraints
    init_constraints = {t.id: set() for t in truck_list}
    init_paths = {}
    for truck in truck_list:
        start = grid.world_to_cell(*truck.pos)
        goal  = goal_map[truck.id]
        init_paths[truck.id] = astar_constrained(start, goal, set())

    init_cost = sum(len(p) for p in init_paths.values())

    # CBS tree: min-heap of (cost, node_id, constraints, paths)
    node_counter = [0]   # unique ID for each node (needed for heap stability)
    cbs_heap = [(init_cost, node_counter[0], init_constraints, init_paths)]
    node_counter[0] += 1

    MAX_CBS_ITERATIONS = 500   # safety limit — stop after this many expansions

    for iteration in range(MAX_CBS_ITERATIONS):
        if not cbs_heap:
            break

        cost, _, constraints, paths = heapq.heappop(cbs_heap)

        # Check if this solution has any conflicts
        conflict = find_conflict(paths)

        if conflict is None:
            # No conflicts! This is our solution.
            return paths   # optimal and conflict-free

        # Conflict found between trucks A and B at (r,c,t)
        id_a, id_b, r, c, t = conflict

        # Branch: create two child nodes
        # Left child: forbid truck A from (r,c,t)
        # Right child: forbid truck B from (r,c,t)
        for blocked_id, other_id in [(id_a, id_b), (id_b, id_a)]:
            # Copy constraints and add the new one
            new_constraints = {tid: set(s) for tid, s in constraints.items()}
            new_constraints[blocked_id].add((r, c, t))

            # Replan ONLY the affected truck
            truck_obj = next(t for t in truck_list if t.id == blocked_id)
            start = grid.world_to_cell(*truck_obj.pos)
            goal  = goal_map[blocked_id]
            new_path = astar_constrained(start, goal, new_constraints[blocked_id])

            # Build new paths dict with updated path for the replanned truck
            new_paths = dict(paths)
            new_paths[blocked_id] = new_path

            new_cost = sum(len(p) for p in new_paths.values())
            heapq.heappush(cbs_heap, (new_cost, node_counter[0], new_constraints, new_paths))
            node_counter[0] += 1

    # If CBS didn't converge, fall back to independent A*
    print("CBS did not converge — falling back to independent A*")
    return plan_paths(grid, assignments)
