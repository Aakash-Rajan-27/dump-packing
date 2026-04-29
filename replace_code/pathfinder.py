# pathfinder.py
# ─────────────────────────────────────────────────────────────
# Path planning for the mixed fleet.
#
# Mixed-fleet changes from original:
#
# 1. PARTIAL cells are IMPASSABLE (same as FILLED).
#    A partial pile is a real physical mound of material.
#    The truck cannot drive through it regardless of height.
#    PARTIAL only differs from FILLED in that another truck
#    can still DUMP there — but no one can DRIVE over it.
#
# 2. Turn-radius-aware step cost.
#    Each move has a base cost of 1.0.
#    A 90-degree turn pays an extra penalty:
#      penalty = (turn_radius / cell_size) * 0.5
#    Cat 793F (33m turn radius, ~11 cells) → penalty ≈ 5.5
#    Cat 773F (22m turn radius, ~7 cells)  → penalty ≈ 3.7
#    Large trucks naturally find straighter routes.
#
# 3. CBS astar_constrained fixed:
#    - PARTIAL now correctly impassable in time-expanded search
#    - Variable name shadow fixed (loop var 'truck_obj' instead of 't')
# ─────────────────────────────────────────────────────────────

import heapq
import numpy as np
from grid_map import CellState


# Cells no truck can drive through, regardless of class
_IMPASSABLE = (CellState.BOUNDARY,
               CellState.PARTIAL,    # real pile — physically blocked
               CellState.FILLED,     # real pile — physically blocked
               CellState.OBSTACLE)
# PROTECTED (entry corridor) and EMPTY and RESERVED are all driveable.


def _turn_cost(prev_rc, next_rc, current_rc, turn_radius_cells):
    """
    Extra cost when the truck changes direction at current_rc.

    Incoming direction: current - prev
    Outgoing direction: next    - current
    Dot product = 1 (straight), 0 (90-deg turn), never -1 in 4-dir grid.

    A 90-degree turn costs turn_radius_cells * 0.5 extra on top of the
    base step cost of 1.0. This makes large trucks prefer long straight
    corridors rather than zigzagging through the paddock.
    """
    if prev_rc is None:
        return 0.0
    dr_in  = current_rc[0] - prev_rc[0]
    dc_in  = current_rc[1] - prev_rc[1]
    dr_out = next_rc[0]    - current_rc[0]
    dc_out = next_rc[1]    - current_rc[1]
    dot = dr_in * dr_out + dc_in * dc_out
    return 0.0 if dot == 1 else turn_radius_cells * 0.5


def astar(grid, start_rc, goal_rc, truck, blocked_cells=frozenset()):
    """
    A* pathfinding, mixed-fleet aware.

    grid          : GridMap
    start_rc      : (row, col) start
    goal_rc       : (row, col) destination (the dump cell)
    truck         : Truck — provides turn_radius for cost calculation
    blocked_cells : set of (row,col) occupied by other trucks

    Returns list of (row,col) from start to goal (not including start).
    Empty list if no path exists.

    State = (r, c, prev_r, prev_c) so we can compute turn cost.
    prev = (-1,-1) at the start node (no incoming direction yet).
    """
    if start_rc == goal_rc:
        return []

    turn_radius_cells = truck.turn_radius / grid.cell_size

    # State encodes position + previous position (for turn cost)
    start_state = (start_rc[0], start_rc[1], -1, -1)

    # heap: (f, g, state)  — g included so equal-f ties break on fewer steps
    open_heap = [(0.0, 0.0, start_state)]
    came_from = {}
    g_cost    = {start_state: 0.0}

    while open_heap:
        f, g, state = heapq.heappop(open_heap)
        r, c, pr, pc = state

        if (r, c) == goal_rc:
            # Reconstruct — extract (row, col) only
            path, cur = [], state
            while cur in came_from:
                path.append((cur[0], cur[1]))
                cur = came_from[cur]
            path.reverse()
            return path

        # Prune stale heap entries
        if g > g_cost.get(state, float('inf')):
            continue

        prev_rc = (pr, pc) if pr != -1 else None

        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < grid.rows and 0 <= nc < grid.cols):
                continue

            # Impassable: any pile (partial or full), boundary, obstacle
            if grid.state[nr, nc] in _IMPASSABLE:
                continue

            # Goal cell is always reachable even if RESERVED for this truck
            if (nr, nc) != goal_rc and (nr, nc) in blocked_cells:
                continue

            turn_extra = _turn_cost(prev_rc, (nr, nc), (r, c),
                                    turn_radius_cells)
            new_g      = g + 1.0 + turn_extra
            new_state  = (nr, nc, r, c)

            if new_g < g_cost.get(new_state, float('inf')):
                g_cost[new_state] = new_g
                h       = abs(nr - goal_rc[0]) + abs(nc - goal_rc[1])
                heapq.heappush(open_heap, (new_g + h, new_g, new_state))
                came_from[new_state] = state

    return []   # no path found


def plan_paths(grid, assignments):
    """
    Plan paths for all assigned trucks using independent A*.
    Other trucks' current grid cells are treated as soft obstacles.

    assignments : list of (Truck, dump_point) pairs
    grid        : GridMap
    Returns     : dict {truck_id: path}
    """
    paths = {}

    all_truck_cells = {
        grid.world_to_cell(*truck.pos)
        for truck, _ in assignments
    }

    for truck, dump_point in assignments:
        truck_cell = grid.world_to_cell(*truck.pos)
        obstacles  = all_truck_cells - {truck_cell}
        path       = astar(grid, truck_cell, dump_point, truck,
                           blocked_cells=obstacles)
        paths[truck.id] = path

    return paths


# ─────────────────────────────────────────────────────────────
# WEEK 2 UPGRADE: plan_paths_cbs() below replaces plan_paths().
# Interface identical — main.py just changes the import alias.
# ─────────────────────────────────────────────────────────────

def plan_paths_cbs(grid, assignments):
    """
    CBS (Conflict-Based Search) — guarantees no two trucks share
    a cell at the same timestep.

    High-level : constraint tree, best-first.
    Low-level  : constrained A* per truck in (row, col, time) space.
    """

    def astar_constrained(start, goal, truck, constraints):
        """
        Time-expanded A*. Forbidden states: set of (r, c, t).
        5 actions: 4 moves + wait-in-place.
        PARTIAL and FILLED are both impassable.
        """
        turn_radius_cells = truck.turn_radius / grid.cell_size

        # State: (r, c, t, prev_r, prev_c)
        start_state = (*start, 0, -1, -1)
        open_heap   = [(0.0, 0.0, start_state)]
        came_from   = {}
        g_cost      = {start_state: 0.0}

        while open_heap:
            f, g, state = heapq.heappop(open_heap)
            r, c, t, pr, pc = state

            if (r, c) == goal:
                path, cur = [], state
                while cur in came_from:
                    path.append((cur[0], cur[1]))
                    cur = came_from[cur]
                path.reverse()
                return path

            if g > g_cost.get(state, float('inf')):
                continue

            prev_rc = (pr, pc) if pr != -1 else None

            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(0,0)]:  # 0,0 = wait
                nr, nc, nt = r+dr, c+dc, t+1

                if not (0 <= nr < grid.rows and 0 <= nc < grid.cols):
                    continue
                if grid.state[nr, nc] in _IMPASSABLE:
                    continue
                if (nr, nc, nt) in constraints:
                    continue

                turn_extra = (_turn_cost(prev_rc, (nr, nc), (r, c),
                                         turn_radius_cells)
                              if (dr, dc) != (0, 0) else 0.0)
                new_g     = g + 1.0 + turn_extra
                new_state = (nr, nc, nt, r, c)

                if new_g < g_cost.get(new_state, float('inf')):
                    g_cost[new_state] = new_g
                    h = abs(nr - goal[0]) + abs(nc - goal[1])
                    heapq.heappush(open_heap,
                                   (new_g + h, new_g, new_state))
                    came_from[new_state] = state

        return []

    def find_conflict(paths):
        """Vertex conflict: two trucks at same (r,c) at same timestep."""
        ids = list(paths.keys())
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                pa = paths[ids[i]]
                pb = paths[ids[j]]
                for t in range(max(len(pa), len(pb))):
                    a = pa[t] if t < len(pa) else (pa[-1] if pa else None)
                    b = pb[t] if t < len(pb) else (pb[-1] if pb else None)
                    if a and b and a == b:
                        return (ids[i], ids[j], a[0], a[1], t)
        return None

    # Build lookup tables
    truck_list = [truck for truck, _ in assignments]
    goal_map   = {truck.id: dp   for truck, dp  in assignments}
    truck_map  = {truck.id: truck for truck, _  in assignments}

    # Initial paths — no constraints
    init_paths = {
        truck.id: astar_constrained(
            grid.world_to_cell(*truck.pos),
            goal_map[truck.id],
            truck,
            set()
        )
        for truck in truck_list
    }
    init_constraints = {truck.id: set() for truck in truck_list}
    init_cost        = sum(len(p) for p in init_paths.values())

    node_id  = 0
    cbs_heap = [(init_cost, node_id, init_constraints, init_paths)]
    node_id += 1

    for _ in range(500):
        if not cbs_heap:
            break

        cost, _, constraints, paths = heapq.heappop(cbs_heap)
        conflict = find_conflict(paths)

        if conflict is None:
            return paths   # conflict-free solution found

        id_a, id_b, r, c, t = conflict

        for blocked_id in (id_a, id_b):
            new_constraints = {tid: set(s) for tid, s in constraints.items()}
            new_constraints[blocked_id].add((r, c, t))

            truck_obj = truck_map[blocked_id]
            new_path  = astar_constrained(
                grid.world_to_cell(*truck_obj.pos),
                goal_map[blocked_id],
                truck_obj,
                new_constraints[blocked_id]
            )

            new_paths       = dict(paths)
            new_paths[blocked_id] = new_path
            new_cost        = sum(len(p) for p in new_paths.values())

            heapq.heappush(cbs_heap,
                           (new_cost, node_id, new_constraints, new_paths))
            node_id += 1

    print("CBS did not converge — falling back to independent A*")
    return plan_paths(grid, assignments)
