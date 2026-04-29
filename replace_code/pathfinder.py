# pathfinder.py
# ─────────────────────────────────────────────────────────────
# FIX: State explosion causing zero-length paths.
#
# Old state: (r, c, prev_r, prev_c) — 4 values.
# At 91x91 grid: 91^4 = 68M possible states.
# A* with this state space runs out of memory/time for paths
# longer than ~45 cells (e.g. entry at row 0 to goal at row 84).
#
# New state: (r, c) — 2 values.
# State space: 91x91 = 8,281 cells. Always finds paths.
# Turn cost preserved as a g-cost penalty: when the truck changes
# direction, extra cost is added — this still steers A* toward
# straighter paths without exploding the state space.
#
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
    """
    Extra cost when the truck changes direction.
    prev_dr/dc: direction of last move. dr/dc: direction of this move.
    A 90-degree change costs turn_radius_cells * 0.3.
    Reduced from 0.5 to 0.3 so turn penalty doesn't overwhelm long paths.
    """
    if prev_dr is None:
        return 0.0
    dot = prev_dr * dr + prev_dc * dc
    return 0.0 if dot == 1 else turn_radius_cells * 0.3


def astar(driveable, grid, start_rc, goal_rc, truck, blocked_cells=frozenset()):
    """
    A* with (r,c) state — fixed from (r,c,pr,pc) to prevent state explosion.

    Turn cost is still applied as a g-cost penalty so large trucks
    prefer straighter routes, but the state space stays manageable.
    """
    if start_rc == goal_rc:
        return []

    turn_radius_cells = truck.turn_radius / grid.cell_size
    rows, cols        = driveable.shape[:2]

    # State: (r, c). Track last direction separately for turn cost.
    open_heap = [(0.0, 0.0, start_rc[0], start_rc[1], None, None)]
    # heap entry: (f, g, r, c, prev_dr, prev_dc)

    came_from = {}   # (r,c) → (pr, pc)
    g_cost    = {start_rc: 0.0}

    while open_heap:
        f, g, r, c, prev_dr, prev_dc = heapq.heappop(open_heap)

        if (r, c) == goal_rc:
            # Reconstruct path
            path = []
            cur  = goal_rc
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return path

        if g > g_cost.get((r, c), float('inf')):
            continue

        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            if (nr, nc) != goal_rc:
                hb = _heading_bucket(dr, dc)
                if not driveable[nr, nc, hb]:
                    continue
                if (nr, nc) in blocked_cells:
                    continue

            tc    = _turn_cost(prev_dr, prev_dc, dr, dc, turn_radius_cells)
            new_g = g + 1.0 + tc

            if new_g < g_cost.get((nr, nc), float('inf')):
                g_cost[(nr, nc)] = new_g
                h = abs(nr - goal_rc[0]) + abs(nc - goal_rc[1])
                heapq.heappush(open_heap,
                               (new_g + h, new_g, nr, nc, dr, dc))
                came_from[(nr, nc)] = (r, c)

    return []


def plan_paths(grid, assignments):
    """
    Each truck class gets its own heading-aware driveable mask.
    Mask cached by truck_class — same class = same dimensions = same mask.
    Reduces mask computation from 9 calls to 3 (one per class).
    """
    if not assignments:
        return {}

    all_truck_cells = {
        grid.world_to_cell(*truck.pos)
        for truck, _ in assignments
    }

    # Cache mask by truck class — all trucks of same class share mask
    mask_cache = {}

    paths = {}
    for truck, dump_point in assignments:
        if truck.truck_class not in mask_cache:
            mask_cache[truck.truck_class] = make_driveable_mask(grid, truck)
        driveable  = mask_cache[truck.truck_class]
        truck_cell = grid.world_to_cell(*truck.pos)
        obstacles  = all_truck_cells - {truck_cell}
        paths[truck.id] = astar(driveable, grid, truck_cell,
                                dump_point, truck,
                                blocked_cells=obstacles)
    return paths


# ── WEEK 2: CBS ────────────────────────────────────────────

def plan_paths_cbs(grid, assignments):
    """CBS with simplified (r,c,t) state space."""

    def astar_constrained(start, goal, truck, constraints):
        driveable = make_driveable_mask(grid, truck)
        rows, cols = driveable.shape[:2]
        turn_radius_cells = truck.turn_radius / grid.cell_size

        open_heap = [(0.0, 0.0, start[0], start[1], 0, None, None)]
        came_from = {}
        g_cost    = {(start[0], start[1], 0): 0.0}

        while open_heap:
            f, g, r, c, t, prev_dr, prev_dc = heapq.heappop(open_heap)

            if (r, c) == goal:
                path, cur = [], (r, c, t)
                while cur in came_from:
                    path.append((cur[0], cur[1]))
                    cur = came_from[cur]
                path.reverse()
                return path

            if g > g_cost.get((r, c, t), float('inf')):
                continue

            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1),(0,0)):
                nr, nc, nt = r+dr, c+dc, t+1
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if (nr, nc) != goal and (dr, dc) != (0, 0):
                    hb = _heading_bucket(dr, dc)
                    if not driveable[nr, nc, hb]:
                        continue
                if (nr, nc, nt) in constraints:
                    continue

                tc    = (_turn_cost(prev_dr, prev_dc, dr, dc, turn_radius_cells)
                         if (dr, dc) != (0, 0) else 0.0)
                new_g = g + 1.0 + tc
                key   = (nr, nc, nt)

                if new_g < g_cost.get(key, float('inf')):
                    g_cost[key] = new_g
                    h = abs(nr-goal[0]) + abs(nc-goal[1])
                    heapq.heappush(open_heap,
                                   (new_g+h, new_g, nr, nc, nt, dr, dc))
                    came_from[key] = (r, c, t)

        return []

    def find_conflict(paths):
        ids = list(paths.keys())
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                pa, pb = paths[ids[i]], paths[ids[j]]
                for t in range(max(len(pa), len(pb))):
                    a = pa[t] if t < len(pa) else (pa[-1] if pa else None)
                    b = pb[t] if t < len(pb) else (pb[-1] if pb else None)
                    if a and b and a == b:
                        return (ids[i], ids[j], a[0], a[1], t)
        return None

    truck_list = [truck for truck, _ in assignments]
    goal_map   = {truck.id: dp    for truck, dp in assignments}
    truck_map  = {truck.id: truck for truck, _  in assignments}

    init_paths = {
        truck.id: astar_constrained(
            grid.world_to_cell(*truck.pos),
            goal_map[truck.id], truck, set()
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
            return paths

        id_a, id_b, r, c, t = conflict
        for blocked_id in (id_a, id_b):
            new_constraints = {tid: set(s) for tid, s in constraints.items()}
            new_constraints[blocked_id].add((r, c, t))
            truck_obj = truck_map[blocked_id]
            new_path  = astar_constrained(
                grid.world_to_cell(*truck_obj.pos),
                goal_map[blocked_id], truck_obj,
                new_constraints[blocked_id]
            )
            new_paths             = dict(paths)
            new_paths[blocked_id] = new_path
            new_cost              = sum(len(p) for p in new_paths.values())
            heapq.heappush(cbs_heap,
                           (new_cost, node_id, new_constraints, new_paths))
            node_id += 1

    print("CBS did not converge — falling back to independent A*")
    return plan_paths(grid, assignments)