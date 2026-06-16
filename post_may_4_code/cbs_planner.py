# cbs_planner.py
# ─────────────────────────────────────────────────────────────
# plan_paths_cbs() — Conflict-Based Search for multi-agent
# dump/exit path planning using space-time A*.
#
# High level  — constraint tree; split on detected conflicts.
# Low level   — astar_st() with hard (r,c,t) forbidden states.
# ─────────────────────────────────────────────────────────────

import heapq
import math
import grid_map
from path_utils import _path_cells, _truck_front_cell, _state_cell
from bicycle_model import interpolate_path_to_truck_states
from astar_core import astar, astar_st
from conflict_detect import _detect_first_conflict
from filters import make_driveable_mask
from config import (DRIVE_CLEARANCE_M, _TAN_REPOSE, ENTRY_POINT,
                    LOCKED_PATH_HORIZON, CBS_MAX_NODES,
                    ENTRY_CORRIDOR_CELLS, ASTAR_WAIT_COST)


def plan_paths_cbs(grid, assignments, locked_paths=None):
    """
    Conflict-Based Search (CBS) for multi-agent path planning.

    High level  — constraint tree; split on detected conflicts.
    Low level   — space-time A* (astar_st) with hard (r,c,t) forbidden states.
                  Trucks wait in place or detour; conflicts are never permitted.

    locked_paths: dict {truck_id: [(r,c), ...]} — remaining waypoints of trucks
      already moving.  Their cells are added as hard (r,c,t) constraints so new
      paths cannot occupy those cells at those exact timesteps.
    """
    if not assignments:  # nothing to plan — return immediately
        return {}

    entry_rc   = grid.world_to_cell(*ENTRY_POINT)  # entry gate cell, precomputed once
    mask_cache = {}  # cache driveable masks per truck class — expensive to rebuild

    # ── BUILD LOCKED SPACE-TIME CONSTRAINTS ─────────────────────────────────────
    locked_st_constraints: set = set()
    if locked_paths:
        for path_entry in locked_paths.values():
            if isinstance(path_entry, tuple):
                raw_path, tail_ticks = path_entry
            else:
                raw_path, tail_ticks = path_entry, 0
            cells = _path_cells(grid, raw_path)
            # Use the full path length as constraints (not just LOCKED_PATH_HORIZON).
            # astar_st is bounded by max_time so extra entries beyond that are free.
            for t, (r, c) in enumerate(cells):
                locked_st_constraints.add((r, c, t))
            if cells and tail_ticks > 0:
                fr, fc = cells[-1]
                for t in range(len(cells), len(cells) + tail_ticks):
                    locked_st_constraints.add((fr, fc, t))

    # ── PER-AGENT SETUP ───────────────────────────────────────────────────────────
    agent_info = {}
    for truck, target_rc in assignments:
        is_exit  = (target_rc == entry_rc)
        mask_key = (truck.truck_class, is_exit)
        if mask_key not in mask_cache:
            # Exit trucks need ignore_path_reserved=True so they can cross active
            # dump corridors on the way back to the gate; CBS time constraints
            # still prevent collisions with the navigating truck.
            mask_cache[mask_key] = make_driveable_mask(grid, truck,
                                                       ignore_path_reserved=is_exit)
        driveable  = mask_cache[mask_key]
        truck_cell = _truck_front_cell(grid, truck)

        # Extended bulldozer: force start cell + radius driveable.
        if is_exit:
            r_pile_m    = (1.2 * truck.payload_volume_m3 / (math.pi * _TAN_REPOSE)) ** (1 / 3)
            bulldozer_r = max(2, int(math.ceil(r_pile_m / grid.cell_size)))
        else:
            bulldozer_r = 2
        for dr in range(-bulldozer_r, bulldozer_r + 1):
            for dc in range(-bulldozer_r, bulldozer_r + 1):
                if abs(dr) + abs(dc) > bulldozer_r:
                    continue
                nr, nc = truck_cell[0] + dr, truck_cell[1] + dc
                if 0 <= nr < grid.rows and 0 <= nc < grid.cols:
                    if grid.state[nr, nc] != grid_map.CellState.BOUNDARY:
                        driveable[nr, nc, :] = True

        if target_rc == entry_rc:
            stop_dist_cells = float(ENTRY_CORRIDOR_CELLS)
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

    # ── HELPERS ──────────────────────────────────────────────────────────────────

    def _at_target_fix(path, aid):
        if not path:
            info = agent_info[aid]
            d = math.hypot(info['start'][0] - info['target'][0],
                           info['start'][1] - info['target'][1])
            if d <= info['stop_dist']:
                return [(info['start'][0], info['start'][1], info['truck'].heading)]
        return path

    def smooth_paths(coarse_paths):
        """Smooth each coarse cell path with the bicycle model."""
        out = {}
        for aid, coarse in coarse_paths.items():
            truck   = agent_info[aid]['truck']
            info    = agent_info[aid]
            is_exit = (info['target'] == entry_rc)

            saved_heading = truck.heading
            if is_exit and coarse:
                fx, fy = grid.cell_to_world(coarse[0][0], coarse[0][1])
                dx_h = fx - truck.pos[0]
                dy_h = fy - truck.pos[1]
                if math.hypot(dx_h, dy_h) > 1e-9:
                    truck.heading = math.atan2(dy_h, dx_h)

            smooth = interpolate_path_to_truck_states(grid, truck, coarse)
            truck.heading = saved_heading

            if not smooth:
                out[aid] = coarse
                continue

            tx, ty = grid.cell_to_world(*info['target'])
            last   = smooth[-1]
            half_l = truck.length / 2.0
            bx     = last[0] + math.cos(last[2]) * half_l
            by     = last[1] + math.sin(last[2]) * half_l

            if math.hypot(bx - tx, by - ty) > grid.cell_size * 4:
                if is_exit:
                    out[aid] = smooth
                else:
                    last_rc = grid.world_to_cell(bx, by)
                    best_d  = float('inf')
                    splice  = len(coarse)
                    for i, wp in enumerate(coarse):
                        d = math.hypot(wp[0] - last_rc[0], wp[1] - last_rc[1])
                        if d < best_d:
                            best_d = d
                            splice = i
                    out[aid] = smooth + list(coarse[splice + 1:])
            else:
                out[aid] = smooth
        return out

    # ── SINGLE-AGENT SHORT-CIRCUIT ────────────────────────────────────────────────
    if len(agent_info) == 1:
        aid  = next(iter(agent_info))
        info = agent_info[aid]
        path = []
        # Only try spatial A* when there are no locked ST constraints — spatial A*
        # has no time dimension and silently ignores every locked (r,c,t) constraint,
        # producing paths that collide with already-moving trucks.
        if not locked_st_constraints:
            path = astar(info['driveable'], grid, info['start'], info['target'],
                         info['truck'], blocked_cells=frozenset(),
                         stop_dist_cells=info['stop_dist'])
            path = _at_target_fix(path, aid)
        if not path:
            path = astar_st(info['driveable'], grid, info['start'], info['target'],
                            info['truck'], locked_st_constraints, info['stop_dist'])
            path = _at_target_fix(path, aid)
        return smooth_paths({aid: path})

    # ── SPATIAL-FIRST (APPROACH B) ────────────────────────────────────────────────
    _sp_paths = {}
    for _aid in agent_info:
        _info = agent_info[_aid]
        _p = astar(_info['driveable'], grid, _info['start'], _info['target'],
                   _info['truck'], blocked_cells=frozenset(),
                   stop_dist_cells=_info['stop_dist'])
        _sp_paths[_aid] = _at_target_fix(_p, _aid)
    _cell_sp = {_aid: [(wp[0], wp[1]) for wp in _p] for _aid, _p in _sp_paths.items()}
    # Only take the spatial shortcut when there are no locked ST constraints either —
    # spatial paths have no time index and cannot respect (r,c,t) locked constraints.
    _sp_clear = not locked_st_constraints
    if _sp_clear and _detect_first_conflict(_cell_sp, truck_map=None, grid=None) is None:
        return smooth_paths(_sp_paths)
    if not _sp_clear:
        # Check the spatial paths against locked constraints (time-indexed).
        _locked_hit = False
        for _aid, _cells in _cell_sp.items():
            for _r, _c, _t in locked_st_constraints:
                if _t < len(_cells) and _cells[_t] == (_r, _c):
                    _locked_hit = True
                    break
            if _locked_hit:
                break
        if not _locked_hit and _detect_first_conflict(_cell_sp, truck_map=None, grid=None) is None:
            return smooth_paths(_sp_paths)
    # ─────────────────────────────────────────────────────────────────────────────

    # ── CBS HIGH-LEVEL SEARCH ─────────────────────────────────────────────────────
    init_constraints = {aid: set() for aid in agent_info}
    init_paths = {}
    for aid in agent_info:
        info = agent_info[aid]
        hard = init_constraints[aid] | locked_st_constraints
        path = astar_st(info['driveable'], grid, info['start'], info['target'],
                        info['truck'], hard, info['stop_dist'])
        init_paths[aid] = _at_target_fix(path, aid)
    init_cost = sum(len(p) for p in init_paths.values())

    _nid = 0
    heap = [(init_cost, _nid, init_constraints, init_paths)]
    MAX_NODES = CBS_MAX_NODES

    for _ in range(MAX_NODES):
        if not heap:
            break
        cost, _, constraints, paths = heapq.heappop(heap)

        truck_map = {aid: agent_info[aid]['truck'] for aid in agent_info}
        conflict = _detect_first_conflict(paths, truck_map=truck_map, grid=grid)

        if conflict is None:
            return smooth_paths(paths)

        if conflict[0] == 'vertex':
            _, ai, aj, ri, ci, rj, cj, t = conflict
            branches = [(ai, ri, ci, t), (aj, rj, cj, t)]
            print(f"\nVertex conflict detected")
        else:
            _, ai, aj, r1, c1, r2, c2, t = conflict
            branches = [(ai, r1, c1, t), (aj, r2, c2, t)]
            print(f"\nEdge conflict detected")

        for branch_agent, r, c, t in branches:
            new_cons = {k: set(v) for k, v in constraints.items()}
            new_cons[branch_agent].add((r, c, t))
            new_paths = dict(paths)
            info = agent_info[branch_agent]
            hard = new_cons[branch_agent] | locked_st_constraints
            bpath = astar_st(info['driveable'], grid, info['start'], info['target'],
                             info['truck'], hard, info['stop_dist'])
            new_paths[branch_agent] = _at_target_fix(bpath, branch_agent)
            new_cost = sum(len(p) for p in new_paths.values())
            _nid += 1
            heapq.heappush(heap, (new_cost, _nid, new_cons, new_paths))

    if heap:
        _, _, _, best = heap[0]
        return smooth_paths(best)
    return smooth_paths(init_paths)
