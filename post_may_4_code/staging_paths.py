# staging_paths.py
# ─────────────────────────────────────────────────────────────
# plan_staging_paths() — CBS-based staging path planner using
# the bicycle-model hybrid A* (hybrid_astar_to_staging_st).
# Used for both dump paths (navigating trucks) and exit paths
# (exiting trucks).
# ─────────────────────────────────────────────────────────────

import heapq
import math
import grid_map
from path_utils import _path_cells, _truck_front_cell, _corridor_cell_set
from filters import make_driveable_mask
from staging import score_staging_candidates
from hybrid_astar import hybrid_astar_to_staging, hybrid_astar_to_staging_st
from conflict_detect import _detect_first_conflict
from config import (LOCKED_PATH_HORIZON, CBS_MAX_NODES, ASTAR_MAX_TIME,
                    ENTRY_CORRIDOR_CELLS)


def plan_staging_paths(grid, assignments, locked_paths=None, max_time=ASTAR_MAX_TIME,
                       ignore_path_reserved=False, precomputed_masks=None):
    """
    CBS-based staging path planner using the bicycle-model A* (hybrid_astar_to_staging_st).
    Used for both dump paths (navigating trucks) and exit paths (exiting trucks).

    ignore_path_reserved: pass True for exit trucks so they can cross dump corridors
      spatially — CBS time constraints handle temporal separation instead.
    max_time: cap on the space-time A* search depth per candidate.
    """
    if not assignments:
        return {}, {}

    # ── LOCKED SPACE-TIME CONSTRAINTS ────────────────────────────────────────────
    # locked_paths values are (cells, tail_ticks) tuples — see plan_paths_cbs.
    locked_st: set = set()
    if locked_paths:
        for path_entry in locked_paths.values():
            if isinstance(path_entry, tuple):
                raw_path, tail_ticks = path_entry
            else:
                raw_path, tail_ticks = path_entry, 0
            cells = _path_cells(grid, raw_path)
            # Use the full path length as constraints (not just LOCKED_PATH_HORIZON).
            # hybrid_astar_to_staging_st is bounded by max_time so extra entries are free.
            for t, (r, c) in enumerate(cells):
                locked_st.add((r, c, t))
            if cells and tail_ticks > 0:
                fr, fc = cells[-1]
                for t in range(len(cells), len(cells) + tail_ticks):
                    locked_st.add((fr, fc, t))

    # ── PER-AGENT SETUP ───────────────────────────────────────────────────────────
    mask_cache = dict(precomputed_masks) if precomputed_masks else {}
    agent_info = {}
    for truck, dump_target in assignments:
        mask_key = (truck.truck_class, ignore_path_reserved)
        if mask_key not in mask_cache:
            mask_cache[mask_key] = make_driveable_mask(
                grid, truck, ignore_path_reserved=ignore_path_reserved)
        driveable  = mask_cache[mask_key]
        truck_cell = _truck_front_cell(grid, truck)

        # Extended bulldozer: force start + 2-cell radius driveable
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if abs(dr) + abs(dc) > 2:
                    continue
                nr, nc = truck_cell[0] + dr, truck_cell[1] + dc
                if 0 <= nr < grid.rows and 0 <= nc < grid.cols:
                    if grid.state[nr, nc] != grid_map.CellState.BOUNDARY:
                        driveable[nr, nc, :] = True

        candidates = score_staging_candidates(grid, truck, dump_target,
                                              ignore_path_reserved=ignore_path_reserved)
        agent_info[truck.id] = {
            'truck':      truck,
            'driveable':  driveable,
            'candidates': candidates,
        }

    # ── LOW-LEVEL PLANNER ─────────────────────────────────────────────────────────
    _st_calls   = [0]   # hybrid_astar_to_staging_st calls (expensive, space-time)
    _spa_calls  = [0]   # hybrid_astar_to_staging calls    (cheap, spatial only)
    _cbs_nodes  = [0]   # CBS high-level nodes expanded

    def _plan_one(aid, cbs_constraints, preferred_pose=None):
        """Plan one truck with combined CBS + locked constraints.
        Tries preferred_pose first, then all scored candidates."""
        info = agent_info[aid]
        hard = cbs_constraints | locked_st
        ordered = (([preferred_pose] if preferred_pose else []) +
                   [c for c in info['candidates'] if c is not preferred_pose])
        for candidate in ordered:
            _st_calls[0] += 1
            path = hybrid_astar_to_staging_st(
                grid, info['truck'], candidate,
                driveable=info['driveable'],
                constraints=hard,
                max_time=max_time,
            )
            if path:
                return path, candidate
        print(f"[STAGING] Truck {info['truck'].id}: no reachable staging pose")
        return [], None

    _agent_ids = list(agent_info.keys())
    _tag = f"T{_agent_ids}" if len(_agent_ids) > 1 else f"T{_agent_ids[0]}"

    # ── SINGLE-AGENT SHORT-CIRCUIT ────────────────────────────────────────────────
    if len(agent_info) == 1:
        aid  = next(iter(agent_info))
        info = agent_info[aid]
        # Skip spatial shortcut when locked ST constraints exist — spatial hybrid A*
        # has no time dimension and silently ignores every locked (r,c,t) constraint.
        if not locked_st:
            for _cand in info['candidates']:
                _spa_calls[0] += 1
                _p = hybrid_astar_to_staging(grid, info['truck'], _cand,
                                              driveable=info['driveable'])
                if _p:
                    print(f"[STAGING] {_tag}: spatial ok "
                          f"(spatial_calls={_spa_calls[0]}, st_calls=0, cbs_nodes=0)")
                    return {aid: _p}, {aid: _cand}
        # spatial failed or skipped — fall back to ST version
        path, pose = _plan_one(aid, set())
        print(f"[STAGING] {_tag}: spatial failed → ST fallback "
              f"(spatial_calls={_spa_calls[0]}, st_calls={_st_calls[0]}, cbs_nodes=0)")
        return {aid: path}, {aid: pose}

    # ── SPATIAL-FIRST (APPROACH B) ────────────────────────────────────────────────
    # Try fast spatial hybrid A* for all agents first.
    # Only fall back to expensive ST/CBS if spatial paths actually conflict.
    # Skip entirely when locked ST constraints exist — spatial paths have no time
    # index so they cannot be checked against (r,c,t) locked constraints.
    if not locked_st:
        _sp_paths, _sp_staging = {}, {}
        for _aid, _info in agent_info.items():
            for _cand in _info['candidates']:
                _spa_calls[0] += 1
                _p = hybrid_astar_to_staging(grid, _info['truck'], _cand,
                                              driveable=_info['driveable'])
                if _p:
                    _sp_paths[_aid]   = _p
                    _sp_staging[_aid] = _cand
                    break
        if len(_sp_paths) == len(agent_info):
            _sp_cells = {_aid: _path_cells(grid, _p) for _aid, _p in _sp_paths.items()}
            if _detect_first_conflict(_sp_cells, truck_map=None, grid=None) is None:
                print(f"[STAGING] {_tag}: spatial ok, no conflict "
                      f"(spatial_calls={_spa_calls[0]}, st_calls=0, cbs_nodes=0)")
                return _sp_paths, _sp_staging   # no conflict — skip ST/CBS entirely
    # ─────────────────────────────────────────────────────────────────────────────

    # ── CBS HIGH-LEVEL SEARCH ─────────────────────────────────────────────────────
    init_cons    = {aid: set() for aid in agent_info}
    init_paths   = {}
    init_staging = {}
    for aid in agent_info:
        p, sp = _plan_one(aid, init_cons[aid])
        init_paths[aid]   = p
        init_staging[aid] = sp

    init_cost  = sum(len(p) for p in init_paths.values())
    truck_map  = {aid: agent_info[aid]['truck'] for aid in agent_info}
    _nid       = 0
    heap       = [(init_cost, _nid, init_cons, init_paths, init_staging)]
    MAX_NODES  = CBS_MAX_NODES   # fewer than exit CBS — hybrid A* is more expensive per call

    for _ in range(MAX_NODES):
        if not heap:
            break
        cost, _, constraints, paths, staging = heapq.heappop(heap)
        _cbs_nodes[0] += 1

        # Convert smooth paths → cell sequences for conflict detection
        cell_paths = {aid: _path_cells(grid, p) for aid, p in paths.items() if p}
        conflict   = _detect_first_conflict(cell_paths, truck_map=truck_map, grid=grid)

        if conflict is None:
            print(f"[STAGING] {_tag}: CBS solved "
                  f"(spatial_calls={_spa_calls[0]}, st_calls={_st_calls[0]}, cbs_nodes={_cbs_nodes[0]})")
            return paths, staging   # conflict-free solution found

        if conflict[0] == 'vertex':
            _, ai, aj, ri, ci, rj, cj, t = conflict
            branches = [(ai, ri, ci, t), (aj, rj, cj, t)]
        else:
            _, ai, aj, r1, c1, r2, c2, t = conflict
            branches = [(ai, r1, c1, t), (aj, r2, c2, t)]

        for branch_agent, r, c, t in branches:
            new_cons  = {k: set(v) for k, v in constraints.items()}
            new_cons[branch_agent].add((r, c, t))
            new_paths   = dict(paths)
            new_staging = dict(staging)
            bp, bsp = _plan_one(branch_agent, new_cons[branch_agent],
                                preferred_pose=staging.get(branch_agent))
            new_paths[branch_agent]   = bp
            new_staging[branch_agent] = bsp
            new_cost = sum(len(p) for p in new_paths.values())
            _nid += 1
            heapq.heappush(heap, (new_cost, _nid, new_cons, new_paths, new_staging))

    # CBS exhausted — return best found
    print(f"[STAGING] {_tag}: CBS exhausted "
          f"(spatial_calls={_spa_calls[0]}, st_calls={_st_calls[0]}, cbs_nodes={_cbs_nodes[0]})")
    if heap:
        _, _, _, best_paths, best_staging = heap[0]
        return best_paths, best_staging
    return init_paths, init_staging
