# staging_paths.py
# ─────────────────────────────────────────────────────────────
# plan_staging_paths() — CBS-based staging path planner using
# the bicycle-model hybrid A* (hybrid_astar_to_staging_st).
#
# Collision model: all constraints are stored as oriented truck
# footprints {t: [(cx,cy,h,hl,hw),...]} and checked via SAT
# (_rect_overlap_2d).  No (r, c, t) cell-occupancy logic remains.
# ─────────────────────────────────────────────────────────────

import heapq
import math
import grid_map
from path_utils import _path_cells, _truck_front_cell, _corridor_cell_set, _make_locked_entry
from filters import make_driveable_mask
from staging import score_staging_candidates
from hybrid_astar import hybrid_astar_to_staging, hybrid_astar_to_staging_st
from conflict_detect import (_detect_first_conflict, _infer_heading_at_t,
                              _build_locked_footprints, _merge_fp_constraints,
                              _footprints_conflict)
from config import (CBS_MAX_NODES, ASTAR_MAX_TIME, ENTRY_CORRIDOR_CELLS)


def plan_staging_paths(grid, assignments, locked_paths=None, max_time=ASTAR_MAX_TIME,
                       ignore_path_reserved=False, precomputed_masks=None,
                       allow_corridor_bypass=True, spatial_only=False):
    """
    CBS-based staging path planner using the bicycle-model A* (hybrid_astar_to_staging_st).

    locked_paths: dict {truck_id: (body_center_poses, tail_ticks, hl, hw)}
      produced by _make_locked_entry().  Footprints are checked via SAT at each
      planner timestep; no (r, c, t) cell-occupancy logic remains.
    """
    if not assignments:
        return {}, {}

    # ── LOCKED FOOTPRINT CONSTRAINTS ─────────────────────────────────────────────
    locked_fp = _build_locked_footprints(locked_paths, grid)

    # ── PER-AGENT SETUP ───────────────────────────────────────────────────────────
    mask_cache = dict(precomputed_masks) if precomputed_masks else {}
    agent_info = {}
    for truck, dump_target in assignments:
        mask_key = (truck.truck_class, ignore_path_reserved)
        if mask_key not in mask_cache:
            mask_cache[mask_key] = make_driveable_mask(
                grid, truck, ignore_path_reserved=ignore_path_reserved)
        driveable  = mask_cache[mask_key]

        # Corridor-bypass mask — always ignore PATH_RESERVED so the truck can wait
        # and then route through a blocked corridor after the other truck has passed.
        mask_key_ipr = (truck.truck_class, True)
        if mask_key_ipr not in mask_cache:
            mask_cache[mask_key_ipr] = make_driveable_mask(
                grid, truck, ignore_path_reserved=True)
        driveable_ipr = mask_cache[mask_key_ipr]

        truck_cell = _truck_front_cell(grid, truck)

        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if abs(dr) + abs(dc) > 2:
                    continue
                nr, nc = truck_cell[0] + dr, truck_cell[1] + dc
                if 0 <= nr < grid.rows and 0 <= nc < grid.cols:
                    if grid.state[nr, nc] != grid_map.CellState.BOUNDARY:
                        driveable[nr, nc, :] = True
                        driveable_ipr[nr, nc, :] = True

        candidates = score_staging_candidates(grid, truck, dump_target,
                                              ignore_path_reserved=ignore_path_reserved)
        agent_info[truck.id] = {
            'truck':         truck,
            'driveable':     driveable,
            'driveable_ipr': driveable_ipr,
            'candidates':    candidates,
        }

    # ── LOW-LEVEL PLANNER ─────────────────────────────────────────────────────────
    _st_calls   = [0]
    _spa_calls  = [0]
    _cbs_nodes  = [0]

    def _plan_one(aid, cbs_constraints, preferred_pose=None):
        info = agent_info[aid]
        hard = _merge_fp_constraints(cbs_constraints, locked_fp)
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

        # Primary plan failed. If temporal footprint constraints exist and corridor
        # bypass is allowed, try routing through corridor cells (waiting for the
        # blocking truck to vacate them).  Gate pre-planning disables this so the
        # entering truck never physically shares cells with an exit corridor.
        if locked_fp and allow_corridor_bypass:
            for candidate in ordered:
                _st_calls[0] += 1
                path = hybrid_astar_to_staging_st(
                    grid, info['truck'], candidate,
                    driveable=info['driveable_ipr'],
                    constraints=hard,
                    max_time=max_time,
                )
                if path:
                    print(f"[STAGING] Truck {info['truck'].id}: corridor-bypass plan — "
                          f"waiting for blocked corridor to clear")
                    return path, candidate

        print(f"[STAGING] Truck {info['truck'].id}: no reachable staging pose")
        return [], None

    _agent_ids = list(agent_info.keys())
    _tag = f"T{_agent_ids}" if len(_agent_ids) > 1 else f"T{_agent_ids[0]}"

    # ── SINGLE-AGENT SHORT-CIRCUIT ────────────────────────────────────────────────
    if len(agent_info) == 1:
        aid  = next(iter(agent_info))
        info = agent_info[aid]
        # Skip spatial shortcut when locked footprint constraints exist.
        if not locked_fp:
            for _cand in info['candidates']:
                _spa_calls[0] += 1
                _p = hybrid_astar_to_staging(grid, info['truck'], _cand,
                                              driveable=info['driveable'])
                if _p:
                    print(f"[STAGING] {_tag}: spatial ok "
                          f"(spatial_calls={_spa_calls[0]}, st_calls=0, cbs_nodes=0)")
                    return {aid: _p}, {aid: _cand}
        # Gate pre-planning: spatial-only mode — if spatial fails, return immediately
        # so the planning thread is never blocked by a slow ST search.
        if spatial_only:
            print(f"[STAGING] {_tag}: spatial failed, spatial_only=True — skipping ST")
            return {aid: []}, {aid: None}
        path, pose = _plan_one(aid, {})
        print(f"[STAGING] {_tag}: spatial failed → ST fallback "
              f"(spatial_calls={_spa_calls[0]}, st_calls={_st_calls[0]}, cbs_nodes=0)")
        return {aid: path}, {aid: pose}

    # ── SPATIAL-FIRST (APPROACH B) ────────────────────────────────────────────────
    if not locked_fp:
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
            _sp_cells  = {_aid: _path_cells(grid, _p) for _aid, _p in _sp_paths.items()}
            _truck_map = {_aid: agent_info[_aid]['truck'] for _aid in agent_info}
            if _detect_first_conflict(_sp_cells, truck_map=_truck_map, grid=grid) is None:
                print(f"[STAGING] {_tag}: spatial ok, no conflict "
                      f"(spatial_calls={_spa_calls[0]}, st_calls=0, cbs_nodes=0)")
                return _sp_paths, _sp_staging

    # ── CBS HIGH-LEVEL SEARCH ─────────────────────────────────────────────────────
    # Constraints per agent: dict {t: [(cx, cy, h, hl, hw), ...]}
    init_cons    = {aid: {} for aid in agent_info}
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
    MAX_NODES  = CBS_MAX_NODES

    for _ in range(MAX_NODES):
        if not heap:
            break
        cost, _, constraints, paths, staging = heapq.heappop(heap)
        _cbs_nodes[0] += 1

        # Convert smooth paths → cell sequences for conflict detection.
        cell_paths = {aid: _path_cells(grid, p) for aid, p in paths.items() if p}
        conflict   = _detect_first_conflict(cell_paths, truck_map=truck_map, grid=grid)

        if conflict is None:
            print(f"[STAGING] {_tag}: CBS solved "
                  f"(spatial_calls={_spa_calls[0]}, st_calls={_st_calls[0]}, cbs_nodes={_cbs_nodes[0]})")
            return paths, staging

        if conflict[0] == 'vertex':
            _, ai, aj, ri, ci, rj, cj, t = conflict
            # Branch for ai: avoid aj's footprint at (rj, cj) at time t.
            # Branch for aj: avoid ai's footprint at (ri, ci) at time t.
            branch_info = [(ai, aj, rj, cj, t), (aj, ai, ri, ci, t)]
        else:
            _, ai, aj, r1, c1, r2, c2, t = conflict
            branch_info = [(ai, aj, r2, c2, t), (aj, ai, r1, c1, t)]

        for branch_agent, other_agent, or_, oc, t in branch_info:
            # Build the opposing truck's footprint at the conflict cell + time.
            ot       = truck_map[other_agent]
            owx, owy = grid.cell_to_world(or_, oc)
            # Use the cell path for heading inference (consistent with conflict detection).
            other_cells = _path_cells(grid, paths.get(other_agent, []))
            oh = _infer_heading_at_t(other_cells, t)
            fp = (owx, owy, oh, ot.length / 2.0, ot.width / 2.0)

            new_cons  = {k: {tt: list(fps) for tt, fps in v.items()}
                         for k, v in constraints.items()}
            branch_t_list = new_cons[branch_agent].setdefault(t, [])
            new_cons[branch_agent][t] = branch_t_list + [fp]

            new_paths   = dict(paths)
            new_staging = dict(staging)
            bp, bsp = _plan_one(branch_agent, new_cons[branch_agent],
                                preferred_pose=staging.get(branch_agent))
            new_paths[branch_agent]   = bp
            new_staging[branch_agent] = bsp
            new_cost = sum(len(p) for p in new_paths.values())
            _nid += 1
            heapq.heappush(heap, (new_cost, _nid, new_cons, new_paths, new_staging))

    print(f"[STAGING] {_tag}: CBS exhausted "
          f"(spatial_calls={_spa_calls[0]}, st_calls={_st_calls[0]}, cbs_nodes={_cbs_nodes[0]})")
    if heap:
        _, _, _, best_paths, best_staging = heap[0]
        return best_paths, best_staging
    return init_paths, init_staging
