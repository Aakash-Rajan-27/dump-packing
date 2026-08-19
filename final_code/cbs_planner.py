# cbs_planner.py
# ─────────────────────────────────────────────────────────────
# plan_paths_cbs() — Conflict-Based Search for multi-agent
# dump/exit path planning using space-time A*.
#
# High level  — constraint tree; split on detected conflicts.
# Low level   — astar_st() with hard footprint-overlap forbidden states.
#
# Collision model: all constraints are stored as oriented truck
# footprints {t: [(cx,cy,h,hl,hw),...]} and checked via SAT
# (_rect_overlap_2d).  No (r, c, t) cell-occupancy logic remains.
# ─────────────────────────────────────────────────────────────

import heapq
import math
import grid_map
from path_utils import _path_cells, _truck_front_cell, _state_cell, _make_locked_entry
from bicycle_model import interpolate_path_to_truck_states
from astar_core import astar, astar_st
from conflict_detect import (_detect_first_conflict, _infer_heading_at_t,
                              _build_locked_footprints, _merge_fp_constraints,
                              _footprints_conflict)
from filters import make_driveable_mask
from config import (DRIVE_CLEARANCE_M, _TAN_REPOSE, ENTRY_POINT,
                    CBS_MAX_NODES, ENTRY_CORRIDOR_CELLS, ASTAR_WAIT_COST)


def plan_paths_cbs(grid, assignments, locked_paths=None):
    """
    Conflict-Based Search (CBS) for multi-agent path planning.

    High level  — constraint tree; split on detected conflicts.
    Low level   — space-time A* (astar_st) with hard footprint constraints.

    locked_paths: dict {truck_id: (body_center_poses, tail_ticks, hl, hw)}
      produced by _make_locked_entry().  Their full footprints are converted to
      a per-timestep dict and used as hard constraints so new paths cannot
      overlap those trucks' bodies at those timesteps.
    """
    if not assignments:
        return {}

    entry_rc   = grid.world_to_cell(*ENTRY_POINT)
    mask_cache = {}

    # ── BUILD LOCKED FOOTPRINT CONSTRAINTS ───────────────────────────────────────
    # {t: [(cx, cy, heading, hl, hw), ...]} — one entry per cell-size of travel.
    locked_footprints = _build_locked_footprints(locked_paths, grid)

    # ── PER-AGENT SETUP ───────────────────────────────────────────────────────────
    agent_info = {}
    for truck, target_rc in assignments:
        is_exit  = (target_rc == entry_rc)
        mask_key = (truck.truck_class, is_exit)
        if mask_key not in mask_cache:
            mask_cache[mask_key] = make_driveable_mask(grid, truck,
                                                       ignore_path_reserved=is_exit)
        driveable  = mask_cache[mask_key]
        truck_cell = _truck_front_cell(grid, truck)

        # Corridor-bypass mask — always ignore PATH_RESERVED so the truck can route
        # through another truck's corridor at timesteps when that truck won't be there.
        mask_key_ipr = (truck.truck_class, True)
        if mask_key_ipr not in mask_cache:
            mask_cache[mask_key_ipr] = make_driveable_mask(grid, truck,
                                                            ignore_path_reserved=True)
        driveable_ipr = mask_cache[mask_key_ipr]

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
                        driveable_ipr[nr, nc, :] = True

        if target_rc == entry_rc:
            stop_dist_cells = float(ENTRY_CORRIDOR_CELLS)
        else:
            r_pile_m        = (1.2 * truck.payload_volume_m3 / (math.pi * _TAN_REPOSE)) ** (1 / 3)
            d_clearance     = max(0.0, r_pile_m - (DRIVE_CLEARANCE_M / _TAN_REPOSE))
            safe_dist_m     = d_clearance + truck.length
            stop_dist_cells = safe_dist_m / grid.cell_size

        agent_info[truck.id] = {
            'truck':         truck,
            'start':         truck_cell,
            'target':        target_rc,
            'driveable':     driveable,
            'driveable_ipr': driveable_ipr,
            'stop_dist':     stop_dist_cells,
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

    # ── CORRIDOR-BYPASS HELPERS ───────────────────────────────────────────────────

    def _at_goal(path, aid):
        """True if the (r,c) path reaches within stop_dist cells of the goal."""
        if not path:
            return False
        info = agent_info[aid]
        last = path[-1]
        return math.hypot(last[0] - info['target'][0],
                          last[1] - info['target'][1]) <= max(info['stop_dist'], 0.5)

    def _corridor_bypass(aid, fp_constraints):
        """Retry astar_st with ignore_path_reserved=True so the truck can wait at
        its current position until the blocking truck's footprint constraints expire,
        then route through the formerly blocked corridor cells.
        Only meaningful when fp_constraints is non-empty (defines when the corridor clears)."""
        if not fp_constraints:
            return []
        info  = agent_info[aid]
        bpath = astar_st(info['driveable_ipr'], grid, info['start'], info['target'],
                         info['truck'], fp_constraints, info['stop_dist'])
        if _at_goal(bpath, aid):
            print(f"[CBS] T{aid}: corridor-bypass plan — truck will wait for blocking "
                  f"corridor to clear then route through")
        return bpath

    # ── SINGLE-AGENT SHORT-CIRCUIT ────────────────────────────────────────────────
    if len(agent_info) == 1:
        aid  = next(iter(agent_info))
        info = agent_info[aid]
        path = []
        # Only try spatial A* when there are no locked footprint constraints.
        if not locked_footprints:
            path = astar(info['driveable'], grid, info['start'], info['target'],
                         info['truck'], stop_dist_cells=info['stop_dist'])
            path = _at_target_fix(path, aid)
        if not path:
            path = astar_st(info['driveable'], grid, info['start'], info['target'],
                            info['truck'], locked_footprints, info['stop_dist'])
            path = _at_target_fix(path, aid)
        # Fallback: corridor blocks prevented reaching goal — wait for it to clear.
        if not _at_goal(path, aid):
            bypass = _corridor_bypass(aid, locked_footprints)
            if _at_goal(bypass, aid):
                path = _at_target_fix(bypass, aid)
        return smooth_paths({aid: path})

    # ── SPATIAL-FIRST (APPROACH B) ────────────────────────────────────────────────
    _truck_map_sp = {_aid: agent_info[_aid]['truck'] for _aid in agent_info}
    _sp_paths = {}
    for _aid in agent_info:
        _info = agent_info[_aid]
        _p = astar(_info['driveable'], grid, _info['start'], _info['target'],
                   _info['truck'], stop_dist_cells=_info['stop_dist'])
        _sp_paths[_aid] = _at_target_fix(_p, _aid)
    _cell_sp = {_aid: [(wp[0], wp[1]) for wp in _p] for _aid, _p in _sp_paths.items()}

    # Only take the spatial shortcut when there are no locked footprint constraints
    # AND the SAT rectangle check confirms no footprint overlaps at any timestep.
    _sp_clear = not locked_footprints
    if _sp_clear and _detect_first_conflict(_cell_sp, truck_map=_truck_map_sp, grid=grid) is None:
        return smooth_paths(_sp_paths)

    if not _sp_clear:
        # Check spatial paths against locked footprint constraints (time-indexed).
        _locked_hit = False
        for _aid, _cells in _cell_sp.items():
            _t_info = agent_info[_aid]
            _hl = _t_info['truck'].length / 2.0
            _hw = _t_info['truck'].width / 2.0
            for _t, _fps in locked_footprints.items():
                if _t < len(_cells):
                    _cr, _cc = _cells[_t]
                    _cwx, _cwy = grid.cell_to_world(_cr, _cc)
                    _ch = _infer_heading_at_t(_cells, _t)
                    if _footprints_conflict(_cwx, _cwy, _ch, _hl, _hw, _t, locked_footprints):
                        _locked_hit = True
                        break
            if _locked_hit:
                break
        if not _locked_hit and _detect_first_conflict(_cell_sp, truck_map=None, grid=None) is None:
            return smooth_paths(_sp_paths)

    # ── CBS HIGH-LEVEL SEARCH ─────────────────────────────────────────────────────
    # Constraints per agent: dict {t: [(cx, cy, h, hl, hw), ...]}
    init_constraints = {aid: {} for aid in agent_info}
    init_paths = {}
    for aid in agent_info:
        info = agent_info[aid]
        hard = _merge_fp_constraints(init_constraints[aid], locked_footprints)
        path = astar_st(info['driveable'], grid, info['start'], info['target'],
                        info['truck'], hard, info['stop_dist'])
        # Corridor blocked all routes — try waiting for it to clear.
        if not _at_goal(path, aid):
            bypass = _corridor_bypass(aid, hard)
            if _at_goal(bypass, aid):
                path = bypass
        init_paths[aid] = _at_target_fix(path, aid)
    init_cost = sum(len(p) for p in init_paths.values())

    truck_map = {aid: agent_info[aid]['truck'] for aid in agent_info}
    _nid = 0
    heap = [(init_cost, _nid, init_constraints, init_paths)]
    MAX_NODES = CBS_MAX_NODES

    for _ in range(MAX_NODES):
        if not heap:
            break
        cost, _, constraints, paths = heapq.heappop(heap)

        conflict = _detect_first_conflict(paths, truck_map=truck_map, grid=grid)

        if conflict is None:
            return smooth_paths(paths)

        if conflict[0] == 'vertex':
            _, ai, aj, ri, ci, rj, cj, t = conflict
            # Branch for ai: avoid aj's footprint at (rj, cj) at time t.
            # Branch for aj: avoid ai's footprint at (ri, ci) at time t.
            branch_info = [(ai, aj, rj, cj, t), (aj, ai, ri, ci, t)]
            print(f"\nVertex conflict detected")
        else:
            _, ai, aj, r1, c1, r2, c2, t = conflict
            # curr_i=(r1,c1), curr_j=(r2,c2).  Each branch avoids the other's position.
            branch_info = [(ai, aj, r2, c2, t), (aj, ai, r1, c1, t)]
            print(f"\nEdge conflict detected")

        for branch_agent, other_agent, or_, oc, t in branch_info:
            # Build the opposing truck's footprint at the conflict cell + time.
            ot        = truck_map[other_agent]
            owx, owy  = grid.cell_to_world(or_, oc)
            oh        = _infer_heading_at_t(paths.get(other_agent, []), t)
            fp        = (owx, owy, oh, ot.length / 2.0, ot.width / 2.0)

            new_cons = {k: {tt: list(fps) for tt, fps in v.items()}
                        for k, v in constraints.items()}
            branch_t_list = new_cons[branch_agent].setdefault(t, [])
            new_cons[branch_agent][t] = branch_t_list + [fp]

            new_paths = dict(paths)
            info  = agent_info[branch_agent]
            hard  = _merge_fp_constraints(new_cons[branch_agent], locked_footprints)
            bpath = astar_st(info['driveable'], grid, info['start'], info['target'],
                             info['truck'], hard, info['stop_dist'])
            # Corridor blocked all routes for this branch — try bypass.
            if not _at_goal(bpath, branch_agent):
                bypass = _corridor_bypass(branch_agent, hard)
                if _at_goal(bypass, branch_agent):
                    bpath = bypass
            new_paths[branch_agent] = _at_target_fix(bpath, branch_agent)
            new_cost = sum(len(p) for p in new_paths.values())
            _nid += 1
            heapq.heappush(heap, (new_cost, _nid, new_cons, new_paths))

    def _apply_bypass_to_paths(paths_dict, constraints_dict):
        """For any truck that didn't reach its goal, try the corridor-bypass path."""
        final = {}
        for aid, path in paths_dict.items():
            if _at_goal(path, aid):
                final[aid] = path
            else:
                merged = _merge_fp_constraints(constraints_dict.get(aid, {}),
                                               locked_footprints)
                bypass = _corridor_bypass(aid, merged)
                final[aid] = bypass if _at_goal(bypass, aid) else path
        return final

    if heap:
        _, _, constraints_ex, best = heap[0]
        return smooth_paths(_apply_bypass_to_paths(best, constraints_ex))
    return smooth_paths(_apply_bypass_to_paths(init_paths, init_constraints))
