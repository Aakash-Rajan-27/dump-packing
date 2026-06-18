# planning_worker.py
# ─────────────────────────────────────────────────────────────
# Background planning thread (Thread 2).
# Receives task dicts from work_q, runs heavy planners on grid/
# truck snapshots, and pushes result dicts to result_q.
#
# Snapshot helpers live here too since they are only called from
# run_simulation when building task dicts for this thread.
# ─────────────────────────────────────────────────────────────

import math
import copy as _copy
import queue as _queue
import traceback as _traceback
import numpy as np
from scipy.ndimage import binary_dilation

import grid_map
from filters import get_raw_candidates, is_accessible, compute_masks
from scoring import score_candidates
from mcts import mcts_select_dump_points
from assignment import assign
from pathfinder import plan_staging_paths, plan_paths_cbs, escape_and_replan_exit
from config import CONFIG_MATERIAL_HEIGHT_THRESHOLD


# ── Grid / truck snapshot helpers ────────────────────────────────────────────

def _make_grid_snapshot(grid):
    """Shallow-copy GridMap with deep-copied numpy arrays.
    The planner reads/writes the copy; the live grid stays untouched."""
    snap               = _copy.copy(grid)
    snap.state         = grid.state.copy()
    snap.z_height      = grid.z_height.copy()
    snap.pheromone     = grid.pheromone.copy()
    snap._path_corridors = {k: list(v) for k, v in grid._path_corridors.items()}
    return snap


def _make_truck_snapshot(truck):
    """Shallow-copy Truck with mutable fields frozen at this instant."""
    snap              = _copy.copy(truck)
    snap.pos          = list(truck.pos)
    snap.heading      = truck.heading
    snap.path         = list(truck.path)
    snap._exit_path   = list(truck._exit_path)
    snap._pre_path    = list(truck._pre_path) if truck._pre_path else None
    snap.dump_target  = truck.dump_target
    snap.staging_pose = truck.staging_pose
    snap._dump_ticks_required = truck._dump_ticks_required
    snap._dump_ticks  = truck._dump_ticks
    return snap


# ── Planning task executor ────────────────────────────────────────────────────

def _execute_planning_task(task, result_q):
    """Run one planning task entirely on snapshots; push result to result_q."""
    ttype     = task['type']
    grid_snap = task['grid_snap']
    entry_rc  = task['entry_rc']

    # ── IDLE TRUCK PLANNING ───────────────────────────────────────────────────
    if ttype == 'idle':
        idle_snaps   = task['idle_trucks']
        locked       = task['locked_paths']
        claimed      = task.get('claimed_targets', set())

        repr_truck       = max(idle_snaps, key=lambda t: t.width * t.length)
        raw              = get_raw_candidates(grid_snap, repr_truck)

        if not raw:
            print(f"[PLANNER] No raw dump candidates left — simulation likely complete.")
            result_q.put({'type': 'idle', 'assignments': {}, 'sim_done': True})
            return

        fp              = grid_snap.fill_pct()
        coarse, _fine   = compute_masks(grid_snap, repr_truck)
        top_cands       = []

        if fp < CONFIG_MATERIAL_HEIGHT_THRESHOLD:
            scores = score_candidates(grid_snap, raw)
            for idx in scores.argsort()[::-1]:
                r, c = raw[idx]
                if is_accessible(grid_snap, r, c, entry_rc, repr_truck,
                                 precomputed_coarse_mask=coarse):
                    top_cands.append((r, c))
                if len(top_cands) >= 50:
                    break
        else:
            acc = [(r, c) for r, c in raw
                   if is_accessible(grid_snap, r, c, entry_rc, repr_truck,
                                    precomputed_coarse_mask=coarse)]
            if acc:
                scores    = score_candidates(grid_snap, acc)
                top_cands = [acc[i] for i in scores.argsort()[-50:][::-1]]

        if not top_cands:
            print(f"[PLANNER] {len(raw)} raw candidates found but none passed BFS accessibility check.")
            result_q.put({'type': 'idle', 'assignments': {}, 'sim_done': False})
            return

        assignments_all = []
        claimed_pts     = set(claimed)
        remaining       = list(idle_snaps)

        for cls in ('large', 'medium', 'small'):
            cls_trucks = [t for t in remaining if t.truck_class == cls]
            if not cls_trucks:
                continue
            avail = [(r, c) for r, c in top_cands if (r, c) not in claimed_pts]
            if not avail:
                break
            dpts = mcts_select_dump_points(grid_snap, avail, cls_trucks[0],
                                           n_trucks=len(cls_trucks), n_sim=200)
            if not dpts:
                continue
            these = assign(cls_trucks[:len(dpts)], dpts, grid_snap)
            assignments_all.extend(these)
            for _, dp in these: claimed_pts.add(dp)
            for t, _ in these: remaining.remove(t)

        if not assignments_all:
            result_q.put({'type': 'idle', 'assignments': {}, 'sim_done': False})
            return

        extra_st = task.get('extra_st_constraints')
        paths, staging = plan_staging_paths(
            grid_snap, assignments_all, locked_paths=locked,
            extra_st_constraints=extra_st,
            precomputed_masks={(repr_truck.truck_class, False): _fine})
        out = {}
        for t_snap, dp in assignments_all:
            out[t_snap.id] = (paths.get(t_snap.id, []), dp,
                              staging.get(t_snap.id))

        result_q.put({'type': 'idle', 'assignments': out, 'sim_done': False})

    # ── EXIT PATH PLANNING ────────────────────────────────────────────────────
    elif ttype == 'exit':
        exit_snaps = task['exit_trucks']
        locked     = task['locked_paths']

        exit_assignments = [(t_snap, entry_rc) for t_snap in exit_snaps]
        extra_st = task.get('extra_st_constraints')
        exit_paths       = plan_paths_cbs(grid_snap, exit_assignments,
                                          locked_paths=locked,
                                          extra_st_constraints=extra_st)

        out = {t_snap.id: exit_paths.get(t_snap.id, []) for t_snap in exit_snaps}
        result_q.put({'type': 'exit', 'paths': out})

    # ── EXIT ESCAPE (fallback for trucks CBS could not route out) ─────────────
    elif ttype == 'exit_escape':
        t_snap     = task['truck_snap']
        all_snaps  = task['all_trucks']
        path = escape_and_replan_exit(t_snap, grid_snap, all_snaps, entry_rc)
        result_q.put({'type': 'exit_escape', 'truck_id': t_snap.id, 'path': path})

    # ── GATE PRE-PLANNING ─────────────────────────────────────────────────────
    elif ttype == 'gate':
        t_snap  = task['truck_snap']
        locked  = task['locked_paths']
        claimed = task.get('claimed_targets', set())

        raw = get_raw_candidates(grid_snap, t_snap)
        if not raw:
            result_q.put({'type': 'gate', 'truck_id': t_snap.id, 'result': None})
            return

        fp             = grid_snap.fill_pct()
        # ignore_path_reserved=True so exit corridors from trucks inside don't block
        # staging candidates for the incoming truck. Locked paths handle temporal separation.
        coarse, _fine  = compute_masks(grid_snap, t_snap, ignore_path_reserved=True)
        top            = []

        if fp < CONFIG_MATERIAL_HEIGHT_THRESHOLD:
            scores = score_candidates(grid_snap, raw)
            for idx in scores.argsort()[::-1]:
                r, c = raw[idx]
                if is_accessible(grid_snap, r, c, entry_rc, t_snap,
                                 precomputed_coarse_mask=coarse):
                    top.append((r, c))
                if len(top) >= 20:
                    break
        else:
            acc = [(r, c) for r, c in raw
                   if is_accessible(grid_snap, r, c, entry_rc, t_snap,
                                    precomputed_coarse_mask=coarse)]
            if acc:
                scores = score_candidates(grid_snap, acc)
                top    = [acc[i] for i in scores.argsort()[-20:][::-1]]

        avail = [(r, c) for r, c in top if (r, c) not in claimed]
        if not avail:
            result_q.put({'type': 'gate', 'truck_id': t_snap.id, 'result': None})
            return

        dpts = mcts_select_dump_points(grid_snap, avail, t_snap,
                                       n_trucks=1, n_sim=100)
        if not dpts:
            result_q.put({'type': 'gate', 'truck_id': t_snap.id, 'result': None})
            return

        asgn = assign([t_snap], dpts[:1], grid_snap)
        if not asgn:
            result_q.put({'type': 'gate', 'truck_id': t_snap.id, 'result': None})
            return

        # Temporarily dilate PATH_RESERVED cells in snapshot for corridor safety
        _hw_buf = int(math.ceil((t_snap.width / 2.0) / grid_snap.cell_size))
        _pr_mask = (grid_snap.state == grid_map.CellState.PATH_RESERVED)
        _struct  = np.array(
            [[math.hypot(dr, dc) <= _hw_buf
              for dc in range(-_hw_buf, _hw_buf + 1)]
             for dr in range(-_hw_buf, _hw_buf + 1)], dtype=bool)
        _expanded = binary_dilation(_pr_mask, structure=_struct)
        _buf      = (_expanded & ~_pr_mask
                     & (grid_snap.state != grid_map.CellState.BOUNDARY)
                     & (grid_snap.state != grid_map.CellState.PROTECTED))
        grid_snap.state[_buf] = grid_map.CellState.PATH_RESERVED

        paths, staging = plan_staging_paths(
            grid_snap, asgn, locked_paths=locked,
            ignore_path_reserved=True,
            precomputed_masks={(t_snap.truck_class, True): _fine})
        dt   = asgn[0][1]
        path = paths.get(t_snap.id, [])
        sp   = staging.get(t_snap.id)

        if path:
            result_q.put({'type': 'gate', 'truck_id': t_snap.id,
                          'result': (path, dt, sp)})
        else:
            result_q.put({'type': 'gate', 'truck_id': t_snap.id, 'result': None})


def _planning_worker(work_q, result_q, stop_evt):
    """Background planning thread — runs until stop_evt is set."""
    while not stop_evt.is_set():
        try:
            task = work_q.get(timeout=0.1)
        except _queue.Empty:
            continue
        try:
            _execute_planning_task(task, result_q)
        except Exception as exc:
            print(f"[PLANNER] Exception in task '{task.get('type','?')}': {exc}")
            _traceback.print_exc()
            # Always push an empty result so _in_flight entries are cleared.
            # Without this, a crash silently leaves trucks stuck forever.
            try:
                ttype = task.get('type', '')
                if ttype == 'idle':
                    result_q.put({'type': 'idle', 'assignments': {}, 'sim_done': False})
                elif ttype == 'exit':
                    out = {s.id: [] for s in task.get('exit_trucks', [])}
                    result_q.put({'type': 'exit', 'paths': out})
                elif ttype == 'gate':
                    snap = task.get('truck_snap')
                    if snap is not None:
                        result_q.put({'type': 'gate', 'truck_id': snap.id, 'result': None})
                elif ttype == 'exit_escape':
                    snap = task.get('truck_snap')
                    if snap is not None:
                        result_q.put({'type': 'exit_escape', 'truck_id': snap.id, 'path': []})
            except Exception:
                pass
        finally:
            work_q.task_done()
