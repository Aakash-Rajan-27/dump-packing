# main.py
# ─────────────────────────────────────────────────────────────
# Threading update: heavy path-planning (A*, CBS, hybrid-A*)
# runs in a background thread so truck movement and rendering
# never pause while paths are being computed.
#
# Thread 1 (main): truck.step(), renderer.draw(), collision
#   guard, deadlock/headlock, conflict detection, pheromone.
# Thread 2 (planner): all plan_staging_paths / plan_paths_cbs
#   calls — works on frozen snapshots of grid + truck state.
# ─────────────────────────────────────────────────────────────

import sys
import time
import math
import threading
import queue   as _queue
import copy    as _copy
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation

sys.stdout.reconfigure(encoding='utf-8')

from config import (POLYGON_BOUNDARY, ENTRY_POINT, CELL_SIZE,
                    FLEET_COMPOSITION, TICK_DELAY, PYGAME_SCALE,
                    PHEROMONE_DECAY, PHEROMONE_SPREAD_SIGMA,
                    CONFIG_MATERIAL_HEIGHT_THRESHOLD, STEPS_PER_TICK,
                    ENTRY_CORRIDOR_CELLS)
import grid_map
from truck      import Truck
from filters    import get_raw_candidates, is_accessible, precompute_coarse_blocked_mask
from scoring    import score_candidates
from mcts       import mcts_select_dump_points
from assignment import assign
from pathfinder import (plan_staging_paths, plan_paths_cbs,
                        _detect_first_conflict, _path_cells,
                        _corridor_cell_set,
                        generate_reverse_retreat, generate_yield_maneuver,
                        escape_and_replan_exit)
from renderer   import Renderer
import random


# ── Grid / truck snapshots for the planning thread ────────────────────────────

def _make_grid_snapshot(grid):
    """Shallow-copy GridMap with deep-copied numpy arrays.
    The planner reads/writes the copy; the live grid stays untouched."""
    snap               = _copy.copy(grid)          # same class, same methods
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


# ── Planning worker (Thread 2) ────────────────────────────────────────────────

def _execute_planning_task(task, result_q):
    """Run one planning task entirely on snapshots; push result to result_q."""
    ttype     = task['type']
    grid_snap = task['grid_snap']
    entry_rc  = task['entry_rc']

    # ── IDLE TRUCK PLANNING ───────────────────────────────────────────────────
    if ttype == 'idle':
        idle_snaps   = task['idle_trucks']      # list[truck_snapshot]
        locked       = task['locked_paths']
        claimed      = task.get('claimed_targets', set())

        repr_truck = max(idle_snaps, key=lambda t: t.width * t.length)
        raw        = get_raw_candidates(grid_snap, repr_truck)

        if not raw:
            result_q.put({'type': 'idle', 'assignments': {}, 'sim_done': True})
            return

        fp           = grid_snap.fill_pct()
        coarse       = precompute_coarse_blocked_mask(grid_snap, repr_truck)
        top_cands    = []

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

        paths, staging = plan_staging_paths(grid_snap, assignments_all,
                                            locked_paths=locked)
        out = {}
        for t_snap, dp in assignments_all:
            out[t_snap.id] = (paths.get(t_snap.id, []), dp,
                              staging.get(t_snap.id))

        result_q.put({'type': 'idle', 'assignments': out, 'sim_done': False})

    # ── EXIT PATH PLANNING ────────────────────────────────────────────────────
    elif ttype == 'exit':
        exit_snaps = task['exit_trucks']     # list[truck_snapshot]
        locked     = task['locked_paths']
        # pre_path_trucks obstacle-stamping removed: it marked waiting trucks'
        # incoming corridors as OBSTACLE cells up to 12 cells wide, creating a
        # solid wall that split the grid and made exit paths impossible to find.
        # Temporal separation is handled by nav_locked space-time constraints;
        # spatial separation by PATH_RESERVED corridors + ignore_path_reserved=True.

        exit_assignments = [(t_snap, entry_rc) for t_snap in exit_snaps]
        exit_paths       = plan_paths_cbs(grid_snap, exit_assignments,
                                          locked_paths=locked)

        out = {t_snap.id: exit_paths.get(t_snap.id, []) for t_snap in exit_snaps}
        result_q.put({'type': 'exit', 'paths': out})

    # ── GATE PRE-PLANNING ─────────────────────────────────────────────────────
    elif ttype == 'gate':
        t_snap  = task['truck_snap']   # snapshot with pos=ENTRY_POINT
        locked  = task['locked_paths']
        claimed = task.get('claimed_targets', set())

        raw = get_raw_candidates(grid_snap, t_snap)
        if not raw:
            result_q.put({'type': 'gate', 'truck_id': t_snap.id, 'result': None})
            return

        fp     = grid_snap.fill_pct()
        coarse = precompute_coarse_blocked_mask(grid_snap, t_snap)
        top    = []

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

        paths, staging = plan_staging_paths(grid_snap, asgn, locked_paths=locked)
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
            print(f"[PLANNER] Exception: {exc}")
        finally:
            work_q.task_done()


# ── Simulation helpers (unchanged) ───────────────────────────────────────────

def initialise_half_full_dump(grid):
    target_pack = 0.00
    valid_cells = np.argwhere(
        (grid.state == grid_map.CellState.EMPTY) |
        (grid.state == grid_map.CellState.PARTIAL)
    )
    while grid.pack_pct() < target_pack:
        r, c = random.choice(valid_cells)
        if grid.is_dumpable(r, c):
            grid.dump_at(r, c, volume_m3=10.0)
    print(f"Initial terrain generated. Pack density = {100*grid.pack_pct():.1f}%")


def _corridor_cells(grid, path, truck):
    half_w = truck.width / 2.0 / grid.cell_size
    half   = int(math.ceil(half_w))
    result = set()
    for r, c in _path_cells(grid, path):
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                if math.hypot(dr, dc) <= half_w:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid.rows and 0 <= nc < grid.cols:
                        result.add((nr, nc))
    return result


def build_fleet():
    from config import TRUCK_CLASSES
    trucks     = []
    truck_id   = 0
    queue_slot = 0
    for cls in ('small', 'medium', 'large'):
        for _ in range(FLEET_COMPOSITION.get(cls, 0)):
            truck_len = TRUCK_CLASSES[cls]['length_m']
            offset    = (queue_slot + 1) * truck_len * 1.5
            home      = (ENTRY_POINT[0], ENTRY_POINT[1] - offset)
            if truck_id == 0:
                trucks.append(Truck(truck_id, cls, home_pos=home))
            else:
                trucks.append(Truck(truck_id, cls, start_pos=home,
                                    waiting=True, home_pos=home))
            queue_slot += 1
            truck_id   += 1
    return trucks


def _try_inplace_replan(ta, tb, grid, entry_rc):
    """Resolve a conflict between two trucks via CBS without forcing idle.
    Stays in Thread 1 (rare, fast-path — cooldown limits frequency)."""
    both_nav  = (ta.status == ta.STATUS_NAVIGATING and tb.status == tb.STATUS_NAVIGATING)
    both_exit = (ta.status == ta.STATUS_EXITING    and tb.status == tb.STATUS_EXITING)
    a_nav     = ta.status == ta.STATUS_NAVIGATING

    if both_nav:
        if not ta.dump_target or not tb.dump_target:
            return False
        ta.clear_all_corridors(grid)
        tb.clear_all_corridors(grid)
        new_paths, new_staging = plan_staging_paths(
            grid, [(ta, ta.dump_target), (tb, tb.dump_target)])
        np_a = new_paths.get(ta.id, [])
        np_b = new_paths.get(tb.id, [])
        if np_a and np_b:
            ta.path = np_a;  ta.stop_target = ta.path[-1];  ta.staging_pose = new_staging.get(ta.id)
            tb.path = np_b;  tb.stop_target = tb.path[-1];  tb.staging_pose = new_staging.get(tb.id)
            ta._mark_dump_corridor(grid, ta.path)
            tb._mark_dump_corridor(grid, tb.path)
            ta._conflict_cooldown = 5;  tb._conflict_cooldown = 5
            print(f"[CBS-REPLAN] T{ta.id}+T{tb.id} nav conflict resolved in-place.")
            return True
        return False

    if both_exit:
        ta._clear_exit_corridor(grid)
        tb._clear_exit_corridor(grid)
        new_paths = plan_paths_cbs(grid, [(ta, entry_rc), (tb, entry_rc)])
        np_a = new_paths.get(ta.id, [])
        np_b = new_paths.get(tb.id, [])
        if np_a and np_b:
            ta._exit_path = np_a;  tb._exit_path = np_b
            ta._mark_exit_corridor(grid, np_a)
            tb._mark_exit_corridor(grid, np_b)
            ta._conflict_cooldown = 5;  tb._conflict_cooldown = 5
            print(f"[CBS-REPLAN] T{ta.id}+T{tb.id} exit conflict resolved in-place.")
            return True
        return False

    nav_t  = ta if a_nav else tb
    exit_t = tb if a_nav else ta

    if nav_t.path:
        locked = {nav_t.id: ([nav_t.front_center_cell(grid)] + list(nav_t.path),
                              nav_t._dump_ticks_required + 2)}
        exit_t._clear_exit_corridor(grid)
        ep = plan_paths_cbs(grid, [(exit_t, entry_rc)], locked_paths=locked)
        np_exit = ep.get(exit_t.id, [])
        if np_exit:
            exit_t._exit_path = np_exit
            exit_t._mark_exit_corridor(grid, np_exit)
            exit_t._conflict_cooldown = 5
            print(f"[CBS-REPLAN] T{exit_t.id} exit replanned around T{nav_t.id}.")
            return True

    if exit_t._exit_path and nav_t.dump_target:
        locked = {exit_t.id: ([exit_t.front_center_cell(grid)] + list(exit_t._exit_path), 0)}
        nav_t.clear_all_corridors(grid)
        np2, ns2 = plan_staging_paths(grid, [(nav_t, nav_t.dump_target)],
                                      locked_paths=locked)
        np_nav = np2.get(nav_t.id, [])
        if np_nav:
            nav_t.path = np_nav;  nav_t.stop_target = nav_t.path[-1]
            nav_t.staging_pose = ns2.get(nav_t.id)
            nav_t._mark_dump_corridor(grid, nav_t.path)
            nav_t._conflict_cooldown = 5
            print(f"[CBS-REPLAN] T{nav_t.id} nav replanned around T{exit_t.id}.")
            return True

    return False


# ── Main simulation loop ──────────────────────────────────────────────────────

def run_simulation():
    print("Initialising grid...")
    grid = grid_map.GridMap(POLYGON_BOUNDARY, CELL_SIZE)
    initialise_half_full_dump(grid)

    valid_cells = np.sum(grid.state == grid_map.CellState.EMPTY)
    print(f"Grid: {grid.rows}x{grid.cols} cells, {valid_cells} valid dump cells")

    entry_rc = grid.world_to_cell(*ENTRY_POINT)
    print(f"Entry point: world{ENTRY_POINT} -> cell{entry_rc}")

    trucks = build_fleet()
    print(f"Fleet: {len(trucks)} trucks")

    corridor_clearance_m = ENTRY_CORRIDOR_CELLS * CELL_SIZE
    truck_map = {t.id: t for t in trucks}

    renderer = Renderer(grid, scale=PYGAME_SCALE)
    print("Renderer ready — starting simulation loop")

    # ── Background planning thread ────────────────────────────────────────────
    _work_q   = _queue.Queue()
    _result_q = _queue.Queue()
    _stop_evt = threading.Event()
    _planner  = threading.Thread(target=_planning_worker,
                                 args=(_work_q, _result_q, _stop_evt),
                                 daemon=True, name="Planner")
    _planner.start()

    # IDs of trucks that have an in-flight planning request (avoid duplicates)
    _in_flight: set = set()

    tick       = 0
    done       = False
    top_candidates: list = []   # kept for metrics display

    while not done:
        if renderer.check_quit():
            break

        # Snapshot each truck position for stagnation tracking
        for _t in trucks:
            _t._pos_snapshot = list(_t.pos)

        # ── APPLY PLANNING RESULTS ────────────────────────────────────────────
        while not _result_q.empty():
            try:
                res = _result_q.get_nowait()
            except _queue.Empty:
                break

            if res['type'] == 'idle':
                for tid, (path, dp, sp) in res['assignments'].items():
                    t = truck_map.get(tid)
                    _in_flight.discard(tid)
                    if t is None:
                        continue
                    if t.status != t.STATUS_IDLE:
                        # Truck was force-replanned or reset — discard stale result
                        if dp:
                            grid.unreserve(*dp)
                        continue
                    # Check target still free
                    if dp and grid.state[dp[0], dp[1]] in (
                            grid_map.CellState.RESERVED, grid_map.CellState.FILLED):
                        continue   # cell claimed in the interim — skip, retry next tick
                    t.set_path(path, dp, grid, staging_pose=sp)

                if res.get('sim_done'):
                    print(f"\nSimulation complete at tick {tick}!")
                    done = True

            elif res['type'] == 'exit':
                for tid, path in res['paths'].items():
                    t = truck_map.get(tid)
                    _in_flight.discard(tid)
                    if t is None:
                        continue
                    if not t.needs_exit_path():
                        continue   # truck already has an exit path or changed state
                    if not path:
                        path = escape_and_replan_exit(t, grid, trucks, entry_rc)
                    t.set_exit_path(path, grid)

            elif res['type'] == 'gate':
                tid = res['truck_id']
                t   = truck_map.get(tid)
                _in_flight.discard(tid)
                if t is None or t.status != t.STATUS_WAITING:
                    continue
                result = res['result']
                if result is None:
                    continue
                path, dp, sp = result
                if dp and grid.state[dp[0], dp[1]] in (
                        grid_map.CellState.RESERVED, grid_map.CellState.FILLED):
                    continue   # target taken — retry next tick
                t.preload_dump_path(path, dp, grid, sp)
                print(f"[GATE] Pre-planned path applied for T{tid}")

        if done:
            break

        # ── GATE LOGIC ────────────────────────────────────────────────────────
        waiting_queue = sorted([t for t in trucks if t.status == t.STATUS_WAITING],
                               key=lambda t: t.id)
        if waiting_queue:
            next_t = waiting_queue[0]

            _trucks_inside = [t for t in trucks
                              if t is not next_t and
                                 t.status in (t.STATUS_NAVIGATING, t.STATUS_EXITING,
                                              t.STATUS_DUMPING,    t.STATUS_REVERSING)]

            # Post gate pre-planning task if none in flight for this truck
            if (not next_t._pre_path
                    and _trucks_inside
                    and next_t.id not in _in_flight):
                g_snap  = _make_grid_snapshot(grid)
                t_snap  = _make_truck_snapshot(next_t)
                t_snap.pos     = list(ENTRY_POINT)
                t_snap.heading = math.pi / 2

                locked = {}
                for ot in _trucks_inside:
                    if ot.status == ot.STATUS_NAVIGATING and ot.path:
                        locked[ot.id] = ([ot.front_center_cell(grid)] + list(ot.path),
                                         ot._dump_ticks_required + 2)
                    elif ot.status == ot.STATUS_EXITING and ot._exit_path:
                        locked[ot.id] = ([ot.front_center_cell(grid)] + list(ot._exit_path), 0)
                    else:
                        rem = max(0, ot._dump_ticks_required - ot._dump_ticks)
                        locked[ot.id] = ([ot.front_center_cell(grid)], rem + 2)

                claimed = {t.dump_target for t in trucks if t.dump_target}

                _work_q.put({'type':            'gate',
                             'grid_snap':        g_snap,
                             'entry_rc':         entry_rc,
                             'truck_snap':       t_snap,
                             'locked_paths':     locked,
                             'claimed_targets':  claimed})
                _in_flight.add(next_t.id)
                print(f"[GATE] Posted pre-planning task for T{next_t.id}")

            # Re-validate pre-path corridor overlap (fast — stays in Thread 1)
            if next_t._pre_path and _trucks_inside:
                _pre_corr = {
                    (r, c) for r, c in _corridor_cells(grid, next_t._pre_path, next_t)
                    if math.hypot(r - entry_rc[0], c - entry_rc[1]) > ENTRY_CORRIDOR_CELLS
                }
                _stale = False
                for ot in _trucks_inside:
                    if ot.status == ot.STATUS_NAVIGATING and ot.path:
                        _oc = {(r, c) for r, c in _corridor_cells(grid, ot.path, ot)
                               if math.hypot(r - entry_rc[0], c - entry_rc[1]) > ENTRY_CORRIDOR_CELLS}
                    elif ot.status == ot.STATUS_EXITING and ot._exit_path:
                        _oc = {(r, c) for r, c in _corridor_cells(grid, ot._exit_path, ot)
                               if math.hypot(r - entry_rc[0], c - entry_rc[1]) > ENTRY_CORRIDOR_CELLS}
                    else:
                        _fc = ot.front_center_cell(grid)
                        _oc = (_corridor_cells(grid, [_fc], ot)
                               if math.hypot(_fc[0] - entry_rc[0], _fc[1] - entry_rc[1])
                               > ENTRY_CORRIDOR_CELLS else set())
                    if _pre_corr & _oc:
                        _stale = True
                        print(f"[GATE] T{next_t.id} pre-path stale — cancelling.")
                        break
                if _stale:
                    next_t.cancel_preload(grid)
                    _in_flight.discard(next_t.id)

            # Release truck if gate is clear
            any_transiting = any(t.status in (t.STATUS_ENTERING, t.STATUS_LEAVING)
                                 for t in trucks)
            _can_enter = next_t._pre_path or not _trucks_inside
            if _can_enter and not any_transiting:
                entry_x, entry_y = ENTRY_POINT
                active = [t for t in trucks if t.status not in (t.STATUS_WAITING,)]
                gate_clear = True
                if active:
                    nearest = min(math.hypot(t.pos[0] - entry_x, t.pos[1] - entry_y)
                                  for t in active)
                    gate_clear = nearest >= corridor_clearance_m
                if gate_clear:
                    next_t.release()
                    print(f"[QUEUE] Released truck {next_t.id} ({next_t.truck_class})")

        # ── EXIT PATH PLANNING ────────────────────────────────────────────────
        need_exit = [t for t in trucks
                     if t.needs_exit_path() and t.id not in _in_flight]
        if need_exit:
            g_snap = _make_grid_snapshot(grid)

            nav_locked = {t.id: ([t.front_center_cell(grid)] + list(t.path),
                                   t._dump_ticks_required + 2)
                          for t in trucks
                          if t.status == t.STATUS_NAVIGATING and t.path}
            for wt in trucks:
                if (wt.status in (wt.STATUS_WAITING, wt.STATUS_ENTERING)
                        and wt._pre_path):
                    nav_locked[wt.id] = (wt._pre_path, wt._dump_ticks_required + 2)

            exit_snaps = [_make_truck_snapshot(t) for t in need_exit]

            _work_q.put({'type':           'exit',
                         'grid_snap':       g_snap,
                         'entry_rc':        entry_rc,
                         'exit_trucks':     exit_snaps,
                         'locked_paths':    nav_locked})
            for t in need_exit:
                _in_flight.add(t.id)

        # ── IDLE TRUCK PLANNING ───────────────────────────────────────────────
        idle_trucks = [t for t in trucks
                       if t.is_idle() and not t._pre_path
                       and t.id not in _in_flight]
        if idle_trucks:
            g_snap = _make_grid_snapshot(grid)
            idle_ids = {t.id for t in idle_trucks}

            all_locked = {}
            for t in trucks:
                if t.id in idle_ids:
                    continue
                if t.status == t.STATUS_NAVIGATING and t.path:
                    all_locked[t.id] = ([t.front_center_cell(grid)] + list(t.path),
                                         t._dump_ticks_required + 2)
                elif t.status == t.STATUS_EXITING and t._exit_path:
                    all_locked[t.id] = ([t.front_center_cell(grid)] + list(t._exit_path), 0)

            claimed = {t.dump_target for t in trucks
                       if t.dump_target and t.id not in idle_ids}

            idle_snaps = [_make_truck_snapshot(t) for t in idle_trucks]

            _work_q.put({'type':           'idle',
                         'grid_snap':       g_snap,
                         'entry_rc':        entry_rc,
                         'idle_trucks':     idle_snaps,
                         'locked_paths':    all_locked,
                         'claimed_targets': claimed})
            for t in idle_trucks:
                _in_flight.add(t.id)

        # ── MOVEMENT + RENDER PHASE ───────────────────────────────────────────
        substep_delay  = TICK_DELAY / max(1, STEPS_PER_TICK)
        cached_fill    = grid.fill_pct() * 100
        cached_pack    = grid.pack_pct() * 100
        active_trucks  = [t for t in trucks if t.status != t.STATUS_WAITING]

        for substep in range(STEPS_PER_TICK):
            if renderer.check_quit():
                done = True
                break

            for truck in trucks:
                prev_x, prev_y     = truck.pos[0], truck.pos[1]
                prev_heading       = truck.heading
                prev_first_wp      = truck.path[0]       if truck.path       else None
                prev_exit_first_wp = truck._exit_path[0] if truck._exit_path else None

                truck.step(grid)

                # Physical collision guard
                collided = False
                for other in active_trucks:
                    if other is truck:
                        continue
                    dist = math.hypot(truck.pos[0] - other.pos[0],
                                      truck.pos[1] - other.pos[1])
                    if dist < (truck.length + other.length) * 0.5:
                        # If both trucks have non-empty, disjoint path corridors
                        # their planned paths don't physically cross — bounding-circle
                        # overlap is a geometry artefact; let both continue.
                        ci = _corridor_cell_set(grid, truck.id)
                        cj = _corridor_cell_set(grid, other.id)
                        if ci and cj and ci.isdisjoint(cj):
                            continue
                        collided = True
                        break

                if collided:
                    truck._stuck_substeps += 1
                    truck.pos[0], truck.pos[1] = prev_x, prev_y
                    truck.heading = prev_heading
                    if prev_first_wp is not None and (
                            not truck.path or truck.path[0] != prev_first_wp):
                        truck.path.insert(0, prev_first_wp)
                    if prev_exit_first_wp is not None and (
                            not truck._exit_path or
                            truck._exit_path[0] != prev_exit_first_wp):
                        truck._exit_path.insert(0, prev_exit_first_wp)
                else:
                    truck._stuck_substeps = 0

            metrics = {
                'tick':       f"{tick}.{substep + 1}/{STEPS_PER_TICK}",
                'fleet':      f"{FLEET_COMPOSITION['small']}S/"
                              f"{FLEET_COMPOSITION['medium']}M/"
                              f"{FLEET_COMPOSITION['large']}L",
                'idle':       len([t for t in trucks if t.is_idle()]),
                'queued':     len(waiting_queue) if waiting_queue else 0,
                'in_flight':  len(_in_flight),
                'candidates': len(top_candidates) if top_candidates else '-',
                'fill%':      f"{cached_fill:.3f}",
                'pack%':      f"{cached_pack:.3f}",
            }
            renderer.draw(trucks, metrics)
            time.sleep(substep_delay)

        # ── PHEROMONE UPDATE ──────────────────────────────────────────────────
        grid.pheromone = 1.0 - (1.0 - grid.pheromone) * PHEROMONE_DECAY
        grid.pheromone = gaussian_filter(grid.pheromone, sigma=PHEROMONE_SPREAD_SIGMA)
        np.clip(grid.pheromone, 0.0, 1.0, out=grid.pheromone)

        # ── DEADLOCK RESOLUTION ───────────────────────────────────────────────
        _STUCK_LIMIT = STEPS_PER_TICK * 4

        stuck_active = [t for t in active_trucks
                        if t._stuck_substeps >= _STUCK_LIMIT
                        and t.status in (t.STATUS_NAVIGATING, t.STATUS_EXITING)]

        _deadlock_handled = False
        if len(stuck_active) >= 2:
            for _si in range(len(stuck_active)):
                for _sj in range(_si + 1, len(stuck_active)):
                    _ta, _tb = stuck_active[_si], stuck_active[_sj]
                    _d = math.hypot(_ta.pos[0] - _tb.pos[0],
                                    _ta.pos[1] - _tb.pos[1])
                    if _d >= (_ta.length + _tb.length) * 0.5:
                        continue

                    _ret_a = generate_reverse_retreat(_ta, grid, num_steps=6)
                    _ret_b = generate_reverse_retreat(_tb, grid, num_steps=6)

                    def _retreat_ok(truck, retreat, other_dump):
                        if not retreat:
                            return False
                        if other_dump is None:
                            return True
                        half = truck.length / 2.0
                        for wp in retreat:
                            bx = wp[0] + math.cos(wp[2]) * half
                            by = wp[1] + math.sin(wp[2]) * half
                            if grid.world_to_cell(bx, by) == other_dump:
                                return False
                        return True

                    _a_ok = _retreat_ok(_ta, _ret_a, _tb.dump_target)
                    _b_ok = _retreat_ok(_tb, _ret_b, _ta.dump_target)

                    if not _a_ok and not _b_ok:
                        continue

                    if _a_ok and (not _b_ok or len(_ret_a) >= len(_ret_b)):
                        _reverser, _retreat_wps = _ta, _ret_a
                        _advancer = _tb
                    else:
                        _reverser, _retreat_wps = _tb, _ret_b
                        _advancer = _ta

                    print(f"[DEADLOCK] T{_ta.id}↔T{_tb.id} mutual block. "
                          f"T{_reverser.id} retreating {len(_retreat_wps)} steps.")

                    if _reverser.status == _reverser.STATUS_NAVIGATING:
                        _reverser.path = _retreat_wps + list(_reverser.path)
                    elif _reverser.status == _reverser.STATUS_EXITING:
                        _reverser._exit_path = _retreat_wps + list(_reverser._exit_path)

                    _reverser._stuck_substeps    = 0
                    _reverser._conflict_cooldown = len(_retreat_wps) + 3
                    _advancer._conflict_cooldown = max(
                        _advancer._conflict_cooldown, len(_retreat_wps))
                    _deadlock_handled = True
                    break
                if _deadlock_handled:
                    break

        if not _deadlock_handled:
            stuck_nav = [t for t in trucks
                         if t._stuck_substeps >= _STUCK_LIMIT
                         and t.status == t.STATUS_NAVIGATING]
            if stuck_nav:
                yielder = max(stuck_nav, key=lambda t: len(t.path))
                yielder._stuck_substeps    = 0
                yielder._conflict_cooldown = 3
                yielder.clear_all_corridors(grid)
                yielder.cancel_preload(grid)
                _in_flight.discard(yielder.id)
                if yielder.dump_target:
                    grid.unreserve(*yielder.dump_target)
                    yielder.dump_target = None
                yielder.status      = yielder.STATUS_IDLE
                yielder.path        = []
                yielder.stop_target = None
                yielder.staging_pose = None

        # ── STAGNATION UPDATE ─────────────────────────────────────────────────
        for _t in active_trucks:
            if _t.status not in (_t.STATUS_NAVIGATING, _t.STATUS_EXITING):
                _t._pos_stagnant_ticks = 0
                continue
            _net = math.hypot(_t.pos[0] - _t._pos_snapshot[0],
                              _t.pos[1] - _t._pos_snapshot[1])
            if _net < 0.5:
                _t._pos_stagnant_ticks += 1
            else:
                _t._pos_stagnant_ticks = 0

        # ── HEADLOCK RESOLUTION ───────────────────────────────────────────────
        _HEADLOCK_TICKS = 20
        _stagnant = [t for t in active_trucks
                     if t._pos_stagnant_ticks >= _HEADLOCK_TICKS
                     and t.status in (t.STATUS_NAVIGATING, t.STATUS_EXITING)
                     and t._conflict_cooldown == 0]

        _headlock_handled = False
        for _si in range(len(_stagnant)):
            if _headlock_handled:
                break
            for _sj in range(_si + 1, len(_stagnant)):
                _ta, _tb = _stagnant[_si], _stagnant[_sj]
                _d = math.hypot(_ta.pos[0] - _tb.pos[0], _ta.pos[1] - _tb.pos[1])
                if _d >= (_ta.length + _tb.length) * 0.75:
                    continue
                if math.cos(_ta.heading - _tb.heading) > -0.5:
                    continue

                _plen_a = (len(_ta.path) if _ta.status == _ta.STATUS_NAVIGATING
                           else len(_ta._exit_path))
                _plen_b = (len(_tb.path) if _tb.status == _tb.STATUS_NAVIGATING
                           else len(_tb._exit_path))
                _reverser = _ta if _plen_a >= _plen_b else _tb
                _advancer = _tb if _reverser is _ta else _ta

                _yield_wps = generate_yield_maneuver(_reverser, grid, _advancer)
                if not _yield_wps:
                    continue

                print(f"[HEADLOCK] T{_ta.id}↔T{_tb.id} nose-to-nose. "
                      f"T{_reverser.id} yielding ({len(_yield_wps)} steps).")

                if _reverser.status == _reverser.STATUS_NAVIGATING:
                    _reverser.path = _yield_wps + list(_reverser.path)
                else:
                    _reverser._exit_path = _yield_wps + list(_reverser._exit_path)

                _reverser._stuck_substeps     = 0
                _reverser._pos_stagnant_ticks = 0
                _reverser._conflict_cooldown  = len(_yield_wps) + 5
                _advancer._conflict_cooldown  = max(
                    _advancer._conflict_cooldown, len(_yield_wps) // 2)
                _headlock_handled = True
                break

        # ── PROACTIVE SPACE-TIME CONFLICT DETECTION ───────────────────────────
        for t in trucks:
            if t._conflict_cooldown > 0:
                t._conflict_cooldown -= 1

        _CONFLICT_HORIZON = 20
        nav_cell_paths = {}
        for t in trucks:
            if t._conflict_cooldown > 0:
                continue
            if t.status == t.STATUS_NAVIGATING and t.path:
                cells = _path_cells(grid, t.path)
                if cells:
                    nav_cell_paths[t.id] = cells
            elif t.status == t.STATUS_EXITING and t._exit_path:
                cells = _path_cells(grid, t._exit_path)
                if cells:
                    nav_cell_paths[t.id] = cells

        if len(nav_cell_paths) >= 2:
            _truck_map_d = {t.id: t for t in trucks}
            conflict = _detect_first_conflict(nav_cell_paths,
                                              truck_map=_truck_map_d, grid=grid)
            if conflict:
                if conflict[0] == 'vertex':
                    _, ai, aj, _ri, _ci, _rj, _cj, t_conf = conflict
                    _in_corr = (
                        math.hypot(_ri - entry_rc[0], _ci - entry_rc[1]) <= ENTRY_CORRIDOR_CELLS or
                        math.hypot(_rj - entry_rc[0], _cj - entry_rc[1]) <= ENTRY_CORRIDOR_CELLS
                    )
                    if _in_corr:
                        conflict = None
                else:
                    _, ai, aj, _r1, _c1, _r2, _c2, t_conf = conflict

                if conflict and t_conf <= _CONFLICT_HORIZON:
                    _cta = next((t for t in trucks if t.id == ai), None)
                    _ctb = next((t for t in trucks if t.id == aj), None)

                    _replanned = False
                    if _cta is not None and _ctb is not None:
                        # Skip in-place replan if either truck has in-flight planning
                        if _cta.id not in _in_flight and _ctb.id not in _in_flight:
                            _replanned = _try_inplace_replan(_cta, _ctb, grid, entry_rc)

                    if not _replanned:
                        len_ai = len(nav_cell_paths.get(ai, []))
                        len_aj = len(nav_cell_paths.get(aj, []))
                        _yielder_id = ai if len_ai >= len_aj else aj
                        for _ct in trucks:
                            if _ct.id != _yielder_id:
                                continue
                            _ct._conflict_cooldown = 4
                            _ct._stuck_substeps    = 0
                            _ct.clear_all_corridors(grid)
                            _ct.cancel_preload(grid)
                            _in_flight.discard(_ct.id)
                            if _ct.status == _ct.STATUS_NAVIGATING:
                                if _ct.dump_target:
                                    grid.unreserve(*_ct.dump_target)
                                    _ct.dump_target = None
                                _ct.status       = _ct.STATUS_IDLE
                                _ct.path         = []
                                _ct.stop_target  = None
                                _ct.staging_pose = None
                            elif _ct.status == _ct.STATUS_EXITING:
                                _ct._clear_exit_corridor(grid)
                                _ct._exit_path = []
                            break

        tick += 1

    # ── Shutdown ──────────────────────────────────────────────────────────────
    _stop_evt.set()
    _planner.join(timeout=5.0)

    print("\n=== Final Results ===")
    print(f"Total ticks:   {tick}")
    print(f"Fill %:        {grid.fill_pct()*100:.3f}%")
    print(f"Pack density:  {grid.pack_pct()*100:.3f}%")

    print("\nClose the window to exit.")
    while not renderer.check_quit():
        renderer.draw(trucks, {'FINAL': 'done',
                               'fill%': f"{grid.fill_pct()*100:.3f}",
                               'pack%': f"{grid.pack_pct()*100:.3f}"})
        time.sleep(0.1)
    renderer.close()


if __name__ == '__main__':
    run_simulation()
