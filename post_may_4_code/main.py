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
import numpy as np
from scipy.ndimage import gaussian_filter

sys.stdout.reconfigure(encoding='utf-8')

from config import (POLYGON_BOUNDARY, ENTRY_POINT, CELL_SIZE,
                    FLEET_COMPOSITION, TICK_DELAY, PYGAME_SCALE,
                    PHEROMONE_DECAY, PHEROMONE_SPREAD_SIGMA,
                    STEPS_PER_TICK, ENTRY_CORRIDOR_CELLS)
import grid_map
from pathfinder import (generate_reverse_retreat, generate_yield_maneuver, _make_locked_entry)
from conflict_detect import _rect_overlap_2d, _detect_first_conflict
from renderer import Renderer

from planning_worker import (_make_grid_snapshot, _make_truck_snapshot,
                             _planning_worker)
from sim_helpers import (initialise_half_full_dump, _corridor_cells,
                         build_fleet)


def _advance_path_to_current(path, truck):
    """Trim a replanned path to start at or ahead of the truck's current position.

    The planner works from a snapshot taken some ticks ago; the truck has since
    moved forward.  Returning the path unchanged would cause the truck to drive
    backward to the stale snapshot position.  We find the closest waypoint and
    then advance past any waypoints that are still behind the truck's heading.
    """
    if not path:
        return path
    tx, ty = truck.pos[0], truck.pos[1]
    cos_h = math.cos(truck.heading)
    sin_h = math.sin(truck.heading)

    # 1. Find the waypoint closest to the truck's current world position.
    best_i, best_dist = 0, float('inf')
    for i, wp in enumerate(path):
        d = math.hypot(wp[0] - tx, wp[1] - ty)
        if d < best_dist:
            best_dist, best_i = d, i

    # 2. Advance past any leading waypoints that are behind the truck (negative
    #    forward-projection), so we never ask the truck to reverse.
    while best_i < len(path) - 1:
        wp = path[best_i]
        dx, dy = wp[0] - tx, wp[1] - ty
        if dx * cos_h + dy * sin_h >= 0:
            break
        best_i += 1

    return path[best_i:] or path[-1:]


def _world_path_to_cell_path(path, grid):
    """Sample a world-coord smooth path at cell-size intervals.
    Returns [(row, col, heading), ...] usable by _detect_first_conflict."""
    if not path:
        return []
    result = []
    prev_x, prev_y = float(path[0][0]), float(path[0][1])
    rc = grid.world_to_cell(prev_x, prev_y)
    result.append((rc[0], rc[1], float(path[0][2])))
    accum = 0.0
    for wp in path[1:]:
        cx, cy = float(wp[0]), float(wp[1])
        dist = math.hypot(cx - prev_x, cy - prev_y)
        accum += dist
        if accum >= grid.cell_size:
            accum -= grid.cell_size
            rc = grid.world_to_cell(cx, cy)
            result.append((rc[0], rc[1], float(wp[2])))
        prev_x, prev_y = cx, cy
    return result


def _build_locked_for_replan(trucks, exclude_ids, grid):
    """Build locked_paths dict for a replan task, excluding trucks being replanned."""
    locked = {}
    for ot in trucks:
        if ot.id in exclude_ids:
            continue
        if ot.status == ot.STATUS_NAVIGATING and ot.path:
            locked[ot.id] = _make_locked_entry(ot, ot.path, ot._dump_ticks_required + 2, grid)
        elif ot.status == ot.STATUS_EXITING and ot._exit_path:
            locked[ot.id] = _make_locked_entry(ot, ot._exit_path, 0, grid)
        elif ot.status in (ot.STATUS_DUMPING, ot.STATUS_REVERSING):
            rem = max(0, ot._dump_ticks_required - ot._dump_ticks)
            locked[ot.id] = _make_locked_entry(ot, [], rem + 2)
        elif ot.status in (ot.STATUS_WAITING, ot.STATUS_ENTERING) and ot._pre_path:
            locked[ot.id] = _make_locked_entry(ot, ot._pre_path, ot._dump_ticks_required + 2, grid)
    return locked


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
    # Per-truck cooldown (ticks) before the gate may re-post a pre-planning task.
    # Prevents rapid-fire retries after staging failures or stale path cancellations.
    _gate_cooldowns: dict = {}

    tick       = 0
    done       = False
    top_candidates: list = []

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
                _newly_assigned_ids = set()
                for tid, (path, dp, sp) in res['assignments'].items():
                    t = truck_map.get(tid)
                    _in_flight.discard(tid)
                    if t is None:
                        continue
                    if t.status != t.STATUS_IDLE:
                        if dp:
                            grid.unreserve(*dp)
                        continue
                    if dp and grid.state[dp[0], dp[1]] in (
                            grid_map.CellState.RESERVED, grid_map.CellState.FILLED):
                        continue
                    t.set_path(path, dp, grid, staging_pose=sp)
                    _newly_assigned_ids.add(tid)

                # Trigger replan for all trucks already inside the polygon.
                # They must now avoid the newly-entered truck's path.
                if _newly_assigned_ids:
                    _replan_trucks = [
                        t for t in trucks
                        if t.status in (t.STATUS_NAVIGATING, t.STATUS_EXITING)
                        and t.id not in _newly_assigned_ids
                        and t.id not in _in_flight
                    ]
                    if _replan_trucks:
                        _replan_ids   = {t.id for t in _replan_trucks}
                        _locked_replan = {}
                        for ot in trucks:
                            if ot.id in _replan_ids:
                                continue   # these trucks are being replanned
                            if ot.status == ot.STATUS_NAVIGATING and ot.path:
                                _locked_replan[ot.id] = _make_locked_entry(
                                    ot, ot.path, ot._dump_ticks_required + 2, grid)
                            elif ot.status == ot.STATUS_EXITING and ot._exit_path:
                                _locked_replan[ot.id] = _make_locked_entry(
                                    ot, ot._exit_path, 0, grid)
                            elif ot.status in (ot.STATUS_DUMPING, ot.STATUS_REVERSING):
                                rem = max(0, ot._dump_ticks_required - ot._dump_ticks)
                                _locked_replan[ot.id] = _make_locked_entry(
                                    ot, [], rem + 2)
                            elif (ot.status in (ot.STATUS_WAITING, ot.STATUS_ENTERING)
                                  and ot._pre_path):
                                _locked_replan[ot.id] = _make_locked_entry(
                                    ot, ot._pre_path, ot._dump_ticks_required + 2, grid)

                        g_snap     = _make_grid_snapshot(grid)
                        nav_snaps  = [(_make_truck_snapshot(t), t.dump_target)
                                      for t in _replan_trucks
                                      if t.status == t.STATUS_NAVIGATING]
                        exit_snaps = [_make_truck_snapshot(t)
                                      for t in _replan_trucks
                                      if t.status == t.STATUS_EXITING]

                        _work_q.put({
                            'type':            'replan',
                            'grid_snap':       g_snap,
                            'entry_rc':        entry_rc,
                            'nav_assignments': nav_snaps,
                            'exit_trucks':     exit_snaps,
                            'locked_paths':    _locked_replan,
                        })
                        for t in _replan_trucks:
                            _in_flight.add(t.id)
                        print(f"[REPLAN] New truck(s) {sorted(_newly_assigned_ids)} entered — "
                              f"triggered replan for {sorted(_replan_ids)}")

                if res.get('sim_done'):
                    print(f"\nSimulation complete at tick {tick}!")
                    done = True

            elif res['type'] == 'exit':
                _newly_exit_ids = set()
                for tid, path in res['paths'].items():
                    t = truck_map.get(tid)
                    _in_flight.discard(tid)
                    if t is None or not t.needs_exit_path():
                        continue
                    if path:
                        t.set_exit_path(path, grid)
                        _newly_exit_ids.add(tid)
                    elif tid not in _in_flight:
                        # CBS failed — post async escape task rather than blocking Thread 1
                        g_snap    = _make_grid_snapshot(grid)
                        t_snap    = _make_truck_snapshot(t)
                        all_snaps = [_make_truck_snapshot(x) for x in trucks]
                        _work_q.put({'type':       'exit_escape',
                                     'grid_snap':   g_snap,
                                     'entry_rc':    entry_rc,
                                     'truck_snap':  t_snap,
                                     'all_trucks':  all_snaps})
                        _in_flight.add(tid)

                # Full replan for all navigating trucks — they must avoid the new exit paths
                if _newly_exit_ids:
                    _nav_replan = [
                        ot for ot in trucks
                        if ot.status == ot.STATUS_NAVIGATING
                        and ot.id not in _newly_exit_ids
                        and ot.id not in _in_flight
                    ]
                    if _nav_replan:
                        _nav_replan_ids = {ot.id for ot in _nav_replan}
                        _locked_nav = _build_locked_for_replan(trucks, _nav_replan_ids, grid)
                        g_snap = _make_grid_snapshot(grid)
                        _work_q.put({
                            'type':            'replan',
                            'grid_snap':       g_snap,
                            'entry_rc':        entry_rc,
                            'nav_assignments': [(_make_truck_snapshot(ot), ot.dump_target)
                                                for ot in _nav_replan],
                            'exit_trucks':     [],
                            'locked_paths':    _locked_nav,
                        })
                        for ot in _nav_replan:
                            _in_flight.add(ot.id)
                        print(f"[EXIT] Trucks {sorted(_newly_exit_ids)} got exit paths — "
                              f"triggered full nav replan for {sorted(_nav_replan_ids)}")

            elif res['type'] == 'exit_escape':
                tid  = res['truck_id']
                t    = truck_map.get(tid)
                _in_flight.discard(tid)
                if t is None or not t.needs_exit_path():
                    continue
                t.set_exit_path(res.get('path', []), grid)

            elif res['type'] == 'replan':
                for tid, (path, sp) in res.get('nav_paths', {}).items():
                    t = truck_map.get(tid)
                    _in_flight.discard(tid)
                    if t is None or t.status != t.STATUS_NAVIGATING:
                        print(f"[REPLAN] T{tid} nav result skipped "
                              f"(status={t.status if t else 'gone'})")
                        continue
                    if path:
                        # Trim to truck's current position — the snapshot was taken
                        # ticks ago; applying from path[0] would cause backward movement.
                        trimmed = _advance_path_to_current(path, t)
                        t.clear_all_corridors(grid)
                        t.set_path(trimmed, t.dump_target, grid, staging_pose=sp)
                        print(f"[REPLAN] T{tid}: nav path updated "
                              f"({len(trimmed)}/{len(path)} waypoints after trim)")
                    else:
                        print(f"[REPLAN] T{tid}: replan returned empty nav path, "
                              f"keeping old path")
                for tid, path in res.get('exit_paths', {}).items():
                    t = truck_map.get(tid)
                    _in_flight.discard(tid)
                    if t is None or t.status != t.STATUS_EXITING:
                        print(f"[REPLAN] T{tid} exit result skipped "
                              f"(status={t.status if t else 'gone'})")
                        continue
                    if path:
                        # Same trim: discard waypoints already behind the truck.
                        trimmed = _advance_path_to_current(path, t)
                        t.clear_all_corridors(grid)
                        t.set_exit_path(trimmed, grid)
                        print(f"[REPLAN] T{tid}: exit path updated "
                              f"({len(trimmed)}/{len(path)} waypoints after trim)")
                    else:
                        print(f"[REPLAN] T{tid}: replan returned empty exit path, "
                              f"keeping old path")

            elif res['type'] == 'gate':
                tid = res['truck_id']
                t   = truck_map.get(tid)
                _in_flight.discard(tid)
                if t is None or t.status != t.STATUS_WAITING:
                    continue
                result = res['result']
                if result is None:
                    _gate_cooldowns[tid] = 5  # staging had no candidates — wait before retrying
                    continue
                path, dp, sp = result
                if dp and grid.state[dp[0], dp[1]] in (
                        grid_map.CellState.RESERVED, grid_map.CellState.FILLED):
                    continue
                t.preload_dump_path(path, dp, grid, sp)
                print(f"[GATE] Pre-planned path applied for T{tid}")

                # Full replan for all trucks inside — route around the incoming truck's path
                _inside_replan = [
                    ot for ot in trucks
                    if ot.status in (ot.STATUS_NAVIGATING, ot.STATUS_EXITING)
                    and ot.id != tid
                    and ot.id not in _in_flight
                ]
                if _inside_replan:
                    _inside_replan_ids = {ot.id for ot in _inside_replan}
                    _locked_gate = _build_locked_for_replan(trucks, _inside_replan_ids, grid)
                    # Add the incoming truck's pre-path as a hard constraint
                    _locked_gate[tid] = _make_locked_entry(t, path, t._dump_ticks_required + 2, grid)
                    g_snap = _make_grid_snapshot(grid)
                    _work_q.put({
                        'type':            'replan',
                        'grid_snap':       g_snap,
                        'entry_rc':        entry_rc,
                        'nav_assignments': [(_make_truck_snapshot(ot), ot.dump_target)
                                            for ot in _inside_replan
                                            if ot.status == ot.STATUS_NAVIGATING],
                        'exit_trucks':     [_make_truck_snapshot(ot)
                                            for ot in _inside_replan
                                            if ot.status == ot.STATUS_EXITING],
                        'locked_paths':    _locked_gate,
                    })
                    for ot in _inside_replan:
                        _in_flight.add(ot.id)
                    print(f"[GATE] T{tid} pre-path applied — "
                          f"triggered full replan for {sorted(_inside_replan_ids)}")

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
                    and next_t.id not in _in_flight
                    and not _gate_cooldowns.get(next_t.id, 0)):
                g_snap  = _make_grid_snapshot(grid)
                t_snap  = _make_truck_snapshot(next_t)
                t_snap.pos     = list(ENTRY_POINT)
                t_snap.heading = math.pi / 2

                locked = {}
                for ot in _trucks_inside:
                    if ot.status == ot.STATUS_NAVIGATING and ot.path:
                        locked[ot.id] = _make_locked_entry(ot, ot.path,
                                                            ot._dump_ticks_required + 2, grid)
                    elif ot.status == ot.STATUS_EXITING and ot._exit_path:
                        locked[ot.id] = _make_locked_entry(ot, ot._exit_path, 0, grid)
                    elif ot.status == ot.STATUS_EXITING:
                        # Stuck exiting with no path — lock current pose for a generous tail.
                        locked[ot.id] = _make_locked_entry(ot, [], 30)
                    else:
                        rem = max(0, ot._dump_ticks_required - ot._dump_ticks)
                        locked[ot.id] = _make_locked_entry(ot, [], rem + 2)

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
                        # Truck has no active path (dumping/reversing/stuck waiting for replan).
                        # Skip it — its position was locked in the CBS planner, so the pre-path
                        # already routes around it.  Using the expanded corridor here is too
                        # conservative and creates an infinite stale-cancel loop.
                        continue
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

            nav_locked = {}
            need_exit_ids = {t.id for t in need_exit}
            for ot in trucks:
                if ot.id in need_exit_ids:
                    continue
                if ot.status == ot.STATUS_NAVIGATING and ot.path:
                    nav_locked[ot.id] = _make_locked_entry(ot, ot.path,
                                                            ot._dump_ticks_required + 2, grid)
                elif ot.status == ot.STATUS_EXITING and ot._exit_path:
                    nav_locked[ot.id] = _make_locked_entry(ot, ot._exit_path, 0, grid)
                elif ot.status in (ot.STATUS_WAITING, ot.STATUS_ENTERING) and ot._pre_path:
                    nav_locked[ot.id] = _make_locked_entry(ot, ot._pre_path,
                                                            ot._dump_ticks_required + 2, grid)
                elif ot.status in (ot.STATUS_DUMPING, ot.STATUS_REVERSING,
                                   ot.STATUS_EXITING, ot.STATUS_NAVIGATING):
                    rem = max(0, ot._dump_ticks_required - ot._dump_ticks)
                    tail = rem + 2 if ot.status == ot.STATUS_DUMPING else 30
                    nav_locked[ot.id] = _make_locked_entry(ot, [], tail)

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
                    all_locked[t.id] = _make_locked_entry(t, t.path,
                                                           t._dump_ticks_required + 2, grid)
                elif t.status == t.STATUS_EXITING and t._exit_path:
                    all_locked[t.id] = _make_locked_entry(t, t._exit_path, 0, grid)
                elif t.status in (t.STATUS_DUMPING, t.STATUS_REVERSING,
                                  t.STATUS_EXITING, t.STATUS_NAVIGATING):
                    rem = max(0, t._dump_ticks_required - t._dump_ticks)
                    tail = rem + 2 if t.status == t.STATUS_DUMPING else 30
                    all_locked[t.id] = _make_locked_entry(t, [], tail)

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

                # Physical collision guard — exact SAT rectangle overlap.
                # Corridor-set overlap was too conservative: paths can share cells
                # at different times, which is fine.  SAT only fires when bodies
                # actually intersect, so trucks on non-conflicting paths can pass
                # by each other without false stops.
                collided = False
                _ex, _ey = ENTRY_POINT
                for other in active_trucks:
                    if other is truck:
                        continue
                    dist = math.hypot(truck.pos[0] - other.pos[0],
                                      truck.pos[1] - other.pos[1])
                    if dist < (truck.length + other.length) * 0.5:
                        # Entry corridor is a natural funnel — all trucks must pass
                        # through the same point.  Skip SAT when either truck is
                        # within the corridor radius, mirroring what CBS conflict
                        # detection already does.  Without this, an IDLE truck at
                        # the entry permanently blocks any EXITING truck from snapping
                        # home (STUCK-EXIT infinite loop).
                        if (math.hypot(truck.pos[0] - _ex, truck.pos[1] - _ey)
                                <= corridor_clearance_m or
                                math.hypot(other.pos[0] - _ex, other.pos[1] - _ey)
                                <= corridor_clearance_m):
                            continue
                        if not _rect_overlap_2d(
                                truck.pos[0], truck.pos[1], truck.heading,
                                truck.length / 2, truck.width / 2,
                                other.pos[0], other.pos[1], other.heading,
                                other.length / 2, other.width / 2):
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

        # ── FUTURE PATH CONFLICT DETECTION ───────────────────────────────────
        # Sample each active truck's remaining path at cell-size intervals and
        # check for future footprint overlaps.  When a conflict is found, both
        # trucks are immediately submitted for a full replan.
        _fp_paths = {}
        _fp_truck_map = {}
        for _ft in trucks:
            if _ft.id in _in_flight:
                continue
            if _ft.status == _ft.STATUS_NAVIGATING and _ft.path:
                _cp = _world_path_to_cell_path(_ft.path, grid)
                if _cp:
                    _fp_paths[_ft.id] = _cp
                    _fp_truck_map[_ft.id] = _ft
            elif _ft.status == _ft.STATUS_EXITING and _ft._exit_path:
                _cp = _world_path_to_cell_path(_ft._exit_path, grid)
                if _cp:
                    _fp_paths[_ft.id] = _cp
                    _fp_truck_map[_ft.id] = _ft

        if len(_fp_paths) >= 2:
            _fc = _detect_first_conflict(_fp_paths, truck_map=_fp_truck_map, grid=grid)
            if _fc is not None:
                _fc_ai, _fc_aj, _fc_t = _fc[1], _fc[2], _fc[-1]
                if _fc_ai not in _in_flight and _fc_aj not in _in_flight:
                    _conflict_ids = {_fc_ai, _fc_aj}
                    _locked_fc = _build_locked_for_replan(trucks, _conflict_ids, grid)
                    g_snap = _make_grid_snapshot(grid)
                    _fc_nav  = [(_make_truck_snapshot(_ft), _ft.dump_target)
                                for _ft in trucks
                                if _ft.id in _conflict_ids
                                and _ft.status == _ft.STATUS_NAVIGATING]
                    _fc_exit = [_make_truck_snapshot(_ft)
                                for _ft in trucks
                                if _ft.id in _conflict_ids
                                and _ft.status == _ft.STATUS_EXITING]
                    if _fc_nav or _fc_exit:
                        _work_q.put({
                            'type':            'replan',
                            'grid_snap':       g_snap,
                            'entry_rc':        entry_rc,
                            'nav_assignments': _fc_nav,
                            'exit_trucks':     _fc_exit,
                            'locked_paths':    _locked_fc,
                        })
                        for _cid in _conflict_ids:
                            _in_flight.add(_cid)
                        # Stop conflicting trucks in place so they don't advance
                        # further toward the collision while waiting for the replan.
                        for _ft in trucks:
                            if _ft.id not in _conflict_ids:
                                continue
                            if _ft.status == _ft.STATUS_NAVIGATING:
                                _ft.path = []
                            elif _ft.status == _ft.STATUS_EXITING:
                                _ft._exit_path = []
                        print(f"[CONFLICT-DETECT] Future {_fc[0]} conflict T{_fc_ai}↔T{_fc_aj} "
                              f"at t={_fc_t} — stopped both, instant replan queued")

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

            # Solo-stuck EXITING trucks have no handler in the pair-deadlock or
            # stuck_nav checks above. Clear the exit path so needs_exit_path()
            # fires next tick and the async worker replans a clear route.
            stuck_exit = [t for t in trucks
                          if t._stuck_substeps >= _STUCK_LIMIT
                          and t.status == t.STATUS_EXITING
                          and t._exit_path]
            for _st in stuck_exit:
                print(f"[STUCK-EXIT] T{_st.id} stuck while exiting — clearing path for replan.")
                _st._stuck_substeps    = 0
                _st._conflict_cooldown = 3
                _st._clear_exit_corridor(grid)
                _st._exit_path = []
                _in_flight.discard(_st.id)

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

        # Tick down cooldowns used by headlock/deadlock yield maneuvers.
        # Trucks with cooldown > 0 are excluded from headlock re-detection so
        # a fresh yield maneuver isn't triggered before the previous one finishes.
        for t in trucks:
            if t._conflict_cooldown > 0:
                t._conflict_cooldown -= 1

        # Tick down gate pre-planning cooldowns (rate-limits staging retries).
        for _gid in list(_gate_cooldowns):
            if _gate_cooldowns[_gid] > 0:
                _gate_cooldowns[_gid] -= 1
            else:
                del _gate_cooldowns[_gid]

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
