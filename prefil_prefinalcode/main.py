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
from conflict_detect import (_detect_physical_conflict,
                             _keeper_forbidden_states)
from renderer import Renderer

from planning_worker import (_make_grid_snapshot, _make_truck_snapshot,
                             _planning_worker)
from sim_helpers import (initialise_prefill, _corridor_cells,
                         build_fleet)


def run_simulation():
    print("Initialising grid...")
    grid = grid_map.GridMap(POLYGON_BOUNDARY, CELL_SIZE)
    initialise_prefill(grid)

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
                if not res['assignments'] and not res.get('sim_done'):
                    print(f"[PLAN-FAIL] idle task returned 0 assignments "
                          f"(idle trucks had no reachable dump candidates)")
                for tid, (path, dp, sp) in res['assignments'].items():
                    t = truck_map.get(tid)
                    _in_flight.discard(tid)
                    if not path:
                        print(f"[PLAN-FAIL] T{tid} idle planner returned empty path "
                              f"(dp={dp})")
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

                if res.get('sim_done'):
                    print(f"\nSimulation complete at tick {tick}!")
                    done = True

            elif res['type'] == 'exit':
                for tid, path in res['paths'].items():
                    t = truck_map.get(tid)
                    _in_flight.discard(tid)
                    if not path:
                        _et = truck_map.get(tid)
                        _epos = [round(p, 1) for p in _et.pos] if _et else '?'
                        print(f"[PLAN-FAIL] T{tid} exit planner returned empty path "
                              f"(pos={_epos}) — posting escape task")
                    if t is None or not t.needs_exit_path():
                        continue
                    if path:
                        t.set_exit_path(path, grid)
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

            elif res['type'] == 'exit_escape':
                tid  = res['truck_id']
                t    = truck_map.get(tid)
                _in_flight.discard(tid)
                if t is None or not t.needs_exit_path():
                    continue
                t.set_exit_path(res.get('path', []), grid)

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
                        locked[ot.id] = ([ot.front_center_cell(grid)] + list(ot.path),
                                         ot._dump_ticks_required + 2)
                    elif ot.status == ot.STATUS_EXITING and ot._exit_path:
                        locked[ot.id] = ([ot.front_center_cell(grid)] + list(ot._exit_path), 0)
                    elif ot.status == ot.STATUS_EXITING:
                        # Stuck exiting with no path — lock current cell for a generous tail
                        # (rem+2 gives only ~2 ticks for a truck that finished dumping; 30 is safer)
                        locked[ot.id] = ([ot.front_center_cell(grid)], 30)
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
                    nav_locked[ot.id] = ([ot.front_center_cell(grid)] + list(ot.path),
                                          ot._dump_ticks_required + 2)
                elif ot.status == ot.STATUS_EXITING and ot._exit_path:
                    nav_locked[ot.id] = ([ot.front_center_cell(grid)] + list(ot._exit_path), 0)
                elif ot.status in (ot.STATUS_WAITING, ot.STATUS_ENTERING) and ot._pre_path:
                    nav_locked[ot.id] = (ot._pre_path, ot._dump_ticks_required + 2)
                elif ot.status in (ot.STATUS_DUMPING, ot.STATUS_REVERSING,
                                   ot.STATUS_EXITING, ot.STATUS_NAVIGATING):
                    # Truck is stopped in place (dumping, reversing, or stuck/waiting
                    # for replan with no path) — lock current cell for remaining dwell.
                    rem = max(0, ot._dump_ticks_required - ot._dump_ticks)
                    tail = rem + 2 if ot.status == ot.STATUS_DUMPING else 30
                    nav_locked[ot.id] = ([ot.front_center_cell(grid)], tail)

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
                elif t.status in (t.STATUS_DUMPING, t.STATUS_REVERSING,
                                  t.STATUS_EXITING, t.STATUS_NAVIGATING):
                    # Truck is stopped in place — lock current cell for remaining dwell.
                    rem = max(0, t._dump_ticks_required - t._dump_ticks)
                    tail = rem + 2 if t.status == t.STATUS_DUMPING else 30
                    all_locked[t.id] = ([t.front_center_cell(grid)], tail)

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

        # ── PROACTIVE CORRIDOR CONFLICT DETECTION ────────────────────────────
        # Every 5 ticks scan all active truck path pairs for physical body
        # overlap before it happens. Lower-priority truck drops its path and
        # replans from current position so CBS can route around the other.
        # Priority: NAVIGATING (entering) > EXITING; then shorter path wins.
        active_trucks  = [t for t in trucks if t.status != t.STATUS_WAITING]
        # Normalise a continuous-world path to one unique cell per step so that
        # hybrid-A* paths (4 waypoints/m) and astar paths (1 waypoint/m) share
        # the same time index when compared in _detect_physical_conflict.
        def _norm_path(raw_wps):
            out = []
            for _wp in raw_wps:
                _rc = grid.world_to_cell(_wp[0], _wp[1])
                if not out or _rc != out[-1]:
                    out.append(_rc)
            return out

        _PROACTIVE_INTERVAL = 5
        if tick % _PROACTIVE_INTERVAL == 0:
            _seg = {}
            _seg_trucks = {}
            for _pt in active_trucks:
                if _pt.id in _in_flight:
                    continue
                if _pt.status == _pt.STATUS_NAVIGATING and _pt.path:
                    _seg[_pt.id] = _norm_path(
                        [(_pt.pos[0], _pt.pos[1], _pt.heading)] + list(_pt.path))
                    _seg_trucks[_pt.id] = _pt
                elif _pt.status == _pt.STATUS_EXITING and _pt._exit_path:
                    _seg[_pt.id] = _norm_path(
                        [(_pt.pos[0], _pt.pos[1], _pt.heading)] + list(_pt._exit_path))
                    _seg_trucks[_pt.id] = _pt

            if len(_seg) >= 2:
                _pro_conflict = _detect_physical_conflict(_seg, _seg_trucks, grid)
                if _pro_conflict:
                    _cai, _caj, _ct = _pro_conflict
                    _pta = truck_map[_cai]
                    _ptb = truck_map[_caj]

                    def _priority(t):
                        # Exiting truck was assigned earlier (already dumped) → higher priority.
                        # Among same-status trucks: shorter remaining path = closer to goal = wins.
                        exiting = 1 if t.status == t.STATUS_EXITING else 0
                        plen    = (len(t.path) if t.status == t.STATUS_NAVIGATING
                                   else len(t._exit_path))
                        return (exiting, -plen)

                    if _priority(_pta) >= _priority(_ptb):
                        _keeper, _rep = _pta, _ptb
                    else:
                        _keeper, _rep = _ptb, _pta

                    print(f"[PROACTIVE] T{_cai}↔T{_caj} corridor conflict at t={_ct}. "
                          f"T{_keeper.id} keeps path → T{_rep.id} replans.")

                    # Build keeper's body-expanded forbidden states so the
                    # replanning truck cannot path through any cell where its
                    # body would physically touch the keeper at the same timestep.
                    _keeper_norm = _seg.get(_keeper.id)
                    if not _keeper_norm:
                        _kp = (_keeper.path if _keeper.status == _keeper.STATUS_NAVIGATING
                               else _keeper._exit_path)
                        _keeper_norm = _norm_path(
                            [(_keeper.pos[0], _keeper.pos[1], _keeper.heading)]
                            + list(_kp))
                    _forbidden = _keeper_forbidden_states(grid, _keeper_norm,
                                                          _keeper, _rep)

                    # Build locked_paths from all other trucks (same as idle/exit loops).
                    _imm_locked = {}
                    for _lt in trucks:
                        if _lt.id == _rep.id:
                            continue
                        if _lt.status == _lt.STATUS_NAVIGATING and _lt.path:
                            _imm_locked[_lt.id] = (
                                [_lt.front_center_cell(grid)] + list(_lt.path),
                                _lt._dump_ticks_required + 2)
                        elif _lt.status == _lt.STATUS_EXITING and _lt._exit_path:
                            _imm_locked[_lt.id] = (
                                [_lt.front_center_cell(grid)] + list(_lt._exit_path), 0)
                        elif _lt.status in (_lt.STATUS_DUMPING, _lt.STATUS_REVERSING):
                            _rem = max(0, _lt._dump_ticks_required - _lt._dump_ticks)
                            _tail = _rem + 2 if _lt.status == _lt.STATUS_DUMPING else 30
                            _imm_locked[_lt.id] = ([_lt.front_center_cell(grid)], _tail)

                    _rep._conflict_cooldown = max(_rep._conflict_cooldown, 8)
                    _in_flight.discard(_rep.id)

                    if _rep.status == _rep.STATUS_NAVIGATING:
                        _rep.clear_all_corridors(grid)
                        _rep.cancel_preload(grid)
                        if _rep.dump_target:
                            grid.unreserve(*_rep.dump_target)
                            _rep.dump_target = None
                        _rep.path         = []
                        _rep.stop_target  = None
                        _rep.staging_pose = None
                        _rep.status       = _rep.STATUS_IDLE

                        _claimed = {t.dump_target for t in trucks
                                    if t.dump_target and t.id != _rep.id}
                        _work_q.put({
                            'type':                'idle',
                            'grid_snap':           _make_grid_snapshot(grid),
                            'entry_rc':            entry_rc,
                            'idle_trucks':         [_make_truck_snapshot(_rep)],
                            'locked_paths':        _imm_locked,
                            'claimed_targets':     _claimed,
                            'extra_st_constraints': _forbidden,
                        })
                        _in_flight.add(_rep.id)

                    elif _rep.status == _rep.STATUS_EXITING:
                        _rep._clear_exit_corridor(grid)
                        _rep._exit_path = []

                        _work_q.put({
                            'type':                'exit',
                            'grid_snap':           _make_grid_snapshot(grid),
                            'entry_rc':            entry_rc,
                            'exit_trucks':         [_make_truck_snapshot(_rep)],
                            'locked_paths':        _imm_locked,
                            'extra_st_constraints': _forbidden,
                        })
                        _in_flight.add(_rep.id)

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
                truck.step(grid)
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

        # ── STAGNATION UPDATE ─────────────────────────────────────────────────
        for _t in active_trucks:
            if _t.status not in (_t.STATUS_NAVIGATING, _t.STATUS_EXITING):
                _t._pos_stagnant_ticks = 0
                continue
            _net = math.hypot(_t.pos[0] - _t._pos_snapshot[0],
                              _t.pos[1] - _t._pos_snapshot[1])
            if _net < 0.5:
                _t._pos_stagnant_ticks += 1
                if _t._pos_stagnant_ticks % 3 == 0:
                    _sr, _sc = grid.world_to_cell(_t.pos[0], _t.pos[1])
                    _szh = grid.z_height[_sr, _sc]
                    _sst = grid_map.CellState(grid.state[_sr, _sc]).name
                    _nxt = (_t.path[0] if _t.path
                            else (_t._exit_path[0] if _t._exit_path else None))
                    _nxt_s = ""
                    if _nxt is not None:
                        if len(_nxt) == 2:
                            _nxt_s = (f" | next_wp=({_nxt[0]},{_nxt[1]})"
                                      f" z={grid.z_height[_nxt[0],_nxt[1]]:.2f}"
                                      f" st={grid_map.CellState(grid.state[_nxt[0],_nxt[1]]).name}")
                        else:
                            _nxt_s = f" | next_wp(world)={_nxt}"
                    print(f"[STAGNANT] T{_t.id} ({_t.status}) "
                          f"x{_t._pos_stagnant_ticks}t "
                          f"pos={[round(p,1) for p in _t.pos]} "
                          f"cell=({_sr},{_sc}) z={_szh:.2f} st={_sst} "
                          f"path={len(_t.path)}wp exit={len(_t._exit_path)}wp "
                          f"inflight={_t.id in _in_flight} cool={_t._conflict_cooldown}"
                          f"{_nxt_s}")
            else:
                _t._pos_stagnant_ticks = 0

        # ── FREEZE REPLAN ─────────────────────────────────────────────────────
        # Any truck that hasn't moved for _FREEZE_TICKS consecutive ticks gets
        # a full reset to IDLE and replans from scratch.  Since trucks pass
        # through each other, stagnation is always caused by an internal issue
        # (bad path, planner failure, unreachable goal) — not proximity.
        _FREEZE_TICKS = 3
        for _gt in list(active_trucks):
            if _gt.status not in (_gt.STATUS_NAVIGATING, _gt.STATUS_EXITING):
                continue
            if _gt._pos_stagnant_ticks < _FREEZE_TICKS:
                continue
            if _gt._conflict_cooldown > 0:
                continue
            if _gt.id in _in_flight:
                continue   # already being replanned

            _gr, _gc = grid.world_to_cell(_gt.pos[0], _gt.pos[1])
            _gzh = grid.z_height[_gr, _gc]
            _gst = grid_map.CellState(grid.state[_gr, _gc]).name
            print(f"[REPLAN] T{_gt.id} ({_gt.status}) frozen {_gt._pos_stagnant_ticks}t → full replan")
            print(f"  pos={[round(p,1) for p in _gt.pos]} "
                  f"hdg={math.degrees(_gt.heading):.0f}° "
                  f"cell=({_gr},{_gc}) z={_gzh:.2f} state={_gst}")
            print(f"  path={len(_gt.path)}wp  exit={len(_gt._exit_path)}wp  "
                  f"dump_target={_gt.dump_target}  "
                  f"staging={'yes' if _gt.staging_pose else 'no'}")
            for _label, _wps in [("path", _gt.path), ("exit", _gt._exit_path)]:
                if _wps:
                    _wp = _wps[0]
                    if len(_wp) == 2:
                        print(f"  first_{_label}_wp=({_wp[0]},{_wp[1]}) "
                              f"z={grid.z_height[_wp[0],_wp[1]]:.2f} "
                              f"state={grid_map.CellState(grid.state[_wp[0],_wp[1]]).name}")
                    else:
                        print(f"  first_{_label}_wp(world)={_wp}")

            _gt._pos_stagnant_ticks = 0
            _gt._stuck_substeps     = 0
            _gt._conflict_cooldown  = 10
            _in_flight.discard(_gt.id)

            # Teleport to entry point and reset to IDLE — clean slate replan from start
            _gt.clear_all_corridors(grid)
            _gt.cancel_preload(grid)
            if _gt.dump_target:
                grid.unreserve(*_gt.dump_target)
                _gt.dump_target  = None
            _gt.path         = []
            _gt._exit_path   = []
            _gt.stop_target  = None
            _gt.staging_pose = None
            _gt.pos[0], _gt.pos[1] = ENTRY_POINT[0], ENTRY_POINT[1]
            _gt.heading      = math.pi / 2
            _gt.status       = _gt.STATUS_IDLE
            print(f"  → teleported T{_gt.id} to entry point, reset to IDLE")

        # Tick down cooldowns so recently-replanned trucks aren't immediately reset again.
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
