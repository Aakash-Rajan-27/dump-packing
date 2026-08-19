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

import os
import sys
import time
import math
import random
import threading
import queue   as _queue
import numpy as np
from scipy.ndimage import gaussian_filter

sys.stdout.reconfigure(encoding='utf-8')

from config import (POLYGON_BOUNDARY, ENTRY_POINT, CELL_SIZE,
                    FLEET_COMPOSITION, TICK_DELAY, PYGAME_SCALE,
                    PHEROMONE_DECAY, PHEROMONE_SPREAD_SIGMA,
                    STEPS_PER_TICK, ENTRY_CORRIDOR_CELLS,
                    ALLOW_COLLISION_BYPASS, METRICS_DIR, METRICS_TICK_EVERY)
# NOTE: imported under an alias — `metrics` is already used below as a
# local variable name for the renderer HUD dict.
import metrics as metrics_sink
import grid_map
from pathfinder import (generate_reverse_retreat, generate_yield_maneuver,
                        generate_deadlock_escape, _make_locked_entry)
from conflict_detect import _rect_overlap_2d, _detect_first_conflict
from renderer import Renderer

from planning_worker import (_make_grid_snapshot, _make_truck_snapshot,
                             _planning_worker, SyncWorkQueue)
from sim_helpers import (initialise_half_full_dump, _corridor_cells,
                         build_fleet, _try_inplace_replan)


class _NullRenderer:
    """Headless stub — no window, no events, never quits."""
    def check_quit(self): return False
    def draw(self, trucks, metrics): pass
    def close(self): pass


def run_simulation(headless=False, max_ticks=0, seed=None, metrics_out=None,
                   allow_collision_bypass=None, sync_planner=False,
                   fleet=None):
    """
    seed                    — seeds random/np.random.  Necessary but NOT
                              sufficient for reproducibility: the background
                              planner thread still lands results on
                              scheduling-dependent ticks.  Pair with
                              sync_planner=True for an exactly repeatable run.
    metrics_out             — path to a .jsonl run log (None = no logging).
    allow_collision_bypass  — override config.ALLOW_COLLISION_BYPASS.  When
                              True the post-step SAT guard COUNTS body overlaps
                              instead of rolling them back (benchmark only).
    sync_planner            — run planning inline instead of on Thread 2.
                              Deterministic; use for regression/equivalence
                              checks, not for performance numbers.
    fleet                   — optional (small, medium, large) override of
                              config.FLEET_COMPOSITION, for the scalability
                              sweep (intrusion rate vs truck count).
    """
    if fleet is not None:
        # Mutate in place — sim_helpers imported the same dict object.
        FLEET_COMPOSITION.update({'small':  fleet[0],
                                  'medium': fleet[1],
                                  'large':  fleet[2]})
        print(f"Fleet override: {fleet[0]}S/{fleet[1]}M/{fleet[2]}L")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        print(f"Seeded RNG with {seed}")

    bypass = (ALLOW_COLLISION_BYPASS if allow_collision_bypass is None
              else allow_collision_bypass)
    if bypass:
        print("[METRICS] COLLISION BYPASS ON — truck bodies may overlap; "
              "intrusions will be counted, not prevented.")

    if metrics_out is None and os.environ.get('DUMP_METRICS_OUT'):
        metrics_out = os.environ['DUMP_METRICS_OUT']

    sink = metrics_sink.MetricsSink(
        path=metrics_out,
        run_id=f"seed{seed}_bypass{int(bool(bypass))}",
        config_snapshot={'seed': seed, 'bypass': bool(bypass),
                         'fleet': dict(FLEET_COMPOSITION),
                         'steps_per_tick': STEPS_PER_TICK,
                         'max_ticks': max_ticks})
    intrusions = metrics_sink.IntrusionTracker(sink)

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

    if headless:
        renderer = _NullRenderer()
        print("Headless mode — starting simulation loop")
    else:
        renderer = Renderer(grid, scale=PYGAME_SCALE)
        print("Renderer ready — starting simulation loop")

    # ── Planning: background thread, or synchronous for reproducibility ───────
    _result_q = _queue.Queue()
    _stop_evt = threading.Event()
    if sync_planner:
        # Deterministic mode — see SyncWorkQueue docstring.  Seeding the RNG
        # alone does NOT make a run reproducible: with the real thread, which
        # tick a plan lands on depends on wall-clock scheduling.
        _work_q  = SyncWorkQueue(_result_q)
        _planner = None
        print("[METRICS] SYNC PLANNER — planning runs inline (reproducible, "
              "not representative of real-time performance).")
    else:
        _work_q  = _queue.Queue()
        _planner = threading.Thread(target=_planning_worker,
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

    # Metric accumulators (see the PACKING / PATH-EFFICIENCY block below).
    _driven:    dict = {}   # truck_id -> metres driven on the current path
    _nav_start: dict = {}   # truck_id -> (x, y) where the current path began
    _dumped_xy: list = []   # world coords of every completed dump, in order

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
                    if t is None or not t.needs_exit_path():
                        continue
                    if path:
                        t.set_exit_path(path, grid)
                        print(f"[EXIT] T{tid}: exit path set ({len(path)} waypoints)")
                    else:
                        print(f"[EXIT] T{tid}: CBS returned empty path — posting escape task")
                    if not path and tid not in _in_flight:
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
                ep = res.get('path', [])
                if ep:
                    print(f"[EXIT-ESC] T{tid}: escape path set ({len(ep)} waypoints)")
                else:
                    print(f"[EXIT-ESC] T{tid}: escape returned empty — T{tid} stuck")
                t.set_exit_path(ep, grid)

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
                        locked[ot.id] = _make_locked_entry(ot, ot.path,
                                                            ot._dump_ticks_required + 2, grid)
                    elif ot.status == ot.STATUS_EXITING and ot._exit_path:
                        locked[ot.id] = _make_locked_entry(ot, ot._exit_path, 0, grid)
                    elif ot.status == ot.STATUS_EXITING:
                        locked[ot.id] = _make_locked_entry(ot, [], 0)
                    else:
                        rem = max(0, ot._dump_ticks_required - ot._dump_ticks)
                        locked[ot.id] = _make_locked_entry(ot, [], rem + 2)
                # Also lock idle trucks physically inside the polygon so the
                # gate pre-plan routes around them (they have no active path).
                for ot in trucks:
                    if ot is next_t or ot.id in locked:
                        continue
                    if ot.status == ot.STATUS_IDLE:
                        locked[ot.id] = _make_locked_entry(ot, [], 5)

                claimed = {t.dump_target for t in trucks if t.dump_target}

                _work_q.put({'type':            'gate',
                             'grid_snap':        g_snap,
                             'entry_rc':         entry_rc,
                             'truck_snap':       t_snap,
                             'locked_paths':     locked,
                             'claimed_targets':  claimed})
                _in_flight.add(next_t.id)
                print(f"[GATE] Posted pre-planning task for T{next_t.id}")

            # NOTE: Stale-check removed. CBS already plans temporally (each truck avoids
            # others at the right time steps). Spatial cell-set comparison was a false
            # positive — paths share cells at *different* times, which is safe — causing
            # immediate cancel-replan loops that prevented trucks from ever entering.

            # Release truck if gate is clear
            any_transiting = any(t.status in (t.STATUS_ENTERING, t.STATUS_LEAVING)
                                 for t in trucks)
            _can_enter = next_t._pre_path or not _trucks_inside
            if _can_enter and not any_transiting:
                entry_x, entry_y = ENTRY_POINT
                # Exclude stuck-exiting trucks (no path) from gate-clear check —
                # they're being replanned and shouldn't block the queue indefinitely.
                active = [t for t in trucks
                          if t.status not in (t.STATUS_WAITING,)
                          and not (t.status == t.STATUS_EXITING and not t._exit_path)]
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
        if tick % 50 == 0:
            exiting = [t for t in trucks if t.status == t.STATUS_EXITING]
            if exiting:
                print(f"[DBG tick={tick}] exiting={[t.id for t in exiting]} "
                      f"need_exit={[t.id for t in need_exit]} "
                      f"in_flight={_in_flight} "
                      f"pos={[(round(t.pos[0],1), round(t.pos[1],1)) for t in exiting]}")
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
                elif ot.status in (ot.STATUS_DUMPING, ot.STATUS_REVERSING):
                    rem = max(0, ot._dump_ticks_required - ot._dump_ticks)
                    tail = rem + 2 if ot.status == ot.STATUS_DUMPING else 5
                    nav_locked[ot.id] = _make_locked_entry(ot, [], tail)
                elif ot.status == ot.STATUS_EXITING:
                    # Stuck-exiting with no path — tail=0 so exiting trucks can plan around it.
                    nav_locked[ot.id] = _make_locked_entry(ot, [], 0)
                elif ot.status == ot.STATUS_NAVIGATING:
                    nav_locked[ot.id] = _make_locked_entry(ot, [], 3)
                elif ot.status == ot.STATUS_IDLE:
                    # Idle truck inside — lock current pos so exit paths route around it.
                    nav_locked[ot.id] = _make_locked_entry(ot, [], 5)

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
                elif t.status in (t.STATUS_DUMPING, t.STATUS_REVERSING):
                    rem  = max(0, t._dump_ticks_required - t._dump_ticks)
                    all_locked[t.id] = _make_locked_entry(t, [], rem + 2)
                elif t.status == t.STATUS_NAVIGATING:
                    # Navigating but no active path — lock briefly.
                    all_locked[t.id] = _make_locked_entry(t, [], 3)
                elif t.status == t.STATUS_EXITING:
                    # Stuck-exiting (no path) — lock ONLY current position (tail=0)
                    # so idle trucks can wait 1 step and route around rather than deadlocking.
                    all_locked[t.id] = _make_locked_entry(t, [], 0)
                elif t.status == t.STATUS_IDLE:
                    # Other idle truck inside — lock its current position so staging
                    # paths route around it rather than planning through its body.
                    all_locked[t.id] = _make_locked_entry(t, [], 5)

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

        # Truck-pairs whose bodies overlapped during ANY substep of this tick.
        # Collected across substeps, resolved into open/closed intrusions once
        # per tick (intrusion durations are measured in ticks, not substeps).
        _overlap_pairs = set()

        # Status/target snapshot for metric transition detection after the
        # substep loop (dump completion, path completion).  Done here rather
        # than inside truck.py so Truck stays untouched.
        _pre_status = {t.id: (t.status, t.dump_target) for t in trucks}

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
                collided   = False
                blocker_id = None
                _ex, _ey = ENTRY_POINT
                for other in active_trucks:
                    if other is truck:
                        continue
                    dist = math.hypot(truck.pos[0] - other.pos[0],
                                      truck.pos[1] - other.pos[1])
                    if dist >= (truck.length + other.length) * 0.5:
                        continue
                    if not _rect_overlap_2d(
                            truck.pos[0], truck.pos[1], truck.heading,
                            truck.length / 2, truck.width / 2,
                            other.pos[0], other.pos[1], other.heading,
                            other.length / 2, other.width / 2):
                        continue

                    # ── Real body overlap. ────────────────────────────────
                    # Recorded for the intrusion metric UNCONDITIONALLY —
                    # including inside the entry corridor.  The corridor
                    # exemption below is a planning tolerance, not a claim
                    # that overlapping bodies there are physically fine, so
                    # it must not silently hide them from the metric.
                    # (in_entry_corridor is logged as a field instead, so
                    # corridor intrusions can be filtered out post-hoc.)
                    _overlap_pairs.add((min(truck.id, other.id),
                                        max(truck.id, other.id)))

                    # Entry corridor is a natural funnel — all trucks must pass
                    # through the same point.  Skip the BLOCK when either truck
                    # is within the corridor radius, mirroring what CBS conflict
                    # detection already does.  Without this, an IDLE truck at
                    # the entry permanently blocks any EXITING truck from snapping
                    # home (STUCK-EXIT infinite loop).
                    if (math.hypot(truck.pos[0] - _ex, truck.pos[1] - _ey)
                            <= corridor_clearance_m or
                            math.hypot(other.pos[0] - _ex, other.pos[1] - _ey)
                            <= corridor_clearance_m):
                        continue

                    collided   = True
                    blocker_id = other.id
                    break

                if collided and not bypass:
                    # ── Normal mode: prevent the overlap (roll the move back).
                    sink.record_collision_block(tick, truck.id, blocker_id)
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
                    # Bypass mode reaches here even when `collided` is True:
                    # the move stands, the bodies overlap, and the overlap is
                    # counted above.  _stuck_substeps stays 0 because the truck
                    # genuinely moved — inflating it would fire the deadlock
                    # maneuvers for a truck that is not actually stuck.
                    truck._stuck_substeps = 0

                # Path-efficiency accounting: distance actually covered this
                # substep (0 when the move was rolled back above).
                _driven[truck.id] = _driven.get(truck.id, 0.0) + math.hypot(
                    truck.pos[0] - prev_x, truck.pos[1] - prev_y)

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
            if headless:
                time.sleep(0.001)   # yield GIL to planning thread
            else:
                time.sleep(substep_delay)

        # ── PACKING / PATH-EFFICIENCY METRICS ─────────────────────────────────
        for _t in trucks:
            _was_status, _was_target = _pre_status[_t.id]

            # Started navigating → begin a fresh driven-distance measurement.
            if (_t.status == _t.STATUS_NAVIGATING
                    and _was_status != _t.STATUS_NAVIGATING):
                _driven[_t.id]   = 0.0
                _nav_start[_t.id] = (_t.pos[0], _t.pos[1])

            # Arrived (NAVIGATING → REVERSING/DUMPING) → log path efficiency.
            if (_was_status == _t.STATUS_NAVIGATING
                    and _t.status in (_t.STATUS_REVERSING, _t.STATUS_DUMPING)
                    and _t.id in _nav_start):
                _sx, _sy = _nav_start.pop(_t.id)
                sink.record_path_completed(
                    tick, _t.id,
                    driven_m=_driven.get(_t.id, 0.0),
                    straight_m=math.hypot(_t.pos[0] - _sx, _t.pos[1] - _sy))

            # Dump finished (DUMPING → EXITING) → log it + spacing to the
            # nearest previously-dumped cell.
            if (_was_status == _t.STATUS_DUMPING
                    and _t.status == _t.STATUS_EXITING
                    and _was_target is not None):
                _dx, _dy = grid.cell_to_world(*_was_target)
                _nn = None
                if _dumped_xy:
                    _nn = min(math.hypot(_dx - _px, _dy - _py)
                              for _px, _py in _dumped_xy)
                _dumped_xy.append((_dx, _dy))
                sink.record_dump(tick, _t.id, _was_target, _nn)

        # ── INTRUSION BOOKKEEPING ─────────────────────────────────────────────
        # Open a new intrusion for each newly-overlapping pair, close any that
        # separated this tick (logged once, with a duration — not once per tick).
        intrusions.update(tick, _overlap_pairs, truck_map,
                          ENTRY_POINT, corridor_clearance_m)

        # ── PHEROMONE UPDATE ──────────────────────────────────────────────────
        grid.pheromone = 1.0 - (1.0 - grid.pheromone) * PHEROMONE_DECAY
        grid.pheromone = gaussian_filter(grid.pheromone, sigma=PHEROMONE_SPREAD_SIGMA)
        np.clip(grid.pheromone, 0.0, 1.0, out=grid.pheromone)

        # ── PROACTIVE CBS CONFLICT DETECTION + REPLAN ────────────────────────
        # Convert active truck paths to cell-space and scan for future conflicts.
        # On the first conflict found, run CBS in-place for the pair so they
        # replan before they ever physically block each other.
        _cbs_paths: dict = {}
        for _t in active_trucks:
            if _t._conflict_cooldown > 0 or _t.id in _in_flight:
                continue
            if _t.status == _t.STATUS_NAVIGATING and _t.path and len(_t.path) > 1:
                _cbs_paths[_t.id] = [grid.world_to_cell(_wp[0], _wp[1])
                                     for _wp in _t.path]
            elif _t.status == _t.STATUS_EXITING and _t._exit_path and len(_t._exit_path) > 1:
                _cbs_paths[_t.id] = [grid.world_to_cell(_wp[0], _wp[1])
                                     for _wp in _t._exit_path]

        if len(_cbs_paths) >= 2:
            _cbs_conflict = _detect_first_conflict(_cbs_paths,
                                                   truck_map=truck_map, grid=grid)
            if _cbs_conflict is not None:
                _cai, _caj = _cbs_conflict[1], _cbs_conflict[2]
                _cta = truck_map.get(_cai)
                _ctb = truck_map.get(_caj)
                if _cta and _ctb:
                    print(f"[CBS-DETECT] Future conflict T{_cai}↔T{_caj} "
                          f"at t={_cbs_conflict[-1]} — replanning.")
                    sink.record_conflict_detected(
                        tick, _cai, _caj, _cbs_conflict[-1], _cbs_conflict[0])
                    _t_replan = time.perf_counter()
                    _ok = _try_inplace_replan(_cta, _ctb, grid, entry_rc, trucks)
                    sink.record_replan(
                        tick, (_cai, _caj), 'reactive_pair',
                        time.perf_counter() - _t_replan, bool(_ok))

        # ── DEADLOCK RESOLUTION (TOP PRIORITY) ───────────────────────────────
        # Two trucks physically jammed: one stays still, the other reverses
        # while arcing ~90° until its body is clear of the blocker's width.
        # Falls back to a straight reverse retreat only if the arc is blocked.
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

                    # Try both orderings; prefer the truck with more remaining
                    # path (it has farther to go, so giving way costs it less).
                    _plen_a = (len(_ta.path) if _ta.status == _ta.STATUS_NAVIGATING
                               else len(_ta._exit_path))
                    _plen_b = (len(_tb.path) if _tb.status == _tb.STATUS_NAVIGATING
                               else len(_tb._exit_path))
                    _candidates = (
                        [(_ta, _tb), (_tb, _ta)] if _plen_a >= _plen_b
                        else [(_tb, _ta), (_ta, _tb)]
                    )

                    _esc_wps   = []
                    _reverser  = None
                    _advancer  = None
                    for _rev_cand, _adv_cand in _candidates:
                        _wps = generate_deadlock_escape(_rev_cand, grid, _adv_cand)
                        if _wps:
                            _esc_wps  = _wps
                            _reverser = _rev_cand
                            _advancer = _adv_cand
                            break

                    if _reverser is not None:
                        # ── 90° reverse-arc escape ────────────────────────────
                        print(f"[DEADLOCK] T{_ta.id}↔T{_tb.id}: "
                              f"T{_reverser.id} executing 90° reverse escape "
                              f"({len(_esc_wps)} steps).")
                        sink.record_deadlock(tick, _ta.id, _tb.id,
                                             'reverse_arc', len(_esc_wps))
                        if _reverser.status == _reverser.STATUS_NAVIGATING:
                            _reverser.clear_all_corridors(grid)
                            if _reverser.dump_target:
                                grid.unreserve(*_reverser.dump_target)
                                _reverser.dump_target = None
                            _reverser.path        = list(_esc_wps)
                            _reverser.stop_target = None  # → auto IDLE after escape
                        elif _reverser.status == _reverser.STATUS_EXITING:
                            _reverser._clear_exit_corridor(grid)
                            _reverser._exit_path = list(_esc_wps)
                        _reverser._stuck_substeps    = 0
                        _reverser._conflict_cooldown = len(_esc_wps) + 6
                        _advancer._stuck_substeps    = 0
                        _advancer._conflict_cooldown = max(
                            _advancer._conflict_cooldown, len(_esc_wps) + 2)
                        _in_flight.discard(_reverser.id)
                        _deadlock_handled = True

                    else:
                        # ── Fallback: straight reverse retreat ────────────────
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

                        print(f"[DEADLOCK-FALLBACK] T{_ta.id}↔T{_tb.id}: "
                              f"T{_reverser.id} straight retreat "
                              f"({len(_retreat_wps)} steps).")
                        sink.record_deadlock(tick, _ta.id, _tb.id,
                                             'straight_retreat',
                                             len(_retreat_wps))

                        if _reverser.status == _reverser.STATUS_NAVIGATING:
                            _reverser.path = _retreat_wps + list(_reverser.path)
                        elif _reverser.status == _reverser.STATUS_EXITING:
                            _reverser._exit_path = (
                                _retreat_wps + list(_reverser._exit_path))

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
                sink.record_force_idle(tick, yielder.id)
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
                sink.record_stuck_exit(tick, _st.id)
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
                sink.record_headlock(tick, _ta.id, _tb.id, _reverser.id,
                                     len(_yield_wps))

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

        if tick % METRICS_TICK_EVERY == 0:
            sink.record_tick(tick,
                             fill_pct=cached_fill, pack_pct=cached_pack,
                             active_trucks=len(active_trucks),
                             active_intrusions=intrusions.active_count(),
                             idle=len([t for t in trucks if t.is_idle()]),
                             in_flight=len(_in_flight))

        if headless and tick % 100 == 0:
            statuses = [f"T{t.id}:{t.status}" for t in trucks]
            print(f"[TICK {tick}] pack={grid.pack_pct()*100:.3f}% | {' '.join(statuses)}")

        tick += 1
        if max_ticks and tick >= max_ticks:
            print(f"\nReached max_ticks={max_ticks}, stopping.")
            break

    # ── Shutdown ──────────────────────────────────────────────────────────────
    _stop_evt.set()
    if _planner is not None:
        _planner.join(timeout=5.0)

    intrusions.close_all(tick, truck_map)
    sink.close(final={'total_ticks': tick,
                      'final_fill_pct': grid.fill_pct() * 100,
                      'final_pack_pct': grid.pack_pct() * 100})

    print("\n=== Final Results ===")
    print(f"Total ticks:   {tick}")
    print(f"Fill %:        {grid.fill_pct()*100:.3f}%")
    print(f"Pack density:  {grid.pack_pct()*100:.3f}%")
    if metrics_out:
        _c = sink.counters
        print(f"Intrusions:    {_c['intrusions_closed']} "
              f"({_c['sustained_intrusions']} sustained) | "
              f"blocks={_c['collision_blocks']} "
              f"deadlocks={_c['deadlocks']} headlocks={_c['headlocks']}")
        print(f"Metrics log:   {metrics_out}")

    if not headless:
        print("\nClose the window to exit.")
        while not renderer.check_quit():
            renderer.draw(trucks, {'FINAL': 'done',
                                   'fill%': f"{grid.fill_pct()*100:.3f}",
                                   'pack%': f"{grid.pack_pct()*100:.3f}"})
            time.sleep(0.1)
        renderer.close()


if __name__ == '__main__':
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument('--headless', action='store_true',
                     help='Run without pygame rendering (fast, for testing)')
    _ap.add_argument('--max-ticks', type=int, default=0,
                     help='Stop after this many ticks (0 = unlimited)')
    _ap.add_argument('--seed', type=int, default=None,
                     help='RNG seed — required for reproducible A/B runs')
    _ap.add_argument('--metrics-out', type=str, default=None,
                     help='Write a .jsonl run log to this path')
    _ap.add_argument('--collision-bypass', action='store_true',
                     help='BENCHMARK ONLY: let truck bodies overlap and count '
                          'the intrusions instead of blocking them')
    _ap.add_argument('--sync-planner', action='store_true',
                     help='Run planning inline instead of on the background '
                          'thread. Makes a seeded run exactly reproducible; '
                          'not representative of real-time performance.')
    _ap.add_argument('--fleet', type=str, default=None,
                     help='Fleet override as S,M,L (e.g. "4,3,2") for the '
                          'scalability sweep')
    _args = _ap.parse_args()

    _fleet = None
    if _args.fleet:
        _parts = [int(x) for x in _args.fleet.split(',')]
        if len(_parts) != 3:
            _ap.error('--fleet expects three comma-separated ints: S,M,L')
        _fleet = tuple(_parts)

    _mo = _args.metrics_out
    if _mo is None and (_args.seed is not None or _args.collision_bypass):
        # Auto-name a log when the run is clearly a measured one.
        os.makedirs(METRICS_DIR, exist_ok=True)
        _mo = os.path.join(
            METRICS_DIR,
            f"seed{_args.seed}_bypass{int(_args.collision_bypass)}.jsonl")

    run_simulation(headless=_args.headless, max_ticks=_args.max_ticks,
                   seed=_args.seed, metrics_out=_mo,
                   allow_collision_bypass=(True if _args.collision_bypass
                                           else None),
                   sync_planner=_args.sync_planner, fleet=_fleet)
