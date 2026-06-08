# main.py
# ─────────────────────────────────────────────────────────────
# THE SIMULATION LOOP — mixed fleet edition.
#
# FIX: Hybrid Orchestrator implemented.
# FIX: Exit paths are automatically planned for trucks.
# FIX: Removed the `continue` trap so active trucks never freeze.
# FIX: Added `STEPS_PER_TICK` loop so trucks don't crawl.
# FIX: Corrected Pheromone Math to fix the "blue smudge" bug.
# ─────────────────────────────────────────────────────────────

import sys
import time
import math
import numpy as np
from scipy.ndimage import gaussian_filter

sys.stdout.reconfigure(encoding='utf-8')

from config import (POLYGON_BOUNDARY, ENTRY_POINT, CELL_SIZE,
                    FLEET_COMPOSITION, TICK_DELAY, PYGAME_SCALE,
                    PHEROMONE_DECAY, PHEROMONE_SPREAD_SIGMA,
                    CONFIG_MATERIAL_HEIGHT_THRESHOLD, STEPS_PER_TICK,
                    ENTRY_CORRIDOR_CELLS, ENTRY_EXIT_HOLD_RADIUS_M)
import grid_map
from truck      import Truck
from filters    import get_raw_candidates, is_accessible, precompute_coarse_blocked_mask
from scoring    import score_candidates
from mcts       import mcts_select_dump_points
from assignment import assign
from pathfinder import (plan_paths_cbs, plan_staging_paths,
                        _detect_first_conflict, _path_cells,
                        generate_reverse_retreat)
from renderer   import Renderer


def build_fleet():
    from config import TRUCK_CLASSES
    trucks = []
    truck_id = 0
    queue_slot = 0  # counts waiting trucks to space them out
    for cls in ('small', 'medium', 'large'):
        for _ in range(FLEET_COMPOSITION.get(cls, 0)):
            truck_len = TRUCK_CLASSES[cls]['length_m']
            # Home position: outside the polygon, stacked below ENTRY_POINT
            offset   = (queue_slot + 1) * truck_len * 1.5
            home     = (ENTRY_POINT[0], ENTRY_POINT[1] - offset)
            if truck_id == 0:
                # First truck starts inside at the entry point, ready to go immediately
                trucks.append(Truck(truck_id, cls, home_pos=home))
            else:
                wait_pos = home
                trucks.append(Truck(truck_id, cls, start_pos=wait_pos, waiting=True,
                                    home_pos=home))
            queue_slot += 1
            truck_id += 1
    return trucks


def _locked_path_for(truck, path):
    """Pair a truck with its current rear-axle pose plus remaining path."""
    rear_x, rear_y = truck.rear_axle_world()
    return truck, [(rear_x, rear_y, truck.heading)] + list(path)


def _try_inplace_replan(ta, tb, grid, entry_rc):
    """
    Resolve a detected conflict between trucks ta and tb by immediately replanning
    both with CBS — without forcing either truck to IDLE or unreserving dump targets.

    Handles three cases:
      • Both NAVIGATING  → plan_staging_paths on both together (CBS resolves internally)
      • Both EXITING     → plan_paths_cbs on both together
      • Mixed            → lock the non-yielding truck, replan the other

    Returns True if a conflict-free replanned solution was found and applied.
    """
    both_nav  = (ta.status == ta.STATUS_NAVIGATING and tb.status == tb.STATUS_NAVIGATING)
    both_exit = (ta.status == ta.STATUS_EXITING    and tb.status == tb.STATUS_EXITING)
    a_nav     = ta.status == ta.STATUS_NAVIGATING
    b_nav     = tb.status == tb.STATUS_NAVIGATING

    if both_nav:
        if not ta.dump_target or not tb.dump_target:
            return False
        # Clear both corridors so replanning sees the grid without them
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
        old_a = list(ta._exit_path)
        old_b = list(tb._exit_path)
        ta._clear_exit_corridor(grid)
        tb._clear_exit_corridor(grid)
        # No exit corridors to clear — exits are not marked PATH_RESERVED
        new_paths = plan_paths_cbs(grid, [(ta, entry_rc), (tb, entry_rc)])
        np_a = new_paths.get(ta.id, [])
        np_b = new_paths.get(tb.id, [])
        if np_a and np_b:
            ta.set_exit_path(np_a, grid)
            tb.set_exit_path(np_b, grid)
            ta._conflict_cooldown = 5;  tb._conflict_cooldown = 5
            print(f"[CBS-REPLAN] T{ta.id}+T{tb.id} exit conflict resolved in-place.")
            return True
        ta._exit_path = old_a
        tb._exit_path = old_b
        if old_a:
            ta._mark_exit_corridor(grid, old_a)
        if old_b:
            tb._mark_exit_corridor(grid, old_b)
        return False

    # Mixed: one navigating, one exiting
    nav_t  = ta if a_nav else tb
    exit_t = tb if a_nav else ta

    # First try: replan exit truck (no exit corridor to clear/re-mark)
    if nav_t.path:
        old_exit_path = list(exit_t._exit_path)
        exit_t._clear_exit_corridor(grid)
        locked = {nav_t.id: _locked_path_for(nav_t, nav_t.path)}
        ep = plan_paths_cbs(grid, [(exit_t, entry_rc)], locked_paths=locked)
        np_exit = ep.get(exit_t.id, [])
        if np_exit:
            exit_t.set_exit_path(np_exit, grid)
            exit_t._conflict_cooldown = 5
            print(f"[CBS-REPLAN] T{exit_t.id} exit replanned around T{nav_t.id}.")
            return True
        exit_t._exit_path = old_exit_path
        if old_exit_path:
            exit_t._mark_exit_corridor(grid, old_exit_path)

    # Second try: replan nav truck around exit truck's existing corridor
    if exit_t._exit_path and nav_t.dump_target:
        locked = {exit_t.id: _locked_path_for(exit_t, exit_t._exit_path)}
        nav_t.clear_all_corridors(grid)
        np, ns = plan_staging_paths(grid, [(nav_t, nav_t.dump_target)], locked_paths=locked)
        np_nav = np.get(nav_t.id, [])
        if np_nav:
            nav_t.path = np_nav;  nav_t.stop_target = nav_t.path[-1];  nav_t.staging_pose = ns.get(nav_t.id)
            nav_t._mark_dump_corridor(grid, nav_t.path)
            nav_t._conflict_cooldown = 5
            print(f"[CBS-REPLAN] T{nav_t.id} nav replanned around T{exit_t.id}.")
            return True

    return False


def _prepend_current_pose_wait(truck, wait_steps):
    """Hold a moving truck at its current rear-axle pose for a few substeps."""
    if wait_steps <= 0:
        return
    rear_x, rear_y = truck.rear_axle_world()
    wait_wps = [(rear_x, rear_y, truck.heading) for _ in range(wait_steps)]
    if truck.status == truck.STATUS_NAVIGATING:
        truck.path = wait_wps + list(truck.path)
    elif truck.status == truck.STATUS_EXITING:
        truck._exit_path = wait_wps + list(truck._exit_path)


def _exit_near_entry_zone(trucks, entry_radius_m):
    entry_x, entry_y = ENTRY_POINT
    hold_radius = ENTRY_CORRIDOR_CELLS * CELL_SIZE + entry_radius_m
    for truck in trucks:
        if (getattr(truck, "on_grid", True)
                and truck.status == truck.STATUS_EXITING
                and math.hypot(truck.pos[0] - entry_x, truck.pos[1] - entry_y) <= hold_radius):
            return True
    return False


def run_simulation():
    print("Initialising grid...")
    grid = grid_map.GridMap(POLYGON_BOUNDARY, CELL_SIZE)

    valid_cells = np.sum(grid.state == grid_map.CellState.EMPTY)
    print(f"Grid: {grid.rows}x{grid.cols} cells, {valid_cells} valid dump cells")

    entry_rc = grid.world_to_cell(*ENTRY_POINT)
    print(f"Entry point: world{ENTRY_POINT} -> cell{entry_rc}")

    trucks = build_fleet()
    total_trucks = len(trucks)
    print(f"Fleet: {total_trucks} trucks")

    corridor_clearance_m = ENTRY_CORRIDOR_CELLS * CELL_SIZE

    renderer = Renderer(grid, scale=PYGAME_SCALE)
    print("Renderer ready - starting simulation loop")

    tick       = 0
    done       = False
    deadlock_attempts = {}

    while not done:
        if renderer.check_quit():
            break

        # ── GATE LOGIC ────────────────────────────────────────────────────────────
        # Before releasing a WAITING truck:
        #   1. Pre-plan a conflict-free dump path for it from ENTRY_POINT
        #   2. Only release once a valid path is found AND the corridor is clear
        # This prevents trucks from entering and getting stuck inside with no route.
        waiting_queue = sorted([t for t in trucks if t.status == t.STATUS_WAITING],
                               key=lambda t: t.id)
        if waiting_queue:
            next_t = waiting_queue[0]

            # ── PRE-PLAN: find a dump path for this truck while it's still outside ──
            if not next_t._pre_path:
                _saved_pos     = list(next_t.pos)
                _saved_heading = next_t.heading
                next_t.pos     = list(ENTRY_POINT)
                next_t.heading = math.pi / 2

                _repr = next_t
                _raw  = get_raw_candidates(grid, _repr)
                if _raw:
                    _fp         = grid.fill_pct()
                    _coarse     = precompute_coarse_blocked_mask(grid, _repr)
                    _top_cands  = []
                    if _fp < CONFIG_MATERIAL_HEIGHT_THRESHOLD:
                        _scores = score_candidates(grid, _raw)
                        for _idx in _scores.argsort()[::-1]:
                            _r, _c = _raw[_idx]
                            if is_accessible(grid, _r, _c, entry_rc, _repr,
                                             precomputed_coarse_mask=_coarse):
                                _top_cands.append((_r, _c))
                            if len(_top_cands) >= 20:
                                break
                    else:
                        _acc = [(_r, _c) for _r, _c in _raw
                                if is_accessible(grid, _r, _c, entry_rc, _repr,
                                                 precomputed_coarse_mask=_coarse)]
                        if _acc:
                            _scores = score_candidates(grid, _acc)
                            _top_cands = [_acc[_i] for _i in _scores.argsort()[-20:][::-1]]

                    if _top_cands:
                        _avail = [(_r, _c) for _r, _c in _top_cands
                                  if (_r, _c) not in {t.dump_target for t in trucks
                                                       if t.dump_target}]
                        if _avail:
                            _dpts = mcts_select_dump_points(grid, _avail, next_t,
                                                            n_trucks=1, n_sim=100)
                            if _dpts:
                                _asgn = assign([next_t], _dpts[:1], grid)
                                if _asgn:
                                    _locked = {}
                                    for _ot in trucks:
                                        if _ot is next_t:
                                            continue
                                        if not getattr(_ot, "on_grid", True):
                                            continue
                                        if _ot.status == _ot.STATUS_NAVIGATING and _ot.path:
                                            _locked[_ot.id] = _locked_path_for(_ot, _ot.path)
                                        elif _ot.status == _ot.STATUS_EXITING and _ot._exit_path:
                                            _locked[_ot.id] = _locked_path_for(_ot, _ot._exit_path)
                                    _paths, _staging = plan_staging_paths(
                                        grid, _asgn, locked_paths=_locked)
                                    _p = _paths.get(next_t.id, [])
                                    _dt = _asgn[0][1]
                                    if _p:
                                        next_t.preload_dump_path(
                                            _p, _dt, grid, _staging.get(next_t.id))
                                        print(f"[GATE] Pre-planned path for T{next_t.id}")
                                    else:
                                        # Path not found — unreserve what assign() may have touched
                                        grid.unreserve(*_dt)

                next_t.pos     = _saved_pos
                next_t.heading = _saved_heading

            # ── RELEASE: gate clear AND pre-plan ready ────────────────────────────
            any_transiting = any(t.status in (t.STATUS_ENTERING, t.STATUS_LEAVING)
                                 for t in trucks)
            exit_near_entry = _exit_near_entry_zone(trucks, ENTRY_EXIT_HOLD_RADIUS_M)
            if next_t._pre_path and not any_transiting and not exit_near_entry:
                entry_x, entry_y = ENTRY_POINT
                active = [t for t in trucks
                          if getattr(t, "on_grid", True)
                          and t.status not in (t.STATUS_WAITING,)]
                if active:
                    nearest_dist = min(
                        math.hypot(t.pos[0] - entry_x, t.pos[1] - entry_y)
                        for t in active
                    )
                    gate_clear = nearest_dist >= corridor_clearance_m
                else:
                    gate_clear = True
                if gate_clear:
                    next_t.release()
                    print(f"[QUEUE] Released truck {next_t.id} ({next_t.truck_class})")

        # Check for trucks needing exit paths
        exiting_trucks = [t for t in trucks if t.needs_exit_path()]
        if exiting_trucks:
            exit_assignments = [(t, entry_rc) for t in exiting_trucks]
            being_exit_planned = {t for t, _ in exit_assignments}
            locked_moving = {}
            for t in trucks:
                if t in being_exit_planned or not getattr(t, "on_grid", True):
                    continue
                if t.status == t.STATUS_NAVIGATING and t.path:
                    locked_moving[t.id] = _locked_path_for(t, t.path)
                elif t.status == t.STATUS_EXITING and t._exit_path:
                    locked_moving[t.id] = _locked_path_for(t, t._exit_path)
            exit_paths = plan_paths_cbs(grid, exit_assignments, locked_paths=locked_moving)
            for t, _ in exit_assignments:
                t.set_exit_path(exit_paths.get(t.id, []), grid)

        # Exclude trucks that already have a pre-planned path committed (they'll
        # consume it when they finish ENTERING — no need to re-assign them).
        idle_trucks = [t for t in trucks if t.is_idle() and not t._pre_path]

        # ── PLANNING PHASE ─────────────────────────────────────────
        if idle_trucks:
            repr_truck = max(idle_trucks, key=lambda t: t.width * t.length)
            raw_candidates = get_raw_candidates(grid, repr_truck)

            if not raw_candidates:
                print(f"\nSimulation complete at tick {tick}!")
                done = True
                break

            fp = grid.fill_pct()
            top_candidates = []
            coarse_mask = precompute_coarse_blocked_mask(grid, repr_truck)

            if fp < CONFIG_MATERIAL_HEIGHT_THRESHOLD:
                scores = score_candidates(grid, raw_candidates)
                top_indices = scores.argsort()[::-1]

                for idx in top_indices:
                    r, c = raw_candidates[idx]
                    if is_accessible(grid, r, c, entry_rc, repr_truck, precomputed_coarse_mask=coarse_mask):
                        top_candidates.append((r, c))
                    if len(top_candidates) >= 50:
                        break
            else:
                accessible_cands = [
                    (r, c) for r, c in raw_candidates
                    if is_accessible(grid, r, c, entry_rc, repr_truck, precomputed_coarse_mask=coarse_mask)
                ]
                if accessible_cands:
                    scores = score_candidates(grid, accessible_cands)
                    top_indices = scores.argsort()[-50:][::-1]
                    top_candidates = [accessible_cands[i] for i in top_indices]

            # FIX: Replaced `continue` with nested execution so active trucks don't freeze
            if top_candidates:
                assignments_all = []
                claimed_points  = set()
                remaining_idle  = list(idle_trucks)

                for cls in ('large', 'medium', 'small'):
                    cls_trucks = [t for t in remaining_idle if t.truck_class == cls]
                    if not cls_trucks:
                        continue

                    avail = [(r, c) for r, c in top_candidates if (r, c) not in claimed_points]
                    if not avail:
                        break

                    n_needed = len(cls_trucks)
                    dump_points = mcts_select_dump_points(grid, avail, cls_trucks[0], n_trucks=n_needed, n_sim=200)

                    if not dump_points:
                        continue

                    these_assignments = assign(cls_trucks[:len(dump_points)], dump_points, grid)
                    assignments_all.extend(these_assignments)
                    for _, dp in these_assignments: claimed_points.add(dp)
                    for t, _ in these_assignments:  remaining_idle.remove(t)

                if assignments_all:
                    # Pass remaining paths of all currently moving trucks (navigating + exiting)
                    # so new dump paths never cross them.
                    being_planned = {t for t, _ in assignments_all}
                    all_locked = {}
                    for t in trucks:
                        if t in being_planned:
                            continue  # this truck IS being planned — don't lock its old path
                        if not getattr(t, "on_grid", True):
                            continue
                        if t.status == t.STATUS_NAVIGATING and t.path:
                            all_locked[t.id] = _locked_path_for(t, t.path)
                        elif t.status == t.STATUS_EXITING and t._exit_path:
                            all_locked[t.id] = _locked_path_for(t, t._exit_path)
                    paths, staging_poses = plan_staging_paths(grid, assignments_all, locked_paths=all_locked)
                    for truck, dump_point in assignments_all:
                        truck_path = paths.get(truck.id, [])
                        truck.set_path(truck_path, dump_point, grid,
                                       staging_pose=staging_poses.get(truck.id))

        # Movement / render phase. Draw every fine truck step; batching several
        # steps before drawing hides the interpolated arc and makes turns look
        # like stop-rotate-go.
        substep_delay = TICK_DELAY / max(1, STEPS_PER_TICK)

        # Compute expensive grid metrics once per tick (not per substep).
        cached_fill = grid.fill_pct() * 100
        cached_pack = grid.pack_pct() * 100

        # Active (non-waiting) trucks for collision checks — computed once per tick.
        active_trucks = [t for t in trucks
                         if getattr(t, "on_grid", True)
                         and t.status != t.STATUS_WAITING]

        for substep in range(STEPS_PER_TICK):
            if renderer.check_quit():
                done = True
                break

            for truck in trucks:
                prev_x, prev_y     = truck.pos[0], truck.pos[1]
                prev_heading       = truck.heading
                prev_first_wp      = truck.path[0]       if truck.path       else None
                prev_exit_first_wp = truck._exit_path[0] if truck._exit_path else None
                was_retreating     = truck._deadlock_retreat_steps > 0

                truck.step(grid)
                if not getattr(truck, "on_grid", True):
                    continue

                # Physical collision guard: CBS plans at cell resolution and treats
                # trucks as single points — path smoothing and physical footprints mean
                # world-space overlaps can still occur. Revert any move that brings
                # this truck's centre closer than the sum of the two half-lengths.
                collided = False
                collision_peer = None
                for other in active_trucks:
                    if other is truck:
                        continue
                    if not getattr(other, "on_grid", True):
                        continue
                    if truck.touches_entry_zone(grid) or other.touches_entry_zone(grid):
                        continue
                    dist = math.hypot(truck.pos[0] - other.pos[0],
                                      truck.pos[1] - other.pos[1])
                    if dist < (truck.length + other.length) * 0.5:
                        collided = True
                        collision_peer = other
                        break

                if collided:
                    truck.pos[0], truck.pos[1] = prev_x, prev_y
                    truck.heading = prev_heading
                    if was_retreating:
                        truck._deadlock_retreat_steps = max(
                            0, truck._deadlock_retreat_steps - 1)
                        truck._stuck_substeps = 0
                        truck._collision_peer = None
                        continue

                    truck._stuck_substeps += 1
                    truck._collision_peer = collision_peer.id if collision_peer else None
                    # Restore the waypoint that step() already popped so the path
                    # stays intact — without this the truck skips a waypoint next
                    # substep and appears to teleport.
                    if prev_first_wp is not None and (
                            not truck.path or truck.path[0] != prev_first_wp):
                        truck.path.insert(0, prev_first_wp)
                        truck._refresh_dump_corridor(grid)
                    if prev_exit_first_wp is not None and (
                            not truck._exit_path or truck._exit_path[0] != prev_exit_first_wp):
                        truck._exit_path.insert(0, prev_exit_first_wp)
                        truck._refresh_exit_corridor(grid)
                else:
                    if was_retreating:
                        truck._deadlock_retreat_steps = max(
                            0, truck._deadlock_retreat_steps - 1)
                    truck._stuck_substeps = 0
                    truck._collision_peer = None

            metrics = {
                'tick':       f"{tick}.{substep + 1}/{STEPS_PER_TICK}",
                'fleet':      f"{FLEET_COMPOSITION['small']}S/{FLEET_COMPOSITION['medium']}M/{FLEET_COMPOSITION['large']}L",
                'idle':       len([t for t in trucks if t.is_idle()]),
                'queued':     len(waiting_queue),
                'candidates': len(top_candidates) if 'top_candidates' in locals() else '-',
                'fill%':      f"{cached_fill:.3f}",
                'pack%':      f"{cached_pack:.3f}",
            }
            renderer.draw(trucks, metrics)

            time.sleep(substep_delay)

        # Pheromone update once per tick (not per substep) — gaussian_filter is expensive.
        grid.pheromone = 1.0 - (1.0 - grid.pheromone) * PHEROMONE_DECAY
        grid.pheromone = gaussian_filter(grid.pheromone, sigma=PHEROMONE_SPREAD_SIGMA)
        np.clip(grid.pheromone, 0.0, 1.0, out=grid.pheromone)

        # ── DEADLOCK RESOLUTION ──────────────────────────────────────────────
        # Step 1: look for mutual deadlocks (two trucks physically blocking each
        # other).  Pick the safer reverser and prepend a short backward retreat
        # to its path so the other truck can advance clear.
        # Step 2: fall back to force-idle for any remaining single stuck trucks.
        _STUCK_LIMIT = STEPS_PER_TICK * 4
        _COLLISION_DEADLOCK_LIMIT = STEPS_PER_TICK

        stuck_active = [t for t in active_trucks
                        if (t._stuck_substeps >= _STUCK_LIMIT
                            or (t._collision_peer is not None
                                and t._stuck_substeps >= _COLLISION_DEADLOCK_LIMIT))
                        and t.status in (t.STATUS_NAVIGATING, t.STATUS_EXITING)]

        _deadlock_handled = False

        def _force_replan_after_deadlock(truck):
            truck._stuck_substeps = 0
            truck._collision_peer = None
            truck._conflict_cooldown = 2
            if truck.status == truck.STATUS_NAVIGATING:
                truck._clear_dump_corridor(grid)
                if truck.dump_target:
                    grid.unreserve(*truck.dump_target)
                    truck.dump_target = None
                truck.path = []
                truck.stop_target = None
                truck.staging_pose = None
                truck.status = truck.STATUS_IDLE
            elif truck.status == truck.STATUS_EXITING:
                truck._clear_exit_corridor(grid)
                truck._exit_path = []

        if len(stuck_active) >= 2:
            for _si in range(len(stuck_active)):
                for _sj in range(_si + 1, len(stuck_active)):
                    _ta, _tb = stuck_active[_si], stuck_active[_sj]
                    _d = math.hypot(_ta.pos[0] - _tb.pos[0],
                                    _ta.pos[1] - _tb.pos[1])
                    if _d >= (_ta.length + _tb.length) * 0.65:
                        continue   # not directly blocking each other

                    _steps_a = max(1, int(math.ceil((_ta.length / 2.0) / grid.cell_size)))
                    _steps_b = max(1, int(math.ceil((_tb.length / 2.0) / grid.cell_size)))
                    _ret_a = generate_reverse_retreat(_ta, grid, num_steps=_steps_a)
                    _ret_b = generate_reverse_retreat(_tb, grid, num_steps=_steps_b)

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
                    _pair_key = tuple(sorted((_ta.id, _tb.id)))

                    if not _a_ok and not _b_ok:
                        deadlock_attempts[_pair_key] = deadlock_attempts.get(_pair_key, 0) + 1
                        if deadlock_attempts[_pair_key] < 2:
                            continue

                        _reverser = max((_ta, _tb),
                                        key=lambda t: len(t.path) + len(t._exit_path))
                        _advancer = _tb if _reverser is _ta else _ta
                        _retreat_wps = _reverser.escape(grid)
                        if not _retreat_wps:
                            _steps = max(1, int(math.ceil((_reverser.length / 2.0)
                                                           / grid.cell_size)))
                            _retreat_wps = generate_reverse_retreat(
                                _reverser, grid, num_steps=_steps)
                        if not _retreat_wps:
                            continue

                        if _reverser.status == _reverser.STATUS_NAVIGATING:
                            _reverser.path = _retreat_wps + list(_reverser.path)
                            _reverser._refresh_dump_corridor(grid)
                        elif _reverser.status == _reverser.STATUS_EXITING:
                            _reverser._exit_path = _retreat_wps + list(_reverser._exit_path)
                            _reverser._refresh_exit_corridor(grid)

                        _force_replan_after_deadlock(_advancer)
                        _reverser._deadlock_retreat_steps = len(_retreat_wps)
                        _reverser._stuck_substeps = 0
                        _reverser._conflict_cooldown = len(_retreat_wps) + 3
                        _ta._last_deadlock_reverser = _reverser.id
                        _tb._last_deadlock_reverser = _reverser.id
                        _ta._collision_peer = None
                        _tb._collision_peer = None
                        deadlock_attempts[_pair_key] = 0
                        print(f"[DEADLOCK] T{_ta.id}<->T{_tb.id} forced fallback. "
                              f"T{_reverser.id} reversing, T{_advancer.id} replanning.")
                        _deadlock_handled = True
                        break

                    _last_reverser = (
                        _ta._last_deadlock_reverser
                        if _ta._last_deadlock_reverser == _tb._last_deadlock_reverser
                        else None
                    )

                    if _last_reverser == _ta.id and _b_ok:
                        _reverser, _retreat_wps = _tb, _ret_b
                        _advancer = _ta
                    elif _last_reverser == _tb.id and _a_ok:
                        _reverser, _retreat_wps = _ta, _ret_a
                        _advancer = _tb
                    elif _a_ok and (not _b_ok or len(_ret_a) >= len(_ret_b)):
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

                    _reverser._deadlock_retreat_steps = len(_retreat_wps)
                    _reverser._stuck_substeps    = 0
                    _reverser._conflict_cooldown = len(_retreat_wps) + 3
                    _ta._last_deadlock_reverser = _reverser.id
                    _tb._last_deadlock_reverser = _reverser.id
                    deadlock_attempts[_pair_key] = 0
                    _ta._collision_peer = None
                    _tb._collision_peer = None
                    # Give advancer a cooldown while reverser backs up
                    _advancer._conflict_cooldown = max(
                        _advancer._conflict_cooldown, len(_retreat_wps))
                    _deadlock_handled = True
                    break
                if _deadlock_handled:
                    break

        if not _deadlock_handled:
            # Single-truck stuck: force it idle so it gets a fresh path next tick
            stuck_nav = [t for t in trucks
                         if t._stuck_substeps >= _STUCK_LIMIT
                         and t.status == t.STATUS_NAVIGATING]
            if stuck_nav:
                yielder = max(stuck_nav, key=lambda t: len(t.path))
                yielder._stuck_substeps    = 0
                yielder._conflict_cooldown = 3
                yielder.clear_all_corridors(grid)
                yielder.cancel_preload(grid)
                if yielder.dump_target:
                    grid.unreserve(*yielder.dump_target)
                    yielder.dump_target = None
                yielder.status       = yielder.STATUS_IDLE
                yielder.path         = []
                yielder.stop_target  = None
                yielder.staging_pose = None

        # ── PROACTIVE SPACE-TIME CONFLICT DETECTION ──────────────────────────
        # Each tick, convert every active truck's remaining path to a cell
        # sequence and run CBS conflict detection.  If two trucks' planned paths
        # will collide within the next _CONFLICT_HORIZON steps, force the
        # lower-priority truck (longer path remaining = farther from its goal)
        # to IDLE immediately so its next replanned path uses the other truck's
        # current path as a space-time constraint.
        # A per-truck cooldown prevents the same truck being force-replanned
        # every tick while it's trying to get a new assignment.
        for t in trucks:
            if t._conflict_cooldown > 0:
                t._conflict_cooldown -= 1

        _CONFLICT_HORIZON = 20
        nav_cell_paths = {}
        for t in trucks:
            if t._conflict_cooldown > 0:
                continue
            if not getattr(t, "on_grid", True):
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
                    # Skip conflicts inside the entry corridor — trucks funnelling
                    # to the single exit point are expected to be close there.
                    _in_corridor = (
                        math.hypot(_ri - entry_rc[0], _ci - entry_rc[1]) <= ENTRY_CORRIDOR_CELLS or
                        math.hypot(_rj - entry_rc[0], _cj - entry_rc[1]) <= ENTRY_CORRIDOR_CELLS
                    )
                    if _in_corridor:
                        conflict = None
                else:
                    _, ai, aj, _r1, _c1, _r2, _c2, t_conf = conflict

                if conflict and t_conf <= _CONFLICT_HORIZON:
                    _cta = next((t for t in trucks if t.id == ai), None)
                    _ctb = next((t for t in trucks if t.id == aj), None)

                    # ── Attempt in-place CBS replan (no idle, no unreserve) ──
                    _replanned = False  # CBS disabled; wait-based conflict handling below
                    if _cta is not None and _ctb is not None:
                        pass

                    # ── Fall back: yield the longer-path truck to IDLE ────────
                    if not _replanned:
                        len_ai = len(nav_cell_paths.get(ai, []))
                        len_aj = len(nav_cell_paths.get(aj, []))
                        _yielder_id = ai if len_ai >= len_aj else aj
                        _wait_steps = max(2, min(STEPS_PER_TICK, t_conf + 2))
                        for _ct in trucks:
                            if _ct.id != _yielder_id:
                                continue
                            _prepend_current_pose_wait(_ct, _wait_steps)
                            _ct._conflict_cooldown = _wait_steps + 2
                            _ct._stuck_substeps    = 0
                            print(f"[WAIT] T{_ct.id} waits {_wait_steps} steps "
                                  f"for T{aj if _ct.id == ai else ai} to pass.")
                            break

        tick += 1
    print("\n=== Final Results ===")
    print(f"Total ticks:   {tick}")
    print(f"Fill %:        {grid.fill_pct()*100:.3f}%")
    print(f"Pack density:  {grid.pack_pct()*100:.3f}%")

    print("\nClose the window to exit.")
    while not renderer.check_quit():
        renderer.draw(trucks, {'FINAL': 'done', 'fill%': f"{grid.fill_pct()*100:.3f}", 'pack%': f"{grid.pack_pct()*100:.3f}"})
        time.sleep(0.1)
    renderer.close()


if __name__ == '__main__':
    run_simulation()
