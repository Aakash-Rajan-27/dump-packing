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
                        generate_reverse_retreat, generate_yield_maneuver,
                        escape_and_replan_exit)
from renderer   import Renderer
import random

def initialise_half_full_dump(grid):
    """
    Generate an initial terrain corresponding to roughly 50%
    packing density using the existing dump physics.
    """

    target_pack = 0.03

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
    """
    Return the set of grid cells physically covered by truck's corridor along path.
    The corridor is a capsule of radius truck.width/2 around each path centre-cell.
    Used for corridor-level conflict detection at the gate.
    """
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
        ta._clear_exit_corridor(grid)
        tb._clear_exit_corridor(grid)
        new_paths = plan_paths_cbs(
            grid, [(ta, entry_rc), (tb, entry_rc)])
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

    # Mixed: one navigating, one exiting
    nav_t  = ta if a_nav else tb
    exit_t = tb if a_nav else ta

    # First try: replan exit truck — clear old corridor, plan, re-mark
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

    # Second try: replan nav truck around exit truck's existing corridor
    if exit_t._exit_path and nav_t.dump_target:
        locked = {exit_t.id: ([exit_t.front_center_cell(grid)] + list(exit_t._exit_path), 0)}
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

def run_simulation():
    print("Initialising grid...")
    grid = grid_map.GridMap(POLYGON_BOUNDARY, CELL_SIZE)

    print("Generating initial 50% dump fill...")
    initialise_half_full_dump(grid)

    valid_cells = np.sum(
        (grid.state == grid_map.CellState.EMPTY) |
        (grid.state == grid_map.CellState.PARTIAL)
    )

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

    while not done:
        if renderer.check_quit():
            break

        # Snapshot each truck's position at the start of this tick so we can
        # measure net progress at the end and detect stagnant (headlocked) trucks.
        for _t in trucks:
            _t._pos_snapshot = list(_t.pos)

        # ── GATE LOGIC ────────────────────────────────────────────────────────────
        # Before releasing a WAITING truck:
        #   1. Pre-plan a conflict-free dump path for it from ENTRY_POINT
        #   2. Only release once a valid path is found AND the corridor is clear
        # This prevents trucks from entering and getting stuck inside with no route.
        # Collect all trucks still waiting outside, ordered by ID so the lowest-ID
        # truck always gets priority — prevents starvation of earlier arrivals.
        waiting_queue = sorted([t for t in trucks if t.status == t.STATUS_WAITING],
                               key=lambda t: t.id)
        if waiting_queue:  # at least one truck is queued outside the polygon
            next_t = waiting_queue[0]  # candidate for release: the first truck in the queue

            # ── PRE-PLAN: find a conflict-free dump path while the truck waits outside ──
            # Only pre-plan when trucks are already inside the polygon (navigating or
            # exiting) — those are the paths that must not be crossed.  If the polygon
            # is empty there is nothing to conflict with and we skip straight to release.

            # Every truck actively occupying the polygon must be accounted for so the
            # pre-path can avoid all of them.  NAVIGATING and EXITING trucks have known
            # future paths; DUMPING and REVERSING trucks are stationary at a fixed cell
            # and their position must still be treated as a hard obstacle.
            _trucks_inside = [t for t in trucks
                              if t is not next_t and                        # exclude the waiting truck itself
                                 t.status in (t.STATUS_NAVIGATING, t.STATUS_EXITING,
                                              t.STATUS_DUMPING, t.STATUS_REVERSING)]  # all trucks currently inside

            # Only run the expensive candidate-search + CBS pre-plan if:
            #   (a) the waiting truck has no pre-path yet, AND
            #   (b) there are trucks inside whose paths could conflict.
            # If the polygon is empty (_trucks_inside == []) we skip straight to release.
            if not next_t._pre_path and _trucks_inside:
                _saved_pos     = list(next_t.pos)    # remember where the truck actually is (outside)
                _saved_heading = next_t.heading       # remember its current heading
                next_t.pos     = list(ENTRY_POINT)   # temporarily move it to the entry point so the
                next_t.heading = math.pi / 2          # pathfinder plans a route starting from the gate

                _repr = next_t  # use this truck as the representative for candidate scoring
                _raw  = get_raw_candidates(grid, _repr)  # all grid cells that are valid dump targets for this truck class
                if _raw:  # at least one dumpable cell exists in the polygon
                    _fp         = grid.fill_pct()                              # current fraction of polygon volume filled (0–1)
                    _coarse     = precompute_coarse_blocked_mask(grid, _repr)  # coarse reachability mask — cheap BFS used to pre-filter unreachable cells
                    _top_cands  = []  # will hold up to 20 reachable, high-score candidate cells
                    if _fp < CONFIG_MATERIAL_HEIGHT_THRESHOLD:
                        # Early/mid fill: score ALL raw candidates first, then BFS-check only the
                        # top scorers.  Scoring is cheap; BFS is expensive, so we avoid calling it
                        # on low-value cells.
                        _scores = score_candidates(grid, _raw)               # heuristic score for every raw cell
                        for _idx in _scores.argsort()[::-1]:                 # iterate cells from highest score downward
                            _r, _c = _raw[_idx]                              # grid coordinates of this candidate
                            if is_accessible(grid, _r, _c, entry_rc, _repr,
                                             precomputed_coarse_mask=_coarse):  # full BFS reachability check from entry
                                _top_cands.append((_r, _c))                  # cell is reachable — keep it
                            if len(_top_cands) >= 20:                        # stop once we have 20 good candidates
                                break
                    else:
                        # Late fill: nearly full, so first BFS-check everything (most of the polygon
                        # is now driveable dirt), then score only the accessible subset.
                        _acc = [(_r, _c) for _r, _c in _raw
                                if is_accessible(grid, _r, _c, entry_rc, _repr,
                                                 precomputed_coarse_mask=_coarse)]  # all reachable raw cells
                        if _acc:
                            _scores = score_candidates(grid, _acc)               # score the accessible subset
                            _top_cands = [_acc[_i] for _i in _scores.argsort()[-20:][::-1]]  # keep top 20 by score

                    if _top_cands:  # we found at least one reachable, scored candidate
                        # Remove cells already reserved by other trucks so we don't double-book.
                        _avail = [(_r, _c) for _r, _c in _top_cands
                                  if (_r, _c) not in {t.dump_target for t in trucks
                                                       if t.dump_target}]  # set of cells already claimed
                        if _avail:  # at least one unreserved candidate remains
                            # MCTS picks the single best dump point from the available candidates,
                            # simulating how well the polygon fills from that position.
                            _dpts = mcts_select_dump_points(grid, _avail, next_t,
                                                            n_trucks=1, n_sim=100)  # 100 rollouts for the waiting truck
                            if _dpts:  # MCTS returned a recommendation
                                # Hungarian assignment pairs the waiting truck to the chosen dump point
                                # and reserves the cell so no other truck can claim it during planning.
                                _asgn = assign([next_t], _dpts[:1], grid)
                                if _asgn:  # assignment succeeded (cell was reservable)
                                    # Build the space-time locked-path dict: every truck already inside
                                    # contributes its remaining waypoints so the CBS planner is forbidden
                                    # from routing the waiting truck through those cells at those times.
                                    _locked = {}
                                    for _ot in _trucks_inside:
                                        if _ot.status == _ot.STATUS_NAVIGATING and _ot.path:
                                            _locked[_ot.id] = (
                                                [_ot.front_center_cell(grid)] + list(_ot.path),
                                                _ot._dump_ticks_required + 2)
                                        elif _ot.status == _ot.STATUS_EXITING and _ot._exit_path:
                                            _locked[_ot.id] = (
                                                [_ot.front_center_cell(grid)] + list(_ot._exit_path),
                                                0)
                                        else:
                                            # DUMPING or REVERSING: lock only for remaining dump ticks.
                                            _remaining = max(0, _ot._dump_ticks_required - _ot._dump_ticks)
                                            _locked[_ot.id] = (
                                                [_ot.front_center_cell(grid)],
                                                _remaining + 2)
                                    # ── CORRIDOR-SAFE PLANNING ───────────────────────────
                                    # The driveable mask already blocks PATH_RESERVED cells
                                    # (existing corridors), so the planned path's CENTRELINE
                                    # avoids them.  But the waiting truck's own corridor
                                    # extends half_width beyond its centreline, so we need
                                    # the centreline to stay at least half_width away from
                                    # every existing corridor.  We achieve this by
                                    # temporarily dilating all PATH_RESERVED cells in the
                                    # grid by the waiting truck's half-width before planning,
                                    # then restoring the grid state immediately after.
                                    _hw_buf = int(math.ceil(
                                        (next_t.width / 2.0) / grid.cell_size))
                                    _pr_mask = (grid.state
                                                == grid_map.CellState.PATH_RESERVED)
                                    _dil_struct = np.array(
                                        [[math.hypot(_bdr, _bdc) <= _hw_buf
                                          for _bdc in range(-_hw_buf, _hw_buf + 1)]
                                         for _bdr in range(-_hw_buf, _hw_buf + 1)],
                                        dtype=bool)
                                    _expanded_pr = binary_dilation(
                                        _pr_mask, structure=_dil_struct)
                                    # Cells to buffer: in the expanded zone but not already
                                    # PATH_RESERVED, BOUNDARY, or PROTECTED.
                                    _buf_mask = (
                                        _expanded_pr & ~_pr_mask
                                        & (grid.state != grid_map.CellState.BOUNDARY)
                                        & (grid.state != grid_map.CellState.PROTECTED))
                                    _buf_orig = grid.state[_buf_mask].copy()
                                    grid.state[_buf_mask] = (
                                        grid_map.CellState.PATH_RESERVED)
                                    # Plan with the widened forbidden zone active.
                                    _paths, _staging = plan_staging_paths(
                                        grid, _asgn, locked_paths=_locked)
                                    # Restore grid — remove the temporary buffer cells.
                                    grid.state[_buf_mask] = _buf_orig
                                    _p = _paths.get(next_t.id, [])   # planned path for the waiting truck (may be empty if CBS failed)
                                    _dt = _asgn[0][1]                # dump-target cell that was assigned and reserved
                                    if _p:  # CBS found a valid conflict-free path
                                        # Commit the path: marks the corridor in the grid and stores
                                        # the path + target on the truck so it can be consumed the
                                        # moment the truck finishes its ENTERING drive to the gate.
                                        next_t.preload_dump_path(
                                            _p, _dt, grid, _staging.get(next_t.id))
                                        print(f"[GATE] Pre-planned path for T{next_t.id}")
                                    else:
                                        # CBS exhausted its node budget without finding a safe path.
                                        # Unreserve the cell assign() reserved so other trucks can
                                        # still claim it; the waiting truck will retry next tick.
                                        grid.unreserve(*_dt)

                next_t.pos     = _saved_pos      # restore real position — truck is still outside waiting
                next_t.heading = _saved_heading  # restore real heading

            # ── RE-VALIDATE pre-path every tick (corridor-level) ─────────────────
            # Inside trucks may have been replanned since the pre-path was committed.
            # We check physical CORRIDOR overlap (not just centreline cells): expand
            # each path by the respective truck's half-width, then test intersection
            # outside the entry corridor.  Any overlap means the stored pre-path is
            # unsafe — cancel it so the block above replans on the next tick.
            if next_t._pre_path and _trucks_inside:
                # Full corridor of the waiting truck's pre-path, stripped of cells
                # inside the entry funnel (all paths converge there legitimately).
                _pre_corridor = {
                    (r, c) for r, c in _corridor_cells(grid, next_t._pre_path, next_t)
                    if math.hypot(r - entry_rc[0], c - entry_rc[1]) > ENTRY_CORRIDOR_CELLS
                }
                _pre_stale = False
                for _ot in _trucks_inside:
                    if _ot.status == _ot.STATUS_NAVIGATING and _ot.path:
                        # Corridor of the navigating truck's remaining dump path.
                        _ot_corridor = {
                            (r, c) for r, c in _corridor_cells(grid, _ot.path, _ot)
                            if math.hypot(r - entry_rc[0], c - entry_rc[1]) > ENTRY_CORRIDOR_CELLS
                        }
                    elif _ot.status == _ot.STATUS_EXITING and _ot._exit_path:
                        # Corridor of the exiting truck's remaining exit path.
                        _ot_corridor = {
                            (r, c) for r, c in _corridor_cells(grid, _ot._exit_path, _ot)
                            if math.hypot(r - entry_rc[0], c - entry_rc[1]) > ENTRY_CORRIDOR_CELLS
                        }
                    else:
                        # DUMPING/REVERSING: treat current body position as a single
                        # cell corridor expanded by the truck's half-width.
                        _fc = _ot.front_center_cell(grid)
                        _ot_corridor = (
                            _corridor_cells(grid, [_fc], _ot)
                            if math.hypot(_fc[0] - entry_rc[0], _fc[1] - entry_rc[1])
                            > ENTRY_CORRIDOR_CELLS else set()
                        )
                    if _pre_corridor & _ot_corridor:
                        _pre_stale = True
                        print(f"[GATE] T{next_t.id} pre-corridor overlaps T{_ot.id} "
                              f"corridor, cancelling pre-path.")
                        break
                if _pre_stale:
                    next_t.cancel_preload(grid)

            # ── RELEASE: gate clear, and either pre-plan ready or polygon is empty ──
            any_transiting = any(t.status in (t.STATUS_ENTERING, t.STATUS_LEAVING)
                                 for t in trucks)
            _can_enter = next_t._pre_path or not _trucks_inside
            if _can_enter and not any_transiting:
                entry_x, entry_y = ENTRY_POINT
                active = [t for t in trucks if t.status not in (t.STATUS_WAITING,)]
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
            # Pass the remaining paths of all currently navigating trucks as locked paths
            # so the new exit routes don't cross through them.
            nav_locked = {t.id: ([t.front_center_cell(grid)] + list(t.path),
                                   t._dump_ticks_required + 2)
                          for t in trucks if t.status == t.STATUS_NAVIGATING and t.path}
            # Also lock the pre-planned entry paths of any waiting/entering trucks.
            for _wt in trucks:
                if (_wt.status in (_wt.STATUS_WAITING, _wt.STATUS_ENTERING)
                        and _wt._pre_path):
                    nav_locked[_wt.id] = (_wt._pre_path, _wt._dump_ticks_required + 2)

            # ── PHYSICAL CORRIDOR BUFFER for exit planning ────────────────────────
            # ignore_path_reserved=True makes the driveable mask treat PATH_RESERVED
            # cells as empty, so CBS space-time constraints alone cannot guarantee
            # corridor-level separation from a waiting truck's pre-path.
            # Fix: temporarily mark every cell inside
            #   (waiting_truck.width/2 + max_exit_truck.width/2)
            # of the waiting truck's path centreline as OBSTACLE — a state that is
            # never bypassed.  This forces the exit planner's centreline to stay at
            # least exit_hw beyond the pre-path's corridor boundary, guaranteeing
            # the exit truck's own corridor cannot clip the waiting truck's corridor.
            # Entry-corridor cells are excluded so all trucks can still reach the gate.
            _exit_hw_buf = max(
                int(math.ceil(t.width / 2.0 / grid.cell_size))
                for t in exiting_trucks
            )
            _exit_buf_saved: dict = {}
            for _wt in trucks:
                if (_wt.status in (_wt.STATUS_WAITING, _wt.STATUS_ENTERING)
                        and _wt._pre_path):
                    _wt_hw       = _wt.width / 2.0 / grid.cell_size
                    _total_half  = int(math.ceil(_wt_hw + _exit_hw_buf))
                    _total_radius = _wt_hw + _exit_hw_buf
                    for _pr, _pc in _path_cells(grid, _wt._pre_path):
                        for _bdr in range(-_total_half, _total_half + 1):
                            for _bdc in range(-_total_half, _total_half + 1):
                                if math.hypot(_bdr, _bdc) > _total_radius:
                                    continue
                                _nr, _nc = _pr + _bdr, _pc + _bdc
                                if not (0 <= _nr < grid.rows and 0 <= _nc < grid.cols):
                                    continue
                                # Don't block the entry funnel — all trucks converge here.
                                if math.hypot(_nr - entry_rc[0], _nc - entry_rc[1]) <= ENTRY_CORRIDOR_CELLS:
                                    continue
                                if (_nr, _nc) not in _exit_buf_saved:
                                    _s = int(grid.state[_nr, _nc])
                                    if _s not in (int(grid_map.CellState.BOUNDARY),
                                                  int(grid_map.CellState.OBSTACLE)):
                                        _exit_buf_saved[(_nr, _nc)] = _s
                                        grid.state[_nr, _nc] = grid_map.CellState.OBSTACLE
            # ─────────────────────────────────────────────────────────────────────

            exit_paths = plan_paths_cbs(
                grid, exit_assignments, locked_paths=nav_locked)

            # Restore cells that were temporarily marked OBSTACLE for corridor safety.
            for (_nr, _nc), _orig_s in _exit_buf_saved.items():
                if grid.state[_nr, _nc] == grid_map.CellState.OBSTACLE:
                    grid.state[_nr, _nc] = _orig_s

            for t, _ in exit_assignments:
                ep = exit_paths.get(t.id, [])
                if not ep:
                    ep = escape_and_replan_exit(t, grid, trucks, entry_rc)
                t.set_exit_path(ep, grid)

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
                        if t.status == t.STATUS_NAVIGATING and t.path:
                            all_locked[t.id] = ([t.front_center_cell(grid)] + list(t.path),
                                                 t._dump_ticks_required + 2)
                        elif t.status == t.STATUS_EXITING and t._exit_path:
                            all_locked[t.id] = ([t.front_center_cell(grid)] + list(t._exit_path),
                                                 0)
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
        active_trucks = [t for t in trucks if t.status != t.STATUS_WAITING]

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

                # Physical collision guard: CBS plans at cell resolution and treats
                # trucks as single points — path smoothing and physical footprints mean
                # world-space overlaps can still occur. Revert any move that brings
                # this truck's centre closer than the sum of the two half-lengths.
                collided = False
                for other in active_trucks:
                    if other is truck:
                        continue
                    dist = math.hypot(truck.pos[0] - other.pos[0],
                                      truck.pos[1] - other.pos[1])
                    if dist < (truck.length + other.length) * 0.5:
                        collided = True
                        break

                if collided:
                    truck._stuck_substeps += 1
                    truck.pos[0], truck.pos[1] = prev_x, prev_y
                    truck.heading = prev_heading
                    # Restore the waypoint that step() already popped so the path
                    # stays intact — without this the truck skips a waypoint next
                    # substep and appears to teleport.
                    if prev_first_wp is not None and (
                            not truck.path or truck.path[0] != prev_first_wp):
                        truck.path.insert(0, prev_first_wp)
                    if prev_exit_first_wp is not None and (
                            not truck._exit_path or truck._exit_path[0] != prev_exit_first_wp):
                        truck._exit_path.insert(0, prev_exit_first_wp)
                else:
                    truck._stuck_substeps = 0

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
                        continue   # not directly blocking each other

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
                        continue   # neither can safely retreat — fall through to idle

                    # Pick the reverser with the longer available retreat
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

        # ── STAGNATION UPDATE ─────────────────────────────────────────────────
        # Compare each truck's position now with the snapshot taken at tick start.
        # Trucks that have moved less than 0.5 m net this tick are "stagnant".
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
        # A headlock is two trucks stuck nose-to-nose for ~1 second (20 ticks).
        # Simple straight-reversal (above) doesn't fix it — one truck needs to
        # turn sideways to open a passing lane for the other.
        _HEADLOCK_TICKS = 20   # ~1 s at TICK_DELAY = 0.05 s

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

                # Must be physically close (within combined length × 0.75)
                _d = math.hypot(_ta.pos[0] - _tb.pos[0], _ta.pos[1] - _tb.pos[1])
                if _d >= (_ta.length + _tb.length) * 0.75:
                    continue

                # Must be facing each other: cos(angle between headings) < -0.5
                # i.e. headings are more than 120° apart → head-on
                if math.cos(_ta.heading - _tb.heading) > -0.5:
                    continue

                # The truck with the longer remaining path yields (it's farther
                # from its goal and has more room to manoeuvre back afterwards).
                _plen_a = len(_ta.path) if _ta.status == _ta.STATUS_NAVIGATING \
                          else len(_ta._exit_path)
                _plen_b = len(_tb.path) if _tb.status == _tb.STATUS_NAVIGATING \
                          else len(_tb._exit_path)
                _reverser = _ta if _plen_a >= _plen_b else _tb
                _advancer = _tb if _reverser is _ta else _ta

                _yield_wps = generate_yield_maneuver(_reverser, grid, _advancer)
                if not _yield_wps:
                    continue

                print(f"[HEADLOCK] T{_ta.id}↔T{_tb.id} nose-to-nose "
                      f"({_ta._pos_stagnant_ticks} stagnant ticks). "
                      f"T{_reverser.id} yielding ({len(_yield_wps)} steps).")

                if _reverser.status == _reverser.STATUS_NAVIGATING:
                    _reverser.path = _yield_wps + list(_reverser.path)
                else:
                    _reverser._exit_path = _yield_wps + list(_reverser._exit_path)

                _reverser._stuck_substeps    = 0
                _reverser._pos_stagnant_ticks = 0
                _reverser._conflict_cooldown  = len(_yield_wps) + 5
                _advancer._conflict_cooldown  = max(
                    _advancer._conflict_cooldown, len(_yield_wps) // 2)
                _headlock_handled = True
                break

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
                    _replanned = False
                    if _cta is not None and _ctb is not None:
                        _replanned = _try_inplace_replan(_cta, _ctb, grid, entry_rc)

                    # ── Fall back: yield the longer-path truck to IDLE ────────
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
                                _ct._exit_path = []  # triggers needs_exit_path() next tick
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
