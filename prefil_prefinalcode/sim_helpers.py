# sim_helpers.py
# ─────────────────────────────────────────────────────────────
# Simulation-level helpers used by run_simulation():
#   • initialise_half_full_dump() — seed initial terrain
#   • _corridor_cells()           — expand path to corridor set
#   • build_fleet()               — create all Truck objects
#   • _try_inplace_replan()       — resolve a 2-truck conflict
#                                   via CBS without forcing idle
# ─────────────────────────────────────────────────────────────

import math
import random
import numpy as np

import grid_map
from truck import Truck
from pathfinder import (plan_staging_paths, plan_paths_cbs, _path_cells)
from config import ENTRY_POINT, FLEET_COMPOSITION


def initialise_prefill(grid):
    """
    Pre-seed the dump area to ~38% pack density — sparse, with clear entry corridors.

    Spatial layout:
      Far zone  (dist > 50% of diagonal): medium cones at ~65% of sites.
                Outer boundary walls partially filled, even spacing.
      Mid zone  (25-50%): smaller cones at ~50% of sites. Visible gaps between
                dump groups so trucks can weave through.
      Entry zone (<25%):  very sparse — only ~20% of sites, tiny cones.
                Wide open near the gate for truck queuing and fanout.

    Navigation corridors (6 m wide strips) radiate from the entry point in
    three directions (straight up, upper-left, upper-right).  Any candidate
    dump centre that falls inside a corridor is always skipped, regardless of
    zone.  This guarantees clear lanes even at higher fill states.

    All heights come from grid.dump_at() physics (angle-of-repose cone + relaxation).
    """
    from config import ENTRY_POINT, _TAN_REPOSE, TARGET_PILE_HEIGHT

    TARGET_PACK   = 0.38
    CORRIDOR_HALF = 6.0 / grid.cell_size   # 6 m clearance either side of corridor axis
    tan_theta     = _TAN_REPOSE

    entry_r, entry_c = grid.world_to_cell(*ENTRY_POINT)
    max_dist_cells    = math.hypot(grid.rows, grid.cols)

    # V = π·tan(θ)·r³/3,  r = H/tan(θ)
    def _vol(H):
        r = H / tan_theta
        return math.pi * tan_theta * r ** 3 / 3.0

    V_far  = _vol(TARGET_PILE_HEIGHT * 0.70)   # ≈ 91 m³ → 3.5 m peak
    V_mid  = _vol(TARGET_PILE_HEIGHT * 0.55)   # ≈ 44 m³ → 2.75 m peak
    V_near = _vol(TARGET_PILE_HEIGHT * 0.35)   # ≈ 11 m³ → 1.75 m peak

    # ── Corridor exclusion: 3 rays from the entry gate ────────────────
    # Each ray is a unit-vector direction.  A candidate cell is "in corridor"
    # if its perpendicular distance to any ray (for the portion ahead of entry)
    # is less than CORRIDOR_HALF cells.
    _corridor_dirs = [
        (1.0,  0.0),          # straight up (main ingress / egress lane)
        (0.75, -0.75),        # upper-left fan  (normalized below)
        (0.75,  0.75),        # upper-right fan
    ]
    _corridor_dirs = [(dr / math.hypot(dr, dc), dc / math.hypot(dr, dc))
                      for dr, dc in _corridor_dirs]

    def _in_corridor(r, c):
        for dr, dc in _corridor_dirs:
            # Project (r,c) onto the ray from entry
            t  = (r - entry_r) * dr + (c - entry_c) * dc
            if t < 0:
                continue   # behind entry — not in this corridor
            perp = math.hypot((r - entry_r) - t * dr,
                              (c - entry_c) - t * dc)
            if perp <= CORRIDOR_HALF:
                return True
        return False

    # ── 1. Candidate centres on 5 m jittered grid ─────────────────────
    spacing = max(1, int(round(5.0 / grid.cell_size)))
    rng     = random.Random(7)

    centers = []
    for r0 in range(spacing // 2, grid.rows, spacing):
        for c0 in range(spacing // 2, grid.cols, spacing):
            jr = max(0, min(r0 + rng.randint(-1, 1), grid.rows - 1))
            jc = max(0, min(c0 + rng.randint(-1, 1), grid.cols - 1))
            if grid.state[jr, jc] not in (grid_map.CellState.EMPTY,
                                           grid_map.CellState.PARTIAL):
                continue
            if _in_corridor(jr, jc):
                continue   # always leave corridor cells empty
            d = math.hypot(jr - entry_r, jc - entry_c)
            centers.append((d, jr, jc))

    # Far-first: outer boundary fills before mid / entry zones
    centers.sort(reverse=True)

    # ── 2. Deposit until target density ───────────────────────────────
    for d, r, c in centers:
        if grid.pack_pct() >= TARGET_PACK:
            break

        if grid.state[r, c] in (grid_map.CellState.BOUNDARY,
                                  grid_map.CellState.PROTECTED,
                                  grid_map.CellState.OBSTACLE,
                                  grid_map.CellState.FILLED):
            continue

        ratio = d / max_dist_cells

        if ratio < 0.25:
            if rng.random() < 0.80:   # keep only ~20% near entry
                continue
            vol = V_near * rng.uniform(0.7, 1.2)
        elif ratio < 0.50:
            if rng.random() < 0.50:   # keep ~50% in mid zone
                continue
            vol = V_mid * rng.uniform(0.8, 1.2)
        else:
            if rng.random() < 0.35:   # keep ~65% in far zone
                continue
            vol = V_far * rng.uniform(0.75, 1.0)

        grid.dump_at(r, c, volume_m3=vol)

    print(f"Pre-fill complete. Pack density = {grid.pack_pct()*100:.1f}%  "
          f"Fill % = {grid.fill_pct()*100:.1f}%")


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
