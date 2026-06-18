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
