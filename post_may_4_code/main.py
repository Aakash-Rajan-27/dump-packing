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
import numpy as np
from scipy.ndimage import gaussian_filter

sys.stdout.reconfigure(encoding='utf-8')

from config import (POLYGON_BOUNDARY, ENTRY_POINT, CELL_SIZE,
                    FLEET_COMPOSITION, TICK_DELAY, PYGAME_SCALE,
                    PHEROMONE_DECAY, PHEROMONE_SPREAD_SIGMA,
                    CONFIG_MATERIAL_HEIGHT_THRESHOLD, STEPS_PER_TICK)
import grid_map
from truck      import Truck
from filters    import get_raw_candidates, is_accessible, precompute_coarse_blocked_mask
from scoring    import score_candidates
from mcts       import mcts_select_dump_points
from assignment import assign
from pathfinder import plan_paths_cbs
from renderer   import Renderer


def build_fleet():
    trucks = []
    truck_id = 0
    for cls in ('small', 'medium', 'large'):
        for _ in range(FLEET_COMPOSITION.get(cls, 0)):
            trucks.append(Truck(truck_id, cls))
            truck_id += 1
    return trucks


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

    renderer = Renderer(grid, scale=PYGAME_SCALE)
    print("Renderer ready - starting simulation loop")

    tick       = 0
    done       = False

    while not done:
        if renderer.check_quit():
            break

        # Check for trucks needing exit paths
        exiting_trucks = [t for t in trucks if t.needs_exit_path()]
        if exiting_trucks:
            exit_assignments = [(t, entry_rc) for t in exiting_trucks]
            # Pass the remaining paths of all currently navigating trucks as locked paths
            # so the new exit routes don't cross through them.
            nav_locked = {t.id: [grid.world_to_cell(*t.pos)] + list(t.path)
                          for t in trucks if t.status == t.STATUS_NAVIGATING and t.path}
            exit_paths = plan_paths_cbs(grid, exit_assignments, locked_paths=nav_locked)
            for t, _ in exit_assignments:
                t.set_exit_path(exit_paths.get(t.id, []))

        idle_trucks = [t for t in trucks if t.is_idle()]

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
                            all_locked[t.id] = [grid.world_to_cell(*t.pos)] + list(t.path)
                        elif t.status == t.STATUS_EXITING and t._exit_path:
                            all_locked[t.id] = [grid.world_to_cell(*t.pos)] + list(t._exit_path)
                    paths = plan_paths_cbs(grid, assignments_all, locked_paths=all_locked)
                    for truck, dump_point in assignments_all:
                        truck_path = paths.get(truck.id, [])
                        truck.set_path(truck_path, dump_point, grid)

        # Movement / render phase. Draw every fine truck step; batching several
        # steps before drawing hides the interpolated arc and makes turns look
        # like stop-rotate-go.
        for _ in range(STEPS_PER_TICK):
            if renderer.check_quit():
                done = True
                break

            for truck in trucks:
                truck.step(grid)

            grid.pheromone = 1.0 - (1.0 - grid.pheromone) * PHEROMONE_DECAY
            grid.pheromone = gaussian_filter(grid.pheromone, sigma=PHEROMONE_SPREAD_SIGMA)
            np.clip(grid.pheromone, 0.0, 1.0, out=grid.pheromone)

            metrics = {
                'tick':       tick,
                'fleet':      f"{FLEET_COMPOSITION['small']}S/{FLEET_COMPOSITION['medium']}M/{FLEET_COMPOSITION['large']}L",
                'idle':       len([t for t in trucks if t.is_idle()]),
                'candidates': len(top_candidates) if 'top_candidates' in locals() else '-',
                'fill%':      f"{grid.fill_pct()*100:.3f}",
                'pack%':      f"{grid.pack_pct()*100:.3f}",
            }
            renderer.draw(trucks, metrics)

            time.sleep(TICK_DELAY)
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