# main.py
# ─────────────────────────────────────────────────────────────
# THE SIMULATION LOOP — mixed fleet edition.
#
# Fleet: 3 small (Cat 773F) + 3 medium (Cat 785D) + 3 large (Cat 793F)
# Each truck class has different physical dimensions, payload,
# turning radius, and per-dump pile height contribution.
#
# Partial pile logic: cells accumulate height over multiple dumps.
# get_candidates() returns EMPTY + PARTIAL cells.
# A cell graduates to FILLED only when z_height >= TARGET_PILE_HEIGHT.
# ─────────────────────────────────────────────────────────────

import sys
import time
import numpy as np
from scipy.ndimage import gaussian_filter

sys.stdout.reconfigure(encoding='utf-8')

from config import (POLYGON_BOUNDARY, ENTRY_POINT, CELL_SIZE,
                    FLEET_COMPOSITION, TICK_DELAY, PYGAME_SCALE,
                    PHEROMONE_DECAY, PHEROMONE_SPREAD_SIGMA)
import grid_map
from truck      import Truck
from filters    import get_candidates, is_accessible
from scoring    import score_candidates
from mcts       import mcts_select_dump_points
from assignment import assign
from pathfinder import plan_paths
from renderer   import Renderer


def build_fleet():
    """
    Instantiate trucks according to FLEET_COMPOSITION.
    Returns a flat list of Truck objects, sorted small → medium → large.
    """
    trucks = []
    truck_id = 0
    for cls in ('small', 'medium', 'large'):
        for _ in range(FLEET_COMPOSITION.get(cls, 0)):
            trucks.append(Truck(truck_id, cls))
            truck_id += 1
    return trucks


def run_simulation():

    # ── World ───────────────────────────────────────────────
    print("Initialising grid...")
    grid = grid_map.GridMap(POLYGON_BOUNDARY, CELL_SIZE)

    valid_cells = np.sum(grid.state == grid_map.CellState.EMPTY)
    print(f"Grid: {grid.rows}x{grid.cols} cells, {valid_cells} valid dump cells")

    entry_rc = grid.world_to_cell(*ENTRY_POINT)
    print(f"Entry point: world{ENTRY_POINT} -> cell{entry_rc}")

    # ── Fleet ───────────────────────────────────────────────
    trucks = build_fleet()
    total_trucks = len(trucks)
    print(f"Fleet: {total_trucks} trucks "
          f"({FLEET_COMPOSITION['small']}S / "
          f"{FLEET_COMPOSITION['medium']}M / "
          f"{FLEET_COMPOSITION['large']}L)")

    # ── Renderer ────────────────────────────────────────────
    renderer = Renderer(grid, scale=PYGAME_SCALE)
    print("Renderer ready - starting simulation loop")

    tick       = 0
    done       = False
    candidates = []

    # ── Main Loop ───────────────────────────────────────────
    while not done:

        if renderer.check_quit():
            print("User quit simulation")
            break

        # Step 1: Find idle trucks
        idle_trucks = [t for t in trucks if t.is_idle()]

        # Step 2-6: Plan for idle trucks
        if idle_trucks:

            # Use the largest idle truck for footprint filtering
            # (if it passes, smaller trucks definitely pass too)
            # Pick representative truck by largest footprint
            repr_truck = max(idle_trucks, key=lambda t: t.width * t.length)

            # Step 2: Fast footprint filter (all dumpable cells)
            candidates = get_candidates(grid, repr_truck, entry_rc)

            if not candidates:
                print(f"\nSimulation complete at tick {tick}!")
                print(f"Final fill:    {grid.fill_pct()*100:.1f}%")
                print(f"Pack density:  {grid.pack_pct()*100:.1f}%")
                done = True
                break

            # Step 3: Score all candidates
            scores = score_candidates(grid, candidates)

            # Step 3b: Top-30 by heuristic score
            top_indices   = scores.argsort()[-30:][::-1]
            top_candidates = [candidates[i] for i in top_indices]

            # Step 3c: Accessibility filter on top-30 only (flood-fill, expensive)
            top_candidates = [
                (r, c) for r, c in top_candidates
                if is_accessible(grid, r, c, entry_rc)
            ]

            if not top_candidates:
                time.sleep(0.1)
                continue

            # Step 4: Group idle trucks by class and plan per-class
            # Each class uses its own truck object for MCTS rollout accuracy.
            # We process classes in order: large first (biggest impact per dump).
            assignments_all = []
            claimed_points  = set()   # prevent two trucks going to same cell
            remaining_idle  = list(idle_trucks)

            for cls in ('large', 'medium', 'small'):
                cls_trucks = [t for t in remaining_idle
                              if t.truck_class == cls]
                if not cls_trucks:
                    continue

                # Filter out already-claimed points
                avail = [(r, c) for r, c in top_candidates
                         if (r, c) not in claimed_points]
                if not avail:
                    break

                n_needed   = len(cls_trucks)
                dump_points = mcts_select_dump_points(
                    grid, avail, cls_trucks[0], n_trucks=n_needed, n_sim=200
                )

                if not dump_points:
                    continue

                these_assignments = assign(
                    cls_trucks[:len(dump_points)], dump_points, grid
                )
                assignments_all.extend(these_assignments)
                for _, dp in these_assignments:
                    claimed_points.add(dp)
                for t, _ in these_assignments:
                    remaining_idle.remove(t)

            if not assignments_all:
                time.sleep(0.1)
                continue

            # Step 6: Path plan
            paths = plan_paths(grid, assignments_all)

            for truck, dump_point in assignments_all:
                truck_path = paths.get(truck.id, [])
                truck.set_path(truck_path, dump_point, grid)

        # Step 7: Move trucks
        for truck in trucks:
            truck.step(grid)

        # Step 8: Pheromone decay + spread
        grid.pheromone *= PHEROMONE_DECAY
        grid.pheromone  = gaussian_filter(grid.pheromone,
                                          sigma=PHEROMONE_SPREAD_SIGMA)
        np.clip(grid.pheromone, 0.0, 1.0, out=grid.pheromone)

        # Step 9: Draw
        metrics = {
            'tick':       tick,
            'fleet':      f"{FLEET_COMPOSITION['small']}S/{FLEET_COMPOSITION['medium']}M/{FLEET_COMPOSITION['large']}L",
            'idle':       len(idle_trucks),
            'candidates': len(candidates) if candidates else '-',
            'fill%':      f"{grid.fill_pct()*100:.1f}",
            'pack%':      f"{grid.pack_pct()*100:.1f}",
        }
        renderer.draw(trucks, metrics)

        time.sleep(TICK_DELAY)
        tick += 1

    # ── Final Stats ─────────────────────────────────────────
    print("\n=== Final Results ===")
    print(f"Total ticks:   {tick}")
    print(f"Cells filled:  {np.sum(grid.state == grid_map.CellState.FILLED)}")
    print(f"Cells partial: {np.sum(grid.state == grid_map.CellState.PARTIAL)}")
    print(f"Fill %:        {grid.fill_pct()*100:.1f}%")
    print(f"Pack density:  {grid.pack_pct()*100:.1f}%")
    print(f"Autonomous baseline spacing: 7.38m")
    print(f"Staffed target spacing:      3.03m")

    print("\nClose the window to exit.")
    while not renderer.check_quit():
        renderer.draw(trucks, {
            'FINAL':  'done',
            'fill%':  f"{grid.fill_pct()*100:.1f}",
            'pack%':  f"{grid.pack_pct()*100:.1f}",
        })
        time.sleep(0.1)

    renderer.close()


if __name__ == '__main__':
    run_simulation()
