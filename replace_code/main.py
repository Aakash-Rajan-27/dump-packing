# main.py

import sys
import time
import numpy as np
from scipy.ndimage import gaussian_filter

sys.stdout.reconfigure(encoding='utf-8')

from config import (POLYGON_BOUNDARY, ENTRY_POINT, CELL_SIZE,
                    FLEET_COMPOSITION, TICK_DELAY, STEPS_PER_TICK,
                    PYGAME_SCALE, PHEROMONE_DECAY, PHEROMONE_SPREAD_SIGMA)
import grid_map
from truck      import Truck
from filters    import get_candidates, is_accessible, make_driveable_mask
from scoring    import score_candidates
from mcts       import mcts_select_dump_points
from assignment import assign
from pathfinder import plan_paths, astar
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
    grid     = grid_map.GridMap(POLYGON_BOUNDARY, CELL_SIZE)
    entry_rc = grid.world_to_cell(*ENTRY_POINT)

    valid_cells = np.sum(grid.state == grid_map.CellState.EMPTY)
    print(f"Grid: {grid.rows}x{grid.cols} cells, {valid_cells} valid dump cells")
    print(f"Entry point: world{ENTRY_POINT} -> cell{entry_rc}")

    trucks = build_fleet()
    print(f"Fleet: {len(trucks)} trucks "
          f"({FLEET_COMPOSITION['small']}S / "
          f"{FLEET_COMPOSITION['medium']}M / "
          f"{FLEET_COMPOSITION['large']}L)")

    renderer   = Renderer(grid, scale=PYGAME_SCALE)
    print("Renderer ready - starting simulation loop")

    tick       = 0
    done       = False
    candidates = []

    while not done:

        if renderer.check_quit():
            print("User quit simulation")
            break

        # Step 1: Find idle trucks
        idle_trucks = [t for t in trucks if t.is_idle()]

        # Step 2-6: Plan dump assignments for idle trucks
        if idle_trucks:
            repr_truck = max(idle_trucks, key=lambda t: t.width * t.length)
            candidates = get_candidates(grid, repr_truck, entry_rc)

            if not candidates:
                print(f"\nSimulation complete at tick {tick}!")
                print(f"Final fill:   {grid.fill_pct()*100:.1f}%")
                print(f"Pack density: {grid.pack_pct()*100:.1f}%")
                done = True
                break

            scores         = score_candidates(grid, candidates)
            top_indices    = scores.argsort()[-30:][::-1]
            top_candidates = [candidates[i] for i in top_indices]

            top_candidates = [
                (r, c) for r, c in top_candidates
                if is_accessible(grid, r, c, entry_rc)
            ]

            if not top_candidates:
                time.sleep(0.1)
                continue

            n_needed    = len(idle_trucks)
            dump_points = mcts_select_dump_points(
                grid, top_candidates, repr_truck,
                n_trucks=n_needed, n_sim=50
            )

            if not dump_points:
                time.sleep(0.1)
                continue

            assignments_all = assign(
                idle_trucks[:len(dump_points)], dump_points, grid)

            if not assignments_all:
                time.sleep(0.1)
                continue

            paths = plan_paths(grid, assignments_all)

            for truck, dump_point in assignments_all:
                truck_path = paths.get(truck.id, [])
                truck.set_path(truck_path, dump_point, grid)

        # FIX: Plan exit paths for trucks that just finished dumping.
        # Cache mask by class to avoid recomputing for same truck type.
        exit_mask_cache = {}
        for truck in trucks:
            if truck.needs_exit_path():
                if truck.truck_class not in exit_mask_cache:
                    exit_mask_cache[truck.truck_class] = make_driveable_mask(
                        grid, truck)
                driveable  = exit_mask_cache[truck.truck_class]
                truck_cell = grid.world_to_cell(*truck.pos)
                exit_path  = astar(driveable, grid, truck_cell,
                                   entry_rc, truck)
                truck.set_exit_path(exit_path)

        # FIX: STEPS_PER_TICK — advance simulation multiple steps per frame
        # so trucks visibly move across the screen each rendered frame.
        for _ in range(STEPS_PER_TICK):
            for truck in trucks:
                truck.step(grid)

            grid.pheromone *= PHEROMONE_DECAY
            grid.pheromone  = gaussian_filter(grid.pheromone,
                                              sigma=PHEROMONE_SPREAD_SIGMA)
            np.clip(grid.pheromone, 0.0, 1.0, out=grid.pheromone)

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
            'FINAL': 'done',
            'fill%': f"{grid.fill_pct()*100:.1f}",
            'pack%': f"{grid.pack_pct()*100:.1f}",
        })
        time.sleep(0.1)

    renderer.close()


if __name__ == '__main__':
    run_simulation()