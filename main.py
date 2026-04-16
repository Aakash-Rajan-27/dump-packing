# main.py
# ─────────────────────────────────────────────────────────────
# THE SIMULATION LOOP — glues every module together.
#
# Each "tick" of the loop:
#   1. Check for idle trucks
#   2. Get valid candidate cells (filter)
#   3. Score candidates (heuristic)
#   4. Run MCTS to pick best dump points
#   5. Assign trucks to dump points (Hungarian)
#   6. Plan collision-free paths (A* / CBS)
#   7. Move trucks one step
#   8. Update grid (pheromone decay)
#   9. Draw frame to screen
#  10. Check if done (no valid candidates left)
# ─────────────────────────────────────────────────────────────

import time
import numpy as np
from scipy.ndimage import gaussian_filter   # for pheromone spatial spreading

# Our modules
from config import (POLYGON_BOUNDARY, ENTRY_POINT, CELL_SIZE,
                    NUM_TRUCKS, TICK_DELAY, PYGAME_SCALE,
                    PHEROMONE_DECAY, PHEROMONE_SPREAD_SIGMA)
from core.grid_map  import GridMap, CellState
from core.truck     import Truck
from core.filters   import get_candidates
from core.scoring   import score_candidates
from core.mcts      import mcts_select_dump_points
from core.assignment import assign
from core.pathfinder import   rt plan_paths   # swap to plan_paths_cbs in week 2
from viz.renderer   import Renderer


def run_simulation():
    """Main simulation entry point. Call this to start everything."""

    # ── Initialise world ───────────────────────────────────
    print("Initialising grid...")
    grid = GridMap(POLYGON_BOUNDARY, CELL_SIZE)

    # Count how many valid cells we have
    valid_cells = np.sum(grid.state == CellState.EMPTY)
    print(f"Grid: {grid.rows}×{grid.cols} cells, {valid_cells} valid dump cells")

    # Find the entry point cell (used as flood-fill root)
    entry_rc = grid.world_to_cell(*ENTRY_POINT)
    print(f"Entry point: world{ENTRY_POINT} → cell{entry_rc}")

    # ── Create trucks ──────────────────────────────────────
    # All trucks start at the entry point, idle
    trucks = [Truck(i, ENTRY_POINT) for i in range(NUM_TRUCKS)]
    print(f"Created {NUM_TRUCKS} trucks")

    # ── Set up visualiser ──────────────────────────────────
    renderer = Renderer(grid, scale=PYGAME_SCALE)
    print("Renderer ready — starting simulation loop")

    tick = 0        # tick counter (increments every loop iteration)
    done = False    # set to True when no valid cells remain

    # ── Main loop ──────────────────────────────────────────
    while not done:

        # Step 0: Check if user closed the window
        if renderer.check_quit():
            print("User quit simulation")
            break

        # ── Step 1: Find idle trucks ───────────────────────
        # Idle = waiting for a new dump assignment
        idle_trucks = [t for t in trucks if t.is_idle()]

        # ── Step 2–6: Plan for idle trucks ────────────────
        if idle_trucks:
            # Get all valid candidate cells (filtered)
            # This is the most expensive step — runs flood-fill for each candidate
            candidates = get_candidates(grid, idle_trucks[0], entry_rc)
            # We pass idle_trucks[0] for footprint dimensions (same for all in week 1)

            if not candidates:
                # No valid cells left → simulation is done
                print(f"\nSimulation complete at tick {tick}!")
                print(f"Final fill: {grid.fill_pct()*100:.1f}%")
                done = True
                break

            # Step 3: Score all candidates with heuristics
            scores = score_candidates(grid, candidates)
            # scores is a 1D array, scores[i] corresponds to candidates[i]

            # Step 3b: Take top-30 by score (no point running MCTS on all 500+ cells)
            # argsort gives indices that would sort the array ascending
            # [-30:] takes last 30 (highest), [::-1] reverses to descending
            top_indices = scores.argsort()[-30:][::-1]
            top_candidates = [candidates[i] for i in top_indices]

            # Step 4: MCTS selects the best N cells from top candidates
            # N = number of idle trucks (one dump point per truck)
            n_needed = len(idle_trucks)
            dump_points = mcts_select_dump_points(
                grid, top_candidates, n_trucks=n_needed, n_sim=200
            )

            if not dump_points:
                # MCTS returned nothing (shouldn't happen if candidates is non-empty)
                time.sleep(0.1)
                continue

            # Step 5: Hungarian assignment — match trucks to dump points
            assignments = assign(idle_trucks[:len(dump_points)], dump_points, grid)

            # Step 6: Plan paths for each assigned truck
            paths = plan_paths(grid, assignments)

            # Give each truck its path
            for truck, dump_point in assignments:
                truck_path = paths.get(truck.id, [])
                truck.set_path(truck_path, dump_point, grid)

        # ── Step 7: Move all trucks one step ──────────────
        for truck in trucks:
            truck.step(grid)   # each truck advances one step along its path

        # ── Step 8: Update pheromone grid ─────────────────
        # Multiply entire pheromone array by decay factor
        # 0.85 means 15% decay per tick — values drift back toward 1 over time
        grid.pheromone *= PHEROMONE_DECAY

        # Spread pheromone spatially — nearby cells also get slightly suppressed
        # gaussian_filter blurs the array with the given sigma (in cell units)
        grid.pheromone = gaussian_filter(grid.pheromone, sigma=PHEROMONE_SPREAD_SIGMA)

        # Clamp pheromone to [0, 1] in case of floating point drift
        np.clip(grid.pheromone, 0.0, 1.0, out=grid.pheromone)

        # ── Step 9: Draw the frame ─────────────────────────
        metrics = {
            'tick':      tick,
            'trucks':    NUM_TRUCKS,
            'idle':      len(idle_trucks),
            'candidates': len(candidates) if idle_trucks else '-',
        }
        renderer.draw(trucks, metrics)

        # ── Step 10: Control simulation speed ─────────────
        # time.sleep pauses the loop so the animation is watchable
        # Remove this line (or set TICK_DELAY=0) to run at full speed
        time.sleep(TICK_DELAY)

        tick += 1   # increment tick counter

    # ── Simulation ended — show final stats ────────────────
    print("\n=== Final Results ===")
    print(f"Total ticks: {tick}")
    print(f"Final fill:  {grid.fill_pct()*100:.1f}%")

    filled_count = np.sum(grid.state == CellState.FILLED)
    print(f"Cells filled: {filled_count}")
    print(f"Autonomous baseline spacing: 7.38m")
    print(f"Staffed target spacing:      3.03m")
    # TODO: add nearest-neighbour distance calculation for your final metric

    # Keep the window open so you can look at the final state
    print("\nClose the window to exit.")
    while not renderer.check_quit():
        renderer.draw(trucks, {'FINAL': 'done', 'fill%': f"{grid.fill_pct()*100:.1f}"})
        time.sleep(0.1)

    renderer.close()


# Standard Python entry point guard:
# Only run if this file is executed directly (not imported as a module)
if __name__ == '__main__':
    run_simulation()
