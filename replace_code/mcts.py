# mcts.py
# ─────────────────────────────────────────────────────────────
# FIX: rollout_score was rebuilding the dumpable cell list from scratch
# inside every simulation (50 sims × 20 depth = 1000 iterations).
# Each rebuild looped all grid.rows × grid.cols = 8281 cells.
# Total: 8.2M cell checks per planning tick just for MCTS.
#
# Fix: pass dumpable list as a parameter, computed once per tick
# in mcts_select_dump_points before the simulation loop.
# ─────────────────────────────────────────────────────────────

import math
import random
import numpy as np
from grid_map import CellState
from config import TARGET_PILE_HEIGHT, TRUCK_CLASSES, FLEET_COMPOSITION

_total_trucks = sum(FLEET_COMPOSITION.values())
_avg_pile_height_per_dump = sum(
    TRUCK_CLASSES[cls]['pile_height_per_dump'] * count
    for cls, count in FLEET_COMPOSITION.items()
) / max(1, _total_trucks)


class MCTSNode:
    def __init__(self, cell, parent=None):
        self.cell     = cell
        self.parent   = parent
        self.children = []
        self.visits   = 0
        self.value    = 0.0

    def ucb1(self, total_sims, c=math.sqrt(2)):
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits +
                c * math.sqrt(math.log(total_sims) / self.visits))


def rollout_score(base_state, base_heights, base_dumpable,
                  initial_cell, dump_add, grid, depth=20):
    """
    FIX: base_dumpable passed in — no longer rebuilt per rollout.
    Simulates dump at initial_cell then depth random future dumps.
    """
    heights = base_heights.copy()
    states  = base_state.copy()

    r0, c0 = initial_cell
    heights[r0, c0] = min(heights[r0, c0] + dump_add, TARGET_PILE_HEIGHT)
    states[r0, c0]  = (CellState.FILLED if heights[r0, c0] >= TARGET_PILE_HEIGHT
                       else CellState.PARTIAL)

    # Use the precomputed dumpable list, filter out now-filled cells
    dumpable = [cell for cell in base_dumpable
                if heights[cell[0], cell[1]] < TARGET_PILE_HEIGHT
                and cell != (r0, c0)]

    for _ in range(min(depth, len(dumpable))):
        if not dumpable:
            break
        idx = random.randint(0, len(dumpable) - 1)
        rr, cc = dumpable[idx]
        heights[rr, cc] = min(heights[rr, cc] + _avg_pile_height_per_dump,
                               TARGET_PILE_HEIGHT)
        if heights[rr, cc] >= TARGET_PILE_HEIGHT:
            states[rr, cc] = CellState.FILLED
            dumpable[idx]  = dumpable[-1]
            dumpable.pop()

    valid_mask = ((states == CellState.EMPTY)   |
                  (states == CellState.PARTIAL)  |
                  (states == CellState.FILLED)   |
                  (states == CellState.RESERVED))
    total_valid = np.sum(valid_mask)
    if total_valid == 0:
        return 0.0

    packing = (np.sum(np.clip(heights, 0, TARGET_PILE_HEIGHT) * valid_mask)
               / (total_valid * TARGET_PILE_HEIGHT))

    filled_set = set(zip(*np.where(heights >= TARGET_PILE_HEIGHT)))
    remaining  = [(r, c) for r, c in dumpable[:15]]
    isolated_penalty = sum(
        1 for rr, cc in remaining
        if sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
               if (rr+dr, cc+dc) in filled_set) >= 3
    )

    score = packing - 0.05 * (isolated_penalty / max(1, len(remaining)))
    return max(0.0, min(1.0, score))


def mcts_select_dump_points(grid, candidates, truck, n_trucks, n_sim=50):
    """
    FIX: dumpable list computed once here, passed into every rollout.
    Old: each rollout rebuilt the list (8281 cells × 1000 rollouts = 8M checks).
    New: one build, passed as parameter.
    """
    if not candidates:
        return []

    n_select     = min(n_trucks, len(candidates))
    nodes        = [MCTSNode(cell) for cell in candidates]
    base_state   = grid.state.copy()
    base_heights = grid.z_height.copy()
    dump_add     = truck.pile_height_per_dump

    # Build dumpable list ONCE
    base_dumpable = [
        (r, c) for r in range(grid.rows) for c in range(grid.cols)
        if base_state[r, c] in (CellState.EMPTY, CellState.PARTIAL,
                                 CellState.RESERVED)
        and base_heights[r, c] < TARGET_PILE_HEIGHT
    ]

    for sim_num in range(n_sim):
        total = sim_num + 1
        node  = max(nodes, key=lambda n: n.ucb1(total))
        score = rollout_score(base_state, base_heights, base_dumpable,
                              node.cell, dump_add, grid, depth=20)
        node.visits += 1
        node.value  += score

    nodes.sort(key=lambda n: n.visits, reverse=True)
    return [nodes[i].cell for i in range(n_select)]