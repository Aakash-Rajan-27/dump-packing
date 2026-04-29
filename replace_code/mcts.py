# mcts.py
# ─────────────────────────────────────────────────────────────
# MCTS forward simulation — updated for partial pile dynamics.
#
# Key change: the rollout now tracks z_height per cell, not just
# a binary filled/empty set. A cell is "done" only when its height
# reaches TARGET_PILE_HEIGHT. A single dump from a large truck may
# fill multiple cells in one shot; a small truck may need several
# dumps to fill one cell. The rollout approximates this by using
# the average pile_height_per_dump across the fleet.
# ─────────────────────────────────────────────────────────────

import math
import random
import numpy as np
from grid_map import CellState
from config import TARGET_PILE_HEIGHT, TRUCK_CLASSES, FLEET_COMPOSITION


# Pre-compute the average pile_height_per_dump across the whole fleet.
# Used in rollouts to estimate how much one "generic" future dump contributes.
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


def rollout_score(base_state, base_heights, initial_cell, dump_add, grid, depth=20):
    """
    From a hypothetical dump at initial_cell, simulate 'depth' more
    random dumps using average fleet dump height. Score the final state.

    base_state   : grid.state snapshot (read-only)
    base_heights : grid.z_height snapshot (read-only)
    initial_cell : (r, c) — the node's dump
    dump_add     : pile_height_per_dump for this specific truck
    grid         : GridMap (for dims)
    depth        : future dumps to simulate
    """
    # Work with height array — tracks partial fill numerically
    heights = base_heights.copy()
    states  = base_state.copy()

    # Apply initial dump
    r0, c0 = initial_cell
    heights[r0, c0] = min(heights[r0, c0] + dump_add, TARGET_PILE_HEIGHT)
    if heights[r0, c0] >= TARGET_PILE_HEIGHT:
        states[r0, c0] = CellState.FILLED
    else:
        states[r0, c0] = CellState.PARTIAL

    # Collect still-dumpable cells
    dumpable = []
    for r in range(grid.rows):
        for c in range(grid.cols):
            if states[r, c] in (CellState.EMPTY, CellState.PARTIAL,
                                 CellState.RESERVED):
                if heights[r, c] < TARGET_PILE_HEIGHT:
                    dumpable.append((r, c))

    # Simulate 'depth' random future dumps with average fleet height
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

    # ── Score the final state ───────────────────────────────
    total_valid = np.sum(
        (states == CellState.EMPTY)    |
        (states == CellState.PARTIAL)  |
        (states == CellState.FILLED)   |
        (states == CellState.RESERVED)
    )
    if total_valid == 0:
        return 0.0

    # Packing density: total height filled / theoretical max
    valid_mask = ((states == CellState.EMPTY)   |
                  (states == CellState.PARTIAL)  |
                  (states == CellState.FILLED)   |
                  (states == CellState.RESERVED))
    packing = (np.sum(np.clip(heights, 0, TARGET_PILE_HEIGHT) * valid_mask)
               / (total_valid * TARGET_PILE_HEIGHT))

    # Isolation penalty (fast heuristic — not full BFS)
    filled_set = set(zip(*np.where(heights >= TARGET_PILE_HEIGHT)))
    remaining  = [(r, c) for r in range(grid.rows)
                  for c in range(grid.cols)
                  if states[r, c] in (CellState.EMPTY, CellState.PARTIAL)
                  and heights[r, c] < TARGET_PILE_HEIGHT]

    isolated_penalty = 0
    for rr, cc in remaining[:15]:
        n_filled_neighbours = sum(
            1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
            if (rr+dr, cc+dc) in filled_set
        )
        if n_filled_neighbours >= 3:
            isolated_penalty += 1

    score = packing - 0.05 * (isolated_penalty / max(1, min(15, len(remaining))))
    return max(0.0, min(1.0, score))


def mcts_select_dump_points(grid, candidates, truck, n_trucks, n_sim=200):
    """
    Select the best n_trucks dump points from candidates using MCTS.

    truck : the Truck object making this planning call — used to get
            pile_height_per_dump for accurate rollout simulation.
    """
    if not candidates:
        return []

    n_select = min(n_trucks, len(candidates))
    nodes    = [MCTSNode(cell) for cell in candidates]

    base_state   = grid.state.copy()
    base_heights = grid.z_height.copy()
    dump_add     = truck.pile_height_per_dump

    for sim_num in range(n_sim):
        total = sim_num + 1
        node  = max(nodes, key=lambda n: n.ucb1(total))
        score = rollout_score(base_state, base_heights,
                              node.cell, dump_add, grid, depth=20)
        node.visits += 1
        node.value  += score

    nodes.sort(key=lambda n: n.visits, reverse=True)
    return [nodes[i].cell for i in range(n_select)]
