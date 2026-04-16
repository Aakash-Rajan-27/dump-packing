# core/mcts.py
# ─────────────────────────────────────────────────────────────
# Monte Carlo Tree Search (MCTS) — makes the cell selection
# smarter by looking AHEAD at future consequences, not just
# scoring the current state.
#
# Without MCTS: pick the highest-scoring cell RIGHT NOW.
# With MCTS: simulate many possible future dump sequences
# and pick the cell that leads to the BEST final density.
#
# How it works:
# 1. Take top-N candidates from heuristic scoring
# 2. Build a tree where each node = "dump at this cell"
# 3. Run 200 simulations: pick a node, randomly dump 20 more cells,
#    score the final state
# 4. Use UCB1 formula to balance exploration vs exploitation
# 5. Return the most-visited node (= most consistently good choice)
# ─────────────────────────────────────────────────────────────

import math    # for math.sqrt, math.log
import random  # for random.choice in rollouts
import numpy as np
from grid_map import CellState


class MCTSNode:
    """
    One node in the MCTS tree.
    Represents the action of dumping at a specific cell.
    """

    def __init__(self, cell, parent=None):
        # The cell this node represents dumping at: (row, col)
        self.cell = cell

        # Parent node (None for root)
        self.parent = parent

        # Children of this node (populated on first visit via expansion)
        self.children = []

        # How many times this node has been visited during simulations
        self.visits = 0

        # Sum of all scores from simulations that passed through this node
        # Average value = self.value / self.visits
        self.value = 0.0

    def ucb1(self, total_sims, c=math.sqrt(2)):
        """
        UCB1 (Upper Confidence Bound 1) formula.
        Balances exploitation (high average value) vs exploration (low visits).

        total_sims: total simulations run so far
        c: exploration constant — sqrt(2) is standard

        Returns: UCB1 score — higher means "visit this node next"
        """
        if self.visits == 0:
            # Never visited → infinite score → always explore unvisited nodes first
            return float('inf')

        # exploitation term: average value of this node
        exploitation = self.value / self.visits

        # exploration term: sqrt(ln(total) / visits) — decreases as visits increase
        exploration = c * math.sqrt(math.log(total_sims) / self.visits)

        return exploitation + exploration
        # A node is chosen if it has high average value OR hasn't been visited enough


def rollout_score(base_state, initial_delta, grid, depth=20):
    """
    From a given state (with one dump applied), simulate 'depth' more
    random dumps and return a score for the final state.

    This is the "rollout" or "simulation" phase of MCTS.
    We use RANDOM future dumps (not optimised) — it's fast and
    statistically good enough when averaged over many simulations.

    base_state: the grid.state array at simulation start (we don't modify it)
    initial_delta: dict {(row,col): CellState} — the hypothetical dump for this node
    grid: GridMap (for dimensions)
    depth: how many random future dumps to simulate

    Returns: float in [0,1] — higher = better packing
    """
    # Build a working copy of state by applying the delta
    # We use a dict to avoid copying the full NumPy array (faster)
    filled = set()  # set of (row,col) cells that are filled in this simulation

    # Add all currently FILLED cells from the real grid
    filled_rows, filled_cols = np.where(base_state == CellState.FILLED)
    for r, c in zip(filled_rows, filled_cols):
        filled.add((r, c))

    # Apply the initial delta (the node's dump)
    for (r, c), state in initial_delta.items():
        if state == CellState.FILLED:
            filled.add((r, c))

    # Find all still-empty cells (inside polygon but not yet filled)
    empty = []
    for r in range(grid.rows):
        for c in range(grid.cols):
            if base_state[r, c] == CellState.EMPTY and (r, c) not in filled:
                empty.append((r, c))

    # Simulate 'depth' random future dumps
    for _ in range(min(depth, len(empty))):
        if not empty:
            break
        # Pick a random empty cell to dump at
        idx = random.randint(0, len(empty) - 1)
        chosen = empty[idx]
        filled.add(chosen)
        # Remove from empty list (swap with last for O(1) removal)
        empty[idx] = empty[-1]
        empty.pop()

    # ── Score the final state ──────────────────────────────
    total_valid = len(filled) + len(empty)  # filled + still empty
    if total_valid == 0:
        return 0.0

    # Primary metric: packing density = fraction of polygon that's filled
    density = len(filled) / total_valid

    # Penalty: count isolated empty cells (crude approximation)
    # A real check would do flood-fill, but that's too slow in a rollout
    # Simple proxy: cells with 4 filled neighbours are "trapped" and hard to fill
    isolated_penalty = 0
    for r, c in empty[:20]:    # only check first 20 for speed
        neighbours_filled = sum(
            1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
            if (r+dr, c+dc) in filled
        )
        if neighbours_filled >= 3:  # 3+ filled neighbours = nearly surrounded
            isolated_penalty += 1

    # Final score: density minus isolation penalty (normalised)
    score = density - 0.05 * (isolated_penalty / max(1, len(empty[:20])))
    return max(0.0, min(1.0, score))   # clamp to [0,1]


def mcts_select_dump_points(grid, candidates, n_trucks, n_sim=200):
    """
    Use MCTS to select the best N dump points from the candidates list.
    N = n_trucks (one dump point per idle truck).

    grid: GridMap
    candidates: list of (row,col) — pre-filtered valid cells
    n_trucks: how many dump points to select
    n_sim: number of MCTS simulations to run

    Returns: list of (row,col) — the selected dump points, length = n_trucks
    """
    if not candidates:
        return []

    # Cap n_trucks to available candidates
    n_select = min(n_trucks, len(candidates))

    # Create one MCTS node per candidate cell
    nodes = [MCTSNode(cell) for cell in candidates]

    # Take a snapshot of current grid state (read-only reference)
    base_state = grid.state   # we never modify this

    # ── Run simulations ────────────────────────────────────
    for sim_num in range(n_sim):
        # Total simulations done so far (for UCB1 calculation)
        total = sim_num + 1

        # SELECT: pick the node with highest UCB1 score
        node = max(nodes, key=lambda n: n.ucb1(total))

        # SIMULATE (rollout): from this node's dump, simulate future dumps
        delta = {node.cell: CellState.FILLED}  # hypothetical: fill this cell
        score = rollout_score(base_state, delta, grid, depth=20)

        # BACKPROPAGATE: update this node's stats
        node.visits += 1       # increment visit count
        node.value  += score   # accumulate score (we use average later)

    # ── Select best nodes ──────────────────────────────────
    # Sort by visit count — most visited = most consistently good
    nodes.sort(key=lambda n: n.visits, reverse=True)

    # Return the top N cells
    return [nodes[i].cell for i in range(n_select)]
