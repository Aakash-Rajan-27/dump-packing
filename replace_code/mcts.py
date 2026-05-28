# mcts.py
# Monte Carlo Tree Search for selecting the best dump sequence.
#
# The goal: given N trucks that need dump assignments, find the sequence of
# N cells (one per truck) that will lead to the best overall packing density
# after many future dumps.
#
# How it works:
#   1. Build a tree rooted at the current grid state.
#   2. Run n_sim simulations. Each simulation:
#        a. SELECTION: Walk the tree, picking children by UCB1 score.
#        b. EXPANSION: If the selected node has been visited but not expanded,
#           grow the tree by scoring the top-M candidate cells as children.
#        c. ROLLOUT: From the selected node, simulate random future dumps and
#           score the resulting packing density.
#        d. BACKPROPAGATION: Push the score back up the path to the root.
#   3. After all simulations, walk the tree greedily (most-visited child at
#      each level) to extract the best N cells — one per truck.

import math
import random
import numpy as np
from grid_map import CellState
from config import TARGET_PILE_HEIGHT, TRUCK_CLASSES, FLEET_COMPOSITION
from scoring import score_candidates

# Pre-compute a fleet-average pile height per dump for use in rollouts.
# Using an average avoids needing per-truck detail during fast simulation.
_total_trucks = sum(FLEET_COMPOSITION.values())
_avg_pile_height_per_dump = sum(
    TRUCK_CLASSES[cls]['pile_height_per_dump'] * count
    for cls, count in FLEET_COMPOSITION.items()
) / max(1, _total_trucks)


class MCTSNode:
    """
    One node in the MCTS tree. Represents choosing `cell` as the next dump location.
    The root node has cell=None (it represents "no dump yet, just the current state").
    """
    def __init__(self, cell, parent=None):
        self.cell     = cell      # (row, col) this node represents, or None for root
        self.parent   = parent
        self.children = []
        self.visits   = 0         # how many simulations passed through this node
        self.value    = 0.0       # accumulated rollout scores through this node
        self.is_expanded = False  # True once children have been added

    def ucb1(self, parent_visits, c=math.sqrt(2)):
        """
        Upper Confidence Bound score for tree traversal.
        Balances exploitation (value/visits) with exploration (sqrt term).
        Unvisited nodes return infinity so they're always tried first.
        """
        if self.visits == 0: return float('inf')
        return (self.value / self.visits +
                c * math.sqrt(math.log(parent_visits) / self.visits))


def get_top_m_next_moves(base_state, base_heights, path_so_far, grid, base_dumpable, dump_add, m=6):
    """
    Given the grid state and the sequence of dumps chosen so far (path_so_far),
    simulate what the grid would look like after those dumps, then return the
    top-M scoring candidate cells for the next dump.

    This is used during EXPANSION to decide which children to add to a node.
    Using score_candidates with a hypothetical state ensures we're scoring
    cells relative to the future grid, not the current one.
    """
    # Build a hypothetical grid state by applying all dumps in path_so_far
    hypothetical_state   = base_state.copy()
    hypothetical_heights = base_heights.copy()
    empty = list(base_dumpable)

    for cell in path_so_far:
        if cell is not None:
            r, c = cell
            hypothetical_heights[r, c] = min(hypothetical_heights[r, c] + dump_add, TARGET_PILE_HEIGHT)
            hypothetical_state[r, c]   = CellState.FILLED if hypothetical_heights[r, c] >= TARGET_PILE_HEIGHT else CellState.PARTIAL
            # Remove cells that just became full from the candidate pool
            if hypothetical_heights[r, c] >= TARGET_PILE_HEIGHT:
                try: empty.remove(cell)
                except ValueError: pass

    if not empty: return []

    # Score remaining candidates against the hypothetical future state
    scores = score_candidates(grid, empty, state_override=hypothetical_state)
    scored_empty = list(zip(scores, empty))
    scored_empty.sort(key=lambda x: x[0], reverse=True)

    return [cell for score, cell in scored_empty[:m]]


def rollout_score_path(base_state, base_heights, path, base_dumpable, dump_add, grid, depth=20):
    """
    Simulate the grid forward from base state, first applying the chosen
    path of dumps, then randomly dumping `depth` more times to approximate
    future packing. Returns a score in [0, 1] where 1 = perfectly packed.

    This is the MCTS rollout (playout) function — it doesn't need to be
    perfect, just fast and directionally correct.
    """
    heights  = base_heights.copy()
    states   = base_state.copy()
    dumpable = list(base_dumpable)

    # Apply the deterministic path first (the sequence chosen by the tree so far)
    for cell in path:
        if cell is not None:
            r0, c0 = cell
            heights[r0, c0] = min(heights[r0, c0] + dump_add, TARGET_PILE_HEIGHT)
            states[r0, c0]  = CellState.FILLED if heights[r0, c0] >= TARGET_PILE_HEIGHT else CellState.PARTIAL
            try: dumpable.remove(cell)
            except ValueError: pass

    # Then do `depth` random future dumps to explore what happens after this path
    for _ in range(min(depth, len(dumpable))):
        if not dumpable: break
        idx = random.randint(0, len(dumpable) - 1)
        rr, cc = dumpable[idx]
        heights[rr, cc] = min(heights[rr, cc] + _avg_pile_height_per_dump, TARGET_PILE_HEIGHT)
        if heights[rr, cc] >= TARGET_PILE_HEIGHT:
            states[rr, cc] = CellState.FILLED
            # Fast remove: swap with last element and pop (avoids O(n) shift)
            dumpable[idx]  = dumpable[-1]
            dumpable.pop()

    # Packing score: average fill fraction across all valid dump cells
    valid_mask = ((states == CellState.EMPTY) | (states == CellState.PARTIAL) |
                  (states == CellState.FILLED) | (states == CellState.RESERVED))
    total_valid = np.sum(valid_mask)
    if total_valid == 0: return 0.0

    packing = (np.sum(np.clip(heights, 0, TARGET_PILE_HEIGHT) * valid_mask)
               / (total_valid * TARGET_PILE_HEIGHT))

    # Penalty for isolated unfilled cells surrounded on 3+ sides by full cells —
    # these are hard to fill later because trucks can't easily reach them.
    filled_set = set(zip(*np.where(heights >= TARGET_PILE_HEIGHT)))
    remaining  = [(r, c) for r, c in dumpable[:15]]
    isolated_penalty = sum(
        1 for rr, cc in remaining
        if sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
               if (rr+dr, cc+dc) in filled_set) >= 3
    )

    score = packing - 0.05 * (isolated_penalty / max(1, len(remaining)))
    return max(0.0, min(1.0, score))


def mcts_select_dump_points(grid, candidates, truck, n_trucks, n_sim=100):
    """
    Run MCTS to pick the best dump cell for each of the n_trucks trucks.

    candidates : list of (row, col) cells pre-filtered as valid dump targets
    truck      : the truck whose pile_height_per_dump drives the simulation
    n_trucks   : how many cells to return (one per truck)
    n_sim      : number of MCTS simulations to run

    Returns a list of up to n_trucks (row, col) cells.
    """
    if not candidates: return []

    base_state   = grid.state.copy()
    base_heights = grid.z_height.copy()
    dump_add     = truck.pile_height_per_dump

    # All cells that could still receive material (empty, partial, or reserved but not full)
    base_dumpable = [
        (r, c) for r in range(grid.rows) for c in range(grid.cols)
        if base_state[r, c] in (CellState.EMPTY, CellState.PARTIAL, CellState.RESERVED)
        and base_heights[r, c] < TARGET_PILE_HEIGHT
    ]

    # Root represents "current state, no dump chosen yet"
    # Its children are the initial candidate cells (one child per candidate)
    root = MCTSNode(cell=None)
    for cell in candidates:
        root.children.append(MCTSNode(cell, parent=root))
    root.is_expanded = True

    for sim_num in range(n_sim):
        node = root
        path = []

        # SELECTION: walk down the tree using UCB1 until we hit an unexpanded node
        while node.is_expanded and node.children:
            node = max(node.children, key=lambda n: n.ucb1(node.visits if node.visits > 0 else 1))
            if node.cell is not None: path.append(node.cell)

        tree_depth = len(path)

        # EXPANSION: if this node has been visited before and isn't expanded yet,
        # grow the tree by adding the top-M scoring children.
        # branch_limits controls how wide we expand at each depth (fewer children deeper down).
        branch_limits = {1: 6, 2: 3}

        if node.visits > 0 and not node.is_expanded:
            if tree_depth in branch_limits:
                m_to_pick = branch_limits[tree_depth]
                top_m = get_top_m_next_moves(base_state, base_heights, path, grid, base_dumpable, dump_add, m=m_to_pick)

                for cell in top_m: node.children.append(MCTSNode(cell, parent=node))
                node.is_expanded = True

                # Immediately select the first new child for this simulation's rollout
                if node.children:
                    node = node.children[0]
                    path.append(node.cell)
            else:
                # At a depth beyond our expansion limits — mark expanded with no children (leaf)
                node.is_expanded = True

        # ROLLOUT: simulate random future dumps from this path and get a score
        score = rollout_score_path(base_state, base_heights, path, base_dumpable, dump_add, grid, depth=20)

        # BACKPROPAGATION: push the score up to every ancestor so UCB1 stays accurate
        curr = node
        while curr is not None:
            curr.visits += 1
            curr.value  += score
            curr = curr.parent

    # EXTRACTION: greedily walk the tree picking the most-visited child at each level.
    # Most-visited = most-explored = most confident estimate of the best choice.
    best_path = []
    curr_node = root

    for _ in range(n_trucks):
        if not curr_node.children:
            break
        best_child = max(curr_node.children, key=lambda n: n.visits)
        best_path.append(best_child.cell)
        curr_node = best_child

    return best_path
