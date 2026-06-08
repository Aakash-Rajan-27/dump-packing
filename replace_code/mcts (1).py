# mcts.py
# ─────────────────────────────────────────────────────────────
# FIX: Merged Deep Tree structure with the base_dumpable loop
#      optimization for extreme speed (prevents 4-million loop lag).
# FIX: Uses `score_candidates` with `state_override` to
#      evaluate true future states inside the tree expansion.
# ─────────────────────────────────────────────────────────────

import math    # math.sqrt for UCB1 exploration constant
import random  # random.randint for random rollout cell selection
import numpy as np  # array operations for grid state copies and score calculations
from grid_map import CellState  # enum: EMPTY, PARTIAL, FILLED, RESERVED — used to filter candidate cells
from config import TARGET_PILE_HEIGHT, TRUCK_CLASSES, FLEET_COMPOSITION  # target height in metres, truck specs, fleet makeup
from scoring import score_candidates  # evaluates a list of candidate cells and returns per-cell scores

_total_trucks = sum(FLEET_COMPOSITION.values())  # total number of trucks across all classes
_avg_pile_height_per_dump = sum(
    TRUCK_CLASSES[cls]['pile_height_per_dump'] * count  # weighted sum of height added per dump for each truck class
    for cls, count in FLEET_COMPOSITION.items()
) / max(1, _total_trucks)  # divide by truck count (guard against zero); used in random rollouts as a "typical" dump size


class MCTSNode:
    def __init__(self, cell, parent=None):
        self.cell     = cell      # (row, col) grid cell this node represents; None for the root
        self.parent   = parent    # parent MCTSNode (None for root)
        self.children = []        # list of child MCTSNodes discovered so far
        self.visits   = 0         # number of times this node has been selected during simulation
        self.value    = 0.0       # accumulated rollout score from all visits (used to compute mean score)
        self.is_expanded = False  # True once this node's children have been generated

    def ucb1(self, parent_visits, c=math.sqrt(2)):
        # Upper Confidence Bound formula: balances exploitation (mean score) vs exploration (rarely visited nodes)
        if self.visits == 0: return float('inf')  # unvisited nodes always get explored first
        return (self.value / self.visits +  # exploitation term: mean reward so far
                c * math.sqrt(math.log(parent_visits) / self.visits))  # exploration term: decreases as visits grow


def get_top_m_next_moves(base_state, base_heights, path_so_far, grid, base_dumpable, dump_add, m=6):
    hypothetical_state   = base_state.copy()    # work on a copy so we don't mutate the real grid state
    hypothetical_heights = base_heights.copy()  # copy of height array for the same reason

    # Fast copy
    empty = list(base_dumpable)  # mutable copy of the list of cells that still need filling

    for cell in path_so_far:  # simulate each dump in the path already chosen above this node
        if cell is not None:
            r, c = cell
            hypothetical_heights[r, c] = min(hypothetical_heights[r, c] + dump_add, TARGET_PILE_HEIGHT)  # add height, cap at target
            hypothetical_state[r, c]   = CellState.FILLED if hypothetical_heights[r, c] >= TARGET_PILE_HEIGHT else CellState.PARTIAL  # update state accordingly

            if hypothetical_heights[r, c] >= TARGET_PILE_HEIGHT:
                try: empty.remove(cell)  # remove fully-filled cells from the candidate list
                except ValueError: pass  # already removed — safe to ignore

    if not empty: return []  # all cells filled; no next moves available

    scores = score_candidates(grid, empty, state_override=hypothetical_state)  # score remaining candidates in the hypothetical future state
    scored_empty = list(zip(scores, empty))  # pair each cell with its score
    scored_empty.sort(key=lambda x: x[0], reverse=True)  # sort best-to-worst

    return [cell for score, cell in scored_empty[:m]]  # return only the top-m cells for tree expansion


def rollout_score_path(base_state, base_heights, path, base_dumpable, dump_add, grid, depth=20):
    heights = base_heights.copy()  # local copy of heights; modified in-place during rollout
    states  = base_state.copy()    # local copy of states

    # Fast copy
    dumpable = list(base_dumpable)  # mutable copy of fillable cells

    for cell in path:  # replay the tree path (selected dump sequence) to reach the leaf node's state
        if cell is not None:
            r0, c0 = cell
            heights[r0, c0] = min(heights[r0, c0] + dump_add, TARGET_PILE_HEIGHT)  # apply dump
            states[r0, c0]  = CellState.FILLED if heights[r0, c0] >= TARGET_PILE_HEIGHT else CellState.PARTIAL

            try: dumpable.remove(cell)  # remove filled cell from remaining candidates
            except ValueError: pass

    for _ in range(min(depth, len(dumpable))):  # random rollout: simulate up to `depth` more random dumps
        if not dumpable: break
        idx = random.randint(0, len(dumpable) - 1)  # pick a random remaining cell index
        rr, cc = dumpable[idx]
        heights[rr, cc] = min(heights[rr, cc] + _avg_pile_height_per_dump, TARGET_PILE_HEIGHT)  # apply average-truck dump
        if heights[rr, cc] >= TARGET_PILE_HEIGHT:
            states[rr, cc] = CellState.FILLED
            dumpable[idx]  = dumpable[-1]  # swap-and-pop: O(1) removal from list
            dumpable.pop()

    # Build a boolean mask for all valid (in-pit) cells to compute packing density
    valid_mask = ((states == CellState.EMPTY) | (states == CellState.PARTIAL) |
                  (states == CellState.FILLED) | (states == CellState.RESERVED))
    total_valid = np.sum(valid_mask)  # total number of valid cells in the dump zone
    if total_valid == 0: return 0.0   # degenerate case: no valid cells (shouldn't happen in practice)

    packing = (np.sum(np.clip(heights, 0, TARGET_PILE_HEIGHT) * valid_mask)
               / (total_valid * TARGET_PILE_HEIGHT))  # fraction of the dump zone that has been filled to target height

    filled_set = set(zip(*np.where(heights >= TARGET_PILE_HEIGHT)))  # set of (r,c) cells that have reached target height
    remaining  = [(r, c) for r, c in dumpable[:15]]                  # look at first 15 remaining unfilled cells
    isolated_penalty = sum(
        1 for rr, cc in remaining
        if sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
               if (rr+dr, cc+dc) in filled_set) >= 3  # count cells surrounded on 3+ sides by filled cells
    )  # isolated unfilled pockets are hard to fill efficiently — penalise them

    score = packing - 0.05 * (isolated_penalty / max(1, len(remaining)))  # final score: high packing minus isolation penalty
    return max(0.0, min(1.0, score))  # clamp to [0, 1] so UCB1 arithmetic stays stable


def mcts_select_dump_points(grid, candidates, truck, n_trucks, n_sim=100):
    if not candidates: return []  # no cells to fill; return immediately

    base_state   = grid.state.copy()     # snapshot of the real grid state at decision time
    base_heights = grid.z_height.copy()  # snapshot of real height map
    dump_add     = truck.pile_height_per_dump  # height increment per dump for the requesting truck

    # ── VECTORIZED base_dumpable ─────────────────────────────────────────────────
    # Was: Python double loop over all grid cells (~8100 iterations) checking state + height.
    # Now: np.where on a vectorised boolean mask — same result, runs in C.
    _fill_states    = (CellState.EMPTY, CellState.PARTIAL, CellState.RESERVED)
    _dumpable_mask  = np.isin(base_state, list(_fill_states)) & (base_heights < TARGET_PILE_HEIGHT)
    _dr, _dc        = np.where(_dumpable_mask)
    base_dumpable   = list(zip(_dr.tolist(), _dc.tolist()))  # list of all globally fillable (r,c) cells

    root = MCTSNode(cell=None)  # root has no cell — it represents the "current state" before any dump
    for cell in candidates:
        root.children.append(MCTSNode(cell, parent=root))  # each candidate becomes a depth-1 child
    root.is_expanded = True  # root is pre-expanded with all candidates

    for sim_num in range(n_sim):  # run n_sim Monte Carlo simulations
        node = root
        path = []

        while node.is_expanded and node.children:  # selection phase: descend tree using UCB1
            node = max(node.children, key=lambda n: n.ucb1(node.visits if node.visits > 0 else 1))  # pick child with best UCB1 score
            if node.cell is not None: path.append(node.cell)  # build the path as we descend

        tree_depth = len(path)   # depth of the selected leaf node
        branch_limits = {1: 6, 2: 3}  # max children to expand at depth 1 (6) and depth 2 (3); deeper nodes don't expand

        if node.visits > 0 and not node.is_expanded:  # expansion phase: only expand nodes visited at least once
            if tree_depth in branch_limits:
                m_to_pick = branch_limits[tree_depth]
                top_m = get_top_m_next_moves(base_state, base_heights, path, grid, base_dumpable, dump_add, m=m_to_pick)  # find top-m next moves in hypothetical future state

                for cell in top_m: node.children.append(MCTSNode(cell, parent=node))  # add top-m cells as children
                node.is_expanded = True  # mark node as expanded so future sims descend through it

                if node.children:
                    node = node.children[0]  # step into the first new child for this simulation
                    path.append(node.cell)   # extend the path to include the new child
            else:
                node.is_expanded = True  # leaf at max depth — mark expanded but add no children (tree terminates here)

        score = rollout_score_path(base_state, base_heights, path, base_dumpable, dump_add, grid, depth=20)  # simulation (rollout) phase: estimate value of this path

        curr = node
        while curr is not None:  # backpropagation phase: update visits and value all the way up to root
            curr.visits += 1
            curr.value  += score
            curr = curr.parent

    best_path = []
    curr_node = root

    for _ in range(n_trucks):  # extract the best sequence of n_trucks dump cells from the tree
        if not curr_node.children:
            break

        best_child = max(curr_node.children, key=lambda n: n.visits)  # pick the most-visited child (exploitation, not UCB1)
        best_path.append(best_child.cell)  # add its cell to the recommended dump sequence
        curr_node = best_child             # descend to pick the next truck's cell from children of this best child

    return best_path  # list of (row, col) cells, one per truck, in recommended dump order
