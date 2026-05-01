# mcts.py
# ─────────────────────────────────────────────────────────────
# FIX: Merged Deep Tree structure (Tapered Branching 50->6->3->0)
#      with the base_dumpable loop optimization for extreme speed.
# FIX: Now utilizes `score_candidates` with `state_override` to 
#      evaluate true future states inside the tree expansion.
# ─────────────────────────────────────────────────────────────

import math
import random
import numpy as np
from grid_map import CellState
from config import TARGET_PILE_HEIGHT, TRUCK_CLASSES, FLEET_COMPOSITION
from scoring import score_candidates

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
        self.is_expanded = False 

    def ucb1(self, parent_visits, c=math.sqrt(2)):
        if self.visits == 0: return float('inf')
        return (self.value / self.visits +
                c * math.sqrt(math.log(parent_visits) / self.visits))


def get_top_m_next_moves(base_state, base_heights, path_so_far, grid, base_dumpable, dump_add, m=6):
    """Uses scoring heuristics to pick top M branches in the deep tree."""
    hypothetical_state   = base_state.copy()
    hypothetical_heights = base_heights.copy()
    
    # Apply path so far
    for cell in path_so_far:
        if cell is not None:
            r, c = cell
            hypothetical_heights[r, c] = min(hypothetical_heights[r, c] + dump_add, TARGET_PILE_HEIGHT)
            hypothetical_state[r, c]   = CellState.FILLED if hypothetical_heights[r, c] >= TARGET_PILE_HEIGHT else CellState.PARTIAL

    # Find still empty/valid cells
    empty = [cell for cell in base_dumpable if hypothetical_heights[cell[0], cell[1]] < TARGET_PILE_HEIGHT]
    if not empty: return []

    # Score using FUTURE goggles (state_override)
    scores = score_candidates(grid, empty, state_override=hypothetical_state)
    scored_empty = list(zip(scores, empty))
    scored_empty.sort(key=lambda x: x[0], reverse=True)
    
    return [cell for score, cell in scored_empty[:m]]


def rollout_score_path(base_state, base_heights, path, base_dumpable, dump_add, grid, depth=20):
    """Simulates the specific MCTS path, then random rollouts."""
    heights = base_heights.copy()
    states  = base_state.copy()

    for cell in path:
        if cell is not None:
            r0, c0 = cell
            heights[r0, c0] = min(heights[r0, c0] + dump_add, TARGET_PILE_HEIGHT)
            states[r0, c0]  = CellState.FILLED if heights[r0, c0] >= TARGET_PILE_HEIGHT else CellState.PARTIAL

    path_cells = set(c for c in path if c is not None)
    dumpable = [cell for cell in base_dumpable
                if heights[cell[0], cell[1]] < TARGET_PILE_HEIGHT
                and cell not in path_cells]

    for _ in range(min(depth, len(dumpable))):
        if not dumpable: break
        idx = random.randint(0, len(dumpable) - 1)
        rr, cc = dumpable[idx]
        heights[rr, cc] = min(heights[rr, cc] + _avg_pile_height_per_dump, TARGET_PILE_HEIGHT)
        if heights[rr, cc] >= TARGET_PILE_HEIGHT:
            states[rr, cc] = CellState.FILLED
            dumpable[idx]  = dumpable[-1]
            dumpable.pop()

    valid_mask = ((states == CellState.EMPTY) | (states == CellState.PARTIAL) | 
                  (states == CellState.FILLED) | (states == CellState.RESERVED))
    total_valid = np.sum(valid_mask)
    if total_valid == 0: return 0.0

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


def mcts_select_dump_points(grid, candidates, truck, n_trucks, n_sim=100):
    if not candidates: return []

    base_state   = grid.state.copy()
    base_heights = grid.z_height.copy()
    dump_add     = truck.pile_height_per_dump

    base_dumpable = [
        (r, c) for r in range(grid.rows) for c in range(grid.cols)
        if base_state[r, c] in (CellState.EMPTY, CellState.PARTIAL, CellState.RESERVED)
        and base_heights[r, c] < TARGET_PILE_HEIGHT
    ]

    root = MCTSNode(cell=None)
    for cell in candidates:
        root.children.append(MCTSNode(cell, parent=root))
    root.is_expanded = True 

    for sim_num in range(n_sim):
        node = root
        path = []

        while node.is_expanded and node.children:
            node = max(node.children, key=lambda n: n.ucb1(node.visits if node.visits > 0 else 1))
            if node.cell is not None: path.append(node.cell)

        tree_depth = len(path)
        branch_limits = {1: 6, 2: 3} 

        if node.visits > 0 and not node.is_expanded:
            if tree_depth in branch_limits:
                m_to_pick = branch_limits[tree_depth]
                top_m = get_top_m_next_moves(base_state, base_heights, path, grid, base_dumpable, dump_add, m=m_to_pick)
                
                for cell in top_m: node.children.append(MCTSNode(cell, parent=node))
                node.is_expanded = True
                
                if node.children:
                    node = node.children[0]
                    path.append(node.cell)
            else:
                node.is_expanded = True

        score = rollout_score_path(base_state, base_heights, path, base_dumpable, dump_add, grid, depth=20)

        curr = node
        while curr is not None:
            curr.visits += 1
            curr.value  += score
            curr = curr.parent

    best_path = []
    curr_node = root
    for _ in range(min(n_trucks, len(candidates))):
        if not curr_node.children: break
        best_child = max(curr_node.children, key=lambda n: n.visits)
        best_path.append(best_child.cell)
        curr_node = best_child
        
    return best_path