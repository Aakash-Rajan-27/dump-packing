# scoring.py
# ─────────────────────────────────────────────────────────────
# Change from previous version:
#
# uniform_filter size parameter changed from hardcoded 8
# to SCORE_FILTER_SIZE from config.
#
# Added 6th Heuristic: Entry Distance (pushes trucks to dump 
# far away from the entry point first).
# ─────────────────────────────────────────────────────────────

import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter
from config import (WEIGHTS_EARLY, WEIGHTS_MID, WEIGHTS_LATE,
                    TARGET_PILE_HEIGHT, SCORE_FILTER_SIZE, ENTRY_POINT)
from grid_map import CellState


def score_candidates(grid, candidate_idxs, state_override=None):
    """
    Score candidate cells. Higher = better dump location.
    candidate_idxs: list of (row, col) — the subsampled candidates from get_candidates()
    Returns: 1D numpy array of scores.
    """
    if not candidate_idxs:
        return np.array([])

    current_state = state_override if state_override is not None else grid.state

    filled_mask  = (current_state == CellState.FILLED)
    partial_mask = (current_state == CellState.PARTIAL)
    reserved_mask = (current_state == CellState.RESERVED)

    # Score 1: Density — dump near existing material
    has_material  = filled_mask | partial_mask | reserved_mask
    dist_to_pile  = distance_transform_edt(~has_material) * grid.cell_size
    spread        = grid.cell_size * 2
    density_score = np.exp(-dist_to_pile / spread)

    # Score 2: Coverage — spread dumps across polygon
    zone_fill      = uniform_filter(has_material.astype(float),
                                    size=SCORE_FILTER_SIZE)
    coverage_score = 1.0 - zone_fill

    # Score 3: Height gap — prioritise cells furthest from TARGET_PILE_HEIGHT
    height_gap      = np.maximum(0.0, TARGET_PILE_HEIGHT - grid.z_height)
    heightgap_score = height_gap / TARGET_PILE_HEIGHT

    # Score 4: Pheromone — avoid recently dumped areas
    phero_score = grid.pheromone

    # Score 5: Boundary priority — fill edges first
    row_idx, col_idx = np.mgrid[0:grid.rows, 0:grid.cols]
    dist_from_centre = np.sqrt(
        (row_idx - grid.rows/2.0)**2 + (col_idx - grid.cols/2.0)**2
    )
    boundary_score = dist_from_centre / (dist_from_centre.max() + 0.01)

    # Score 6: Entry distance — dump far from entry first
    entry_r, entry_c = grid.world_to_cell(*ENTRY_POINT)
    # Calculate physical distance in metres
    dist_from_entry = np.sqrt(
        (row_idx - entry_r)**2 + (col_idx - entry_c)**2
    ) * grid.cell_size 
    
    # Normalise using exponential decay (scaled by 50m)
    entry_score = 1.0 - np.exp(-dist_from_entry / 50.0)

    # Adaptive weights based on fill percentage
    fp = grid.fill_pct()
    if   fp < 0.30: w = WEIGHTS_EARLY
    elif fp < 0.70: w = WEIGHTS_MID
    else:           w = WEIGHTS_LATE

    combined = (w[0] * density_score   +
                w[1] * coverage_score  +
                w[2] * heightgap_score +
                w[3] * phero_score     +
                w[4] * boundary_score  +
                w[5] * entry_score) # <--- Added 6th weight here
    
    # ─── ROUND 2 PPT HEATMAP GENERATOR ───
    # 1. Setup a counter attached to the function
    if not hasattr(score_candidates, "call_count"):
        score_candidates.call_count = 0
    
    score_candidates.call_count += 1

    # 2. Trigger the heatmap exactly on the 10th planning phase
    if score_candidates.call_count == 10:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        
        # Mask out boundaries so they don't skew the color scale
        heatmap_data = np.where(current_state == CellState.BOUNDARY, np.nan, combined)
        
        plt.imshow(heatmap_data, cmap='inferno', interpolation='nearest')
        plt.colorbar(label='Heuristic Score Utility')
        plt.title("Spatial Scoring Distribution (Pre-MCTS)")
        
        plt.savefig("scoring_heatmap.png", bbox_inches='tight', dpi=300)
        print("\n[DEBUG] Saved high-res scoring_heatmap.png to your folder! Put this on Slide 4.")
    # ──────────────────────────────────────────

    rows = [rc[0] for rc in candidate_idxs]
    cols = [rc[1] for rc in candidate_idxs]
    return combined[rows, cols]