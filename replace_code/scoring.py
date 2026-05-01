# scoring.py
# ─────────────────────────────────────────────────────────────
# Change from previous version:
#
# uniform_filter size parameter changed from hardcoded 8
# to SCORE_FILTER_SIZE from config.
#
# Old: size=8 at CELL_SIZE=3.0m → 24m physical window
# New: size=SCORE_FILTER_SIZE at CELL_SIZE=0.5m
#      SCORE_FILTER_SIZE = int(round(24.0 / 0.5)) = 48
#      → same 24m physical window
#
# This keeps the scoring heuristics physically meaningful
# at the finer cell resolution.
#
# All other scoring logic unchanged.
# ─────────────────────────────────────────────────────────────

import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter
from config import (WEIGHTS_EARLY, WEIGHTS_MID, WEIGHTS_LATE,
                    TARGET_PILE_HEIGHT, SCORE_FILTER_SIZE)
from grid_map import CellState


def score_candidates(grid, candidate_idxs):
    """
    Score candidate cells. Higher = better dump location.
    candidate_idxs: list of (row, col) — the subsampled candidates from get_candidates()
    Returns: 1D numpy array of scores.
    """
    if not candidate_idxs:
        return np.array([])

    filled_mask  = (grid.state == CellState.FILLED)
    partial_mask = (grid.state == CellState.PARTIAL)

    # Score 1: Density — dump near existing material
    has_material  = filled_mask | partial_mask
    dist_to_pile  = distance_transform_edt(~has_material) * grid.cell_size
    spread        = grid.cell_size * 2
    density_score = np.exp(-dist_to_pile / spread)

    # Score 2: Coverage — spread dumps across polygon
    # CHANGED: size=SCORE_FILTER_SIZE (was hardcoded 8)
    # At 0.5m cells: size=48 → same 24m physical window as before
    zone_fill      = uniform_filter(filled_mask.astype(float),
                                    size=SCORE_FILTER_SIZE)
    coverage_score = 1.0 - zone_fill

    # Score 3: Height gap — prioritise cells furthest from TARGET_PILE_HEIGHT
    height_gap     = np.maximum(0.0, TARGET_PILE_HEIGHT - grid.z_height)
    heightgap_score = height_gap / TARGET_PILE_HEIGHT

    # Score 4: Pheromone — avoid recently dumped areas
    phero_score = grid.pheromone

    # Score 5: Boundary priority — fill edges first
    row_idx, col_idx = np.mgrid[0:grid.rows, 0:grid.cols]
    dist_from_centre = np.sqrt(
        (row_idx - grid.rows/2.0)**2 + (col_idx - grid.cols/2.0)**2
    )
    boundary_score = dist_from_centre / (dist_from_centre.max() + 0.01)

    # Adaptive weights
    fp = grid.fill_pct()
    if   fp < 0.30: w = WEIGHTS_EARLY
    elif fp < 0.70: w = WEIGHTS_MID
    else:           w = WEIGHTS_LATE

    combined = (w[0] * density_score   +
                w[1] * coverage_score  +
                w[2] * heightgap_score +
                w[3] * phero_score     +
                w[4] * boundary_score)

    rows = [rc[0] for rc in candidate_idxs]
    cols = [rc[1] for rc in candidate_idxs]
    return combined[rows, cols]
