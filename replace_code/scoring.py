# scoring.py
# ─────────────────────────────────────────────────────────────
# Vectorized heuristic scoring for candidate dump cells.
# Updated for partial pile dynamics:
#   - PARTIAL cells are scored — they still need more material
#   - Added "height gap" score: prioritises cells far from full
#     so trucks top up the deepest partial piles first
#   - All 5 scores still adapt with fill percentage
# ─────────────────────────────────────────────────────────────

import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter
from config import WEIGHTS_EARLY, WEIGHTS_MID, WEIGHTS_LATE, TARGET_PILE_HEIGHT
from grid_map import CellState


def score_candidates(grid, candidate_idxs):
    """
    Score a list of candidate cells. Higher = better dump location.

    candidate_idxs : list of (row, col) tuples
    Returns        : 1D numpy array, same length, higher = better
    """
    if not candidate_idxs:
        return np.array([])

    filled_mask  = (grid.state == CellState.FILLED)
    partial_mask = (grid.state == CellState.PARTIAL)
    empty_mask   = (grid.state == CellState.EMPTY)

    # ── Score 1: Density ────────────────────────────────────
    # Dump near existing piles (any height) so material merges
    has_material = filled_mask | partial_mask
    dist_to_pile = distance_transform_edt(~has_material) * grid.cell_size
    spread       = grid.cell_size * 2
    density_score = np.exp(-dist_to_pile / spread)

    # ── Score 2: Coverage ───────────────────────────────────
    # Spread dumps across polygon — prefer less-filled zones
    zone_fill    = uniform_filter(filled_mask.astype(float), size=8)
    coverage_score = 1.0 - zone_fill

    # ── Score 3: Height Gap (replaces low-spot for mixed fleet) ─
    # Prioritise cells that are furthest from TARGET_PILE_HEIGHT.
    # EMPTY cells = gap of TARGET_PILE_HEIGHT (maximum priority).
    # PARTIAL cells = gap proportional to remaining height needed.
    # FILLED cells = 0 gap (already excluded from candidates).
    height_gap   = np.maximum(0.0, TARGET_PILE_HEIGHT - grid.z_height)
    heightgap_score = height_gap / TARGET_PILE_HEIGHT
    # 1.0 = completely empty, 0.5 = half filled, 0.0 = full

    # ── Score 4: Pheromone ──────────────────────────────────
    # Avoid areas recently dumped on — implicit multi-truck coordination
    phero_score = grid.pheromone   # 1.0 = untouched, 0.0 = just dumped

    # ── Score 5: Boundary Priority ──────────────────────────
    # Fill cells near polygon edge first (outside-in strategy)
    row_idx, col_idx = np.mgrid[0:grid.rows, 0:grid.cols]
    dist_from_centre = np.sqrt(
        (row_idx - grid.rows/2.0)**2 + (col_idx - grid.cols/2.0)**2
    )
    boundary_score = dist_from_centre / (dist_from_centre.max() + 0.01)

    # ── Adaptive Weight Blending ────────────────────────────
    fp = grid.fill_pct()
    if   fp < 0.30:  w = WEIGHTS_EARLY
    elif fp < 0.70:  w = WEIGHTS_MID
    else:            w = WEIGHTS_LATE
    # w = [density, coverage, heightgap, pheromone, boundary]

    combined = (w[0] * density_score   +
                w[1] * coverage_score  +
                w[2] * heightgap_score +
                w[3] * phero_score     +
                w[4] * boundary_score)

    rows = [rc[0] for rc in candidate_idxs]
    cols = [rc[1] for rc in candidate_idxs]
    return combined[rows, cols]
