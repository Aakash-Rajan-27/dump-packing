# core/scoring.py
# ─────────────────────────────────────────────────────────────
# Scores every candidate cell using 5 heuristics combined
# into one number. Higher score = better dump location.
# All operations are vectorized (NumPy arrays) — no Python loops
# over individual cells, so it runs fast even on large grids.
# ─────────────────────────────────────────────────────────────

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, uniform_filter
# distance_transform_edt: for each cell, finds distance to nearest filled cell
# gaussian_filter: blurs an array (used for pheromone spreading)
# uniform_filter: local average in a sliding window (used for zone fill %)

from config import WEIGHTS_EARLY, WEIGHTS_MID, WEIGHTS_LATE
from core.grid_map import CellState


def score_candidates(grid, candidate_idxs):
    """
    Score a list of candidate cells.

    grid: GridMap object
    candidate_idxs: list of (row, col) tuples — the cells to score

    Returns: numpy array of scores, one per candidate, same order as input.
    Higher = better dump location.
    """

    if not candidate_idxs:
        return np.array([])  # nothing to score

    # ── Build boolean masks ────────────────────────────────
    # A mask is a True/False array the same size as the grid.

    # True where material has already been dumped
    filled_mask = (grid.state == CellState.FILLED)

    # True where the cell is empty and dumpable
    empty_mask = (grid.state == CellState.EMPTY)

    # ── Score 1: Density ───────────────────────────────────
    # Measures: how close is this cell to existing piles?
    # Goal: dump NEAR existing piles so material merges, reducing gaps.
    #
    # distance_transform_edt: for every cell, computes the Euclidean
    # distance (in cell units) to the nearest True cell in ~filled_mask.
    # So if filled_mask is True where piles are, the distance is 0 AT piles,
    # and increases as you move away.
    #
    # We INVERT the mask so distance is measured FROM empty cells TO piles.
    dist_to_pile = distance_transform_edt(~filled_mask)  # shape: (rows, cols)

    # Convert from cell units to metres
    dist_to_pile_m = dist_to_pile * grid.cell_size

    # Exponential decay: score is 1.0 right next to a pile, drops off fast.
    # spread = half the dump spread radius → controls how quickly score drops
    spread = grid.cell_size * 2
    density_score = np.exp(-dist_to_pile_m / spread)
    # Result: 1.0 adjacent to pile, ~0.37 at 1 spread distance, ~0 far away

    # ── Score 2: Coverage ──────────────────────────────────
    # Measures: how empty is the local zone around this cell?
    # Goal: spread dumps across the whole polygon, not cluster in one corner.
    #
    # uniform_filter computes a local average in an 8×8 cell sliding window.
    # This gives us the fraction of cells filled in each neighbourhood.
    zone_fill_pct = uniform_filter(filled_mask.astype(float), size=8)
    # zone_fill_pct[r,c] ≈ fraction of 8×8 neighbourhood that is filled

    # Coverage score: high where zone is LESS full (1 - fill%)
    coverage_score = 1.0 - zone_fill_pct
    # 1.0 = this zone is completely empty (spread here!)
    # 0.0 = this zone is completely full (don't pile on more)

    # ── Score 3: Low Spot ──────────────────────────────────
    # Measures: is this cell lower than its surroundings?
    # Goal: level out depressions in the dump surface.
    #
    # uniform_filter on z_height gives average height of each cell's neighbourhood
    avg_neighbour_height = uniform_filter(grid.z_height, size=3)

    # Depression = how much lower this cell is vs its neighbours
    # max(0,...) ensures we only score actual depressions, not hills
    depression = np.maximum(0.0, avg_neighbour_height - grid.z_height)

    # Normalize relative to average height (avoid division by near-zero)
    lowspot_score = depression / (avg_neighbour_height + 0.1)
    # High score = this cell is a significant dip → good to fill it

    # ── Score 4: Pheromone ─────────────────────────────────
    # Measures: has this area been used recently?
    # Goal: avoid sending multiple trucks to the same zone simultaneously.
    #
    # grid.pheromone is 1.0 where nothing has happened recently,
    # and drops toward 0.0 after a truck dumps there (decays back over ticks).
    # Score = (1 - pheromone)² → strongly suppresses very recent dumps
    phero_score = (1.0 - grid.pheromone) ** 2
    # 0.0 where recently dumped (pheromone ≈ 1.0 after decay ... wait)
    # Actually: pheromone is SET TO 0 after dump, then DECAYS BACK to 1.
    # So right after dump: pheromone=0 → score=(1-0)²=1.0 ... hmm.
    # Correction: pheromone starts at 1 (untouched) and we set it to 0
    # at dump point. So "recently dumped" = low pheromone = LOW score.
    # score = pheromone itself is fine:
    phero_score = grid.pheromone  # high (1.0) = not recently used = preferred

    # ── Score 5: Boundary Priority ─────────────────────────
    # Measures: how far is this cell from the polygon centroid?
    # Goal: fill cells near the boundary FIRST — this makes manoeuvring
    # easier later because trucks can approach from outside-in.
    #
    # Build a grid of (row, col) indices for every cell
    row_idx, col_idx = np.mgrid[0:grid.rows, 0:grid.cols]

    # Centre of the grid in cell coordinates
    centre_r = grid.rows / 2.0
    centre_c = grid.cols / 2.0

    # Euclidean distance from each cell to the grid centre
    dist_from_centre = np.sqrt((row_idx - centre_r)**2 + (col_idx - centre_c)**2)

    # Normalize to [0,1] range
    max_dist = dist_from_centre.max()
    boundary_score = dist_from_centre / (max_dist + 0.01)
    # 1.0 at far edges, 0.0 at centre → fills boundary first

    # ── Combine scores with adaptive weights ───────────────
    # Weights change depending on how full the polygon is.
    # Early on: spread out (coverage matters most)
    # Late on: pack tight (density matters most)
    fp = grid.fill_pct()  # 0.0 to 1.0

    if fp < 0.30:
        w = WEIGHTS_EARLY   # [density, coverage, lowspot, pheromone, boundary]
    elif fp < 0.70:
        w = WEIGHTS_MID
    else:
        w = WEIGHTS_LATE

    # Combined score grid: weighted sum of all 5 score arrays
    combined = (w[0] * density_score  +
                w[1] * coverage_score +
                w[2] * lowspot_score  +
                w[3] * phero_score    +
                w[4] * boundary_score)
    # combined is shape (rows, cols) — one score per cell

    # ── Extract scores for just our candidates ─────────────
    # candidate_idxs is a list of (row,col) tuples — we only need their scores
    rows = [rc[0] for rc in candidate_idxs]   # list of row indices
    cols = [rc[1] for rc in candidate_idxs]   # list of col indices

    # Fancy indexing: combined[rows, cols] grabs exactly the cells we want
    return combined[rows, cols]
    # Returns 1D array: scores[i] corresponds to candidate_idxs[i]
