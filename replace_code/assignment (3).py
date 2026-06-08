# assignment.py
# ─────────────────────────────────────────────────────────────
# Hungarian algorithm for truck-to-dump-point assignment.
# Mixed fleet update: the heading weight is scaled by truck class
# because large trucks (33m turn radius) pay a higher penalty for
# misalignment than small trucks (22m turn radius).
# ─────────────────────────────────────────────────────────────

import numpy as np
from scipy.optimize import linear_sum_assignment
from config import W_DISTANCE, W_HEADING


# Maximum turn radius in fleet (Cat 793F = 33m) — used for normalisation
_MAX_TURN_RADIUS = 33.0


def build_cost_matrix(trucks, dump_points, grid):
    """
    Build N×M cost matrix.  
    C[i][j] = cost of sending truck i to dumgit addp point j.

    Cost components:
      - Normalised Euclidean distance (primary)
      - Normalised heading misalignment, scaled by truck turn radius
        (a large truck pays more for bad alignment than a small truck)
    """
    n = len(trucks)
    m = len(dump_points)
    C = np.zeros((n, m))

    max_dist = np.hypot(grid.cols * grid.cell_size,
                        grid.rows * grid.cell_size)

    for i, truck in enumerate(trucks):
        tx, ty = truck.pos
        # Scale heading weight by relative turn radius:
        # large truck (33m) → heading_scale=1.0; small truck (22m) → ~0.67
        heading_scale = truck.turn_radius / _MAX_TURN_RADIUS

        for j, (dr, dc) in enumerate(dump_points):
            px, py = grid.cell_to_world(dr, dc)

            # Distance component
            dist      = np.hypot(px - tx, py - ty)
            norm_dist = dist / max_dist

            # Heading component
            bearing      = np.arctan2(py - ty, px - tx)
            delta        = abs(bearing - truck.heading)
            delta        = min(delta, 2 * np.pi - delta)
            norm_heading = delta / np.pi

            C[i, j] = (W_DISTANCE * norm_dist +
                       W_HEADING * heading_scale * norm_heading)
    print("Hungarian Cost Matrix (Distance + Heading):")
    print(np.round(C, 2))

    return C


def assign(trucks, dump_points, grid):
    """Optimally assign idle trucks to dump points via Hungarian algorithm."""
    if not trucks or not dump_points:
        return []
    C = build_cost_matrix(trucks, dump_points, grid)
    row_ind, col_ind = linear_sum_assignment(C)
    return [(trucks[r], dump_points[c]) for r, c in zip(row_ind, col_ind)]
