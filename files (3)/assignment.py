# core/assignment.py
# ─────────────────────────────────────────────────────────────
# Hungarian Algorithm for truck-to-dump-point assignment.
#
# Problem: We have N idle trucks and N candidate dump points.
# Naive approach (greedy): each truck picks its nearest point.
# Problem with greedy: two trucks might both pick the same point,
# leaving a farther point unassigned.
#
# Hungarian Algorithm: finds the GLOBALLY optimal assignment
# (minimises total cost across ALL trucks simultaneously).
# scipy gives us this in 2 lines.
# ─────────────────────────────────────────────────────────────

import numpy as np
from scipy.optimize import linear_sum_assignment
# linear_sum_assignment: scipy's implementation of the Hungarian algorithm
# Input: cost matrix C where C[i][j] = cost of assigning truck i to point j
# Output: (row_indices, col_indices) — the optimal pairing

from config import W_DISTANCE, W_HEADING


def build_cost_matrix(trucks, dump_points, grid):
    """
    Build the N×M cost matrix where:
      C[i][j] = cost of sending truck i to dump point j

    Cost = weighted sum of:
      - normalised distance (primary factor)
      - normalised heading misalignment (secondary factor)

#FIX -> <add truck - point compatability, e.g. small trucks prefer low spots>

    trucks: list of Truck objects
    dump_points: list of (row, col) candidate dump cells
    grid: GridMap (for cell_to_world conversion)

    Returns: 2D numpy array shape (len(trucks), len(dump_points))
    """
    n = len(trucks)       # number of trucks (rows in cost matrix)
    m = len(dump_points)  # number of dump points (columns)

    # Initialise cost matrix with zeros
    C = np.zeros((n, m))

    # Estimate max possible distance for normalisation
    # (diagonal of the grid bounding box)
    max_dist = np.hypot(
        grid.cols * grid.cell_size,   # grid width in metres
        grid.rows * grid.cell_size    # grid height in metres
    )

    for i, truck in enumerate(trucks):
        tx, ty = truck.pos          # truck current position (metres)

        for j, (dr, dc) in enumerate(dump_points):
            # Get real-world coordinates of this dump point
            px, py = grid.cell_to_world(dr, dc)

            # ── Component 1: Distance ──────────────────────
            # Straight-line Euclidean distance in metres
            dist = np.hypot(px - tx, py - ty)  # √((px-tx)² + (py-ty)²)

            # Normalise to [0,1] range by dividing by max possible distance
            norm_dist = dist / max_dist

            # ── Component 2: Heading misalignment ─────────
            # Bearing = direction FROM truck TO dump point (in radians)
            bearing = np.arctan2(py - ty, px - tx)
            # atan2(dy, dx) gives angle in radians in range [-π, π]

            # Angular difference between truck's current heading and the bearing
            delta = abs(bearing - truck.heading)

            # Wrap to [0, π] — angles are circular, so 350° vs 10° = 20° difference
            delta = min(delta, 2 * np.pi - delta)

            # Normalise to [0,1] by dividing by π (max possible difference)
            norm_heading = delta / np.pi

            # ── Combined cost ──────────────────────────────
            C[i, j] = W_DISTANCE * norm_dist + W_HEADING * norm_heading #FIX - WEIGHTS of cost matrix
            # e.g. 1.0 * 0.3 + 0.4 * 0.5 = 0.5 total cost

    return C


def assign(trucks, dump_points, grid):
    """
    Optimally assign each idle truck to a unique dump point.

    trucks: list of idle Truck objects
    dump_points: list of (row,col) — the selected dump targets (same length as trucks)
    grid: GridMap

    Returns: list of (truck, dump_point) pairs — the optimal assignment
    """
    if not trucks or not dump_points:
        return []  # nothing to assign

    # Build cost matrix
    C = build_cost_matrix(trucks, dump_points, grid)

    # Run the Hungarian algorithm
    # row_ind[k] = truck index, col_ind[k] = dump point index for that pairing
    row_ind, col_ind = linear_sum_assignment(C)
    # linear_sum_assignment minimises the sum of C[row_ind[k], col_ind[k]] over k

    # Build result: list of (truck, dump_point) pairs
    result = []
    for r, c in zip(row_ind, col_ind):
        result.append((trucks[r], dump_points[c]))

    return result
