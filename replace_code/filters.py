# filters.py
# ─────────────────────────────────────────────────────────────
# Filter candidate dump cells for a given truck.
# Updated for mixed fleet:
#   - footprint_ok tests the truck's actual width & length
#     (large trucks have bigger footprints → fewer valid cells)
#   - get_candidates now includes PARTIAL cells (not just EMPTY),
#     since a partially-filled cell can still receive more dumps
#   - is_accessible unchanged — flood-fill isolation check
# ─────────────────────────────────────────────────────────────

import numpy as np
from collections import deque
from shapely.geometry import Point
from grid_map import CellState


def footprint_ok(grid, r, c, truck):
    """
    Check that a truck of this class can physically fit at cell (r,c)
    from at least one of 4 cardinal approach angles.

    Larger trucks (Cat 793F: 9.8m × 14.8m) are more likely to fail
    near polygon corners/edges than smaller trucks (Cat 773F: 5.5m × 8.7m).

    Returns True if ANY heading orientation keeps all 4 truck corners
    inside the polygon.
    """
    cx, cy   = grid.cell_to_world(r, c)
    half_w   = truck.width  / 2.0
    half_l   = truck.length / 2.0

    for heading in [0, np.pi/2, np.pi, 3*np.pi/2]:
        rev   = heading + np.pi
        fwd_x = np.cos(heading);  
        fwd_y = np.sin(heading)
        sid_x = fwd_y;       
        sid_y = -fwd_x

        corners = [
            (cx + ls*half_l*fwd_x + ws*half_w*sid_x,
             cy + ls*half_l*fwd_y + ws*half_w*sid_y)
            for ls in (-1, +1) for ws in (-1, +1)
        ]
        if all(grid.polygon.contains(Point(px, py)) for px, py in corners):
            return True

    return False


def is_accessible(grid, r, c, entry_rc):
    """
    BFS flood-fill: hypothetically fill cell (r,c) and check that no
    EMPTY or PARTIAL cell in the polygon gets cut off from entry_rc.

    Returns True if nothing is isolated (safe to dump here).
    """
    temp_state = grid.state.copy()
    temp_state[r, c] = CellState.FILLED

    # BFS from entry
    visited = set()
    queue   = deque([entry_rc])
    # PARTIAL = physical pile — impassable, same as FILLED
    driveable = (CellState.EMPTY, CellState.RESERVED, CellState.PROTECTED)

    while queue:
        cr, cc = queue.popleft()
        if (cr, cc) in visited:
            continue
        visited.add((cr, cc))
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = cr+dr, cc+dc
            if not (0 <= nr < grid.rows and 0 <= nc < grid.cols):
                continue
            if temp_state[nr, nc] not in driveable:
                continue
            if (nr, nc) not in visited:
                queue.append((nr, nc))

    # All cells that should be reachable
    all_reachable = set(zip(*np.where(
    (temp_state == CellState.EMPTY)     |
    (temp_state == CellState.RESERVED)  |
    (temp_state == CellState.PROTECTED)
)))

    return len(all_reachable - visited) == 0

    
def get_candidates(grid, truck, entry_rc):
    """
    Return all cells that:
      1. Are dumpable (EMPTY or PARTIAL and below TARGET_PILE_HEIGHT)
      2. Pass the footprint check for this specific truck class

    Accessibility (flood-fill) is applied later in main.py on top-30 only.

    Note: PARTIAL cells are included — they still need more material
    to reach TARGET_PILE_HEIGHT, so they're valid dump targets.
    """
    # Dumpable = EMPTY or PARTIAL and not yet at max height
    dumpable_mask = np.zeros((grid.rows, grid.cols), dtype=bool)
    for r in range(grid.rows):
        for c in range(grid.cols):
            dumpable_mask[r, c] = grid.is_dumpable(r, c)

    rows, cols = np.where(dumpable_mask)
    all_dumpable = list(zip(rows.tolist(), cols.tolist()))

    valid = []
    for r, c in all_dumpable:
        if footprint_ok(grid, r, c, truck):
            valid.append((r, c))

    return valid
