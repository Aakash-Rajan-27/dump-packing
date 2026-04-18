# core/filters.py
# ─────────────────────────────────────────────────────────────
# Before scoring any candidate cell, we FILTER OUT invalid ones.
# Three filters:
#   1. Footprint check — truck body stays inside polygon
#   2. Flood-fill check — dumping here doesn't isolate any region
#   3. (Turning radius check is deferred to CBS path planner)
# Only cells that pass all filters get scored and considered.
# ─────────────────────────────────────────────────────────────

import numpy as np
from collections import deque          # deque = fast queue for BFS
from shapely.geometry import Point     # for polygon containment test
from grid_map import CellState


def footprint_ok(grid, r, c, truck):
    """
    Check that when the truck reverses to dump at cell (r,c),
    every corner of the truck body stays inside the polygon.

    grid: GridMap
    r,c: the target dump cell
    truck: Truck object (has .width, .length, .heading)

    Returns True if the truck fits, False if it sticks outside.
    """
    # Get the real-world centre of the target cell
    cx, cy = grid.cell_to_world(r, c)

    # Truck reverses to dump — its REAR end is at (cx, cy).
    # We need to check that the full truck rectangle fits.
    half_w = truck.width / 2.0    # half-width in metres
    half_l = truck.length / 2.0   # half-length in metres

    # The truck heading when reversing is the OPPOSITE of its forward heading.
    # heading is in radians. Reversing direction = heading + π
    reverse_heading = truck.heading + np.pi

    # Direction vectors along and across the truck body
    # fwd = unit vector pointing forward (along truck length)
    fwd_x = np.cos(truck.heading)
    fwd_y = np.sin(truck.heading)
    # side = unit vector pointing to the right of the truck
    side_x = fwd_y   # perpendicular to fwd (rotate 90°)
    side_y = -fwd_x

    # Calculate the 4 corners of the truck rectangle:
    # rear-left, rear-right, front-left, front-right
    corners = []
    for length_sign in [-1, +1]:        # rear = -1, front = +1
        for width_sign in [-1, +1]:     # left = -1, right = +1
            corner_x = cx + length_sign * half_l * fwd_x + width_sign * half_w * side_x
            corner_y = cy + length_sign * half_l * fwd_y + width_sign * half_w * side_y
            corners.append((corner_x, corner_y))

    # Check every corner is inside the polygon
    for corner_x, corner_y in corners:
        if not grid.polygon.contains(Point(corner_x, corner_y)): #FIX - KEEP EXTRA GAP WITH BOUNDAY AS BUFFER, ALSO THIS IS JUST SUBSET OF TURNING RADIUS CHECK??
            return False  # this corner is outside — reject this cell

    return True  # all 4 corners are inside — this cell is valid


def is_accessible(grid, r, c, entry_rc):
    """
    Check that dumping at (r,c) doesn't ISOLATE any part of the polygon.
    If we dump here and create a wall, trucks can never reach the blocked area.

    Uses BFS (flood fill) from the entry point:
    Hypothetically mark (r,c) as FILLED, then check all EMPTY cells
    are still reachable from the entry point.

    grid: GridMap
    r,c: the candidate dump cell
    entry_rc: (row,col) of the entry point — the flood fill starts here

    Returns True if nothing gets isolated, False if some area would be cut off.
    """
    # Make a temporary copy of the state array so we don't modify the real grid
    temp_state = grid.state.copy()

    # Hypothetically fill this cell
    temp_state[r, c] = CellState.FILLED

    # BFS flood fill from entry point
    # We mark every cell reachable from entry as visited
    visited = set()
    queue = deque([entry_rc])   # start at the entry point

    while queue:
        cr, cc = queue.popleft()    # take the next cell to process

        if (cr, cc) in visited:
            continue                # already processed this cell, skip
        visited.add((cr, cc))

        # Check all 4 neighbours (up, down, left, right)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr = cr + dr    # neighbour row
            nc = cc + dc    # neighbour col

            # Skip if out of grid bounds
            if not (0 <= nr < grid.rows and 0 <= nc < grid.cols):
                continue

            # Skip if neighbour is not driveable
            # EMPTY and RESERVED are driveable; FILLED, BOUNDARY, PROTECTED are not
            state = temp_state[nr, nc]
            if state not in (CellState.EMPTY, CellState.RESERVED):
                continue

            if (nr, nc) not in visited:
                queue.append((nr, nc))  # add to queue

    # Find ALL cells that should be reachable (empty or reserved, inside polygon)
    all_reachable = set(
        zip(*np.where(
            (temp_state == CellState.EMPTY) |
            (temp_state == CellState.RESERVED)
        ))
    )
    # zip(*np.where(...)) gives us a set of (row,col) tuples for True cells

    # If every reachable cell was visited by BFS → nothing isolated → safe to dump
    # If some cells weren't visited → they're cut off → DON'T dump here
    isolated = all_reachable - visited  # cells in all_reachable but NOT visited
    return len(isolated) == 0           # True = nothing isolated = safe


def get_candidates(grid, truck, entry_rc):
    """
    Get all valid candidate dump cells for a given truck.
    Applies footprint and accessibility filters.
    Returns a list of (row, col) tuples that passed all filters.

    grid: GridMap
    truck: Truck object
    entry_rc: (row, col) of the entry point
    """
    # Start with all EMPTY cells in the grid
    # np.where returns (array_of_rows, array_of_cols) where condition is True
    empty_rows, empty_cols = np.where(grid.state == CellState.EMPTY)

    # Zip them into (row, col) pairs
    all_empty = list(zip(empty_rows.tolist(), empty_cols.tolist()))

    # Apply filters — keep only cells that pass BOTH checks
    valid = []
    for r, c in all_empty:
        # Filter 1: truck body must stay inside polygon
        if not footprint_ok(grid, r, c, truck):
            continue    # skip this cell

        # Filter 2: must not isolate any region (flood fill check)
        # Note: this is expensive for every cell — in production we'd only
        # run this on the top-N scored candidates. For now, run on all.
        if not is_accessible(grid, r, c, entry_rc):
            continue    # skip this cell

        valid.append((r, c))  # this cell passed all filters!

    return valid
