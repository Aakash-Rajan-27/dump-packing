# path_utils.py
# ─────────────────────────────────────────────────────────────
# Low-level path utilities shared across all planner modules:
#   • 8-directional heading bucket tables
#   • Cell/world conversion helpers for trucks and paths
#   • _truck_inside_boundary — hard polygon check used by bicycle model
#   • _corridor_cell_set    — frozenset of cells reserved by a truck corridor
# ─────────────────────────────────────────────────────────────

import math
import numpy as np
import shapely
import grid_map
from config import ENTRY_POINT, POSE_HEADING_BUCKETS


# ── 8-directional heading bucket tables ──────────────────────────────────────
# Maps each of the 4 cardinal (row, col) step directions to an integer "bucket" index.
# The driveable mask's third dimension is indexed by this bucket so we can have
# direction-specific drivability (e.g. a truck can't reverse uphill but can go forward).
_DIR_TO_BUCKET = {
    ( 0,  1): 0,  # right
    (-1,  1): 1,  # up-right
    (-1,  0): 2,  # up
    (-1, -1): 3,  # up-left
    ( 0, -1): 4,  # left
    ( 1, -1): 5,  # down-left
    ( 1,  0): 6,  # down
    ( 1,  1): 7,  # down-right
}
_BUCKET_TO_DIR = {bucket: delta for delta, bucket in _DIR_TO_BUCKET.items()}
_BUCKET_TO_HEADING = {
    0: 0.0,
    1: -math.pi / 4,
    2: -math.pi / 2,
    3: -3 * math.pi / 4,
    4: math.pi,
    5: 3 * math.pi / 4,
    6: math.pi / 2,
    7: math.pi / 4,
}


def _heading_bucket(dr, dc):
    # Convert a (row_delta, col_delta) step into its driveable-mask bucket index.
    # Falls back to bucket 0 if the direction isn't in the lookup table.
    return _DIR_TO_BUCKET.get((dr, dc), 0)


def _angle_diff_signed(target, current):
    return (target - current + math.pi) % (2 * math.pi) - math.pi


def _bucket_from_heading(heading):
    best_bucket = 0
    best_delta = float('inf')
    for bucket, bucket_heading in _BUCKET_TO_HEADING.items():
        delta = abs(_angle_diff_signed(bucket_heading, heading))
        if delta < best_delta:
            best_bucket = bucket
            best_delta = delta
    return best_bucket


def _heading_for_bucket(bucket):
    return _BUCKET_TO_HEADING[bucket % 8]


# ── Path / cell helpers ───────────────────────────────────────────────────────

def _state_cell(state):
    return (state[0], state[1])


def _path_cells(grid, path):
    cells = []
    for wp in path or []:
        if len(wp) == 3 and not isinstance(wp[0], (int, np.integer)):
            cells.append(grid.world_to_cell(wp[0], wp[1]))
        else:
            cells.append((wp[0], wp[1]))
    return cells


def _dedup_path_cells(grid, path):
    """Like _path_cells but removes consecutive duplicate cells.

    Hybrid-A* paths have ~4 waypoints per metre (TRUCK_MOVE_STEP_M=0.25).
    Using raw _path_cells for locked (r,c,t) constraints causes a 4× time-axis
    mismatch against astar paths (1 cell per step), so the lock never fires at
    the right timestep.  Deduplication normalises both representations to
    ~1 unique cell per step before building ST constraints.
    """
    cells = []
    for wp in path or []:
        if len(wp) == 3 and not isinstance(wp[0], (int, np.integer)):
            rc = grid.world_to_cell(wp[0], wp[1])
        else:
            rc = (wp[0], wp[1])
        if not cells or rc != cells[-1]:
            cells.append(rc)
    return cells


def _truck_front_cell(grid, truck):
    if hasattr(truck, "front_center_cell"):
        return truck.front_center_cell(grid)
    front_x = truck.pos[0] + math.cos(truck.heading) * (truck.length / 2.0)
    front_y = truck.pos[1] + math.sin(truck.heading) * (truck.length / 2.0)
    return grid.world_to_cell(front_x, front_y)


def _truck_rear_pose(truck):
    if hasattr(truck, "rear_axle_world"):
        rear_x, rear_y = truck.rear_axle_world()
    else:
        half_len = truck.length / 2.0
        rear_x = truck.pos[0] - math.cos(truck.heading) * half_len
        rear_y = truck.pos[1] - math.sin(truck.heading) * half_len
    return float(rear_x), float(rear_y), float(truck.heading)


def _truck_front_world(truck):
    if hasattr(truck, "front_axle_world"):
        return truck.front_axle_world()
    return truck.front_center_world()


# ── Boundary / polygon check ──────────────────────────────────────────────────

def _truck_inside_boundary(polygon, body_x, body_y, heading, half_l, half_w):
    """Hard boundary check: all 4 truck body corners must be inside the polygon.
    Skipped within (truck_length + 2m) of ENTRY_POINT to allow gate crossing only."""
    if math.hypot(body_x - ENTRY_POINT[0], body_y - ENTRY_POINT[1]) <= half_l * 2.0 + 2.0:
        return True
    hcos, hsin = math.cos(heading), math.sin(heading)
    scos, ssin = -hsin, hcos
    for ls in (-1, 1):
        for ws in (-1, 1):
            cx = body_x + ls * half_l * hcos + ws * half_w * scos
            cy = body_y + ls * half_l * hsin + ws * half_w * ssin
            if not shapely.contains_xy(polygon, cx, cy):
                return False
    return True


# ── Corridor cell set ─────────────────────────────────────────────────────────

def _corridor_cell_set(grid, truck_id):
    """Return the frozenset of (r,c) cells reserved by this truck's path corridors."""
    cells = set()
    for key in (f"exit_{truck_id}", f"dump_{truck_id}"):
        for r, c, _ in grid._path_corridors.get(key, []):
            cells.add((r, c))
    return frozenset(cells)
