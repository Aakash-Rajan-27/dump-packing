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


def _make_locked_entry(truck, path, tail_ticks, grid=None):
    """Build a locked_paths entry for one truck.

    Returns (body_center_poses, tail_ticks, half_length, half_width) where
    body_center_poses is a list of (cx, cy, heading) world positions measured
    at the body centre, suitable for footprint-overlap checking.

    Path waypoints can be:
      (rear_x, rear_y, heading) floats — smooth bicycle path output
      (r, c, heading) int r/c        — coarse A* path with heading
      (r, c)          int r/c        — coarse A* path without heading
    All are converted to body-centre world coords.  Cell waypoints without an
    explicit heading use the heading of the nearest forward smooth waypoint, or
    the direction between consecutive cells, so the footprint is approximately
    oriented correctly even for coarse-path segments.
    """
    hl = truck.length / 2.0
    hw = truck.width / 2.0

    # Build raw list of (wx, wy, heading_or_None) body-centre world positions.
    # First entry is the truck's current body centre (always known exactly).
    raw = [(float(truck.pos[0]), float(truck.pos[1]), float(truck.heading))]

    for wp in (path or []):
        if len(wp) >= 3 and not isinstance(wp[0], (int, np.integer)):
            # Smooth bicycle waypoint: (rear_x, rear_y, heading)
            rx, ry, h = float(wp[0]), float(wp[1]), float(wp[2])
            raw.append((rx + math.cos(h) * hl, ry + math.sin(h) * hl, h))
        elif len(wp) >= 3 and grid is not None:
            # Coarse A* waypoint: (r, c, heading)
            r, c, h = int(wp[0]), int(wp[1]), float(wp[2])
            wx, wy = grid.cell_to_world(r, c)
            raw.append((wx, wy, h))
        elif len(wp) >= 2 and grid is not None:
            # Coarse A* waypoint: (r, c) — heading inferred later
            r, c = int(wp[0]), int(wp[1])
            wx, wy = grid.cell_to_world(r, c)
            raw.append((wx, wy, None))
        elif len(wp) >= 2:
            raw.append((float(wp[0]), float(wp[1]), None))

    # Fill in None headings: prefer the nearest forward pose with a known heading;
    # fall back to the direction between consecutive positions.
    for i in range(len(raw)):
        if raw[i][2] is not None:
            continue
        h = 0.0
        for fwd in range(i + 1, min(i + 5, len(raw))):
            if raw[fwd][2] is not None:
                h = raw[fwd][2]
                break
            dx = raw[fwd][0] - raw[i][0]
            dy = raw[fwd][1] - raw[i][1]
            if dx * dx + dy * dy > 1e-9:
                h = math.atan2(dy, dx)
                break
        raw[i] = (raw[i][0], raw[i][1], h)

    return (raw, tail_ticks, hl, hw)


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
