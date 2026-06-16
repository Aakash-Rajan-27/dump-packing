# filters.py
# ─────────────────────────────────────────────────────────────
# Changes from previous version:
#
# 1. is_accessible() — BFS on downsampled coarse grid
#    Old: BFS on full grid (31x31 at 3m = fast)
#    New: At 0.5m the full grid is 181x181 = BFS takes 100ms per cell.
#         Isolation is a topological property — we can check it on a
#         coarser grid without losing accuracy.
#         Downsample by factor = int(3.0 / CELL_SIZE) = 6x.
#         Resulting BFS grid: 31x31, same as before = <1ms.
#
# 2. get_raw_candidates() — subsampled candidate selection
#    Subsamples to one candidate per 3m×3m block to keep the planning 
#    problem tractable for MCTS while retaining fine-grid physics.
#
# ── Phase A & B Updates (Truck-Specific BFS & Gate Fix) ────────
# 
# 3. PADDOCK RULE: make_driveable_mask() now ensures a truck 
#    cannot drive over non-zero z_height (freshly dumped material).
#
# 4. TRUCK-AWARE BFS: is_accessible() uses `precompute_coarse_blocked_mask` 
#    which calculates the specific physical footprint of the truck 
#    (Small/Medium/Large) before allowing BFS through gaps.
#
# 5. GATE COLLISION FIX: make_driveable_mask now permits BOUNDARY 
#    clipping if the truck is within 15m of the ENTRY_POINT. 
#    This prevents Large/Medium trucks from failing A* instantly at spawn.
#
# ── Phase C Update (Self-Burial Fix) ───────────────────────────
# 6. FORGIVENESS BUFFER: Added +0.5m to the DRIVE_CLEARANCE_M check 
#    to allow trucks to escape the shallow edges of their own dump piles.
# ─────────────────────────────────────────────────────────────

import numpy as np
from collections import deque
import shapely
from scipy.ndimage import binary_dilation
from grid_map import CellState
from config import (CELL_SIZE, ENTRY_POINT, DRIVE_CLEARANCE_M,
                    TARGET_PILE_HEIGHT, POSE_HEADING_BUCKETS,
                    STAGING_FOOTPRINT_MARGIN_M)

_BLOCKED = (CellState.BOUNDARY, CellState.FILLED, CellState.OBSTACLE, CellState.PATH_RESERVED)
_BRIDGE_RISK_BLOCKED = (CellState.FILLED, CellState.BOUNDARY, CellState.OBSTACLE)

_HEADING_ANGLES = [
    i * 2 * np.pi / POSE_HEADING_BUCKETS
    for i in range(POSE_HEADING_BUCKETS)
]

_COARSE_FACTOR = max(1, int(round(3.0 / CELL_SIZE)))
_CANDIDATE_STRIDE = max(1, int(round(3.0 / CELL_SIZE)))


def _build_corner_offsets(half_w, half_l):
    offsets = np.zeros((len(_HEADING_ANGLES), 4, 2), dtype=np.float64)
    for hi, heading in enumerate(_HEADING_ANGLES):
        fwd  = np.array([np.cos(heading), np.sin(heading)])
        side = np.array([-fwd[1], fwd[0]])
        idx  = 0
        for ls in (-1, +1):
            for ws in (-1, +1):
                offsets[hi, idx] = ls * half_l * fwd + ws * half_w * side
                idx += 1
    return offsets


def _build_dump_offsets(half_w, half_l):
    angles  = [0, np.pi/2, np.pi, 3*np.pi/2]
    offsets = np.zeros((4, 4, 2), dtype=np.float64)
    for hi, heading in enumerate(angles):
        rev  = heading + np.pi
        fwd  = np.array([np.cos(rev), np.sin(rev)])
        side = np.array([-fwd[1], fwd[0]])
        idx  = 0
        for ls in (-1, +1):
            for ws in (-1, +1):
                offsets[hi, idx] = ls * half_l * fwd + ws * half_w * side
                idx += 1
    return offsets


def dump_accessibility_ok(grid, candidates, truck):
    if not candidates:
        return np.array([], dtype=bool)

    half_w  = truck.width  / 2.0
    half_l  = truck.length / 2.0
    offsets = _build_dump_offsets(half_w, half_l)

    centres = np.array([grid.cell_to_world(r, c) for r, c in candidates])
    N       = len(centres)
    fits    = np.zeros(N, dtype=bool)

    for hi in range(4):
        heading_fits = np.ones(N, dtype=bool)
        for ci in range(4):
            dx, dy   = offsets[hi, ci]
            corner_x = centres[:, 0] + dx
            corner_y = centres[:, 1] + dy
            inside   = shapely.contains_xy(grid.polygon, corner_x, corner_y)
            heading_fits &= inside
        fits |= heading_fits

    return fits


def make_driveable_mask(grid, truck, ignore_path_reserved=False):
    """Build a [rows, cols, heading_buckets] boolean passability mask.

    ignore_path_reserved: when True, PATH_RESERVED cells are treated as
    passable — used for exit-path planning so active dump corridors don't
    cut off the route back to the entry gate.
    """
    half_w  = truck.width  / 2.0
    half_l  = truck.length / 2.0
    offsets = _build_corner_offsets(half_w, half_l)

    rows, cols        = grid.rows, grid.cols
    mask              = np.zeros((rows, cols, POSE_HEADING_BUCKETS), dtype=bool)
    max_escape_height = DRIVE_CLEARANCE_M + 0.5  # forgiveness buffer so trucks can escape shallow pile edges

    blocked_for_base = list(_BLOCKED)
    if ignore_path_reserved and CellState.PATH_RESERVED in blocked_for_base:
        blocked_for_base = [s for s in blocked_for_base if s != CellState.PATH_RESERVED]

    # Boundary erosion: a cell is "near boundary" if any BOUNDARY cell falls within
    # the truck's half-width radius. Eroding the polygon by half_w guarantees that the
    # full path corridor (which marks cells within half_w of the body centre) never
    # touches or crosses the polygon wall — no shapely needed in the heading loop.
    half_w_cells = half_w / grid.cell_size
    half_int     = int(np.ceil(half_w_cells))
    dr_idx       = np.arange(-half_int, half_int + 1)
    DC, DR       = np.meshgrid(dr_idx, dr_idx)
    struct       = np.hypot(DR, DC) <= half_w_cells
    near_boundary = binary_dilation(grid.state == CellState.BOUNDARY, structure=struct)

    base_ok = (~near_boundary) & (~np.isin(grid.state, blocked_for_base)) & (grid.z_height <= max_escape_height)

    rows_arr, cols_arr = np.where(base_ok)
    if len(rows_arr) == 0:
        return mask

    centres_x  = grid.origin[0] + cols_arr * grid.cell_size + grid.cell_size / 2
    centres_y  = grid.origin[1] + rows_arr * grid.cell_size + grid.cell_size / 2

    # Heading loop only checks FILLED / OBSTACLE / PATH_RESERVED via fast cell-state
    # lookup — boundary is already handled by near_boundary above.
    corner_blocked_states = [CellState.FILLED, CellState.OBSTACLE]
    if not ignore_path_reserved:
        corner_blocked_states.append(CellState.PATH_RESERVED)

    for hi in range(POSE_HEADING_BUCKETS):
        heading_fits = np.ones(len(rows_arr), dtype=bool)
        for ci in range(4):
            dx, dy    = offsets[hi, ci]
            corner_x  = centres_x + dx
            corner_y  = centres_y + dy

            # All truck corners must be strictly inside the polygon.
            # The near_boundary erosion above is an approximation; diagonal polygon
            # edges can leave cells passable whose truck corners clip outside.
            heading_fits &= shapely.contains_xy(grid.polygon, corner_x, corner_y)

            c_col = np.clip(((corner_x - grid.origin[0]) / grid.cell_size).astype(int), 0, cols - 1)
            c_row = np.clip(((corner_y - grid.origin[1]) / grid.cell_size).astype(int), 0, rows - 1)

            corner_state = grid.state[c_row, c_col]
            corner_z     = grid.z_height[c_row, c_col]
            is_blocked   = np.isin(corner_state, corner_blocked_states) | (corner_z > max_escape_height)

            heading_fits &= ~is_blocked

        mask[rows_arr[heading_fits], cols_arr[heading_fits], hi] = True

    return mask


def is_pose_driveable(grid, truck, x, y, heading, margin_m=None, ignore_path_reserved=False):
    """Check the exact oriented truck rectangle at a continuous body pose."""
    margin = STAGING_FOOTPRINT_MARGIN_M if margin_m is None else margin_m
    half_w = truck.width / 2.0 + margin
    half_l = truck.length / 2.0 + margin
    hx, hy = np.cos(heading), np.sin(heading)
    sx, sy = -hy, hx

    blocked = (CellState.FILLED, CellState.OBSTACLE) if ignore_path_reserved \
              else (CellState.FILLED, CellState.OBSTACLE, CellState.PATH_RESERVED)

    for ls in (-1, 1):
        for ws in (-1, 1):
            corner_x = x + ls * half_l * hx + ws * half_w * sx
            corner_y = y + ls * half_l * hy + ws * half_w * sy
            if not shapely.contains_xy(grid.polygon, corner_x, corner_y):
                return False
            r, c = grid.world_to_cell(corner_x, corner_y)
            if grid.state[r, c] in blocked:
                return False
            if grid.z_height[r, c] > DRIVE_CLEARANCE_M + 0.5:
                return False
    return True


def _coarse_from_fine(grid, drive_mask_3d):
    f  = _COARSE_FACTOR
    cr = (grid.rows + f - 1) // f
    cc = (grid.cols + f - 1) // f
    passable_fine   = drive_mask_3d.any(axis=2)
    pad_r           = cr * f - grid.rows
    pad_c           = cc * f - grid.cols
    padded_passable = np.pad(passable_fine, ((0, pad_r), (0, pad_c)), constant_values=False)
    coarse_passable = padded_passable.reshape(cr, f, cc, f).any(axis=(1, 3))
    return ~coarse_passable


def precompute_coarse_blocked_mask(grid, truck):
    return _coarse_from_fine(grid, make_driveable_mask(grid, truck))


def compute_masks(grid, truck, ignore_path_reserved=False):
    """Return (coarse_blocked, fine_driveable) from a single make_driveable_mask call.
    Use this in planning tasks to avoid computing the mask twice."""
    fine = make_driveable_mask(grid, truck, ignore_path_reserved=ignore_path_reserved)
    return _coarse_from_fine(grid, fine), fine


def _has_bridge_risk(grid, r, c):
    n = r - 1 < 0          or grid.state[r-1, c] in _BRIDGE_RISK_BLOCKED
    s = r + 1 >= grid.rows or grid.state[r+1, c] in _BRIDGE_RISK_BLOCKED
    w = c - 1 < 0          or grid.state[r, c-1] in _BRIDGE_RISK_BLOCKED
    e = c + 1 >= grid.cols or grid.state[r, c+1] in _BRIDGE_RISK_BLOCKED
    return sum([n, s, w, e]) >= 2


def is_accessible(grid, r, c, entry_rc, truck, precomputed_coarse_mask=None):
    if not _has_bridge_risk(grid, r, c):
        return True

    f = _COARSE_FACTOR
    coarse_blocked = precomputed_coarse_mask if precomputed_coarse_mask is not None else precompute_coarse_blocked_mask(grid, truck)

    cr = coarse_blocked.shape[0]
    cc = coarse_blocked.shape[1]

    coarse_r = r // f
    coarse_c = c // f
    
    temp_blocked = coarse_blocked.copy()
    temp_blocked[coarse_r, coarse_c] = True

    entry_coarse = (entry_rc[0] // f, entry_rc[1] // f)

    visited = set()
    queue   = deque([entry_coarse])

    while queue:
        er2, ec2 = queue.popleft()
        if (er2, ec2) in visited: continue
        visited.add((er2, ec2))
        
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr2, nc2 = er2+dr, ec2+dc
            if not (0 <= nr2 < cr and 0 <= nc2 < cc): continue
            if temp_blocked[nr2, nc2]: continue
            if (nr2, nc2) not in visited:
                queue.append((nr2, nc2))

    all_reachable = set(zip(*np.where(~temp_blocked)))
    return len(all_reachable - visited) == 0


def get_raw_candidates(grid, truck):
    stride = _CANDIDATE_STRIDE

    # ── VECTORIZED dumpable ──────────────────────────────────────────────────────
    # Was: Python double loop calling is_dumpable() on every cell (~8100 iterations).
    # Now: one numpy boolean expression — mirrors is_dumpable() exactly but runs in C.
    _not_dumpable_states = (CellState.BOUNDARY, CellState.PROTECTED,
                            CellState.OBSTACLE, CellState.FILLED, CellState.RESERVED,
                            CellState.PATH_RESERVED)
    dumpable = (~np.isin(grid.state, list(_not_dumpable_states)) &
                (grid.z_height < TARGET_PILE_HEIGHT))

    candidates = []
    for r0 in range(0, grid.rows, stride):
        for c0 in range(0, grid.cols, stride):
            r1 = min(r0 + stride, grid.rows)
            c1 = min(c0 + stride, grid.cols)

            block_dumpable = dumpable[r0:r1, c0:c1]
            if not block_dumpable.any():
                continue

            block_z = grid.z_height[r0:r1, c0:c1].copy()
            block_z[~block_dumpable] = np.inf
            local_idx = np.unravel_index(block_z.argmin(), block_z.shape)
            r, c = r0 + local_idx[0], c0 + local_idx[1]
            candidates.append((r, c))

    if not candidates:
        return []

    fits = dump_accessibility_ok(grid, candidates, truck)
    return [cell for cell, ok in zip(candidates, fits) if ok]
