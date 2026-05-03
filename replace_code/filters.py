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
# ─────────────────────────────────────────────────────────────

# filters.py
# ─────────────────────────────────────────────────────────────
# HARD OBSTACLES RESTORED: Dirt > 0.4m is once again a strict,
# impassable wall in the driveable mask. The precise math in truck.py
# guarantees we will never accidentally park inside a wall.
# ─────────────────────────────────────────────────────────────

import numpy as np
from collections import deque
import shapely
from grid_map import CellState
from config import CELL_SIZE, ENTRY_POINT, DRIVE_CLEARANCE_M

_BLOCKED = (CellState.BOUNDARY, CellState.FILLED, CellState.OBSTACLE)
_BRIDGE_RISK_BLOCKED = (CellState.FILLED, CellState.BOUNDARY, CellState.OBSTACLE)

_HEADING_ANGLES = [i * np.pi / 4 for i in range(8)]

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


def make_driveable_mask(grid, truck):
    half_w  = truck.width  / 2.0
    half_l  = truck.length / 2.0
    offsets = _build_corner_offsets(half_w, half_l)

    rows, cols = grid.rows, grid.cols
    mask = np.zeros((rows, cols, 8), dtype=bool)

    base_ok = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            s = grid.state[r, c]
            
            if s in _BLOCKED:
                continue
                
            if grid.z_height[r, c] > DRIVE_CLEARANCE_M:
                continue
                
            base_ok[r, c] = True

    candidate_cells = list(zip(*np.where(base_ok)))
    if not candidate_cells:
        return mask

    centres = np.array([grid.cell_to_world(r, c) for r, c in candidate_cells])
    N = len(centres)
    ex, ey = ENTRY_POINT

    for hi in range(8):
        heading_fits = np.ones(N, dtype=bool)
        for ci in range(4):
            dx, dy   = offsets[hi, ci]
            corner_x = centres[:, 0] + dx
            corner_y = centres[:, 1] + dy

            c_col = np.clip(((corner_x - grid.origin[0]) / grid.cell_size).astype(int), 0, cols - 1)
            c_row = np.clip(((corner_y - grid.origin[1]) / grid.cell_size).astype(int), 0, rows - 1)
            
            corner_state = grid.state[c_row, c_col]
            corner_z     = grid.z_height[c_row, c_col]
            dist_to_entry = np.hypot(corner_x - ex, corner_y - ey)
            
            is_blocked  = np.isin(corner_state, [CellState.FILLED, CellState.OBSTACLE]) | (corner_z > DRIVE_CLEARANCE_M)
            is_boundary = (corner_state == CellState.BOUNDARY) & (dist_to_entry > 15.0)
            
            heading_fits &= ~(is_blocked | is_boundary)

        for i, (r, c) in enumerate(candidate_cells):
            if heading_fits[i]:
                mask[r, c, hi] = True

    return mask


def precompute_coarse_blocked_mask(grid, truck):
    f = _COARSE_FACTOR
    cr = (grid.rows + f - 1) // f
    cc = (grid.cols + f - 1) // f

    drive_mask_3d = make_driveable_mask(grid, truck)
    passable_fine = drive_mask_3d.any(axis=2)

    pad_r = cr * f - grid.rows
    pad_c = cc * f - grid.cols
    padded_passable = np.pad(passable_fine, ((0, pad_r), (0, pad_c)), constant_values=False)

    coarse_passable = padded_passable.reshape(cr, f, cc, f).any(axis=(1, 3))
    return ~coarse_passable


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

    dumpable = np.zeros((grid.rows, grid.cols), dtype=bool)
    for r in range(grid.rows):
        for c in range(grid.cols):
            dumpable[r, c] = grid.is_dumpable(r, c)

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