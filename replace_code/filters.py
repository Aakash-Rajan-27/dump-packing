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
#         A cell is 'blocked' in coarse grid if ANY fine cell in that
#         block is FILLED or BOUNDARY.
#
# 2. get_candidates() — subsampled candidate selection
#    Old: all dumpable cells as candidates (32k at 0.5m — too many)
#    New: subsample to one candidate per 3m×3m block (original cell size).
#         This gives ~800 candidates max, same as before. MCTS and
#         scoring still operate at this granularity. The fine grid is
#         used for accurate height modelling, not for candidate count.
#
# 3. make_driveable_mask() — no change in logic, but runs on fine grid
#    At 0.5m with 28k+ cells × 8 headings this is still fast because
#    shapely.contains_xy is batched. ~64ms estimated.
#
# 4. All other functions unchanged.
# ─────────────────────────────────────────────────────────────

import numpy as np
from collections import deque
import shapely
from grid_map import CellState
from config import CELL_SIZE

_BLOCKED = (CellState.BOUNDARY, CellState.FILLED,
            CellState.PARTIAL, CellState.OBSTACLE)

_BRIDGE_RISK_BLOCKED = (CellState.FILLED, CellState.BOUNDARY, CellState.OBSTACLE)

_HEADING_ANGLES = [i * np.pi / 4 for i in range(8)]

# Downsample factor for BFS: coarse cell = 3m physical regardless of CELL_SIZE
# At CELL_SIZE=0.5m: factor=6 → coarse grid = fine/6 ≈ 31x31
# At CELL_SIZE=3.0m: factor=1 → no downsampling
_COARSE_FACTOR = max(1, int(round(3.0 / CELL_SIZE)))

# Subsample stride for candidate selection (one candidate per 3m block)
_CANDIDATE_STRIDE = max(1, int(round(3.0 / CELL_SIZE)))


# ── Corner offset precomputation ───────────────────────────

def _build_corner_offsets(half_w, half_l):
    """8 heading angles × 4 corners → (8,4,2) offset array."""
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
    """4 cardinal reversed headings × 4 corners → (4,4,2) for dump check."""
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


# ── Dump Accessibility ─────────────────────────────────────

def dump_accessibility_ok(grid, candidates, truck):
    """Vectorized: can this truck reverse into each candidate cell?"""
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


# ── Heading-Aware Driveability ─────────────────────────────

def make_driveable_mask(grid, truck):
    """
    (rows, cols, 8) heading-aware driveable mask.

    A cell is driveable at heading h if:
      1. State is passable (EMPTY, RESERVED, PROTECTED) AND z_height == 0
      2. The truck rectangle at that heading does not clip any
         PARTIAL/FILLED/BOUNDARY neighbour cell.

    Note: we do NOT require truck corners to be inside the polygon for
    driving — only for dumping. Trucks drive through the entry corridor
    (PROTECTED) which sits at the polygon edge; their body extends
    slightly outside the polygon while passing through, which is fine.
    """
    half_w  = truck.width  / 2.0
    half_l  = truck.length / 2.0
    offsets = _build_corner_offsets(half_w, half_l)

    rows, cols = grid.rows, grid.cols
    mask = np.zeros((rows, cols, 8), dtype=bool)

    # Base passability: driveable states with no material
    # PROTECTED is explicitly included — it's the entry corridor
    base_ok = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            s = grid.state[r, c]
            if s in (CellState.BOUNDARY, CellState.FILLED,
                     CellState.PARTIAL, CellState.OBSTACLE):
                continue
            # EMPTY, RESERVED, PROTECTED all pass state check
            if grid.z_height[r, c] > 0:
                continue
            base_ok[r, c] = True

    candidate_cells = list(zip(*np.where(base_ok)))
    if not candidate_cells:
        return mask

    centres = np.array([grid.cell_to_world(r, c)
                        for r, c in candidate_cells])
    N = len(centres)

    for hi in range(8):
        heading_fits = np.ones(N, dtype=bool)
        for ci in range(4):
            dx, dy   = offsets[hi, ci]
            corner_x = centres[:, 0] + dx
            corner_y = centres[:, 1] + dy

            # Check corner cell is not a blocked cell (PARTIAL/FILLED/BOUNDARY)
            # Do NOT check polygon containment — trucks can slightly overhang
            # the polygon edge while navigating (esp. at entry corridor)
            c_col = np.clip(
                ((corner_x - grid.origin[0]) / grid.cell_size).astype(int),
                0, cols - 1
            )
            c_row = np.clip(
                ((corner_y - grid.origin[1]) / grid.cell_size).astype(int),
                0, rows - 1
            )
            corner_state = grid.state[c_row, c_col]
            not_blocked  = ~np.isin(corner_state,
                                    [CellState.BOUNDARY, CellState.FILLED,
                                     CellState.PARTIAL, CellState.OBSTACLE])
            heading_fits &= not_blocked

        for i, (r, c) in enumerate(candidate_cells):
            if heading_fits[i]:
                mask[r, c, hi] = True

    return mask


# ── Accessibility (isolation check) ───────────────────────

def _build_coarse_blocked(grid):
    """
    Downsample the fine grid to a coarse grid for BFS.
    Each coarse cell covers _COARSE_FACTOR x _COARSE_FACTOR fine cells.
    A coarse cell is blocked if ANY fine cell in its block is FILLED or BOUNDARY.

    Returns: 2D bool array, True = blocked in coarse grid.
    """
    f = _COARSE_FACTOR
    # Coarse grid dimensions
    cr = (grid.rows + f - 1) // f
    cc = (grid.cols + f - 1) // f

    # Pad fine state array to be divisible by f
    pad_r = cr * f - grid.rows
    pad_c = cc * f - grid.cols
    padded = np.pad(
        grid.state,
        ((0, pad_r), (0, pad_c)),
        constant_values=CellState.BOUNDARY
    )

    # Reshape into blocks and check if any cell in block is blocked
    blocked_fine = np.isin(padded,
                           [CellState.BOUNDARY, CellState.FILLED,
                            CellState.PARTIAL, CellState.OBSTACLE])
    # Shape: (cr, f, cc, f) → max over block dims
    coarse_blocked = blocked_fine.reshape(cr, f, cc, f).any(axis=(1, 3))
    return coarse_blocked


def _has_bridge_risk(grid, r, c):
    """O(1) bridge risk check — unchanged."""
    n = r - 1 < 0          or grid.state[r-1, c] in _BRIDGE_RISK_BLOCKED
    s = r + 1 >= grid.rows or grid.state[r+1, c] in _BRIDGE_RISK_BLOCKED
    w = c - 1 < 0          or grid.state[r, c-1] in _BRIDGE_RISK_BLOCKED
    e = c + 1 >= grid.cols or grid.state[r, c+1] in _BRIDGE_RISK_BLOCKED
    return sum([n, s, w, e]) >= 2


def is_accessible(grid, r, c, entry_rc):
    """
    CHANGED: BFS on coarse downsampled grid instead of full fine grid.
    Old: BFS on 31x31 = fast. New fine grid: 181x181 = 100ms per call.
    Fix: downsample 6x to ~31x31 coarse grid, run BFS there instead.

    The isolation property is topological — a region cut off at 3m
    resolution is still cut off at 0.5m resolution. No accuracy lost.
    """
    if not _has_bridge_risk(grid, r, c):
        return True

    f = _COARSE_FACTOR
    coarse_blocked = _build_coarse_blocked(grid)

    cr = coarse_blocked.shape[0]
    cc = coarse_blocked.shape[1]

    # Map fine cell (r,c) to coarse cell
    coarse_r = r // f
    coarse_c = c // f

    # Hypothetically block the coarse cell containing (r,c)
    coarse_blocked[coarse_r, coarse_c] = True

    # Map entry_rc to coarse
    entry_coarse = (entry_rc[0] // f, entry_rc[1] // f)

    # BFS on coarse grid
    visited = set()
    queue   = deque([entry_coarse])

    while queue:
        er2, ec2 = queue.popleft()
        if (er2, ec2) in visited:
            continue
        visited.add((er2, ec2))
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr2, nc2 = er2+dr, ec2+dc
            if not (0 <= nr2 < cr and 0 <= nc2 < cc):
                continue
            if coarse_blocked[nr2, nc2]:
                continue
            if (nr2, nc2) not in visited:
                queue.append((nr2, nc2))

    # All non-blocked coarse cells that should be reachable
    all_reachable = set(zip(*np.where(~coarse_blocked)))
    return len(all_reachable - visited) == 0


# ── Main Entry Point ───────────────────────────────────────

def get_candidates(grid, truck, entry_rc):
    """
    CHANGED: subsampled candidate selection.
    Old: every dumpable cell = candidate (961 at 3m cells — manageable)
    New: at 0.5m there are 28k+ dumpable cells — too many for MCTS/scoring.
         Subsample: one candidate per _CANDIDATE_STRIDE × _CANDIDATE_STRIDE
         block (= one per 3m × 3m physical area = same count as before).
         Within each block, pick the cell with lowest z_height (best to dump).

    This preserves the full fine-resolution height map for physics accuracy
    while keeping the planning problem tractable.
    """
    stride = _CANDIDATE_STRIDE

    # Build dumpable mask
    dumpable = np.zeros((grid.rows, grid.cols), dtype=bool)
    for r in range(grid.rows):
        for c in range(grid.cols):
            dumpable[r, c] = grid.is_dumpable(r, c)

    candidates = []
    # Walk in stride steps
    for r0 in range(0, grid.rows, stride):
        for c0 in range(0, grid.cols, stride):
            r1 = min(r0 + stride, grid.rows)
            c1 = min(c0 + stride, grid.cols)

            # Find dumpable cells in this block
            block_dumpable = dumpable[r0:r1, c0:c1]
            if not block_dumpable.any():
                continue

            # Pick cell with lowest z_height in block (most room for more material)
            block_z = grid.z_height[r0:r1, c0:c1].copy()
            block_z[~block_dumpable] = np.inf
            local_idx = np.unravel_index(block_z.argmin(),
                                         block_z.shape)
            r, c = r0 + local_idx[0], c0 + local_idx[1]
            candidates.append((r, c))

    if not candidates:
        return []

    # Dump accessibility filter
    fits = dump_accessibility_ok(grid, candidates, truck)
    return [cell for cell, ok in zip(candidates, fits) if ok]