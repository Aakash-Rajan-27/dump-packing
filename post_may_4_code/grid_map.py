# grid_map.py
# ─────────────────────────────────────────────────────────────
# Changes from previous version:
#
# 1. _classify_cells() — vectorized with shapely.contains_xy
#    Old: per-point shapely.contains(Point) loop — 410ms at 0.5m
#    New: batch shapely.contains_xy on all cell centres — 9ms
#
# 2. _mark_entry_corridor() — uses ENTRY_CORRIDOR_CELLS from config
#    Old: hardcoded range(-1, 2) = 3-cell wide corridor (9m at 3m cells)
#    New: range(-ENTRY_CORRIDOR_CELLS, ENTRY_CORRIDOR_CELLS+1) = 6 cells
#         same physical 3m buffer at new 0.5m cell size
#
# 3. fill_pct() and pack_pct() — no logic change, work on finer grid
#    The finer grid means these metrics are now much more meaningful:
#    fill_pct = fraction of 0.5m cells at TARGET_PILE_HEIGHT
#    pack_pct = average z_height / TARGET across all valid cells
#
# 4. dump_at() — no logic change, cone geometry unchanged
#    At 0.5m cells it now naturally affects many more cells:
#    small truck: ~9 cells, medium: ~83 cells, large: ~262 cells
#    States (PARTIAL/FILLED) assigned per-cell from z_height — this
#    is now a true continuous height map, not a discrete cell fill.
#
# 5. All other methods unchanged.
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# REALISTIC SANDPILE PHYSICS ADDED
# Replaces simple block-dumping with volumetric cone spreading 
# and angle-of-repose relaxation. Vectorized classification kept.
# ─────────────────────────────────────────────────────────────

# grid_map.py
# ─────────────────────────────────────────────────────────────
# REALISTIC SANDPILE PHYSICS ADDED
# Replaces simple block-dumping with volumetric cone spreading 
# and angle-of-repose relaxation. Vectorized classification kept.
# Includes "Anti-Freeze" check in the physics loop to stop infinite Zeno paradoxes.
# ─────────────────────────────────────────────────────────────

import math
import numpy as np
import shapely
import shapely.geometry
from enum import IntEnum
from config import (TARGET_PILE_HEIGHT, DRIVE_CLEARANCE_M,
                    _TAN_REPOSE, ENTRY_CORRIDOR_CELLS)


class CellState(IntEnum):
    BOUNDARY      = 0
    EMPTY         = 1
    PARTIAL       = 2
    FILLED        = 3
    RESERVED      = 4
    PROTECTED     = 5
    OBSTACLE      = 6
    PATH_RESERVED = 7  # temporary: marks a planned-path corridor; cleared when truck arrives


class GridMap:
    def __init__(self, polygon_coords, cell_size):
        self.cell_size = cell_size
        self.polygon   = shapely.geometry.Polygon(polygon_coords)

        minx, miny, maxx, maxy = self.polygon.bounds
        self.cols   = int((maxx - minx) / cell_size) + 1
        self.rows   = int((maxy - miny) / cell_size) + 1
        self.origin = (minx, miny)

        self.state     = np.full((self.rows, self.cols),
                                 CellState.BOUNDARY, dtype=np.int8)
        self.z_height  = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.pheromone = np.ones ((self.rows, self.cols), dtype=np.float32)

        self._path_corridors: dict = {}  # key -> list of (r, c, original_state)
        self._path_corridor_counts = {}
        self._path_corridor_base = {}

        self._classify_cells()
        self._mark_entry_corridor()

    def _classify_cells(self):
        rows_idx, cols_idx = np.mgrid[0:self.rows, 0:self.cols]
        xs = self.origin[0] + cols_idx * self.cell_size + self.cell_size / 2
        ys = self.origin[1] + rows_idx * self.cell_size + self.cell_size / 2

        inside = shapely.contains_xy(
            self.polygon,
            xs.ravel(),
            ys.ravel()
        ).reshape(self.rows, self.cols)

        self.state[inside] = CellState.EMPTY

    def _mark_entry_corridor(self):
        from config import ENTRY_POINT
        er, ec = self.world_to_cell(*ENTRY_POINT)
        half = ENTRY_CORRIDOR_CELLS
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                nr, nc = er + dr, ec + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if self.state[nr, nc] == CellState.EMPTY:
                        self.state[nr, nc] = CellState.PROTECTED

    # ── PATH CORRIDOR MANAGEMENT ──────────────────────────────────────────────
    _CORRIDOR_RESTORABLE = frozenset([
        CellState.EMPTY, CellState.PARTIAL, CellState.RESERVED
    ])

    def mark_path_corridor(self, key, body_cells, half_w_cells):
        """Mark a truck-width corridor around body_cells as PATH_RESERVED.
        Saves original states so clear_path_corridor can restore them.
        key: unique string (e.g. 'dump_0', 'exit_1') per truck/path."""
        half = int(math.ceil(half_w_cells))
        saved = []
        seen = set()
        for (r, c) in body_cells:
            for dr in range(-half, half + 1):
                for dc in range(-half, half + 1):
                    if math.hypot(dr, dc) > half_w_cells:
                        continue
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                        continue
                    if (nr, nc) in seen:
                        continue
                    seen.add((nr, nc))
                    orig = int(self.state[nr, nc])
                    if orig in self._CORRIDOR_RESTORABLE or orig == CellState.PATH_RESERVED:
                        saved.append((nr, nc, orig))
                        if (nr, nc) not in self._path_corridor_counts:
                            self._path_corridor_base[(nr, nc)] = orig
                        self._path_corridor_counts[(nr, nc)] = (
                            self._path_corridor_counts.get((nr, nc), 0) + 1)
                        self.state[nr, nc] = CellState.PATH_RESERVED
        self._path_corridors[key] = saved

    def clear_path_corridor(self, key):
        """Restore cells that were marked PATH_RESERVED under this key.
        Safe to call even if the key was never registered."""
        for r, c, _orig in self._path_corridors.pop(key, []):
            count = self._path_corridor_counts.get((r, c), 0) - 1
            if count > 0:
                self._path_corridor_counts[(r, c)] = count
                continue
            self._path_corridor_counts.pop((r, c), None)
            orig = self._path_corridor_base.pop((r, c), CellState.EMPTY)
            if self.state[r, c] == CellState.PATH_RESERVED:
                self.state[r, c] = orig
                self._update_state(r, c)  # reclassify from z_height if EMPTY/PARTIAL

    def cell_to_world(self, r, c):
        x = self.origin[0] + c * self.cell_size + self.cell_size / 2
        y = self.origin[1] + r * self.cell_size + self.cell_size / 2
        return (x, y)

    def world_to_cell(self, x, y):
        c = int((x - self.origin[0]) / self.cell_size)
        r = int((y - self.origin[1]) / self.cell_size)
        c = max(0, min(c, self.cols - 1))
        r = max(0, min(r, self.rows - 1))
        return (r, c)

    def fill_pct(self):
        valid  = np.sum((self.state == CellState.EMPTY)   |
                        (self.state == CellState.PARTIAL)  |
                        (self.state == CellState.FILLED)   |
                        (self.state == CellState.RESERVED))
        filled = np.sum(self.state == CellState.FILLED)
        return filled / max(1, valid)

    def pack_pct(self):
        valid_mask = ((self.state == CellState.EMPTY)   |
                      (self.state == CellState.PARTIAL)  |
                      (self.state == CellState.FILLED)   |
                      (self.state == CellState.RESERVED))
        total_valid = np.sum(valid_mask)
        if total_valid == 0:
            return 0.0
        return float(
            np.sum(np.clip(self.z_height, 0, TARGET_PILE_HEIGHT) * valid_mask)
            / (total_valid * TARGET_PILE_HEIGHT)
        )

    def dump_at(self, r, c, volume_m3):
        tan_theta = _TAN_REPOSE
        r_pile_m  = (3 * volume_m3 / (np.pi * tan_theta)) ** (1/3)
        r_cells   = r_pile_m / self.cell_size

        rmin = max(0, r - int(math.ceil(r_cells)))
        rmax = min(self.rows, r + int(math.ceil(r_cells)) + 1)
        cmin = max(0, c - int(math.ceil(r_cells)))
        cmax = min(self.cols, c + int(math.ceil(r_cells)) + 1)

        # Vectorized cone deposition — replaces the Python double loop.
        ii = np.arange(rmin, rmax)
        jj = np.arange(cmin, cmax)
        II, JJ = np.meshgrid(ii, jj, indexing='ij')
        dist_m = np.hypot(II - r, JJ - c) * self.cell_size
        cone_mask = (dist_m < r_pile_m) & (self.state[rmin:rmax, cmin:cmax] != CellState.BOUNDARY)
        self.z_height[rmin:rmax, cmin:cmax] += np.where(cone_mask, (r_pile_m - dist_m) * tan_theta, 0.0).astype(np.float32)

        max_dz = tan_theta * self.cell_size
        rx_min, rx_max = max(1, rmin - 4), min(self.rows - 1, rmax + 4)
        cx_min, cx_max = max(1, cmin - 4), min(self.cols - 1, cmax + 4)

        # Relaxation: cap at 25 passes to prevent unbounded spin on large piles.
        for _ in range(25):
            changed = False
            for i in range(rx_min, rx_max):
                for j in range(cx_min, cx_max):
                    if self.state[i, j] in (CellState.FILLED, CellState.BOUNDARY, CellState.OBSTACLE, CellState.PROTECTED):
                        continue

                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj

                        if self.state[ni, nj] in (CellState.BOUNDARY, CellState.OBSTACLE, CellState.PROTECTED):
                            continue

                        dz = self.z_height[i, j] - self.z_height[ni, nj]
                        if dz > max_dz:
                            transfer = (dz - max_dz) / 2.0
                            if transfer > 0.005:
                                self.z_height[i, j]   -= transfer
                                self.z_height[ni, nj] += transfer
                                changed = True
            if not changed:
                break

        for i in range(rx_min, rx_max):
            for j in range(cx_min, cx_max):
                self._update_state(i, j)
                if self.z_height[i, j] > 0:
                    self.pheromone[i, j] = 0.0
                

    def deposit_trail(self, r, c, radius_cells, strength):
        rad = int(math.ceil(radius_cells))
        r_min = max(0, r - rad);  r_max = min(self.rows, r + rad + 1)
        c_min = max(0, c - rad);  c_max = min(self.cols, c + rad + 1)
        rs = np.arange(r_min, r_max)
        cs = np.arange(c_min, c_max)
        rr, cc = np.meshgrid(rs, cs, indexing='ij')
        dist2  = (rr - r) ** 2 + (cc - c) ** 2
        sigma2 = max(1.0, radius_cells / 2.0) ** 2
        drop   = strength * np.exp(-dist2 / (2.0 * sigma2)).astype(np.float32)
        self.pheromone[r_min:r_max, c_min:c_max] = np.maximum(
            0.0, self.pheromone[r_min:r_max, c_min:c_max] - drop
        )

    def _update_state(self, r, c):
        if self.state[r, c] in (CellState.BOUNDARY, CellState.PROTECTED, CellState.OBSTACLE,
                                 CellState.RESERVED, CellState.PATH_RESERVED):
            return
            
        z = self.z_height[r, c]
        if z >= TARGET_PILE_HEIGHT:
            self.state[r, c] = CellState.FILLED
        elif z > 0:
            self.state[r, c] = CellState.PARTIAL
        else:
            self.state[r, c] = CellState.EMPTY

    def reserve(self, r, c):
        if self.state[r, c] in (CellState.EMPTY, CellState.PARTIAL):
            self.state[r, c] = CellState.RESERVED

    def unreserve(self, r, c):
        if self.state[r, c] == CellState.RESERVED:
            self._update_state(r, c)

    def is_dumpable(self, r, c):
        if self.state[r, c] in (CellState.BOUNDARY, CellState.PROTECTED,
                                 CellState.OBSTACLE, CellState.FILLED,
                                 CellState.RESERVED, CellState.PATH_RESERVED):
            return False
        return self.z_height[r, c] < TARGET_PILE_HEIGHT
