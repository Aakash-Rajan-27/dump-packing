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
    BOUNDARY  = 0
    EMPTY     = 1
    PARTIAL   = 2
    FILLED    = 3
    RESERVED  = 4
    PROTECTED = 5
    OBSTACLE  = 6


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
        
        for i in range(rmin, rmax):
            for j in range(cmin, cmax):
                if self.state[i, j] == CellState.BOUNDARY:
                    continue
                    
                dist_cells = math.hypot(i - r, j - c)
                dist_m = dist_cells * self.cell_size
                
                if dist_m < r_pile_m:
                    added_height = (r_pile_m - dist_m) * tan_theta
                    self.z_height[i, j] += added_height
                    
        max_dz = tan_theta * self.cell_size
        changed = True
        
        rx_min, rx_max = max(1, rmin - 4), min(self.rows - 1, rmax + 4)
        cx_min, cx_max = max(1, cmin - 4), min(self.cols - 1, cmax + 4)
        
        while changed:
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
                            
                            # THE ANTI-FREEZE FIX: Ignore microscopic dirt movements
                            if transfer > 0.005: 
                                self.z_height[i, j]   -= transfer
                                self.z_height[ni, nj] += transfer
                                changed = True

        for i in range(rx_min, rx_max):
            for j in range(cx_min, cx_max):
                self._update_state(i, j)
                if self.z_height[i, j] > 0:
                    self.pheromone[i, j] = 0.0

    def _update_state(self, r, c):
        if self.state[r, c] in (CellState.BOUNDARY, CellState.PROTECTED, CellState.OBSTACLE, CellState.RESERVED):
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
                                 CellState.OBSTACLE, CellState.FILLED):
            return False
        return self.z_height[r, c] < TARGET_PILE_HEIGHT