# grid_map.py
# ─────────────────────────────────────────────────────────────
# THE DATA MODEL. Every other module reads from and writes to
# this. Updated for mixed fleet + partial pile dynamics:
#
#   PARTIAL cells have been dumped on but haven't reached
#   TARGET_PILE_HEIGHT yet. Any truck can dump there again.
#   FILLED cells are at or above TARGET_PILE_HEIGHT — done.
# ─────────────────────────────────────────────────────────────

import numpy as np
import shapely.geometry
from enum import IntEnum


class CellState(IntEnum):
    BOUNDARY  = 0   # outside polygon
    EMPTY     = 1   # inside polygon, nothing dumped yet
    PARTIAL   = 2   # dumped on, but below TARGET_PILE_HEIGHT — can dump again
    FILLED    = 3   # at or above TARGET_PILE_HEIGHT — fully packed
    RESERVED  = 4   # a truck is heading here — don't re-assign
    PROTECTED = 5   # entry corridor — never dump here
    OBSTACLE  = 6   # blocked cell


class GridMap:
    def __init__(self, polygon_coords, cell_size):
        self.cell_size = cell_size
        self.polygon   = shapely.geometry.Polygon(polygon_coords)

        minx, miny, maxx, maxy = self.polygon.bounds
        self.cols   = int((maxx - minx) / cell_size) + 1
        self.rows   = int((maxy - miny) / cell_size) + 1
        self.origin = (minx, miny)

        # Core arrays — all (rows × cols)
        self.state     = np.full((self.rows, self.cols),
                                 CellState.BOUNDARY, dtype=np.int8)
        self.z_height  = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.pheromone = np.ones ((self.rows, self.cols), dtype=np.float32)

        self._classify_cells()
        self._mark_entry_corridor()

    # ── Setup ──────────────────────────────────────────────

    def _classify_cells(self):
        for r in range(self.rows):
            for c in range(self.cols):
                cx, cy = self.cell_to_world(r, c)
                if self.polygon.contains(shapely.geometry.Point(cx, cy)):
                    self.state[r, c] = CellState.EMPTY

    def _mark_entry_corridor(self):
        from config import ENTRY_POINT
        er, ec = self.world_to_cell(*ENTRY_POINT)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = er + dr, ec + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if self.state[nr, nc] == CellState.EMPTY:
                        self.state[nr, nc] = CellState.PROTECTED

    # ── Coordinate Conversion ──────────────────────────────

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

    # ── Metrics ────────────────────────────────────────────

    def fill_pct(self):
        """Fraction of valid cells that are FULLY filled."""
        valid  = np.sum((self.state == CellState.EMPTY)   |
                        (self.state == CellState.PARTIAL)  |
                        (self.state == CellState.FILLED)   |
                        (self.state == CellState.RESERVED))
        filled = np.sum(self.state == CellState.FILLED)
        return filled / max(1, valid)

    def pack_pct(self):
        """Overall packing: weighted by fill height / target height."""
        valid_mask = ((self.state == CellState.EMPTY)   |
                      (self.state == CellState.PARTIAL)  |
                      (self.state == CellState.FILLED)   |
                      (self.state == CellState.RESERVED))
        total_valid = np.sum(valid_mask)
        if total_valid == 0:
            return 0.0
        from config import TARGET_PILE_HEIGHT
        packing = np.sum(
            np.clip(self.z_height, 0, TARGET_PILE_HEIGHT) * valid_mask
        ) / (total_valid * TARGET_PILE_HEIGHT)
        return float(packing)

    # ── State Mutations ────────────────────────────────────

    def dump_at(self, r, c, pile_height_per_dump):
        """
        Add material to cell (r,c). Called when a truck finishes dumping.

        pile_height_per_dump: metres of height this truck's dump adds.
        The cell transitions:
          EMPTY / RESERVED → PARTIAL  (if still below TARGET_PILE_HEIGHT)
          PARTIAL           → FILLED   (if it hits TARGET_PILE_HEIGHT)

        Also spreads a small height bonus to neighbouring cells to simulate
        the angle-of-repose spreading of material.
        """
        from config import TARGET_PILE_HEIGHT, ANGLE_OF_REPOSE
        import math

        # Add height at the dump cell
        self.z_height[r, c] += pile_height_per_dump

        # Spread material to neighbours proportional to angle of repose.
        # tan(repose_angle) = height / distance → at 1 cell away, spread fraction:
        spread_fraction = math.tan(math.radians(ANGLE_OF_REPOSE)) * self.cell_size
        neighbour_add   = pile_height_per_dump * 0.15   # 15% bleeds to each neighbour

        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.state[nr, nc] in (CellState.EMPTY,
                                          CellState.PARTIAL,
                                          CellState.RESERVED):
                    self.z_height[nr, nc] = min(
                        self.z_height[nr, nc] + neighbour_add,
                        TARGET_PILE_HEIGHT
                    )
                    # Promote partial neighbour if it's now full
                    if (self.z_height[nr, nc] >= TARGET_PILE_HEIGHT and
                            self.state[nr, nc] != CellState.RESERVED):
                        self.state[nr, nc] = CellState.FILLED

        # Update state of the dumped cell based on new height
        if self.z_height[r, c] >= TARGET_PILE_HEIGHT:
            self.state[r, c] = CellState.FILLED
        else:
            # Still below target — mark as PARTIAL so another truck can top it up
            if self.state[r, c] in (CellState.EMPTY, CellState.RESERVED):
                self.state[r, c] = CellState.PARTIAL

        # Deposit fresh pheromone — set to 0 (recently used), decays back to 1
        self.pheromone[r, c] = 0.0

    def reserve(self, r, c):
        """Mark cell as RESERVED — a truck is heading here."""
        if self.state[r, c] in (CellState.EMPTY, CellState.PARTIAL):
            self.state[r, c] = CellState.RESERVED

    def unreserve(self, r, c):
        """Release reservation. Revert to PARTIAL if has height, else EMPTY."""
        if self.state[r, c] == CellState.RESERVED:
            from config import TARGET_PILE_HEIGHT
            if self.z_height[r, c] > 0:
                self.state[r, c] = CellState.PARTIAL
            else:
                self.state[r, c] = CellState.EMPTY

    def is_dumpable(self, r, c):
        """
        Can a truck dump at this cell?
        Yes if: EMPTY, PARTIAL (not yet full), or RESERVED (reserved but not full).
        """
        from config import TARGET_PILE_HEIGHT
        s = self.state[r, c]
        if s == CellState.FILLED:
            return False
        if s in (CellState.BOUNDARY, CellState.PROTECTED, CellState.OBSTACLE):
            return False
        # EMPTY, PARTIAL, RESERVED are all dumpable as long as not at max height
        return self.z_height[r, c] < TARGET_PILE_HEIGHT
