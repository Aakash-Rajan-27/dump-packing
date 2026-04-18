# core/grid_map.py
# ─────────────────────────────────────────────────────────────
# THE DATA MODEL. Every other module reads from and writes to
# this. It represents the dump polygon as a 2D grid of cells,
# where each cell knows its state (empty/filled/reserved etc.),
# its height, and its pheromone level.
# ─────────────────────────────────────────────────────────────

import numpy as np                          # numpy: fast array math
import shapely.geometry # shapely: geometry/polygon operations
from enum import IntEnum                    # IntEnum: named integers (cleaner than raw numbers)


# ── Cell states ────────────────────────────────────────────
# Each cell in the grid is always in exactly one of these states.
# IntEnum means BOUNDARY==0, EMPTY==1, etc. — lets us use them
# as array values and compare with == easily.
class CellState(IntEnum):
    BOUNDARY  = 0   # outside the polygon — trucks never go here
    EMPTY     = 1   # inside polygon, no material dumped yet — valid dump spot
    FILLED    = 2   # material already dumped here — pile exists
    RESERVED  = 3   # a truck is on its way to dump here — don't assign to another truck
    PROTECTED = 4   # entry/exit corridor — never dump here, always keep driveable
    OBSTACLE  = 5   # something blocking this cell (e.g. a stuck truck)


# ── The main class ─────────────────────────────────────────
class GridMap:
    """
    Wraps the dump polygon in a rectangular grid.
    Each cell is CELL_SIZE × CELL_SIZE metres in real life.
    Stores: cell state, pile height (z), pheromone level.
    """

    def __init__(self, polygon_coords, cell_size):
        # polygon_coords: list of (x,y) tuples defining the boundary
        # cell_size: real-world size of one cell in metres

        self.cell_size = cell_size

        # Build a Shapely Polygon object from the coordinate list.
        # Shapely lets us do .contains(point) checks easily later.
        self.polygon = shapely.geometry.Polygon(polygon_coords)

        # Get the axis-aligned bounding box of the polygon.
        # minx,miny = bottom-left corner, maxx,maxy = top-right corner.
        minx, miny, maxx, maxy = self.polygon.bounds

        # Calculate how many columns and rows the grid needs.
        # +1 ensures we don't lose a partial cell at the edge.
        self.cols = int((maxx - minx) / cell_size) + 1
        self.rows = int((maxy - miny) / cell_size) + 1

        # Store the real-world origin (bottom-left of bounding box).
        # We need this to convert between cell indices and metres.
        self.origin = (minx, miny)

        # ── Core data arrays ───────────────────────────────
        # All arrays are (rows × cols) — one value per cell.

        # State array: starts as all BOUNDARY (0), then we classify cells below.
        self.state = np.full((self.rows, self.cols), CellState.BOUNDARY, dtype=np.int8)

        # Height array: 0.0 everywhere initially. Increases when material is dumped.
        self.z_height = np.zeros((self.rows, self.cols), dtype=np.float32)

        # Pheromone array: 1.0 everywhere initially (high = not recently used).
        # Decays over time to discourage trucks from clustering in one area.
        self.pheromone = np.ones((self.rows, self.cols), dtype=np.float32)

        # Now actually classify each cell as EMPTY or BOUNDARY
        self._classify_cells()

        # Mark the entry corridor as PROTECTED so it never gets dumped on
        self._mark_entry_corridor()

    def _classify_cells(self):#FIX THIS - CONSIDERS ONLY XY POSITION OF CELL CENTER AND CHECKS IF INSIDE BOUNDARY OR NO  
                                #IT SHOULD CHECK ALL 4 EXTREMES OF THE SQUARE CELL IF INSIDE BOUNDARY OR NO
        """
        Loop over every cell, find its real-world centre coordinate,
        and test whether it's inside the polygon.
        If yes → EMPTY. If no → stays BOUNDARY.
        """
        for r in range(self.rows):
            for c in range(self.cols):
                # Convert cell (r,c) to real-world (x,y) metres
                cx, cy = self.cell_to_world(r, c)

                # Shapely point-in-polygon test.
                # .contains() returns True if the point is strictly inside.
                if self.polygon.contains(shapely.geometry.Point(cx, cy)):
                    self.state[r, c] = CellState.EMPTY

    def _mark_entry_corridor(self):
        """
        The entry point and the 2 cells around it are PROTECTED.
        Trucks need a clear path to enter and exit — we never dump here.
        """
        from config import ENTRY_POINT
        # Find which cell the entry point falls in
        er, ec = self.world_to_cell(*ENTRY_POINT)

        # Mark a 3-cell wide corridor (the entry cell and immediate neighbours)
        for dr in range(-1, 2):         # rows: -1, 0, +1
            for dc in range(-1, 2):     # cols: -1, 0, +1
                nr, nc = er + dr, ec + dc
                # Make sure we don't go out of array bounds
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    # Only protect cells that are inside the polygon
                    if self.state[nr, nc] == CellState.EMPTY:
                        self.state[nr, nc] = CellState.PROTECTED

    # ── Coordinate conversion helpers ──────────────────────

    def cell_to_world(self, r, c):
        """
        Convert grid cell (row, col) → real-world (x, y) in metres.
        Returns the centre of the cell.
        """
        x = self.origin[0] + c * self.cell_size + self.cell_size / 2
        y = self.origin[1] + r * self.cell_size + self.cell_size / 2
        return (x, y)

    def world_to_cell(self, x, y):
        """
        Convert real-world (x, y) metres → grid cell (row, col).
        Clamps to valid range so we never get an out-of-bounds index.
        """
        c = int((x - self.origin[0]) / self.cell_size)
        r = int((y - self.origin[1]) / self.cell_size)
        # Clamp to grid bounds just in case of floating point edge cases
        c = max(0, min(c, self.cols - 1))
        r = max(0, min(r, self.rows - 1))
        return (r, c)

    # ── Metrics ────────────────────────────────────────────

    def fill_pct(self):
        """
        Returns the fraction of valid (non-boundary, non-protected) cells
        that are filled. 0.0 = empty, 1.0 = completely full.
        """
        # Count cells that are either EMPTY or FILLED (valid polygon cells)
        valid = np.sum((self.state == CellState.EMPTY) |
                       (self.state == CellState.FILLED))
        # Count just the filled ones
        filled = np.sum(self.state == CellState.FILLED)

        # Avoid division by zero if no valid cells
        return filled / max(1, valid)

    def dump_at(self, r, c, truck_payload):
        """
        Mark cell (r,c) as FILLED and update its height.
        Also deposit pheromone to discourage other trucks from coming here soon.
        truck_payload: weight of material dumped (tonnes) — affects pile height.
        """
        # Mark the cell as filled
        self.state[r, c] = CellState.FILLED

        # Increase height based on payload (simplified — real calc uses volume)
        from config import CELL_SIZE, TARGET_PILE_HEIGHT
        self.z_height[r, c] += truck_payload * 0.1  # scale factor#FIX - HOW WE KNOW EXACT SCALING FACTOR BASED ON PAYLOAD

        # Clamp height so it doesn't exceed the physical target
        self.z_height[r, c] = min(self.z_height[r, c], TARGET_PILE_HEIGHT)

        # Deposit fresh pheromone: set to 0 (recently used) at dump point.
        # It will decay back toward 1 over subsequent ticks.
        self.pheromone[r, c] = 0.0

    def reserve(self, r, c):
        """Mark cell as RESERVED — a truck is heading here."""
        if self.state[r, c] == CellState.EMPTY:
            self.state[r, c] = CellState.RESERVED

    def unreserve(self, r, c):
        """Release a reservation (e.g. if a truck path changes)."""
        if self.state[r, c] == CellState.RESERVED:
            self.state[r, c] = CellState.EMPTY
