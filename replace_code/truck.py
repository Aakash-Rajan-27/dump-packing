# truck.py
# ─────────────────────────────────────────────────────────────
# Represents a single truck in the simulation.
# Now supports mixed fleet: small / medium / large classes,
# each with different physical dimensions, payload, turning
# radius, and dump behaviour.
# ─────────────────────────────────────────────────────────────

import numpy as np
from config import TRUCK_CLASSES, ENTRY_POINT


class Truck:
    """
    One truck. Physical specs come from TRUCK_CLASSES[truck_class].
    Status machine: IDLE → NAVIGATING → REVERSING → DUMPING → IDLE
    """

    STATUS_IDLE       = 'IDLE'
    STATUS_NAVIGATING = 'NAVIGATING'
    STATUS_REVERSING  = 'REVERSING'
    STATUS_DUMPING    = 'DUMPING'
    STATUS_EXITING    = 'EXITING'

    def __init__(self, truck_id, truck_class, start_pos=None):
        """
        truck_id    : unique integer (0 … N-1)
        truck_class : 'small', 'medium', or 'large'
        start_pos   : (x, y) metres — defaults to ENTRY_POINT
        """
        self.id          = truck_id
        self.truck_class = truck_class

        # Load specs from config dict
        specs = TRUCK_CLASSES[truck_class]
        self.payload_t           = specs['payload_t']
        self.width               = specs['width_m']
        self.length              = specs['length_m']
        self.turn_radius         = specs['turn_radius_m']
        self.dump_radius         = specs['dump_radius_m']
        self.pile_height_per_dump = specs['pile_height_per_dump']
        self._dump_ticks_required = specs['dump_ticks']
        self.colour              = specs['colour']
        self.label               = specs['label']   # 'S', 'M', or 'L'

        # Position and heading
        self.pos     = list(start_pos if start_pos else ENTRY_POINT)
        self.heading = np.pi / 2   # initially facing into polygon (upward)

        # State machine
        self.status      = self.STATUS_IDLE
        self.path        = []
        self.dump_target = None
        self._dump_ticks = 0

    # ── Path / Assignment ──────────────────────────────────

    def set_path(self, path, dump_target, grid):
        """Give this truck its route and dump target."""
        self.path        = list(path)
        self.dump_target = dump_target
        self.status      = self.STATUS_NAVIGATING
        if dump_target:
            grid.reserve(*dump_target)

    # ── Tick ───────────────────────────────────────────────

    def step(self, grid):
        """Advance truck by one simulation tick."""

        if self.status == self.STATUS_IDLE:
            return

        elif self.status == self.STATUS_NAVIGATING:
            if self.path:
                r, c = self.path.pop(0)
                tx, ty = grid.cell_to_world(r, c)
                dx, dy = tx - self.pos[0], ty - self.pos[1]
                if abs(dx) > 0.01 or abs(dy) > 0.01:
                    self.heading = np.arctan2(dy, dx)
                self.pos[0], self.pos[1] = tx, ty
                if not self.path:
                    self.status = self.STATUS_REVERSING
            else:
                self.status = self.STATUS_REVERSING

        elif self.status == self.STATUS_REVERSING:
            # One tick to reverse into dump position
            self.status      = self.STATUS_DUMPING
            self._dump_ticks = 0

        elif self.status == self.STATUS_DUMPING:
            self._dump_ticks += 1
            if self._dump_ticks >= self._dump_ticks_required:
                if self.dump_target:
                    r, c = self.dump_target
                    grid.dump_at(r, c, self.pile_height_per_dump)
                    self.dump_target = None
                # Return to entry and go idle
                self.pos    = list(ENTRY_POINT)
                self.status = self.STATUS_IDLE

    # ── Queries ────────────────────────────────────────────

    def is_idle(self):
        return self.status == self.STATUS_IDLE

    def __repr__(self):
        return (f"Truck({self.id}, class={self.truck_class}, "
                f"pos={self.pos}, status={self.status})")
