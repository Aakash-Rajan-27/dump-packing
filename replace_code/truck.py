# truck.py
# ─────────────────────────────────────────────────────────────
# FIX: STATUS_EXITING now actually navigates back to entry.
#
# Old: after dumping, truck teleported instantly to ENTRY_POINT
#      → visually disappeared and reappeared at entry
#      → never showed the exit journey
#
# New: after dumping, truck sets an exit path back to entry_rc
#      using its existing path mechanism, then goes IDLE on arrival.
#      exit_path is set by set_exit_path() called from main.py
#      after dump completes.
#
# Also: heading is updated during exit navigation same as forward nav.
# ─────────────────────────────────────────────────────────────

import numpy as np
from config import TRUCK_CLASSES, ENTRY_POINT


class Truck:

    STATUS_IDLE       = 'IDLE'
    STATUS_NAVIGATING = 'NAVIGATING'
    STATUS_REVERSING  = 'REVERSING'
    STATUS_DUMPING    = 'DUMPING'
    STATUS_EXITING    = 'EXITING'

    def __init__(self, truck_id, truck_class, start_pos=None):
        self.id          = truck_id
        self.truck_class = truck_class

        specs = TRUCK_CLASSES[truck_class]
        self.payload_t            = specs['payload_t']
        self.width                = specs['width_m']
        self.length               = specs['length_m']
        self.turn_radius          = specs['turn_radius_m']
        self.pile_height_per_dump = specs['pile_height_per_dump']
        self._dump_ticks_required = specs['dump_ticks']
        self.colour               = specs['colour']
        self.label                = specs['label']

        self.pos     = list(start_pos if start_pos else ENTRY_POINT)
        self.heading = np.pi / 2

        self.status      = self.STATUS_IDLE
        self.path        = []
        self.dump_target = None
        self._dump_ticks = 0
        self._exit_path  = []   # NEW: path back to entry after dumping

    def set_path(self, path, dump_target, grid):
        self.path        = list(path)
        self.dump_target = dump_target
        self.status      = self.STATUS_NAVIGATING
        if dump_target:
            grid.reserve(*dump_target)

    def set_exit_path(self, exit_path):
        """
        NEW: Called from main.py after dump completes.
        Gives the truck a path back to the entry point.
        If path is empty (already at entry), go idle immediately.
        """
        if exit_path:
            self._exit_path = list(exit_path)
            self.status     = self.STATUS_EXITING
        else:
            self.status = self.STATUS_IDLE

    def step(self, grid):
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
            self.status      = self.STATUS_DUMPING
            self._dump_ticks = 0

        elif self.status == self.STATUS_DUMPING:
            self._dump_ticks += 1
            if self._dump_ticks >= self._dump_ticks_required:
                if self.dump_target:
                    r, c = self.dump_target
                    grid.dump_at(r, c, self.pile_height_per_dump)
                    self.dump_target = None
                # FIX: don't teleport — go to EXITING and wait for exit path
                # main.py will call set_exit_path() on this truck
                self.status = self.STATUS_EXITING

        elif self.status == self.STATUS_EXITING:
            # FIX: navigate back to entry along exit path
            if self._exit_path:
                r, c = self._exit_path.pop(0)
                tx, ty = grid.cell_to_world(r, c)
                dx, dy = tx - self.pos[0], ty - self.pos[1]
                if abs(dx) > 0.01 or abs(dy) > 0.01:
                    self.heading = np.arctan2(dy, dx)
                self.pos[0], self.pos[1] = tx, ty
                if not self._exit_path:
                    self.status = self.STATUS_IDLE
            else:
                # No exit path set yet (waiting for main.py to set it)
                # or path was empty — go idle at current position
                self.pos    = list(ENTRY_POINT)
                self.status = self.STATUS_IDLE

    def needs_exit_path(self):
        """True when truck has just finished dumping and needs an exit path."""
        return (self.status == self.STATUS_EXITING
                and not self._exit_path
                and self.dump_target is None)

    def is_idle(self):
        return self.status == self.STATUS_IDLE

    def __repr__(self):
        return (f"Truck({self.id}, class={self.truck_class}, "
                f"pos={self.pos}, status={self.status})")