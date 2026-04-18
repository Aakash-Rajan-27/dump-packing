# core/truck.py
# ─────────────────────────────────────────────────────────────
# Represents a single truck in the simulation.
# Stores its physical state (position, heading, payload)
# and handles moving along its planned path step by step.
# ─────────────────────────────────────────────────────────────

import numpy as np  # for angle math


class Truck:
    """
    One truck. Has a position, a heading, a payload size,
    and a status telling us what it's currently doing.
    """

    # All possible statuses a truck can be in at any moment
    STATUS_IDLE       = 'IDLE'        # waiting for a dump point assignment
    STATUS_NAVIGATING = 'NAVIGATING'  # following its planned path to a dump point
    STATUS_REVERSING  = 'REVERSING'   # backing up to the dump point
    STATUS_DUMPING    = 'DUMPING'     # actively dumping material (takes a few ticks)
    STATUS_EXITING    = 'EXITING'     # heading back to entry corridor

    def __init__(self, truck_id, start_pos_metres, payload=100.0):
        # Unique ID for this truck (0, 1, 2, 3...)
        self.id = truck_id

        # Real-world position in metres (x, y)
        self.pos = list(start_pos_metres)

        # Heading in radians. 0 = facing right (+x), pi/2 = facing up (+y).
        # Trucks start facing into the polygon (upward if entry is at bottom)
        self.heading = np.pi / 2

        # How much material this truck carries per trip (tonnes).
        # Mixed fleet: different trucks have different payloads.
        self.payload = payload

        # Physical constraints — from config, but stored per-truck
        # so mixed fleet works (different trucks can have different radii)
        from config import TRUCK_WIDTH, TRUCK_LENGTH, MIN_TURN_RADIUS
        self.width        = TRUCK_WIDTH
        self.length       = TRUCK_LENGTH
        self.turn_radius  = MIN_TURN_RADIUS

        # Current status
        self.status = self.STATUS_IDLE

        # The planned path — a list of (row, col) cell indices to follow.
        # CBS fills this in. We pop from the front each tick.
        self.path = []

        # The final destination cell (row, col) where we dump
        self.dump_target = None

        # How many ticks the truck has been in DUMPING state.
        # Dumping takes 3 ticks to simulate the physical dump action.
        self._dump_ticks = 0
        self._dump_ticks_required = 3

    def set_path(self, path, dump_target, grid):
        """
        Called by the path planner to give the truck its route.
        path: list of (row,col) — the cells to traverse in order
        dump_target: (row,col) — the cell to dump at (end of path)
        grid: the GridMap — so we can mark cells reserved
        """
        self.path = list(path)          # copy the path
        self.dump_target = dump_target  # remember where we're dumping
        self.status = self.STATUS_NAVIGATING

        # Mark the dump target as RESERVED so no other truck gets assigned it
        if dump_target:
            grid.reserve(*dump_target)

    def step(self, grid):
        """
        Advance the truck by one simulation tick.
        Called every tick from main.py's loop.
        Depending on status, the truck moves, reverses, dumps, or waits.
        """

        if self.status == self.STATUS_IDLE:
            # Nothing to do — waiting for assignment from the planner
            return

        elif self.status == self.STATUS_NAVIGATING:
            # Move one step along the planned path
            if self.path:
                # Take the next cell from the front of the path
                next_cell = self.path.pop(0)

                # Convert that cell to real-world coords and move there
                r, c = next_cell
                target_x, target_y = grid.cell_to_world(r, c)

                # Update heading to face the direction of movement
                dx = target_x - self.pos[0]
                dy = target_y - self.pos[1]
                if abs(dx) > 0.01 or abs(dy) > 0.01:  # avoid atan2(0,0)
                    self.heading = np.arctan2(dy, dx)

                # Move to the new position
                self.pos[0] = target_x
                self.pos[1] = target_y

                # If path is now empty, we've arrived — start reversing to dump
                if not self.path:
                    self.status = self.STATUS_REVERSING

            else:
                # Path was empty already (shouldn't happen, but be safe)
                self.status = self.STATUS_REVERSING

        elif self.status == self.STATUS_REVERSING:
            # Truck backs up to the exact dump point.
            # In a real system this is more complex — simplified here.
            # After 1 tick of reversing, we start dumping.
            self.status = self.STATUS_DUMPING
            self._dump_ticks = 0

        elif self.status == self.STATUS_DUMPING:
            # Dumping takes _dump_ticks_required ticks to complete
            self._dump_ticks += 1

            if self._dump_ticks >= self._dump_ticks_required:
                # Dump is complete — update the grid
                if self.dump_target:
                    r, c = self.dump_target
                    grid.dump_at(r, c, self.payload)  # marks FILLED, updates height
                    self.dump_target = None
                # Now head back to the entry corridor
                self.status = self.STATUS_EXITING
                # In a full system we'd plan a path back.
                # For now: teleport back to entry (simplification for week 1)
                #FIX -> <plan path back to entry point>
                from config import ENTRY_POINT
                self.pos = list(ENTRY_POINT)
                self.status = self.STATUS_IDLE  # ready for next assignment

    def is_idle(self):
        """Convenience check — is this truck available for a new assignment?"""
        return self.status == self.STATUS_IDLE

    def __repr__(self):
        return f"Truck({self.id}, pos={self.pos}, status={self.status})"
