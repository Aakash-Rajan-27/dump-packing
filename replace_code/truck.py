# truck.py
# Defines the Truck class and its state machine.
# Each truck cycles through: IDLE -> NAVIGATING -> REVERSING -> DUMPING -> EXITING -> IDLE

import numpy as np
import math
from config import TRUCK_CLASSES, ENTRY_POINT, _TAN_REPOSE


class Truck:

    # The five phases of a truck's life cycle
    STATUS_IDLE       = 'IDLE'        # waiting for a dump assignment
    STATUS_NAVIGATING = 'NAVIGATING'  # driving forward along its planned path
    STATUS_REVERSING  = 'REVERSING'   # one-tick pause to face the dump cell before dumping
    STATUS_DUMPING    = 'DUMPING'     # actively depositing material over multiple ticks
    STATUS_EXITING    = 'EXITING'     # driving back to the entry gate after dumping

    def __init__(self, truck_id, truck_class, start_pos=None):
        self.id          = truck_id
        self.truck_class = truck_class

        # Pull specs (payload, size, turning radius, etc.) from config
        specs = TRUCK_CLASSES[truck_class]
        self.payload_t            = specs['payload_t']
        self.width                = specs['width_m']
        self.length               = specs['length_m']
        self.turn_radius          = specs['turn_radius_m']
        self.pile_height_per_dump = specs['pile_height_per_dump']  # used by MCTS rollouts
        self._dump_ticks_required = specs['dump_ticks']
        self.colour               = specs['colour']
        self.label                = specs['label']

        # Convert payload tonnage to m³ (bulk density ~1.8 t/m³), then spread across ticks
        self.payload_volume_m3 = self.payload_t / 1.8
        self.volume_per_tick   = self.payload_volume_m3 / self._dump_ticks_required

        # Start at the entry gate, facing up (π/2 = north)
        self.pos     = list(start_pos if start_pos else ENTRY_POINT)
        self.heading = np.pi / 2

        self.status      = self.STATUS_IDLE
        self.path        = []         # forward path cells (popped one per step)
        self.dump_target = None       # (row, col) of the cell this truck will dump into
        self.stop_target = None       # last cell on the forward path (where reversing begins)
        self._dump_ticks = 0          # counts how many ticks we've been dumping
        self._exit_path  = []         # path back to the entry gate (set after dumping)

    def set_path(self, path, dump_target, grid):
        """
        Assign a forward drive path and a dump target to this truck.
        Reserves the target cell on the grid so no other truck claims it.
        If the path is empty, immediately idles the truck and unreserves the target.
        """
        self.path        = list(path)
        self.dump_target = dump_target

        if not self.path:
            # Nothing to drive — release the reservation and go back to idle
            if self.dump_target:
                grid.unreserve(*self.dump_target)
                self.dump_target = None
            self.status = self.STATUS_IDLE
            return

        self.stop_target = self.path[-1]  # remember where we need to end up before reversing
        self.status = self.STATUS_NAVIGATING
        if dump_target:
            grid.reserve(*dump_target)    # lock the cell so others don't steal it

    def set_exit_path(self, exit_path):
        """
        Give the truck its route home after dumping.
        If the pathfinder couldn't find a route (empty list), do nothing and
        wait — main.py will retry on the next assignment cycle.
        """
        if exit_path:
            self._exit_path = list(exit_path)
            self.status     = self.STATUS_EXITING
        else:
            # Pathfinder failed (truck probably on top of a pile). Stay put and retry next tick.
            pass

    def step(self, grid):
        """
        Advance the truck by one simulation tick.
        Called once per tick from main.py for every truck.
        """
        if self.status != 'IDLE':
            print(f"DEBUG: Truck {self.id} Status: {self.status} Pos: {self.pos} Path Len: {len(self.path)}")

        if self.status == self.STATUS_IDLE:
            return
        if self.status == self.STATUS_IDLE:
            return

        elif self.status == self.STATUS_NAVIGATING:
            # Pop the next cell from the path and teleport the truck there (one cell per tick)
            if self.path:
                r, c = self.path.pop(0)
                tx, ty = grid.cell_to_world(r, c)
                dx, dy = tx - self.pos[0], ty - self.pos[1]
                # Update heading to face the direction of travel (skip if barely moving)
                if abs(dx) > 0.01 or abs(dy) > 0.01:
                    self.heading = np.arctan2(dy, dx)
                self.pos[0], self.pos[1] = tx, ty

            if not self.path:
                # Path exhausted — check if we actually landed on our stop target
                tr, tc = grid.world_to_cell(*self.pos)
                if self.stop_target and (tr, tc) == self.stop_target:
                    # We're in position. Turn to face the dump cell, then start reversing.
                    if self.dump_target:
                        dr, dc = self.dump_target
                        dtx, dty = grid.cell_to_world(dr, dc)
                        self.heading = np.arctan2(self.pos[1] - dty, self.pos[0] - dtx)
                    self.status = self.STATUS_REVERSING
                else:
                    # We ran out of path but didn't reach the stop target — abort and idle
                    if self.dump_target:
                        grid.unreserve(*self.dump_target)
                        self.dump_target = None
                    self.status = self.STATUS_IDLE

        elif self.status == self.STATUS_REVERSING:
            # Single-tick "reverse" phase — in future this can animate the truck backing up.
            # For now it just immediately transitions to dumping.
            self.status      = self.STATUS_DUMPING
            self._dump_ticks = 0

        elif self.status == self.STATUS_DUMPING:
            # Each tick deposits volume_per_tick m³ into the dump cell.
            # The grid's dump_at() handles pile physics (spreading, height clamping).
            self._dump_ticks += 1
            if self.dump_target:
                r, c = self.dump_target
                grid.dump_at(r, c, self.volume_per_tick)

            if self._dump_ticks >= self._dump_ticks_required:
                # Full payload delivered — release the cell and move to exit phase
                if self.dump_target:
                    grid.unreserve(*self.dump_target)
                    self.dump_target = None
                self.status = self.STATUS_EXITING

        elif self.status == self.STATUS_EXITING:
            # Follow the exit path home, one cell per tick
            if self._exit_path:
                r, c = self._exit_path.pop(0)
                tx, ty = grid.cell_to_world(r, c)
                dx, dy = tx - self.pos[0], ty - self.pos[1]
                if abs(dx) > 0.01 or abs(dy) > 0.01:
                    self.heading = np.arctan2(dy, dx)
                self.pos[0], self.pos[1] = tx, ty

            # Check if we're close enough to the entry gate to call it done.
            # angle_diff < 10.0 is effectively always true (10 radians ≈ 573°),
            # so arrival is purely distance-based.
            # The fallback `or not self._exit_path` catches trucks whose exit path
            # planning failed — they snap home rather than freezing forever.
            dist_to_entry = math.hypot(self.pos[0] - ENTRY_POINT[0],
                                       self.pos[1] - ENTRY_POINT[1])
            angle_diff = abs((self.heading - np.pi/2 + np.pi) % (2*np.pi) - np.pi)

            if dist_to_entry < 3.0 and (angle_diff < 10.0 or not self._exit_path):
                print(f"[DEBUG] Truck {self.id} reached entry radius. Snapping to IDLE.")
                # Snap cleanly to the gate and reset all state for the next assignment
                self.pos         = list(ENTRY_POINT)
                self.heading     = np.pi / 2
                self.status      = self.STATUS_IDLE
                self.path        = []
                self.dump_target = None
                self.stop_target = None

    def needs_exit_path(self):
        """
        Returns True if the truck just finished dumping and is waiting for an exit route.
        main.py checks this each tick to know when to call plan_paths for the exit.
        """
        return (self.status == self.STATUS_EXITING
                and not self._exit_path
                and self.dump_target is None)

    def is_idle(self):
        return self.status == self.STATUS_IDLE

    def __repr__(self):
        return (f"Truck({self.id}, class={self.truck_class}, "
                f"pos={self.pos}, status={self.status})")
