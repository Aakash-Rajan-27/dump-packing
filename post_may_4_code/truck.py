# truck.py
# ─────────────────────────────────────────────────────────────
# FIX: Added `grid.unreserve()` calls.
#      Previously, trucks permanently locked cells in the RESERVED state.
#      Because grid_map ignores state updates for RESERVED cells, the
#      renderer couldn't "see" the dirt that was being dumped.
# ─────────────────────────────────────────────────────────────

# truck.py
# ─────────────────────────────────────────────────────────────
# THE "STOP SHORT" NAVIGATION UPDATE & VOLUME CONVERSION
# Trucks now stop with their rear aligned to the target cell,
# dump the correct payload volume behind them, and exit cleanly.
# ─────────────────────────────────────────────────────────────

# truck.py
# ─────────────────────────────────────────────────────────────
# THE "STOP SHORT" NAVIGATION UPDATE & VOLUME CONVERSION
# Fixed: Re-added pile_height_per_dump for the MCTS fast-rollouts
# ─────────────────────────────────────────────────────────────

# truck.py
# ─────────────────────────────────────────────────────────────
# EXACT CLEARANCE MATH & "INSIDE OBSTACLE" BUG FIX
#
# THE BUG: When a truck dumped dirt, the cellular sandpile physics
# caused the dirt to spread under the truck's rear wheels. If the
# dirt under the truck exceeded 0.4m, the A* pathfinder saw the truck
# as "starting inside a hard obstacle," panicked, and returned no path.
# This caused the truck to trigger its teleport failsafe or freeze.
#
# THE FIX: We calculate the exact radius of the sandpile where the
# height drops to exactly 0.4m (DRIVE_CLEARANCE_M). The truck slices
# its path to park its rear wheels safely OUTSIDE that radius, plus
# a 0.5m safety buffer. When the dirt settles, the truck is always
# sitting on driveable terrain (<0.4m), so A* can safely route it out.
# ─────────────────────────────────────────────────────────────

# truck.py
# ─────────────────────────────────────────────────────────────
# EXACT CLEARANCE MATH & "SNAP TO START" FIX
# ─────────────────────────────────────────────────────────────

# truck.py
# ─────────────────────────────────────────────────────────────
# SIMPLIFIED NAVIGATION: Pathfinder handles the Radius Goal.
# ENTRY RADIUS & ANGLE CHECK: Validates arrival at loading zone.
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# AGENT OBSTACLE RESOLUTION: From Teleport to Physical Navigation
#
# DOCUMENTATION:
# Previously, the simulation relied on immediate coordinate resets (teleportation).
# Switching to physical navigation introduced "Inside Obstacle" errors because
# A* would fail if the destination or start was inside a dirt pile (>0.4m).
#
# RESOLUTION:
# 1. RADIUS GOAL: The Pathfinder now stops searching once the truck is within
#    a safe radius of the dump or entry point, solving the "Inside Obstacle" bug.
# 2. ENTRY SNAP: Because trucks now drive to the gate, we use a wide Radius Goal
#    and high Angle Tolerance to ensure they successfully transition to IDLE.
# ─────────────────────────────────────────────────────────────

import numpy as np  # numerical operations (arctan2, pi)
import math  # math.hypot for Euclidean distance
from config import TRUCK_CLASSES, ENTRY_POINT, _TAN_REPOSE, TRAIL_STRENGTH, TRAIL_RADIUS_M


class Truck:

    # String constants for the truck's state machine — avoids typos and makes comparisons readable
    STATUS_IDLE       = 'IDLE'        # truck is at the entry point, waiting for an assignment
    STATUS_NAVIGATING = 'NAVIGATING'  # truck is driving forward toward the dump stop position
    STATUS_REVERSING  = 'REVERSING'   # truck has arrived; one-tick state that kicks off dumping
    STATUS_DUMPING    = 'DUMPING'     # truck is actively depositing dirt over multiple ticks
    STATUS_EXITING    = 'EXITING'     # truck is driving back to the entry point after dumping

    def __init__(self, truck_id, truck_class, start_pos=None):
        self.id          = truck_id     # unique identifier for this truck (used in logs and path dicts)
        self.truck_class = truck_class  # string key into TRUCK_CLASSES config dict (e.g. 'large', 'small')

        specs = TRUCK_CLASSES[truck_class]                     # look up the spec dictionary for this truck type
        self.payload_t            = specs['payload_t']         # payload capacity in metric tonnes
        self.width                = specs['width_m']           # truck width in metres (used for collision checks)
        self.length               = specs['length_m']          # truck length in metres (used for stop-short offset)
        self.turn_radius          = specs['turn_radius_m']     # minimum turning radius; feeds A* turn cost
        self.pile_height_per_dump = specs['pile_height_per_dump']  # metres of height added per full dump (used in MCTS fast-rollouts)
        self._dump_ticks_required = specs['dump_ticks']        # how many simulation ticks a dump takes
        self.colour               = specs['colour']            # RGB colour for renderer
        self.label                = specs['label']             # display string shown in the visualiser

        self.payload_volume_m3 = self.payload_t / 1.8          # convert tonnes to m³ (bulk density ~1.8 t/m³)
        self.volume_per_tick   = self.payload_volume_m3 / self._dump_ticks_required  # volume deposited each tick so total sums to payload

        self.pos     = list(start_pos if start_pos else ENTRY_POINT)  #CHANGE THIS FOR MULTI-DUMPS IN ONE TRIP [x, y] world position; default to entry gate
        self.heading = np.pi / 2  # initial heading: pointing "up" (north) in world coords

        self.status      = self.STATUS_IDLE  # start idle, waiting for an assignment
        self.path        = []                # ordered list of (row, col) waypoints to follow
        self.dump_target = None              # (row, col) of the grid cell being filled this run
        self.stop_target = None              # (row, col) where the truck stops before reversing
        self._dump_ticks = 0                 # tick counter for the current dump operation
        self._exit_path  = []                # waypoints for the return trip to the entry point

    def _waypoint_to_pose(self, grid, waypoint):
        if len(waypoint) == 3:
            return waypoint

        r, c = waypoint
        tx, ty = grid.cell_to_world(r, c)
        dx, dy = tx - self.pos[0], ty - self.pos[1]
        heading = self.heading
        if abs(dx) > 0.01 or abs(dy) > 0.01:
            heading = np.arctan2(dy, dx)
        return tx, ty, heading

    def _waypoint_to_cell(self, grid, waypoint):
        if len(waypoint) == 3:
            return grid.world_to_cell(waypoint[0], waypoint[1])
        return waypoint

    def set_path(self, path, dump_target, grid):
        self.path        = list(path)   # copy the path so mutations outside don't affect us
        self.dump_target = dump_target  # remember which cell we're filling this trip

        if not self.path:  # if A* returned an empty path (truck is already there or fully blocked)
            if self.dump_target:
                grid.unreserve(*self.dump_target)  # release the cell reservation so another truck can use it
                self.dump_target = None             # clear the target reference
            self.status = self.STATUS_IDLE          # stay idle — nothing to do
            return

        self.stop_target = self.path[-1]    # the last waypoint is where the truck parks before reversing
        self.status = self.STATUS_NAVIGATING  # begin driving toward the dump position
        if dump_target:
            grid.reserve(*dump_target)  # mark the target cell as reserved so other trucks don't also plan to dump there

    def set_exit_path(self, exit_path):
        if exit_path:  # only switch state if a valid exit path was found
            self._exit_path = list(exit_path)      # store a copy of the exit waypoints
            self.status     = self.STATUS_EXITING  # truck will start following the exit path next tick
        else:
            # CBS returned an empty path — leave status as EXITING so the main loop retries next tick.
            # The pathfinder's extended bulldozer (which forces a multi-cell radius around the truck
            # to be driveable) should prevent this from happening in practice.
            pass

    def step(self, grid):
        print(f"Truck {self.id}: {self.status} | exit_path_len: {len(self._exit_path)}")
        if self.status != 'IDLE':
            print(f"DEBUG: Truck {self.id} Status: {self.status} Pos: {self.pos} Path Len: {len(self.path)}")  # log every active tick for debugging

        if self.status == self.STATUS_IDLE:  # nothing to do this tick
            return

        elif self.status == self.STATUS_NAVIGATING:
            if self.path:
                waypoint = self.path.pop(0)           # consume the next waypoint from the front of the list
                tx, ty, heading = self._waypoint_to_pose(grid, waypoint)
                self.pos[0], self.pos[1] = tx, ty     # snap position to the waypoint (no interpolation between ticks)
                self.heading = heading
                r, c = grid.world_to_cell(tx, ty)
                grid.deposit_trail(r, c, TRAIL_RADIUS_M / grid.cell_size, TRAIL_STRENGTH)

            if not self.path:  # path just ran out — check whether we reached the intended stop
                tr, tc = grid.world_to_cell(*self.pos)  # convert current world pos back to grid cell
                stop_cell = self._waypoint_to_cell(grid, self.stop_target) if self.stop_target else None
                if stop_cell and (tr, tc) == stop_cell:  # did we land exactly on the stop cell?
                    if self.dump_target:
                        dr, dc = self.dump_target
                        dtx, dty = grid.cell_to_world(dr, dc)
                        self.heading = np.arctan2(self.pos[1] - dty, self.pos[0] - dtx)  # face away from dump cell (truck will reverse into it)
                    self.status = self.STATUS_REVERSING  # trigger the one-tick reversing state
                else:  # path ran out but we didn't reach the stop — something went wrong
                    if self.dump_target:
                        grid.unreserve(*self.dump_target)  # free the reservation so the cell isn't permanently locked
                        self.dump_target = None
                    self.status = self.STATUS_IDLE  # fall back to idle; the assignment loop will try again

        elif self.status == self.STATUS_REVERSING:
            self.status      = self.STATUS_DUMPING  # immediately advance to DUMPING (reversing is instantaneous in this model)
            self._dump_ticks = 0                    # reset tick counter for the fresh dump

        elif self.status == self.STATUS_DUMPING:
            self._dump_ticks += 1  # count this tick toward the required dump duration
            if self.dump_target:
                r, c = self.dump_target
                grid.dump_at(r, c, self.volume_per_tick)  # add one tick's worth of dirt to the target cell

            if self._dump_ticks >= self._dump_ticks_required:  # dump is complete
                if self.dump_target:
                    grid.unreserve(*self.dump_target)  # release reservation now that this cell has been filled
                    self.dump_target = None             # clear target so needs_exit_path() triggers correctly
                self.status = self.STATUS_EXITING       # start the return journey

        elif self.status == self.STATUS_EXITING:
            if self._exit_path:
                waypoint = self._exit_path.pop(0)      # consume the next exit waypoint
                tx, ty, heading = self._waypoint_to_pose(grid, waypoint)
                self.pos[0], self.pos[1] = tx, ty
                self.heading = heading
                r, c = grid.world_to_cell(tx, ty)
                grid.deposit_trail(r, c, TRAIL_RADIUS_M / grid.cell_size, TRAIL_STRENGTH)

                if not self._exit_path:                # just consumed the last waypoint — path complete
                    self.pos         = list(ENTRY_POINT)
                    self.status      = self.STATUS_IDLE
                    self.path        = []
                    self.dump_target = None
                    self.stop_target = None
            # if _exit_path is already empty, waiting for the main loop to assign one — do nothing

    def needs_exit_path(self):
        # Returns True when truck has finished dumping and has no exit route yet — signals main loop to call pathfinder
        return (self.status == self.STATUS_EXITING
                and not self._exit_path
                and self.dump_target is None)

    def is_idle(self):
        return self.status == self.STATUS_IDLE  # convenience predicate used by the assignment loop

    def __repr__(self):
        # human-readable string for debugging print statements
        return (f"Truck({self.id}, class={self.truck_class}, "
                f"pos={self.pos}, status={self.status})")
