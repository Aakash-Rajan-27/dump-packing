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
from config import (TRUCK_CLASSES, ENTRY_POINT, _TAN_REPOSE, TRAIL_STRENGTH,
                    TRAIL_RADIUS_M, REVERSE_DUMP_CLOSE_ENOUGH_M,
                    REVERSE_DUMP_STEP_M, TRUCK_MOVE_STEP_M,
                    ENTRY_CORRIDOR_CELLS)
from filters import is_pose_driveable
from staging import required_dump_clearance_m


class Truck:

    # String constants for the truck's state machine — avoids typos and makes comparisons readable
    STATUS_WAITING    = 'WAITING'     # truck is queued outside the polygon, not yet released
    STATUS_ENTERING   = 'ENTERING'    # truck is driving from its waiting slot to the entry point
    STATUS_IDLE       = 'IDLE'        # truck is at the entry point, waiting for an assignment
    STATUS_NAVIGATING = 'NAVIGATING'  # truck is driving forward toward the dump stop position
    STATUS_REVERSING  = 'REVERSING'   # truck has arrived; one-tick state that kicks off dumping
    STATUS_DUMPING    = 'DUMPING'     # truck is actively depositing dirt over multiple ticks
    STATUS_EXITING    = 'EXITING'     # truck is driving back to the entry point after dumping
    STATUS_LEAVING    = 'LEAVING'     # truck drives out through the entry corridor to its home position

    def __init__(self, truck_id, truck_class, start_pos=None, waiting=False, home_pos=None):
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

        self.pos      = list(start_pos if start_pos else ENTRY_POINT)
        self.home_pos = list(home_pos if home_pos else ENTRY_POINT)  # outside parking position
        self.heading  = np.pi / 2  # initial heading: pointing "up" (north) in world coords

        self.status      = self.STATUS_WAITING if waiting else self.STATUS_IDLE  # WAITING = held outside until released
        self.path        = []                # ordered list of (row, col) waypoints to follow
        self.dump_target = None              # (row, col) of the grid cell being filled this run
        self.stop_target = None              # (row, col) where the truck stops before reversing
        self._dump_ticks = 0                 # tick counter for the current dump operation
        self._exit_path  = []                # waypoints for the return trip to the entry point
        self.staging_pose = None             # outward-facing pose reached before reversing
        self._reverse_stuck = 0             # consecutive steps where reverse was blocked
        self._stuck_substeps = 0            # consecutive substeps reverted by collision guard
        self._conflict_cooldown = 0         # ticks before this truck can be force-replanned again
        self._dump_corridor_marked = False   # True while the dump-path corridor is active in the grid
        self._exit_corridor_marked = False   # True while the exit-path corridor is active in the grid
        self._leaving_waypoints    = []      # [ENTRY_POINT, home_pos] driven during STATUS_LEAVING
        # Pre-planned dump path for waiting trucks (planned before entering the polygon)
        self._pre_path         = None        # path pre-planned while truck is WAITING
        # Stagnation tracking for headlock detection
        self._pos_snapshot        = list(self.pos)  # position at start of last tick
        self._pos_stagnant_ticks  = 0               # consecutive ticks with < 0.5 m net progress
        self._pre_dump_target  = None        # dump target reserved during pre-planning
        self._pre_staging_pose = None        # staging pose for the pre-planned path

    def front_center_world(self):
        half_len = self.length / 2.0
        return (
            self.pos[0] + math.cos(self.heading) * half_len,
            self.pos[1] + math.sin(self.heading) * half_len,
        )

    def front_center_cell(self, grid):
        return grid.world_to_cell(*self.front_center_world())

    def rear_axle_world(self):
        half_len = self.length / 2.0
        return (
            self.pos[0] - math.cos(self.heading) * half_len,
            self.pos[1] - math.sin(self.heading) * half_len,
        )

    def front_axle_world(self):
        return self.front_center_world()

    def _body_pose_for_front_target(self, front_x, front_y):
        half_len = self.length / 2.0
        hx, hy = math.cos(self.heading), math.sin(self.heading)

        dx = front_x - self.pos[0]
        dy = front_y - self.pos[1]
        forward = dx * hx + dy * hy
        dist2 = dx * dx + dy * dy
        lateral2 = max(0.0, dist2 - forward * forward)

        if lateral2 <= half_len * half_len:
            along_offset = math.sqrt(max(0.0, half_len * half_len - lateral2))
            candidates = [forward - along_offset, forward + along_offset]
            forward_candidates = [s for s in candidates if s >= -0.01]
            travel = min(forward_candidates) if forward_candidates else max(candidates)
            body_x = self.pos[0] + travel * hx
            body_y = self.pos[1] + travel * hy
        else:
            heading_to_front = math.atan2(dy, dx)
            body_x = front_x - math.cos(heading_to_front) * half_len
            body_y = front_y - math.sin(heading_to_front) * half_len

        new_heading = math.atan2(front_y - body_y, front_x - body_x)
        return body_x, body_y, new_heading

    def _waypoint_to_pose(self, grid, waypoint):
        if len(waypoint) == 3:
            rear_x, rear_y, heading = waypoint
            half_len = self.length / 2.0
            return (
                rear_x + math.cos(heading) * half_len,
                rear_y + math.sin(heading) * half_len,
                heading,
            )

        r, c = waypoint
        tx, ty = grid.cell_to_world(r, c)
        return self._body_pose_for_front_target(tx, ty)

    def _waypoint_to_cell(self, grid, waypoint):
        if len(waypoint) == 3:
            rear_x, rear_y, heading = waypoint
            return grid.world_to_cell(
                rear_x + math.cos(heading) * self.length,
                rear_y + math.sin(heading) * self.length,
            )
        return waypoint

    def _corridor_cells_from_path(self, grid, path):
        """Convert a list of (rear_x, rear_y, heading) waypoints to body-centre grid cells."""
        half_len = self.length / 2.0
        seen = set()
        cells = []
        for wp in path:
            if len(wp) == 3:
                rx, ry, h = wp
                bx = rx + math.cos(h) * half_len
                by = ry + math.sin(h) * half_len
            else:
                bx, by = grid.cell_to_world(wp[0], wp[1])
            cell = grid.world_to_cell(bx, by)
            if cell not in seen:
                seen.add(cell)
                cells.append(cell)
        return cells

    def _mark_dump_corridor(self, grid, path):
        if self._dump_corridor_marked:
            grid.clear_path_corridor(f"dump_{self.id}")
        cells = self._corridor_cells_from_path(grid, path)
        half_w_cells = (self.width / 2.0) / grid.cell_size
        grid.mark_path_corridor(f"dump_{self.id}", cells, half_w_cells)
        self._dump_corridor_marked = True

    def _clear_dump_corridor(self, grid):
        if self._dump_corridor_marked:
            grid.clear_path_corridor(f"dump_{self.id}")
            self._dump_corridor_marked = False

    def _mark_exit_corridor(self, grid, exit_path):
        if self._exit_corridor_marked:
            grid.clear_path_corridor(f"exit_{self.id}")
        cells = self._corridor_cells_from_path(grid, exit_path)
        half_w_cells = (self.width / 2.0) / grid.cell_size
        grid.mark_path_corridor(f"exit_{self.id}", cells, half_w_cells)
        self._exit_corridor_marked = True

    def _clear_exit_corridor(self, grid):
        if self._exit_corridor_marked:
            grid.clear_path_corridor(f"exit_{self.id}")
            self._exit_corridor_marked = False

    def clear_all_corridors(self, grid):
        """Clear both corridors — call when force-idling or force-replanning this truck."""
        self._clear_dump_corridor(grid)
        self._clear_exit_corridor(grid)

    def preload_dump_path(self, path, dump_target, grid, staging_pose=None):
        """Store a dump path pre-planned while the truck is still WAITING outside.
        Marks the corridor and reserves the target cell so the path is committed
        before the truck physically enters the polygon."""
        # Clear any stale pre-plan first
        self.cancel_preload(grid)
        self._pre_path         = list(path)
        self._pre_dump_target  = dump_target
        self._pre_staging_pose = staging_pose
        if dump_target:
            grid.reserve(*dump_target)
        self._mark_dump_corridor(grid, self._pre_path)

    def cancel_preload(self, grid):
        """Discard a pre-planned path (e.g. it became invalid). Frees all reservations."""
        if self._pre_dump_target:
            grid.unreserve(*self._pre_dump_target)
            self._pre_dump_target = None
        self._clear_dump_corridor(grid)
        self._pre_path         = None
        self._pre_staging_pose = None

    def set_path(self, path, dump_target, grid, staging_pose=None):
        self.path        = list(path)   # copy the path so mutations outside don't affect us
        self.dump_target = dump_target  # remember which cell we're filling this trip
        self.staging_pose = staging_pose

        if not self.path:  # if A* returned an empty path (truck is already there or fully blocked)
            if self.dump_target:
                grid.unreserve(*self.dump_target)  # release the cell reservation so another truck can use it
                self.dump_target = None             # clear the target reference
            self._clear_dump_corridor(grid)
            self.status = self.STATUS_IDLE          # stay idle — nothing to do
            return

        self.stop_target = self.path[-1]    # the last waypoint is where the truck parks before reversing
        self.status = self.STATUS_NAVIGATING  # begin driving toward the dump position
        if dump_target:
            grid.reserve(*dump_target)  # mark the target cell as reserved so other trucks don't also plan to dump there
        self._mark_dump_corridor(grid, self.path)

    def set_exit_path(self, exit_path, grid):
        if exit_path:  # only switch state if a valid exit path was found
            self._exit_path = list(exit_path)      # store a copy of the exit waypoints
            self.status     = self.STATUS_EXITING  # truck will start following the exit path next tick
            # Mark exit corridor so navigating dump trucks route around us.
            # Exit trucks themselves still use ignore_path_reserved=True so they can
            # reach the entry regardless of other corridors; CBS time constraints
            # handle temporal separation between simultaneously-exiting trucks.
            self._mark_exit_corridor(grid, self._exit_path)
        else:
            # CBS returned an empty path — leave status as EXITING so the main loop retries next tick.
            pass

    def step(self, grid):
        if self.status in (self.STATUS_IDLE, self.STATUS_WAITING):  # nothing to do this tick
            return

        elif self.status == self.STATUS_ENTERING:
            ex, ey = ENTRY_POINT
            dx = ex - self.pos[0]
            dy = ey - self.pos[1]
            dist = math.hypot(dx, dy)
            step = TRUCK_MOVE_STEP_M
            if dist <= step:
                self.pos[0], self.pos[1] = ex, ey
                self.heading = math.pi / 2  # face into the polygon (north)
                if self._pre_path:
                    # Consume the pre-planned dump path — skip IDLE entirely
                    self.path         = self._pre_path
                    self.dump_target  = self._pre_dump_target
                    self.staging_pose = self._pre_staging_pose
                    self.stop_target  = self._pre_path[-1]
                    self._pre_path         = None
                    self._pre_dump_target  = None
                    self._pre_staging_pose = None
                    self.status = self.STATUS_NAVIGATING
                else:
                    self.status = self.STATUS_IDLE
            else:
                self.pos[0] += (dx / dist) * step
                self.pos[1] += (dy / dist) * step
                self.heading = math.atan2(dy, dx)
            return

        elif self.status == self.STATUS_LEAVING:
            # Drive through waypoints: always [ENTRY_POINT, home_pos] so the truck
            # physically passes through the entry corridor before going outside.
            if not self._leaving_waypoints:
                self.status = self.STATUS_WAITING
                return
            tx, ty = self._leaving_waypoints[0]
            dx = tx - self.pos[0]
            dy = ty - self.pos[1]
            dist = math.hypot(dx, dy)
            step = TRUCK_MOVE_STEP_M
            if dist <= step:
                self.pos[0], self.pos[1] = tx, ty
                self.heading = math.atan2(dy, dx) if dist > 1e-9 else self.heading
                self._leaving_waypoints.pop(0)
                if not self._leaving_waypoints:
                    self.heading = math.pi / 2  # face inward when parked outside
                    self._clear_exit_corridor(grid)  # truck is fully outside — release the corridor
                    self.status  = self.STATUS_WAITING
            else:
                self.pos[0] += (dx / dist) * step
                self.pos[1] += (dy / dist) * step
                self.heading = math.atan2(dy, dx)
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
                tr, tc = self.front_center_cell(grid)  # path targets the truck's front centre, not its body centre
                stop_cell = self._waypoint_to_cell(grid, self.stop_target) if self.stop_target else None
                if stop_cell and (tr, tc) == stop_cell:  # did we land exactly on the stop cell?
                    if self.dump_target and self.staging_pose is None:
                        dr, dc = self.dump_target
                        dtx, dty = grid.cell_to_world(dr, dc)
                        self.heading = np.arctan2(self.pos[1] - dty, self.pos[0] - dtx)  # face away from dump cell (truck will reverse into it)
                    self._clear_dump_corridor(grid)   # path consumed — release corridor for others
                    self.status = self.STATUS_REVERSING  # trigger the one-tick reversing state
                else:  # path ran out but we didn't reach the stop — something went wrong
                    if self.dump_target:
                        grid.unreserve(*self.dump_target)  # free the reservation so the cell isn't permanently locked
                        self.dump_target = None
                    self._clear_dump_corridor(grid)
                    self.status = self.STATUS_IDLE  # fall back to idle; the assignment loop will try again

        elif self.status == self.STATUS_REVERSING:
            if self.dump_target is None or self.staging_pose is None:
                self.status      = self.STATUS_DUMPING
                self._dump_ticks = 0
                return

            dump_x, dump_y = grid.cell_to_world(*self.dump_target)
            rear_x, rear_y = self.rear_axle_world()
            clearance = required_dump_clearance_m(self)
            if math.hypot(rear_x - dump_x, rear_y - dump_y) <= clearance + REVERSE_DUMP_CLOSE_ENOUGH_M:
                self.status      = self.STATUS_DUMPING
                self._dump_ticks = 0
                return

            next_x = self.pos[0] - math.cos(self.heading) * REVERSE_DUMP_STEP_M
            next_y = self.pos[1] - math.sin(self.heading) * REVERSE_DUMP_STEP_M
            if is_pose_driveable(grid, self, next_x, next_y, self.heading):
                self._reverse_stuck = 0
                self.pos[0], self.pos[1] = next_x, next_y
                r, c = grid.world_to_cell(next_x, next_y)
                grid.deposit_trail(r, c, TRAIL_RADIUS_M / grid.cell_size, TRAIL_STRENGTH)
            else:
                self._reverse_stuck += 1
                if self._reverse_stuck >= 5:
                    # Terrain blocked reverse for 5 consecutive steps — dump in place
                    self._reverse_stuck = 0
                    self.status      = self.STATUS_DUMPING
                    self._dump_ticks = 0

        elif self.status == self.STATUS_DUMPING:
            self._dump_ticks += 1  # count this tick toward the required dump duration
            if self.dump_target:
                r, c = self.dump_target
                grid.dump_at(r, c, self.volume_per_tick)  # add one tick's worth of dirt to the target cell

            if self._dump_ticks >= self._dump_ticks_required:  # dump is complete
                if self.dump_target:
                    grid.unreserve(*self.dump_target)  # release reservation now that this cell has been filled
                    self.dump_target = None             # clear target so needs_exit_path() triggers correctly
                    self.staging_pose = None
                self.status = self.STATUS_EXITING       # start the return journey

        elif self.status == self.STATUS_EXITING:
            if self._exit_path:
                waypoint = self._exit_path.pop(0)      # consume the next exit waypoint
                tx, ty, heading = self._waypoint_to_pose(grid, waypoint)
                self.pos[0], self.pos[1] = tx, ty
                self.heading = heading
                r, c = grid.world_to_cell(tx, ty)
                grid.deposit_trail(r, c, TRAIL_RADIUS_M / grid.cell_size, TRAIL_STRENGTH)

                # As soon as ANY part of the truck touches the entry corridor,
                # snap to home — check both body centre and front so the trigger
                # fires even when the smooth path's rear-axle pose stops just
                # outside the corridor while the front is already inside it.
                entry_rc = grid.world_to_cell(*ENTRY_POINT)
                cur_r, cur_c = grid.world_to_cell(self.pos[0], self.pos[1])
                front_r, front_c = self.front_center_cell(grid)
                if (math.hypot(cur_r - entry_rc[0], cur_c - entry_rc[1]) <= ENTRY_CORRIDOR_CELLS
                        or math.hypot(front_r - entry_rc[0], front_c - entry_rc[1]) <= ENTRY_CORRIDOR_CELLS):
                    self._clear_exit_corridor(grid)
                    self._exit_path      = []
                    self.pos[0], self.pos[1] = self.home_pos[0], self.home_pos[1]
                    self.heading         = math.pi / 2
                    self.path            = []
                    self.dump_target     = None
                    self.stop_target     = None
                    self.status          = self.STATUS_WAITING
                    return

                if not self._exit_path:                # path exhausted before reaching corridor (fallback)
                    self._leaving_waypoints = [tuple(ENTRY_POINT), tuple(self.home_pos)]
                    self.status      = self.STATUS_LEAVING
                    self.path        = []
                    self.dump_target = None
                    self.stop_target = None
            # if _exit_path is already empty, waiting for the main loop to assign one — do nothing

    def needs_exit_path(self):
        # Returns True when truck has finished dumping and has no exit route yet — signals main loop to call pathfinder
        return (self.status == self.STATUS_EXITING
                and not self._exit_path
                and self.dump_target is None)

    def release(self):
        """Move truck from WAITING to ENTERING so it drives to the entry point."""
        if self.status == self.STATUS_WAITING:
            self.status = self.STATUS_ENTERING

    def is_idle(self):
        return self.status == self.STATUS_IDLE  # convenience predicate used by the assignment loop

    def __repr__(self):
        # human-readable string for debugging print statements
        return (f"Truck({self.id}, class={self.truck_class}, "
                f"pos={self.pos}, status={self.status})")
