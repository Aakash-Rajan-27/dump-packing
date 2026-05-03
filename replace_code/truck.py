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

import numpy as np
import math
from config import TRUCK_CLASSES, ENTRY_POINT, _TAN_REPOSE


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
        
        self.payload_volume_m3 = self.payload_t / 1.8 
        self.volume_per_tick   = self.payload_volume_m3 / self._dump_ticks_required

        self.pos     = list(start_pos if start_pos else ENTRY_POINT)
        self.heading = np.pi / 2

        self.status      = self.STATUS_IDLE
        self.path        = []
        self.dump_target = None
        self.stop_target = None 
        self._dump_ticks = 0
        self._exit_path  = []   

    def set_path(self, path, dump_target, grid):
        self.path        = list(path)
        self.dump_target = dump_target
        
        if not self.path:
            if self.dump_target:
                grid.unreserve(*self.dump_target)
                self.dump_target = None
            self.status = self.STATUS_IDLE
            return
            
        self.stop_target = self.path[-1]
        self.status = self.STATUS_NAVIGATING
        if dump_target:
            grid.reserve(*dump_target)

    def set_exit_path(self, exit_path):
        if exit_path:
            self._exit_path = list(exit_path)
            self.status     = self.STATUS_EXITING
        else:
            # Trapped/Blocked: wait for pathfinder retry
            pass

    def step(self, grid):
        if self.status != 'IDLE':
            print(f"DEBUG: Truck {self.id} Status: {self.status} Pos: {self.pos} Path Len: {len(self.path)}")

        if self.status == self.STATUS_IDLE:
            return
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
                tr, tc = grid.world_to_cell(*self.pos)
                if self.stop_target and (tr, tc) == self.stop_target:
                    if self.dump_target:
                        dr, dc = self.dump_target
                        dtx, dty = grid.cell_to_world(dr, dc)
                        self.heading = np.arctan2(self.pos[1] - dty, self.pos[0] - dtx)
                    self.status = self.STATUS_REVERSING
                else:
                    if self.dump_target:
                        grid.unreserve(*self.dump_target)
                        self.dump_target = None
                    self.status = self.STATUS_IDLE

        elif self.status == self.STATUS_REVERSING:
            self.status      = self.STATUS_DUMPING
            self._dump_ticks = 0

        elif self.status == self.STATUS_DUMPING:
            self._dump_ticks += 1
            if self.dump_target:
                r, c = self.dump_target
                grid.dump_at(r, c, self.volume_per_tick)

            if self._dump_ticks >= self._dump_ticks_required:
                if self.dump_target:
                    grid.unreserve(*self.dump_target)  
                    self.dump_target = None
                self.status = self.STATUS_EXITING

        elif self.status == self.STATUS_EXITING:
            if self._exit_path:
                r, c = self._exit_path.pop(0)
                tx, ty = grid.cell_to_world(r, c)
                dx, dy = tx - self.pos[0], ty - self.pos[1]
                if abs(dx) > 0.01 or abs(dy) > 0.01:
                    self.heading = np.arctan2(dy, dx)
                self.pos[0], self.pos[1] = tx, ty
                
            # ─── RADIUS & LOOSE ANGLE ENTRY CHECK ───
            dist_to_entry = math.hypot(self.pos[0] - ENTRY_POINT[0], 
                                       self.pos[1] - ENTRY_POINT[1])
            
            # Heading difference calculation
            angle_diff = abs((self.heading - np.pi/2 + np.pi) % (2*np.pi) - np.pi)
            
            # CHANGE 1: Set angle tolerance to 10.0 (effectively disabling it)
            # CHANGE 2: Ensure distance (3.0m) is larger than pathfinder stop (2.0m)
            if dist_to_entry < 3.0 and (angle_diff < 10.0 or not self._exit_path):
                print(f"[DEBUG] Truck {self.id} reached entry radius. Snapping to IDLE.")
                self.pos         = list(ENTRY_POINT)
                self.heading     = np.pi / 2
                self.status      = self.STATUS_IDLE
                self.path        = []
                self.dump_target = None
                self.stop_target = None

    def needs_exit_path(self):
        return (self.status == self.STATUS_EXITING
                and not self._exit_path
                and self.dump_target is None)

    def is_idle(self):
        return self.status == self.STATUS_IDLE

    def __repr__(self):
        return (f"Truck({self.id}, class={self.truck_class}, "
                f"pos={self.pos}, status={self.status})")