# deadlock.py
# ─────────────────────────────────────────────────────────────
# Deadlock and headlock escape maneuvers:
#   • generate_reverse_retreat()  — back up along current heading
#   • generate_yield_maneuver()   — reverse + 90° turn to yield
#   • escape_and_replan_exit()    — fallback exit planner for
#     trucks CBS could not route out
# ─────────────────────────────────────────────────────────────

import math
from path_utils import _truck_front_cell
from bicycle_model import interpolate_path_to_truck_states
from filters import make_driveable_mask
from config import DRIVE_CLEARANCE_M, _TAN_REPOSE, ENTRY_CORRIDOR_CELLS, TRUCK_MOVE_STEP_M


def generate_reverse_retreat(truck, grid, num_steps=6):
    """
    Generate a short list of (rear_x, rear_y, heading) waypoints that move the
    truck backward along its current heading, for use in deadlock escape.

    The truck heading stays constant — only body position changes backward.
    Stops early if a BOUNDARY cell is encountered or the polygon edge is hit.
    Returns an empty list if no safe retreat is possible.
    """
    import grid_map as _gm
    rear_x, rear_y = truck.rear_axle_world()
    half_len = truck.length / 2.0
    # Backward direction = opposite of heading
    bcos = -math.cos(truck.heading)
    bsin = -math.sin(truck.heading)
    step = grid.cell_size

    retreat = []
    rx, ry = rear_x, rear_y

    for _ in range(num_steps):
        rx += bcos * step
        ry += bsin * step
        # Body centre when rear is at (rx, ry) facing truck.heading
        bx = rx + math.cos(truck.heading) * half_len
        by = ry + math.sin(truck.heading) * half_len
        nr, nc = grid.world_to_cell(rx, ry)
        br, bc = grid.world_to_cell(bx, by)
        if not (0 <= nr < grid.rows and 0 <= nc < grid.cols):
            break
        if not (0 <= br < grid.rows and 0 <= bc < grid.cols):
            break
        if grid.state[nr, nc] == _gm.CellState.BOUNDARY:
            break
        if grid.state[br, bc] == _gm.CellState.BOUNDARY:
            break
        retreat.append((rx, ry, truck.heading))

    return retreat


def generate_yield_maneuver(truck, grid, other_truck, num_reverse=5, num_turn=12):
    """
    Yield maneuver for head-on (headlock) deadlocks.

    Phase 1 — reverse straight to open clearance between the trucks.
    Phase 2 — turn ~90° away from the blocker while advancing, so the
              other truck has a clear lane to pass.

    Turn direction chosen so the yielder moves AWAY from the side where
    the other truck sits, maximising lateral clearance.

    Returns list of (rear_x, rear_y, heading) waypoints, or [] if blocked.
    """
    import grid_map as _gm
    half_len = truck.length / 2.0

    # ── Phase 1: straight reverse ────────────────────────────────────────────
    retreat = generate_reverse_retreat(truck, grid, num_steps=num_reverse)
    if not retreat:
        return []

    # ── Choose turn direction ─────────────────────────────────────────────────
    # Cross product of truck heading × vector-to-other: positive → other is to
    # our LEFT → turn right (clockwise, negative sign); negative → turn left.
    dx = other_truck.pos[0] - truck.pos[0]
    dy = other_truck.pos[1] - truck.pos[1]
    cross = math.cos(truck.heading) * dy - math.sin(truck.heading) * dx
    turn_sign = -1.0 if cross >= 0.0 else 1.0

    # ── Phase 2: forward + 90° turn ──────────────────────────────────────────
    cur_x, cur_y, cur_h = retreat[-1]
    d_h = turn_sign * (math.pi / 2.0) / num_turn   # heading increment per step
    step_m = TRUCK_MOVE_STEP_M

    turn_poses = []
    for _ in range(num_turn):
        mid_h = cur_h + d_h / 2.0
        nx = cur_x + math.cos(mid_h) * step_m
        ny = cur_y + math.sin(mid_h) * step_m
        nh = cur_h + d_h

        bx = nx + math.cos(nh) * half_len
        by = ny + math.sin(nh) * half_len
        nr, nc = grid.world_to_cell(nx, ny)
        br, bc = grid.world_to_cell(bx, by)
        if not (0 <= nr < grid.rows and 0 <= nc < grid.cols):
            break
        if not (0 <= br < grid.rows and 0 <= bc < grid.cols):
            break
        if grid.state[nr, nc] == _gm.CellState.BOUNDARY:
            break
        if grid.state[br, bc] == _gm.CellState.BOUNDARY:
            break
        cur_x, cur_y, cur_h = nx, ny, nh
        turn_poses.append((cur_x, cur_y, cur_h))

    return retreat + turn_poses


def escape_and_replan_exit(truck, grid, all_trucks, entry_rc):
    """
    Fallback exit planner for a truck that CBS could not route out.

    Phase 1 — back up until clear of the pile or other trucks.
    Phase 2 — replan from the escape endpoint with a fresh driveable mask.
    Phase 3 — prepend the retreat waypoints to the smoothed A* path.
    """
    import grid_map as _gm
    from astar_core import astar_st
    half_len = truck.length / 2.0

    # ── Phase 1: reverse retreat ──────────────────────────────────────────────
    retreat = generate_reverse_retreat(truck, grid, num_steps=12)

    trimmed = []
    for pose in retreat:
        rx, ry, rh = pose
        bx = rx + math.cos(rh) * half_len
        by = ry + math.sin(rh) * half_len
        # Stop if too close to another truck
        for other in all_trucks:
            if other is truck:
                continue
            if math.hypot(bx - other.pos[0], by - other.pos[1]) < (truck.length) * 0.5:
                retreat = trimmed
                break
        else:
            # Stop if body cell z_height has dropped below clearance (pile is behind us)
            br, bc = grid.world_to_cell(bx, by)
            if (0 <= br < grid.rows and 0 <= bc < grid.cols
                    and grid.z_height[br, bc] < DRIVE_CLEARANCE_M):
                retreat = trimmed
                break
            trimmed.append(pose)
            continue
        break
    else:
        retreat = trimmed

    # Escape endpoint (body centre + heading)
    if retreat:
        esc_rx, esc_ry, esc_h = retreat[-1]
        esc_bx = esc_rx + math.cos(esc_h) * half_len
        esc_by = esc_ry + math.sin(esc_h) * half_len
    else:
        esc_bx, esc_by, esc_h = truck.pos[0], truck.pos[1], truck.heading

    # ── Phase 2: replan from escape endpoint ─────────────────────────────────
    saved_pos     = list(truck.pos)
    saved_heading = truck.heading
    truck.pos     = [esc_bx, esc_by]
    truck.heading = esc_h

    driveable = make_driveable_mask(grid, truck, ignore_path_reserved=True)

    r_pile_m    = (1.2 * truck.payload_volume_m3 / (math.pi * _TAN_REPOSE)) ** (1 / 3)
    bulldozer_r = max(2, int(math.ceil(r_pile_m / grid.cell_size)))
    esc_cell    = _truck_front_cell(grid, truck)
    for dr in range(-bulldozer_r, bulldozer_r + 1):
        for dc in range(-bulldozer_r, bulldozer_r + 1):
            if abs(dr) + abs(dc) > bulldozer_r:
                continue
            nr, nc = esc_cell[0] + dr, esc_cell[1] + dc
            if 0 <= nr < grid.rows and 0 <= nc < grid.cols:
                if grid.state[nr, nc] != _gm.CellState.BOUNDARY:
                    driveable[nr, nc, :] = True

    astar_path = astar_st(driveable, grid, esc_cell, entry_rc, truck,
                          constraints=frozenset(),
                          stop_dist_cells=float(ENTRY_CORRIDOR_CELLS),
                          max_time=500)

    truck.pos     = saved_pos
    truck.heading = saved_heading

    if not astar_path and not retreat:
        return []

    # ── Phase 3: assemble path ────────────────────────────────────────────────
    # Smooth only the A* coarse portion; prepend the already-fine retreat poses.
    if astar_path:
        truck.pos     = [esc_bx, esc_by]
        truck.heading = esc_h
        # Snap heading toward first coarse cell (mirrors smooth_paths exit logic)
        fx, fy = grid.cell_to_world(astar_path[0][0], astar_path[0][1])
        dx_h = fx - esc_bx
        dy_h = fy - esc_by
        if math.hypot(dx_h, dy_h) > 1e-9:
            truck.heading = math.atan2(dy_h, dx_h)
        smooth_part = interpolate_path_to_truck_states(grid, truck, astar_path)
        truck.pos     = saved_pos
        truck.heading = saved_heading
        if not smooth_part:
            smooth_part = astar_path
    else:
        smooth_part = []

    return retreat + smooth_part
