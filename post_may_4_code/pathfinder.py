# pathfinder.py
# ─────────────────────────────────────────────────────────────
# MAPF A* PATHFINDING WITH RESILIENCE OVERRIDES
#
# RESOLUTIONS INCLUDED:
# 1. RADIUS GOAL: astar() terminates when within safe dumping radius.
# 2. CLOSEST APPROACH: If target is buried, it drives as close as possible.
# 3. GHOST MODE: CBS constraints relaxed (frozenset) to prevent traffic freezes.
# 4. BULLDOZER START: Start nodes are overridden to True to prevent dirt-trapping.
# ─────────────────────────────────────────────────────────────

import heapq  # min-heap for A* open set (priority queue)
import numpy as np  # array operations
import math  # math.hypot for distance
import shapely
import grid_map  # needed for CellState.BOUNDARY check in bulldozer
from filters import is_pose_driveable, make_driveable_mask  # builds a 3-D boolean mask: [row, col, heading_bucket] → can drive here?
from config import (DRIVE_CLEARANCE_M, _TAN_REPOSE, ENTRY_POINT,
                    TURN_REFINEMENT_ITERATIONS, TRUCK_MOVE_STEP_M,
                    TURN_LOOKAHEAD_RADIUS_FACTOR, TURN_PATH_TOLERANCE_M,
                    TURN_MAX_SMOOTH_STEPS, POSE_HEADING_BUCKETS,
                    ASTAR_WAIT_COST, LOCKED_PATH_HORIZON,
                    ENTRY_CORRIDOR_CELLS)
from staging import score_staging_candidates


def _truck_inside_boundary(polygon, body_x, body_y, heading, half_l, half_w):
    """Hard boundary check: all 4 truck body corners must be inside the polygon.
    Skipped within 20 m of ENTRY_POINT so trucks can cross the gate threshold."""
    if math.hypot(body_x - ENTRY_POINT[0], body_y - ENTRY_POINT[1]) <= 20.0:
        return True
    hcos, hsin = math.cos(heading), math.sin(heading)
    scos, ssin = -hsin, hcos
    for ls in (-1, 1):
        for ws in (-1, 1):
            cx = body_x + ls * half_l * hcos + ws * half_w * scos
            cy = body_y + ls * half_l * hsin + ws * half_w * ssin
            if not shapely.contains_xy(polygon, cx, cy):
                return False
    return True


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


# Maps each of the 4 cardinal (row, col) step directions to an integer "bucket" index.
# The driveable mask's third dimension is indexed by this bucket so we can have
# direction-specific drivability (e.g. a truck can't reverse uphill but can go forward).
_DIR_TO_BUCKET = {
    ( 0,  1): 0,  # right
    (-1,  1): 1,  # up-right
    (-1,  0): 2,  # up
    (-1, -1): 3,  # up-left
    ( 0, -1): 4,  # left
    ( 1, -1): 5,  # down-left
    ( 1,  0): 6,  # down
    ( 1,  1): 7,  # down-right
}
_BUCKET_TO_DIR = {bucket: delta for delta, bucket in _DIR_TO_BUCKET.items()}
_BUCKET_TO_HEADING = {
    0: 0.0,
    1: -math.pi / 4,
    2: -math.pi / 2,
    3: -3 * math.pi / 4,
    4: math.pi,
    5: 3 * math.pi / 4,
    6: math.pi / 2,
    7: math.pi / 4,
}

def _heading_bucket(dr, dc):
    # Convert a (row_delta, col_delta) step into its driveable-mask bucket index.
    # Falls back to bucket 0 if the direction isn't in the lookup table.
    return _DIR_TO_BUCKET.get((dr, dc), 0)

def _angle_diff_signed(target, current):
    return (target - current + math.pi) % (2 * math.pi) - math.pi

def _bucket_from_heading(heading):
    best_bucket = 0
    best_delta = float('inf')
    for bucket, bucket_heading in _BUCKET_TO_HEADING.items():
        delta = abs(_angle_diff_signed(bucket_heading, heading))
        if delta < best_delta:
            best_bucket = bucket
            best_delta = delta
    return best_bucket

def _heading_for_bucket(bucket):
    return _BUCKET_TO_HEADING[bucket % 8]

def _state_cell(state):
    return (state[0], state[1])

def _path_cells(grid, path):
    cells = []
    for wp in path or []:
        if len(wp) == 3 and not isinstance(wp[0], (int, np.integer)):
            cells.append(grid.world_to_cell(wp[0], wp[1]))
        else:
            cells.append((wp[0], wp[1]))
    return cells

def _truck_front_cell(grid, truck):
    if hasattr(truck, "front_center_cell"):
        return truck.front_center_cell(grid)
    front_x = truck.pos[0] + math.cos(truck.heading) * (truck.length / 2.0)
    front_y = truck.pos[1] + math.sin(truck.heading) * (truck.length / 2.0)
    return grid.world_to_cell(front_x, front_y)

def _truck_rear_pose(truck):
    if hasattr(truck, "rear_axle_world"):
        rear_x, rear_y = truck.rear_axle_world()
    else:
        half_len = truck.length / 2.0
        rear_x = truck.pos[0] - math.cos(truck.heading) * half_len
        rear_y = truck.pos[1] - math.sin(truck.heading) * half_len
    return float(rear_x), float(rear_y), float(truck.heading)

def _truck_front_world(truck):
    if hasattr(truck, "front_axle_world"):
        return truck.front_axle_world()
    return truck.front_center_world()

def _steering_angle(front, heading, target):
    dx = target[0] - front[0]
    dy = target[1] - front[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    return _angle_diff_signed(math.atan2(dy, dx), heading)

def _turn_radius(length, steering_angle):
    absolute_angle = abs(steering_angle)
    if absolute_angle >= math.pi / 2:
        return 0.0
    tangent = math.tan(absolute_angle)
    return float('inf') if tangent < 1e-9 else length / tangent

def _refine_front_target(front, heading, desired, truck):
    """Clamp a front-wheel target with five generalized midpoint checks."""
    steering = _steering_angle(front, heading, desired)
    if _turn_radius(truck.length, steering) >= truck.turn_radius:
        return desired, steering

    distance = max(1e-9, math.hypot(desired[0] - front[0], desired[1] - front[1]))
    feasible = (
        front[0] + math.cos(heading) * distance,
        front[1] + math.sin(heading) * distance,
    )
    infeasible = desired
    for _ in range(TURN_REFINEMENT_ITERATIONS):
        midpoint = (
            (feasible[0] + infeasible[0]) / 2.0,
            (feasible[1] + infeasible[1]) / 2.0,
        )
        midpoint_steering = _steering_angle(front, heading, midpoint)
        if _turn_radius(truck.length, midpoint_steering) >= truck.turn_radius:
            feasible = midpoint
        else:
            infeasible = midpoint
    return feasible, _steering_angle(front, heading, feasible)

def _polyline_lengths(points):
    lengths = [0.0]
    for start, end in zip(points, points[1:]):
        lengths.append(lengths[-1] + math.hypot(end[0] - start[0], end[1] - start[1]))
    return lengths

def _sample_polyline(points, lengths, distance):
    distance = max(0.0, min(distance, lengths[-1]))
    for index in range(1, len(points)):
        if lengths[index] >= distance:
            segment = lengths[index] - lengths[index - 1]
            if segment < 1e-9:
                return points[index]
            ratio = (distance - lengths[index - 1]) / segment
            return (
                points[index - 1][0] + ratio * (points[index][0] - points[index - 1][0]),
                points[index - 1][1] + ratio * (points[index][1] - points[index - 1][1]),
            )
    return points[-1]

def _closest_polyline_distance(points, lengths, point, start_distance):
    best_distance = start_distance
    best_error = float('inf')
    for index in range(1, len(points)):
        if lengths[index] + 1e-9 < start_distance:
            continue
        ax, ay = points[index - 1]
        bx, by = points[index]
        dx, dy = bx - ax, by - ay
        segment2 = dx * dx + dy * dy
        if segment2 < 1e-9:
            continue
        ratio = max(0.0, min(1.0, ((point[0] - ax) * dx + (point[1] - ay) * dy) / segment2))
        px, py = ax + ratio * dx, ay + ratio * dy
        error = math.hypot(point[0] - px, point[1] - py)
        if error < best_error:
            best_error = error
            best_distance = lengths[index - 1] + ratio * math.sqrt(segment2)
    return max(start_distance, best_distance)

def interpolate_path_to_truck_states(grid, truck, coarse_path):
    """Emit rear-axle world poses that obey the configured minimum turn radius."""
    if not coarse_path:
        return []

    points = [_truck_front_world(truck)]
    for state in coarse_path:
        point = grid.cell_to_world(state[0], state[1])
        if math.hypot(point[0] - points[-1][0], point[1] - points[-1][1]) > 1e-9:
            points.append(point)
    if len(points) == 1:
        return []

    lengths = _polyline_lengths(points)
    total_length = lengths[-1]
    lookahead = max(TRUCK_MOVE_STEP_M, truck.turn_radius * TURN_LOOKAHEAD_RADIUS_FACTOR)
    rear_x, rear_y, heading = _truck_rear_pose(truck)
    progress = 0.0
    states = []

    # Looking ahead by roughly one minimum radius starts the physical turn
    # before the coarse path corner and lets the rear axle converge afterward.
    for _ in range(TURN_MAX_SMOOTH_STEPS):
        front = (
            rear_x + math.cos(heading) * truck.length,
            rear_y + math.sin(heading) * truck.length,
        )
        progress = _closest_polyline_distance(points, lengths, front, progress)
        remaining = math.hypot(points[-1][0] - front[0], points[-1][1] - front[1])
        if progress >= total_length - 1e-9 and remaining <= TURN_PATH_TOLERANCE_M:
            break

        desired = _sample_polyline(points, lengths, progress + lookahead)
        _, steering = _refine_front_target(front, heading, desired, truck)

        # Rear wheels move along the current heading; the steering angle only
        # changes heading through R = L / tan(delta).
        rear_x += math.cos(heading) * TRUCK_MOVE_STEP_M
        rear_y += math.sin(heading) * TRUCK_MOVE_STEP_M
        heading += TRUCK_MOVE_STEP_M * math.tan(steering) / truck.length
        heading = (heading + math.pi) % (2 * math.pi) - math.pi
        body_x = rear_x + math.cos(heading) * (truck.length / 2.0)
        body_y = rear_y + math.sin(heading) * (truck.length / 2.0)
        if not _truck_inside_boundary(grid.polygon, body_x, body_y, heading,
                                      truck.length / 2.0, truck.width / 2.0):
            break
        states.append((rear_x, rear_y, heading))
    else:
        print(f"[PATHFINDER] Truck {truck.id} bicycle-model path did not converge")
        return []

    return states


def _pose_bucket(heading):
    step = 2.0 * math.pi / POSE_HEADING_BUCKETS
    return int(round(heading / step)) % POSE_HEADING_BUCKETS


def _pose_heading(bucket):
    return bucket * 2.0 * math.pi / POSE_HEADING_BUCKETS


def _pose_allowed(grid, truck, x, y, heading):
    # Allow the truck body to straddle the boundary only while it is literally
    # crossing the gate threshold (within 1 truck-length of ENTRY_POINT).
    if math.hypot(x - ENTRY_POINT[0], y - ENTRY_POINT[1]) <= truck.length:
        return True
    return is_pose_driveable(grid, truck, x, y, heading)


def _mask_pose_allowed(grid, driveable, x, y, heading):
    # 5 m grace zone — enough to cross the gate line, not enough to roam the boundary.
    if math.hypot(x - ENTRY_POINT[0], y - ENTRY_POINT[1]) <= 5.0:
        return True
    r, c = grid.world_to_cell(x, y)
    return bool(driveable[r, c, _pose_bucket(heading)])


def _hybrid_primitive(grid, truck, driveable, x, y, heading, turn):
    """Generate one forward bicycle primitive and its intermediate body poses."""
    heading_step = 2.0 * math.pi / POSE_HEADING_BUCKETS
    travel = grid.cell_size if turn == 0 else max(grid.cell_size, truck.turn_radius * heading_step)
    samples = max(1, int(math.ceil(travel / TRUCK_MOVE_STEP_M)))
    ds = travel / samples
    d_heading = 0.0 if turn == 0 else turn * heading_step / samples
    poses = []

    for step in range(samples):
        mid_heading = heading + d_heading / 2.0
        x += math.cos(mid_heading) * ds
        y += math.sin(mid_heading) * ds
        heading = (heading + d_heading + math.pi) % (2.0 * math.pi) - math.pi
        if step % 3 == 0 or step == samples - 1:
            if not _mask_pose_allowed(grid, driveable, x, y, heading):
                return None
        poses.append((x, y, heading))
    return poses


def _route_exactly_driveable(grid, truck, path):
    half_len = truck.length / 2.0
    for rear_x, rear_y, heading in path:
        body_x = rear_x + math.cos(heading) * half_len
        body_y = rear_y + math.sin(heading) * half_len
        if not _pose_allowed(grid, truck, body_x, body_y, heading):
            return False
    return True


def hybrid_astar_to_staging(grid, truck, staging_pose, blocked_cells=frozenset(), driveable=None):
    """Plan forward bicycle arcs to an outward-facing staging pose."""
    if driveable is None:
        driveable = make_driveable_mask(grid, truck)

    start_rc = grid.world_to_cell(*truck.pos)
    start = (start_rc[0], start_rc[1], _pose_bucket(truck.heading))
    open_heap = [(0.0, 0.0, start)]
    g_cost = {start: 0.0}
    came_from = {}
    actions = {}
    state_pose = {start: (truck.pos[0], truck.pos[1], truck.heading)}
    terminal = None

    while open_heap:
        _, g, state = heapq.heappop(open_heap)
        if g > g_cost.get(state, float('inf')):
            continue
        r, c, hb = state
        x, y, heading = state_pose[state]

        heading_error = abs(_angle_diff_signed(staging_pose.heading, heading))
        if (math.hypot(x - staging_pose.x, y - staging_pose.y) <= 1.5 * grid.cell_size
                and heading_error <= 2.0 * math.pi / POSE_HEADING_BUCKETS):
            terminal = state
            break

        for turn in (0, -1, 1):
            primitive = _hybrid_primitive(grid, truck, driveable, x, y, heading, turn)
            if not primitive:
                continue
            nx, ny, nh = primitive[-1]
            nr, nc = grid.world_to_cell(nx, ny)
            next_state = (nr, nc, _pose_bucket(nh))
            if next_state == state:
                continue
            if (nr, nc) in blocked_cells:  # hard block — another truck is here, never enter
                continue
            turn_cost = 0.0 if turn == 0 else 0.35 * truck.turn_radius * (2.0 * math.pi / POSE_HEADING_BUCKETS)
            new_g = g + len(primitive) * TRUCK_MOVE_STEP_M + turn_cost
            if new_g >= g_cost.get(next_state, float('inf')):
                continue
            g_cost[next_state] = new_g
            came_from[next_state] = state
            actions[next_state] = primitive
            state_pose[next_state] = (nx, ny, nh)
            h = math.hypot(nx - staging_pose.x, ny - staging_pose.y)
            h += truck.turn_radius * abs(_angle_diff_signed(staging_pose.heading, nh))
            heapq.heappush(open_heap, (new_g + h, new_g, next_state))

    if terminal is None:
        return []

    segments = []
    current = terminal
    while current in came_from:
        segments.append(actions[current])
        current = came_from[current]
    segments.reverse()

    body_poses = [pose for segment in segments for pose in segment]
    connector_start = body_poses[-1] if body_poses else state_pose[start]
    dx = staging_pose.x - connector_start[0]
    dy = staging_pose.y - connector_start[1]
    distance = math.hypot(dx, dy)
    connector_steps = max(1, int(math.ceil(distance / TRUCK_MOVE_STEP_M)))
    heading_delta = _angle_diff_signed(staging_pose.heading, connector_start[2])
    for index in range(1, connector_steps + 1):
        ratio = index / connector_steps
        x = connector_start[0] + ratio * dx
        y = connector_start[1] + ratio * dy
        heading = connector_start[2] + ratio * heading_delta
        if index % 3 == 1 or index == connector_steps:
            if not _mask_pose_allowed(grid, driveable, x, y, heading):
                return []
        body_poses.append((x, y, heading))
    half_len = truck.length / 2.0
    return [
        (x - math.cos(heading) * half_len,
         y - math.sin(heading) * half_len,
         heading)
        for x, y, heading in body_poses
    ]


def hybrid_astar_to_staging_st(grid, truck, staging_pose, blocked_cells=frozenset(),
                                driveable=None, constraints=frozenset(), max_time=250):
    """
    Space-time variant of hybrid_astar_to_staging.
    State: (r, c, heading_bucket, t).
    constraints: set of (r, c, t) — forbidden positions at specific timesteps (CBS).
    Wait action added so the planner can hold position to let another truck pass.
    Wait poses are included in the output path so the cell-sequence is time-accurate.
    """
    if driveable is None:
        driveable = make_driveable_mask(grid, truck)

    start_rc = grid.world_to_cell(*truck.pos)
    start_hb = _pose_bucket(truck.heading)
    start    = (start_rc[0], start_rc[1], start_hb, 0)

    open_heap  = [(0.0, 0.0, start_rc[0], start_rc[1], start_hb, 0)]
    g_cost     = {start: 0.0}
    came_from  = {}
    actions    = {}          # state → primitive poses list (move) or None (wait)
    state_pose = {start: (truck.pos[0], truck.pos[1], truck.heading)}
    terminal   = None

    while open_heap:
        _, g, r, c, hb, t = heapq.heappop(open_heap)
        state = (r, c, hb, t)

        if g > g_cost.get(state, float('inf')):
            continue
        if t > max_time:
            continue

        x, y, heading = state_pose[state]

        # Terminal: close enough to staging pose with correct heading
        heading_error = abs(_angle_diff_signed(staging_pose.heading, heading))
        if (math.hypot(x - staging_pose.x, y - staging_pose.y) <= 1.5 * grid.cell_size
                and heading_error <= 2.0 * math.pi / POSE_HEADING_BUCKETS):
            terminal = state
            break

        nt = t + 1
        if nt > max_time:
            continue

        # ── WAIT ─────────────────────────────────────────────────────────────────
        if (r, c, nt) not in constraints:
            wait_state = (r, c, hb, nt)
            wait_g = g + ASTAR_WAIT_COST
            if wait_g < g_cost.get(wait_state, float('inf')):
                g_cost[wait_state]     = wait_g
                came_from[wait_state]  = state
                actions[wait_state]    = None          # sentinel: wait
                state_pose[wait_state] = (x, y, heading)
                h = (math.hypot(x - staging_pose.x, y - staging_pose.y)
                     + truck.turn_radius * abs(_angle_diff_signed(staging_pose.heading, heading)))
                heapq.heappush(open_heap, (wait_g + h, wait_g, r, c, hb, nt))

        # ── MOVE (3 bicycle primitives) ───────────────────────────────────────────
        for turn in (0, -1, 1):
            primitive = _hybrid_primitive(grid, truck, driveable, x, y, heading, turn)
            if not primitive:
                continue
            nx, ny, nh = primitive[-1]
            nr, nc     = grid.world_to_cell(nx, ny)
            nhb        = _pose_bucket(nh)
            next_state = (nr, nc, nhb, nt)

            if (nr, nc, nhb) == (r, c, hb):   # primitive went nowhere
                continue
            if (nr, nc) in blocked_cells:
                continue
            if (nr, nc, nt) in constraints:    # CBS hard constraint
                continue

            turn_cost = (0.0 if turn == 0
                         else 0.35 * truck.turn_radius * (2.0 * math.pi / POSE_HEADING_BUCKETS))
            new_g = g + len(primitive) * TRUCK_MOVE_STEP_M + turn_cost

            if new_g < g_cost.get(next_state, float('inf')):
                g_cost[next_state]     = new_g
                came_from[next_state]  = state
                actions[next_state]    = primitive
                state_pose[next_state] = (nx, ny, nh)
                h = (math.hypot(nx - staging_pose.x, ny - staging_pose.y)
                     + truck.turn_radius * abs(_angle_diff_signed(staging_pose.heading, nh)))
                heapq.heappush(open_heap, (new_g + h, new_g, nr, nc, nhb, nt))

    if terminal is None:
        return []

    # ── PATH RECONSTRUCTION ───────────────────────────────────────────────────────
    # Wait actions produce a repeated body pose so that _path_cells() gives the
    # correct time-indexed cell sequence for CBS conflict detection.
    segments = []
    current  = terminal
    while current in came_from:
        parent = came_from[current]
        action = actions[current]
        if action is not None:
            segments.append(action)                        # bicycle-arc poses
        else:
            segments.append([state_pose[parent]])          # one wait pose (repeated cell)
        current = parent
    segments.reverse()

    body_poses = [pose for seg in segments for pose in seg]

    # Connector: linear interpolation to exact staging pose (same as original)
    connector_start = body_poses[-1] if body_poses else state_pose[start]
    dx = staging_pose.x - connector_start[0]
    dy = staging_pose.y - connector_start[1]
    distance        = math.hypot(dx, dy)
    connector_steps = max(1, int(math.ceil(distance / TRUCK_MOVE_STEP_M)))
    heading_delta   = _angle_diff_signed(staging_pose.heading, connector_start[2])
    for index in range(1, connector_steps + 1):
        ratio   = index / connector_steps
        cx      = connector_start[0] + ratio * dx
        cy      = connector_start[1] + ratio * dy
        ch      = connector_start[2] + ratio * heading_delta
        if index % 3 == 1 or index == connector_steps:
            if not _mask_pose_allowed(grid, driveable, cx, cy, ch):
                return []
        body_poses.append((cx, cy, ch))

    half_len = truck.length / 2.0
    return [
        (x - math.cos(heading) * half_len,
         y - math.sin(heading) * half_len,
         heading)
        for x, y, heading in body_poses
    ]


def plan_staging_paths(grid, assignments, locked_paths=None, max_time=250,
                       ignore_path_reserved=False):
    """
    CBS-based staging path planner using the bicycle-model A* (hybrid_astar_to_staging_st).
    Used for both dump paths (navigating trucks) and exit paths (exiting trucks).

    ignore_path_reserved: pass True for exit trucks so they can cross dump corridors
      spatially — CBS time constraints handle temporal separation instead.
    max_time: cap on the space-time A* search depth per candidate.
    """
    if not assignments:
        return {}, {}

    # ── LOCKED SPACE-TIME CONSTRAINTS ────────────────────────────────────────────
    # locked_paths values are (cells, tail_ticks) tuples — see plan_paths_cbs.
    locked_st: set = set()
    if locked_paths:
        for path_entry in locked_paths.values():
            if isinstance(path_entry, tuple):
                raw_path, tail_ticks = path_entry
            else:
                raw_path, tail_ticks = path_entry, LOCKED_PATH_HORIZON
            cells = _path_cells(grid, raw_path)
            for t, (r, c) in enumerate(cells):
                if t > LOCKED_PATH_HORIZON:
                    break
                locked_st.add((r, c, t))
            if cells and tail_ticks > 0:
                fr, fc = cells[min(len(cells) - 1, LOCKED_PATH_HORIZON)]
                end_t = min(len(cells) + tail_ticks, LOCKED_PATH_HORIZON + 1)
                for t in range(len(cells), end_t):
                    locked_st.add((fr, fc, t))

    # ── PER-AGENT SETUP ───────────────────────────────────────────────────────────
    mask_cache = {}
    agent_info = {}
    for truck, dump_target in assignments:
        mask_key = (truck.truck_class, ignore_path_reserved)
        if mask_key not in mask_cache:
            mask_cache[mask_key] = make_driveable_mask(
                grid, truck, ignore_path_reserved=ignore_path_reserved)
        driveable  = mask_cache[mask_key]
        truck_cell = _truck_front_cell(grid, truck)

        # Extended bulldozer: force start + 2-cell radius driveable
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if abs(dr) + abs(dc) > 2:
                    continue
                nr, nc = truck_cell[0] + dr, truck_cell[1] + dc
                if 0 <= nr < grid.rows and 0 <= nc < grid.cols:
                    if grid.state[nr, nc] != grid_map.CellState.BOUNDARY:
                        driveable[nr, nc, :] = True

        candidates = score_staging_candidates(grid, truck, dump_target)
        agent_info[truck.id] = {
            'truck':      truck,
            'driveable':  driveable,
            'candidates': candidates,
        }

    # ── LOW-LEVEL PLANNER ─────────────────────────────────────────────────────────
    def _plan_one(aid, cbs_constraints, preferred_pose=None):
        """Plan one truck with combined CBS + locked constraints.
        Tries preferred_pose first, then all scored candidates."""
        info = agent_info[aid]
        hard = cbs_constraints | locked_st
        ordered = (([preferred_pose] if preferred_pose else []) +
                   [c for c in info['candidates'] if c is not preferred_pose])
        for candidate in ordered:
            path = hybrid_astar_to_staging_st(
                grid, info['truck'], candidate,
                driveable=info['driveable'],
                constraints=hard,
                max_time=max_time,
            )
            if path:
                return path, candidate
        print(f"[STAGING] Truck {info['truck'].id}: no reachable staging pose")
        return [], None

    # ── SINGLE-AGENT SHORT-CIRCUIT ────────────────────────────────────────────────
    if len(agent_info) == 1:
        aid        = next(iter(agent_info))
        path, pose = _plan_one(aid, set())
        return {aid: path}, {aid: pose}

    # ── CBS HIGH-LEVEL SEARCH ─────────────────────────────────────────────────────
    init_cons    = {aid: set() for aid in agent_info}
    init_paths   = {}
    init_staging = {}
    for aid in agent_info:
        p, sp = _plan_one(aid, init_cons[aid])
        init_paths[aid]   = p
        init_staging[aid] = sp

    init_cost  = sum(len(p) for p in init_paths.values())
    truck_map  = {aid: agent_info[aid]['truck'] for aid in agent_info}
    _nid       = 0
    heap       = [(init_cost, _nid, init_cons, init_paths, init_staging)]
    MAX_NODES  = 100   # fewer than exit CBS — hybrid A* is more expensive per call

    for _ in range(MAX_NODES):
        if not heap:
            break
        cost, _, constraints, paths, staging = heapq.heappop(heap)

        # Convert smooth paths → cell sequences for conflict detection
        cell_paths = {aid: _path_cells(grid, p) for aid, p in paths.items() if p}
        conflict   = _detect_first_conflict(cell_paths, truck_map=truck_map, grid=grid)

        if conflict is None:
            return paths, staging   # conflict-free solution found

        if conflict[0] == 'vertex':
            _, ai, aj, ri, ci, rj, cj, t = conflict
            branches = [(ai, ri, ci, t), (aj, rj, cj, t)]
        else:
            _, ai, aj, r1, c1, r2, c2, t = conflict
            branches = [(ai, r1, c1, t), (aj, r2, c2, t)]

        for branch_agent, r, c, t in branches:
            new_cons  = {k: set(v) for k, v in constraints.items()}
            new_cons[branch_agent].add((r, c, t))
            new_paths   = dict(paths)
            new_staging = dict(staging)
            bp, bsp = _plan_one(branch_agent, new_cons[branch_agent],
                                preferred_pose=staging.get(branch_agent))
            new_paths[branch_agent]   = bp
            new_staging[branch_agent] = bsp
            new_cost = sum(len(p) for p in new_paths.values())
            _nid += 1
            heapq.heappush(heap, (new_cost, _nid, new_cons, new_paths, new_staging))

    # CBS exhausted — return best found
    if heap:
        _, _, _, best_paths, best_staging = heap[0]
        return best_paths, best_staging
    return init_paths, init_staging

def _turn_cost(prev_dr, prev_dc, dr, dc, turn_radius_cells):
    # Penalise direction changes to discourage zig-zag paths.
    # No penalty for the first step (prev is None) or for going straight (dot product == 1).
    # A turn costs turn_radius * 0.3 to approximate the real-world arc length penalty.
    if prev_dr is None: return 0.0
    dot = prev_dr * dr + prev_dc * dc  # 1 = straight ahead, 0 = 90° turn, -1 = U-turn
    return 0.0 if dot == 1 else turn_radius_cells * 0.3

def astar(driveable, grid, start_rc, goal_rc, truck, blocked_cells=frozenset(), stop_dist_cells=0.0):
    turn_radius_cells = truck.turn_radius / grid.cell_size
    rows, cols        = driveable.shape[:2]

    start_hb    = _bucket_from_heading(truck.heading)
    start_state = (start_rc[0], start_rc[1], start_hb)

    open_heap = [(0.0, 0.0, start_rc[0], start_rc[1], start_hb)]
    came_from = {}
    g_cost    = {start_state: 0.0}

    closest_state = start_state
    min_dist_to_target = math.hypot(start_rc[0] - goal_rc[0], start_rc[1] - goal_rc[1])

    while open_heap:
        f, g, r, c, hb = heapq.heappop(open_heap)
        state = (r, c, hb)

        dist_to_target = math.hypot(r - goal_rc[0], c - goal_rc[1])
        if dist_to_target < min_dist_to_target:
            min_dist_to_target = dist_to_target
            closest_state = state

        if dist_to_target <= stop_dist_cells:
            closest_state = state
            break

        if g > g_cost.get(state, float('inf')):
            continue

        next_states = []
        for turn in (-1, 0, 1):
            next_hb = (hb + turn) % 8
            dr, dc = _BUCKET_TO_DIR[next_hb]
            turn_cost = 0.0 if turn == 0 else 0.35 * turn_radius_cells
            next_states.append((r + dr, c + dc, next_hb, math.hypot(dr, dc) + turn_cost))

        for nr, nc, nhb, action_cost in next_states:
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            if not driveable[nr, nc, nhb]:
                continue

            if (nr, nc) in blocked_cells:  # hard block — another truck is here, never enter
                continue
            new_g = g + action_cost
            next_state = (nr, nc, nhb)

            if new_g < g_cost.get(next_state, float('inf')):
                g_cost[next_state] = new_g
                h = abs(nr - goal_rc[0]) + abs(nc - goal_rc[1])
                heapq.heappush(open_heap, (new_g + h, new_g, nr, nc, nhb))
                came_from[next_state] = state

    path, cur = [], closest_state
    while cur in came_from:
        path.append((cur[0], cur[1], _heading_for_bucket(cur[2])))
        cur = came_from[cur]
    path.reverse()

    return path
def plan_paths(grid, assignments, existing_paths=None):
    if not assignments: return {}  # nothing to plan

    current_truck_cells = set()  # cells occupied by trucks — hard blocks for subsequent planners
    if existing_paths is not None:
        for p in existing_paths.values():
            cells = _path_cells(grid, p)
            if cells: current_truck_cells.add(cells[0])

    for truck, _ in assignments:
        current_truck_cells.add(_truck_front_cell(grid, truck))

    mask_cache, paths = {}, {}  # cache driveable masks per truck class (expensive to recompute)
    entry_rc = grid.world_to_cell(*ENTRY_POINT)  # precompute the entry gate cell once

    for truck, target_rc in assignments:
        if truck.truck_class not in mask_cache:
            mask_cache[truck.truck_class] = make_driveable_mask(grid, truck)  # build (and cache) terrain mask for this truck type

        driveable  = mask_cache[truck.truck_class]        # the [row, col, bucket] boolean mask for this truck
        truck_cell = _truck_front_cell(grid, truck)       # current front-centre cell of the truck
        obstacles  = current_truck_cells - {truck_cell}   # other trucks' cells (this truck excluded)

        # ─── BULLDOZER MODE: FORCE START CELL TO BE VALID ───
        # If the truck is standing in a restricted zone, this forces the pathfinder
        # to let it drive out, rather than permanently freezing it.
        if 0 <= truck_cell[0] < grid.rows and 0 <= truck_cell[1] < grid.cols:
            driveable[truck_cell[0], truck_cell[1], :] = True  # override all directions at start cell to driveable
        # ────────────────────────────────────────────────────

        # ─── CALCULATE RADIUS STOPPING DISTANCE ───
        if target_rc == entry_rc:
            stop_dist_cells = 0.0  # plan all the way to the entry cell — no early stop
        else:
            # Safe distance for dumping to avoid "Inside Obstacle" failures
            r_pile_m = (1.2* truck.payload_volume_m3 / (math.pi * _TAN_REPOSE)) ** (1/3)  # radius where sandpile height equals the repose limit
            d_clearance = max(0.0, r_pile_m - (DRIVE_CLEARANCE_M / _TAN_REPOSE))           # extra clearance so the pile never buries the truck
            safe_dist_m = d_clearance + truck.length/2                                       # front centre is one full truck length ahead of the rear
            stop_dist_cells = safe_dist_m / grid.cell_size                                  # convert metres to cells

        coarse_path = astar(driveable, grid, truck_cell, target_rc, truck,
                            blocked_cells=frozenset(obstacles), stop_dist_cells=stop_dist_cells)
        path = interpolate_path_to_truck_states(grid, truck, coarse_path)
        paths[truck.id] = path

        if coarse_path:
            current_truck_cells.add(_state_cell(coarse_path[0]))

    return paths

def astar_st(driveable, grid, start_rc, goal_rc, truck, constraints,
             stop_dist_cells=0.0, max_time=250):
    """
    Space-time A* for CBS.
    State: (r, c, hb, t) — arrival heading bucket included so turn costs are
    always computed against the actual arrival direction, eliminating zigzag paths.
    8-connected movement via heading bucket transitions (±1 bucket per step),
    matching the same scheme used by astar().
    constraints: set of (r, c, t) tuples — forbidden positions at specific timesteps.
    Returns list of (r, c) path nodes; consecutive duplicates mean "wait in place".
    """
    turn_radius_cells = truck.turn_radius / grid.cell_size
    rows, cols = driveable.shape[:2]

    start_hb    = _bucket_from_heading(truck.heading)
    start_state = (start_rc[0], start_rc[1], start_hb, 0)

    open_heap = [(0.0, 0.0, start_rc[0], start_rc[1], start_hb, 0)]
    came_from = {}
    g_cost    = {start_state: 0.0}

    closest_node = start_state
    min_dist = math.hypot(start_rc[0] - goal_rc[0], start_rc[1] - goal_rc[1])

    while open_heap:
        f, g, r, c, hb, t = heapq.heappop(open_heap)
        state = (r, c, hb, t)

        if t > max_time:
            continue

        dist = math.hypot(r - goal_rc[0], c - goal_rc[1])
        if dist < min_dist:
            min_dist = dist
            closest_node = state

        if dist <= stop_dist_cells:
            closest_node = state
            break

        if g > g_cost.get(state, float('inf')):
            continue

        nt = t + 1

        # ── WAIT ACTION ──────────────────────────────────────────────────────────
        if nt <= max_time and (r, c, nt) not in constraints:
            wait_state = (r, c, hb, nt)
            wait_g = g + ASTAR_WAIT_COST
            if wait_g < g_cost.get(wait_state, float('inf')):
                g_cost[wait_state] = wait_g
                abs_dr = abs(r - goal_rc[0])
                abs_dc = abs(c - goal_rc[1])
                h = max(abs_dr, abs_dc) + (math.sqrt(2) - 1) * min(abs_dr, abs_dc)
                heapq.heappush(open_heap, (wait_g + h, wait_g, r, c, hb, nt))
                came_from[wait_state] = state

        # ── MOVE ACTIONS (8-connected via heading buckets) ───────────────────────
        for turn in (-1, 0, 1):
            next_hb    = (hb + turn) % 8
            dr, dc     = _BUCKET_TO_DIR[next_hb]
            nr, nc     = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if not driveable[nr, nc, next_hb]:
                continue
            if nt > max_time or (nr, nc, nt) in constraints:
                continue
            turn_cost  = 0.0 if turn == 0 else 0.35 * turn_radius_cells
            move_cost  = math.hypot(dr, dc)   # 1.0 orthogonal, √2 diagonal
            new_g      = g + move_cost + turn_cost
            next_state = (nr, nc, next_hb, nt)
            if new_g < g_cost.get(next_state, float('inf')):
                g_cost[next_state] = new_g
                abs_dr = abs(nr - goal_rc[0])
                abs_dc = abs(nc - goal_rc[1])
                h = max(abs_dr, abs_dc) + (math.sqrt(2) - 1) * min(abs_dr, abs_dc)
                heapq.heappush(open_heap, (new_g + h, new_g, nr, nc, next_hb, nt))
                came_from[next_state] = state

    path, cur = [], closest_node
    while cur in came_from:
        path.append((cur[0], cur[1]))
        cur = came_from[cur]
    path.reverse()
    return path


def _rect_overlap_2d(cx1, cy1, h1, hl1, hw1, cx2, cy2, h2, hl2, hw2):
    """Separating-axis theorem overlap test for two oriented rectangles in world space.
    Returns True when the rectangles overlap (edge-touching counts as overlap)."""
    cos1, sin1 = math.cos(h1), math.sin(h1)
    cos2, sin2 = math.cos(h2), math.sin(h2)
    axes = ((cos1, sin1), (-sin1, cos1), (cos2, sin2), (-sin2, cos2))
    c1 = [(cx1 + s * hl1 * cos1 + t * hw1 * (-sin1),
           cy1 + s * hl1 * sin1 + t * hw1 * cos1)
          for s in (-1, 1) for t in (-1, 1)]
    c2 = [(cx2 + s * hl2 * cos2 + t * hw2 * (-sin2),
           cy2 + s * hl2 * sin2 + t * hw2 * cos2)
          for s in (-1, 1) for t in (-1, 1)]
    for ax, ay in axes:
        p1 = [x * ax + y * ay for x, y in c1]
        p2 = [x * ax + y * ay for x, y in c2]
        if max(p1) < min(p2) or max(p2) < min(p1):
            return False
    return True


def _infer_heading_at_t(path, t):
    """Approximate truck heading at timestep t from the cell-path direction of travel.
    Heading convention: atan2(Δrow, Δcol) matches the grid's _BUCKET_TO_HEADING mapping."""
    if not path:
        return 0.0
    end = len(path) - 1
    t0  = min(t, end)
    # Clamp both neighbour candidates to [0, end] to guard against t >> path length.
    for t1 in (min(t + 1, end), min(max(t - 1, 0), end)):
        if t1 == t0:
            continue
        dr = path[t1][0] - path[t0][0] if t1 > t0 else path[t0][0] - path[t1][0]
        dc = path[t1][1] - path[t0][1] if t1 > t0 else path[t0][1] - path[t1][1]
        if dr or dc:
            return math.atan2(dr, dc)
    return 0.0


def _detect_first_conflict(paths_dict, truck_map=None, grid=None):
    """
    Scan all agent paths for the earliest vertex or edge conflict.
    Paths are padded: after reaching the end, an agent stays at its final cell.

    When truck_map ({aid: Truck}) and grid are provided, vertex conflicts use an
    exact SAT (separating-axis theorem) rectangle overlap test on each truck's
    oriented bounding box — heading inferred from consecutive path cells.  This
    avoids the false positives produced by the old bounding-circle approach when
    trucks travel in adjacent, non-intersecting corridors.

    Returns:
      ('vertex', aid_i, aid_j, ri, ci, rj, cj, t)  — truck rectangles overlap at t;
                                                       (ri,ci) is aid_i's cell,
                                                       (rj,cj) is aid_j's cell.
      ('edge',   aid_i, aid_j, r1,c1, r2,c2, t)    — agents swap cells between t-1 and t.
      None if no conflict found.
    """
    agent_ids = list(paths_dict.keys())
    if len(agent_ids) < 2:
        return None

    max_t = max((len(p) for p in paths_dict.values()), default=0)
    if max_t == 0:
        return None

    def pos_at(path, t):
        if not path:
            return None
        return _state_cell(path[min(t, len(path) - 1)])

    entry_rc_local = grid.world_to_cell(*ENTRY_POINT) if grid is not None else None

    for t in range(max_t + 1):

        # ── VERTEX CONFLICT CHECK (exact oriented-rectangle SAT) ─────────────────
        # Infer each truck's heading from consecutive path cells, convert the cell
        # centre to world coordinates, then run a separating-axis test on the two
        # oriented rectangles.  Bounding-circle checks fire whenever trucks are
        # within one truck-length of each other — even in separate corridor lanes —
        # because length >> width.  SAT uses the actual footprint, so parallel-lane
        # travel never triggers a false conflict.
        positions = {}
        for aid in agent_ids:
            p = pos_at(paths_dict[aid], t)
            if p is not None:
                positions[aid] = p

        aids_present = list(positions.keys())
        for i in range(len(aids_present)):
            for j in range(i + 1, len(aids_present)):
                ai, aj = aids_present[i], aids_present[j]
                pi, pj = positions[ai], positions[aj]

                if truck_map is not None and grid is not None:
                    ti = truck_map.get(ai)
                    tj = truck_map.get(aj)
                    if ti is not None and tj is not None:
                        hi = _infer_heading_at_t(paths_dict[ai], t)
                        hj = _infer_heading_at_t(paths_dict[aj], t)
                        wxi, wyi = grid.cell_to_world(pi[0], pi[1])
                        wxj, wyj = grid.cell_to_world(pj[0], pj[1])
                        print(ai, pi, math.degrees(hi))
                        print(aj, pj, math.degrees(hj))
                        print(ti.length, ti.width)
                        print(pi)
                        print(grid.cell_to_world(pi[0], pi[1]))
                        print("A cell", pi)
                        print("A world", wxi, wyi)

                        print("B cell", pj)
                        print("B world", wxj, wyj)

                        print("headings",
                            math.degrees(hi),
                            math.degrees(hj))
                        dist_world = math.hypot(wxi - wxj, wyi - wyj)
                        print(f"world dist: {dist_world:.2f}  half-lengths: {ti.length/2:.2f} {tj.length/2:.2f}  sum: {ti.length/2 + tj.length/2:.2f}")
                        if not _rect_overlap_2d(wxi, wyi, hi, ti.length / 2, ti.width / 2,
                                                wxj, wyj, hj, tj.length / 2, tj.width / 2):
                            
                            continue
                    else:
                        if pi != pj:
                            continue
                else:
                    if pi != pj:
                        continue

                # Skip conflicts inside the entry corridor — trucks funnel through
                # a single exit point so proximity there is expected and unresolvable.
                if entry_rc_local is not None:
                    di = math.hypot(pi[0] - entry_rc_local[0], pi[1] - entry_rc_local[1])
                    dj = math.hypot(pj[0] - entry_rc_local[0], pj[1] - entry_rc_local[1])
                    if di <= ENTRY_CORRIDOR_CELLS or dj <= ENTRY_CORRIDOR_CELLS:
                        continue
                print(f"[CONFLICT] VERTEX: trucks {ai} and {aj} rectangles overlap at t={t} "
                      f"— T{ai}@({pi[0]},{pi[1]}) T{aj}@({pj[0]},{pj[1]})")
                return ('vertex', ai, aj, pi[0], pi[1], pj[0], pj[1], t)

        # ── EDGE (SWAP) CONFLICT CHECK ────────────────────────────────────────────
        # Two trucks passing through each other is physically impossible at any size.
        # Guards:
        #   prev_i != prev_j — require trucks at DIFFERENT cells before the swap.
        #     Without this, two trucks both parked at entry_rc (path-padded) satisfy
        #     prev_i==curr_j and prev_j==curr_i trivially, causing an infinite CBS loop.
        #   entry corridor skip — trucks converging at the single exit funnel are
        #     expected to be close there; flagging it would prevent any exit path.
        #   SAT physical check — confirm the truck rectangles at both cells actually
        #     overlap at the midpoint of the swap; filters cell-model artefacts where
        #     discrete paths cross but the physical bodies never touch.
        if t >= 1:
            n = len(agent_ids)
            for i in range(n):
                for j in range(i + 1, n):
                    ai, aj = agent_ids[i], agent_ids[j]
                    prev_i = pos_at(paths_dict[ai], t - 1)
                    curr_i = pos_at(paths_dict[ai], t)
                    prev_j = pos_at(paths_dict[aj], t - 1)
                    curr_j = pos_at(paths_dict[aj], t)
                    if prev_i == curr_j and prev_j == curr_i and prev_i != prev_j:
                        # Skip swaps at the entry corridor (funnel point)
                        if entry_rc_local is not None:
                            di = math.hypot(curr_i[0] - entry_rc_local[0], curr_i[1] - entry_rc_local[1])
                            dj = math.hypot(curr_j[0] - entry_rc_local[0], curr_j[1] - entry_rc_local[1])
                            if di <= ENTRY_CORRIDOR_CELLS or dj <= ENTRY_CORRIDOR_CELLS:
                                continue
                        # SAT check at the swap midpoint to reject geometric artefacts
                        if truck_map is not None and grid is not None:
                            ti = truck_map.get(ai)
                            tj = truck_map.get(aj)
                            if ti is not None and tj is not None:
                                hi = math.atan2(curr_i[0] - prev_i[0], curr_i[1] - prev_i[1]) if curr_i != prev_i else 0.0
                                hj = math.atan2(curr_j[0] - prev_j[0], curr_j[1] - prev_j[1]) if curr_j != prev_j else 0.0
                                wpi = grid.cell_to_world(prev_i[0], prev_i[1])
                                wci = grid.cell_to_world(curr_i[0], curr_i[1])
                                wpj = grid.cell_to_world(prev_j[0], prev_j[1])
                                wcj = grid.cell_to_world(curr_j[0], curr_j[1])
                                mid_ix = (wpi[0] + wci[0]) / 2
                                mid_iy = (wpi[1] + wci[1]) / 2
                                mid_jx = (wpj[0] + wcj[0]) / 2
                                mid_jy = (wpj[1] + wcj[1]) / 2
                                if not _rect_overlap_2d(mid_ix, mid_iy, hi, ti.length / 2, ti.width / 2,
                                                        mid_jx, mid_jy, hj, tj.length / 2, tj.width / 2):
                                    continue
                        print(f"[CONFLICT] EDGE SWAP: trucks {ai} and {aj} swapped at t={t} "
                              f"— {prev_i}↔{prev_j}")
                        return ('edge', ai, aj,
                                curr_i[0], curr_i[1],
                                curr_j[0], curr_j[1],
                                t)

    return None


def plan_paths_cbs(grid, assignments, locked_paths=None):
    """
    Conflict-Based Search (CBS) for multi-agent path planning.

    High level  — constraint tree; split on detected conflicts.
    Low level   — space-time A* (astar_st) with hard (r,c,t) forbidden states.
                  Trucks wait in place or detour; conflicts are never permitted.

    locked_paths: dict {truck_id: [(r,c), ...]} — remaining waypoints of trucks
      already moving.  Their cells are added as hard (r,c,t) constraints so new
      paths cannot occupy those cells at those exact timesteps.
    """
    if not assignments:  # nothing to plan — return immediately
        return {}

    entry_rc   = grid.world_to_cell(*ENTRY_POINT)  # entry gate cell, precomputed once
    mask_cache = {}  # cache driveable masks per truck class — expensive to rebuild

    # ── BUILD LOCKED SPACE-TIME CONSTRAINTS ─────────────────────────────────────
    # locked_paths values are (cells, tail_ticks) tuples where tail_ticks is how
    # many extra timesteps to hold the final cell after the path ends.
    # tail_ticks=0 means the cell is free immediately (e.g. exiting trucks).
    # tail_ticks>0 covers the dump duration so planners don't route through a
    # truck that is still depositing, but release the cell once it drives away.
    locked_st_constraints: set = set()
    if locked_paths:
        for path_entry in locked_paths.values():
            if isinstance(path_entry, tuple):
                raw_path, tail_ticks = path_entry
            else:
                raw_path, tail_ticks = path_entry, LOCKED_PATH_HORIZON
            cells = _path_cells(grid, raw_path)
            for t, (r, c) in enumerate(cells):
                if t > LOCKED_PATH_HORIZON:
                    break
                locked_st_constraints.add((r, c, t))
            if cells and tail_ticks > 0:
                final_r, final_c = cells[min(len(cells) - 1, LOCKED_PATH_HORIZON)]
                end_t = min(len(cells) + tail_ticks, LOCKED_PATH_HORIZON + 1)
                for t in range(len(cells), end_t):
                    locked_st_constraints.add((final_r, final_c, t))


    # ── PER-AGENT SETUP ───────────────────────────────────────────────────────────
    agent_info = {}
    for truck, target_rc in assignments:
        is_exit = (target_rc == entry_rc)
        # Exit paths ignore PATH_RESERVED: all exits converge to entry and CBS
        # space-time constraints already prevent temporal collisions there.
        # Dump paths must respect PATH_RESERVED corridors of other trucks.
        mask_key = (truck.truck_class, is_exit)
        if mask_key not in mask_cache:
            mask_cache[mask_key] = make_driveable_mask(grid, truck,
                                                       ignore_path_reserved=is_exit)
        driveable  = mask_cache[mask_key]
        truck_cell = _truck_front_cell(grid, truck)

        # Extended bulldozer: force start cell + radius driveable.
        # After dumping, the pile spreads and blocks immediate neighbours —
        # this ensures the truck always has at least one valid first step out.
        # Exit agents use a pile-size radius so they can always escape the mound.
        # BOUNDARY cells are never overridden — they are hard walls in all cases.
        if is_exit:
            r_pile_m    = (1.2 * truck.payload_volume_m3 / (math.pi * _TAN_REPOSE)) ** (1 / 3)
            bulldozer_r = max(2, int(math.ceil(r_pile_m / grid.cell_size)))
        else:
            bulldozer_r = 2
        for dr in range(-bulldozer_r, bulldozer_r + 1):
            for dc in range(-bulldozer_r, bulldozer_r + 1):
                if abs(dr) + abs(dc) > bulldozer_r:
                    continue
                nr, nc = truck_cell[0] + dr, truck_cell[1] + dc
                if 0 <= nr < grid.rows and 0 <= nc < grid.cols:
                    if grid.state[nr, nc] != grid_map.CellState.BOUNDARY:
                        driveable[nr, nc, :] = True

        if target_rc == entry_rc:
            stop_dist_cells = float(ENTRY_CORRIDOR_CELLS)  # stop at the inner corridor edge
        else:
            r_pile_m        = (1.2 * truck.payload_volume_m3 / (math.pi * _TAN_REPOSE)) ** (1 / 3)
            d_clearance     = max(0.0, r_pile_m - (DRIVE_CLEARANCE_M / _TAN_REPOSE))
            safe_dist_m     = d_clearance + truck.length
            stop_dist_cells = safe_dist_m / grid.cell_size

        agent_info[truck.id] = {
            'truck':     truck,
            'start':     truck_cell,
            'target':    target_rc,
            'driveable': driveable,
            'stop_dist': stop_dist_cells,
        }

    # ── HELPERS ──────────────────────────────────────────────────────────────────

    def _at_target_fix(path, aid):
        # astar_st returns [] when the truck is already within stop_dist.
        # Emit a 1-cell hold so set_path() transitions to REVERSING rather than
        # bouncing back to IDLE and replanning forever.
        if not path:
            info = agent_info[aid]
            d = math.hypot(info['start'][0] - info['target'][0],
                           info['start'][1] - info['target'][1])
            if d <= info['stop_dist']:
                return [(info['start'][0], info['start'][1], info['truck'].heading)]
        return path

    def smooth_paths(coarse_paths):
        """Smooth each coarse cell path with the bicycle model.
        If the smoother breaks early (boundary check near polygon walls), splice
        the remaining coarse cells back on so the path always reaches the target.
        Trucks handle (r, c) waypoints just as well as (rear_x, rear_y, heading) ones."""
        out = {}
        for aid, coarse in coarse_paths.items():
            truck   = agent_info[aid]['truck']
            info    = agent_info[aid]
            is_exit = (info['target'] == entry_rc)

            # Exit paths: after dumping, truck.heading points INTO the polygon.
            # Snap heading toward the first coarse cell so the bicycle smoother
            # starts facing the right direction instead of arcing forward first.
            saved_heading = truck.heading
            if is_exit and coarse:
                fx, fy = grid.cell_to_world(coarse[0][0], coarse[0][1])
                dx_h = fx - truck.pos[0]
                dy_h = fy - truck.pos[1]
                if math.hypot(dx_h, dy_h) > 1e-9:
                    truck.heading = math.atan2(dy_h, dx_h)

            smooth = interpolate_path_to_truck_states(grid, truck, coarse)
            truck.heading = saved_heading  # restore physical heading unconditionally

            if not smooth:
                out[aid] = coarse
                continue

            tx, ty = grid.cell_to_world(*info['target'])
            last   = smooth[-1]
            half_l = truck.length / 2.0
            bx     = last[0] + math.cos(last[2]) * half_l
            by     = last[1] + math.sin(last[2]) * half_l

            if math.hypot(bx - tx, by - ty) > grid.cell_size * 4:
                if is_exit:
                    out[aid] = smooth
                else:
                    last_rc = grid.world_to_cell(bx, by)
                    best_d  = float('inf')
                    splice  = len(coarse)
                    for i, wp in enumerate(coarse):
                        d = math.hypot(wp[0] - last_rc[0], wp[1] - last_rc[1])
                        if d < best_d:
                            best_d = d
                            splice = i
                    out[aid] = smooth + list(coarse[splice + 1:])
            else:
                out[aid] = smooth
        return out

    # ── SINGLE-AGENT SHORT-CIRCUIT ────────────────────────────────────────────────
    # One truck: no inter-agent conflict possible — skip CBS tree, one astar_st call.
    if len(agent_info) == 1:
        aid  = next(iter(agent_info))
        info = agent_info[aid]
        path = astar_st(info['driveable'], grid, info['start'], info['target'],
                        info['truck'], locked_st_constraints, info['stop_dist'])
        path = _at_target_fix(path, aid)
        if not path:
            # astar_st exhausted max_time under locked constraints — fall back to spatial astar.
            path = astar(info['driveable'], grid, info['start'], info['target'],
                         info['truck'], blocked_cells=frozenset(),
                         stop_dist_cells=info['stop_dist'])
            path = _at_target_fix(path, aid)
        return smooth_paths({aid: path})

    # ── CBS HIGH-LEVEL SEARCH ─────────────────────────────────────────────────────
    # Root: plan every truck independently (no CBS constraints yet).
    # Each replan is a direct astar_st call — hard (r,c,t) constraints, no exceptions.
    init_constraints = {aid: set() for aid in agent_info}
    init_paths = {}
    for aid in agent_info:
        info = agent_info[aid]
        hard = init_constraints[aid] | locked_st_constraints
        path = astar_st(info['driveable'], grid, info['start'], info['target'],
                        info['truck'], hard, info['stop_dist'])
        init_paths[aid] = _at_target_fix(path, aid)
    init_cost = sum(len(p) for p in init_paths.values())

    _nid = 0
    heap = [(init_cost, _nid, init_constraints, init_paths)]
    MAX_NODES = 100

    for _ in range(MAX_NODES):
        if not heap:
            break
        cost, _, constraints, paths = heapq.heappop(heap)

        truck_map = {aid: agent_info[aid]['truck'] for aid in agent_info}
        conflict = _detect_first_conflict(paths, truck_map=truck_map, grid=grid)

        if conflict is None:
            return smooth_paths(paths)

        if conflict[0] == 'vertex':
            # Each truck is constrained to avoid its own cell at the conflict time,
            # not a shared cell — outer-edge conflicts can involve different cells.
            _, ai, aj, ri, ci, rj, cj, t = conflict
            branches = [(ai, ri, ci, t), (aj, rj, cj, t)]
            print(f"\nVertex conflict detected")
        else:
            _, ai, aj, r1, c1, r2, c2, t = conflict
            branches = [(ai, r1, c1, t), (aj, r2, c2, t)]
            print(f"\nEdge conflict detected")

        for branch_agent, r, c, t in branches:
            new_cons = {k: set(v) for k, v in constraints.items()}
            new_cons[branch_agent].add((r, c, t))
            new_paths = dict(paths)
            info = agent_info[branch_agent]
            hard = new_cons[branch_agent] | locked_st_constraints
            bpath = astar_st(info['driveable'], grid, info['start'], info['target'],
                             info['truck'], hard, info['stop_dist'])
            new_paths[branch_agent] = _at_target_fix(bpath, branch_agent)
            new_cost = sum(len(p) for p in new_paths.values())
            _nid += 1
            heapq.heappush(heap, (new_cost, _nid, new_cons, new_paths))

    if heap:
        _, _, _, best = heap[0]
        return smooth_paths(best)
    return smooth_paths(init_paths)


def escape_and_replan_exit(truck, grid, all_trucks, entry_rc):
    """
    Fallback exit planner for a truck that CBS could not route out.

    Phase 1 — back up until clear of the pile or other trucks.
    Phase 2 — replan from the escape endpoint with a fresh driveable mask.
    Phase 3 — prepend the retreat waypoints to the smoothed A* path.
    """
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
                if grid.state[nr, nc] != grid_map.CellState.BOUNDARY:
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
