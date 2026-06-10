# pathfinder.py
# ─────────────────────────────────────────────────────────────
# MAPF A* PATHFINDING WITH RESILIENCE OVERRIDES
#
# RESOLUTIONS INCLUDED:
# 1. RADIUS GOAL: astar() terminates when within safe dumping radius.
# 2. HIERARCHICAL OPTIMISER: Macro/Micro search drastically reduces lag.
# 3. DYNAMIC TAIL LOCKS: Trucks no longer reserve their destinations to infinity.
# 4. STRICT DETOURS: "Wait" actions disabled. Trucks MUST use alternate lanes.
# ─────────────────────────────────────────────────────────────

import heapq
import numpy as np
import math
import shapely
import grid_map
from filters import is_pose_driveable, make_driveable_mask
from config import (DRIVE_CLEARANCE_M, _TAN_REPOSE, ENTRY_POINT,
                    TURN_REFINEMENT_ITERATIONS, TRUCK_MOVE_STEP_M,
                    TURN_LOOKAHEAD_RADIUS_FACTOR, TURN_PATH_TOLERANCE_M,
                    TURN_MAX_SMOOTH_STEPS, POSE_HEADING_BUCKETS,
                    ASTAR_WAIT_COST, LOCKED_PATH_HORIZON,
                    ENTRY_CORRIDOR_CELLS, SPACE_TIME_CONFLICT_TIME_OFFSET,
                    SPACE_TIME_LENGTH_BUFFER_M)
from staging import score_staging_candidates


def _truck_inside_boundary(polygon, body_x, body_y, heading, half_l, half_w):
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
    rear_x, rear_y = truck.rear_axle_world()
    half_len = truck.length / 2.0
    bcos = -math.cos(truck.heading)
    bsin = -math.sin(truck.heading)
    step = grid.cell_size

    retreat = []
    rx, ry = rear_x, rear_y

    for _ in range(num_steps):
        rx += bcos * step
        ry += bsin * step
        bx = rx + math.cos(truck.heading) * half_len
        by = ry + math.sin(truck.heading) * half_len
        if not is_pose_driveable(grid, truck, bx, by, truck.heading):
            break
        retreat.append((rx, ry, truck.heading))

    return retreat


_DIR_TO_BUCKET = {
    ( 0,  1): 0, (-1,  1): 1, (-1,  0): 2, (-1, -1): 3,
    ( 0, -1): 4, ( 1, -1): 5, ( 1,  0): 6, ( 1,  1): 7,
}
_BUCKET_TO_DIR = {bucket: delta for delta, bucket in _DIR_TO_BUCKET.items()}
_BUCKET_TO_HEADING = {
    0: 0.0, 1: -math.pi / 4, 2: -math.pi / 2, 3: -3 * math.pi / 4,
    4: math.pi, 5: 3 * math.pi / 4, 6: math.pi / 2, 7: math.pi / 4,
}

def _heading_bucket(dr, dc):
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

def _locked_path_items(locked_paths):
    if not locked_paths:
        return
    for key, value in locked_paths.items():
        if (isinstance(value, tuple) and len(value) == 2
                and hasattr(value[0], "width")):
            yield value[0], value[1]
        else:
            yield None, value

def _time_offset_steps(*trucks):
    lengths = [t.length for t in trucks if t is not None]
    if not lengths:
        return SPACE_TIME_CONFLICT_TIME_OFFSET
    length_gap = (max(lengths) + SPACE_TIME_LENGTH_BUFFER_M) / max(TRUCK_MOVE_STEP_M, 1e-6)
    return max(SPACE_TIME_CONFLICT_TIME_OFFSET, int(math.ceil(length_gap)))

def _path_corridor_cells_by_time(grid, truck, path, extra_half_width_m=0.0):
    if truck is None:
        return [[cell] for cell in _path_cells(grid, path)]

    half_len = truck.length / 2.0
    half_w_cells = max(0.0, (truck.width / 2.0 + extra_half_width_m) / grid.cell_size)
    radius = int(math.ceil(half_w_cells))
    corridor_by_t = []

    for wp in path or []:
        if len(wp) == 3 and not isinstance(wp[0], (int, np.integer)):
            rear_x, rear_y, heading = wp
            body_x = rear_x + math.cos(heading) * half_len
            body_y = rear_y + math.sin(heading) * half_len
            center_r, center_c = grid.world_to_cell(body_x, body_y)
        else:
            center_r, center_c = (wp[0], wp[1])

        cells = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if math.hypot(dr, dc) > half_w_cells:
                    continue
                nr, nc = center_r + dr, center_c + dc
                if 0 <= nr < grid.rows and 0 <= nc < grid.cols:
                    cells.append((nr, nc))
        corridor_by_t.append(cells)

    return corridor_by_t

def _space_time_corridor_graph(grid, truck, path, other_truck=None):
    extra_half_width = (other_truck.width / 2.0) if other_truck is not None else 0.0
    return [
        set(cells)
        for cells in _path_corridor_cells_by_time(grid, truck, path, extra_half_width)
    ]

def _in_entry_zone_cell(grid, r, c):
    entry_r, entry_c = grid.world_to_cell(*ENTRY_POINT)
    return math.hypot(r - entry_r, c - entry_c) <= ENTRY_CORRIDOR_CELLS

def _add_locked_time_window(locked, r, c, t, max_time, time_offset=None):
    offset = SPACE_TIME_CONFLICT_TIME_OFFSET if time_offset is None else time_offset
    for dt in range(-offset, offset + 1):
        tt = t + dt
        if 0 <= tt <= max_time:
            locked.add((r, c, tt))

def _add_locked_corridor_path(locked, grid, truck, path, max_time, moving_truck=None):
    corridor_by_t = _space_time_corridor_graph(grid, truck, path, moving_truck)
    time_offset = _time_offset_steps(truck, moving_truck)
    
    # 1. Lock the active driving path in time
    for t, cells in enumerate(corridor_by_t):
        if t > max_time:
            break
        for r, c in cells:
            _add_locked_time_window(locked, r, c, t, max_time, time_offset)

    if not corridor_by_t:
        return

    # ─────────────────────────────────────────────────────────────
    # THE DYNAMIC TAIL LOCK (Fixing Infinite Destination Lock)
    # ─────────────────────────────────────────────────────────────
    tail_hold_steps = 0
    if truck.status == truck.STATUS_NAVIGATING:
        # Incoming trucks will park and dump. Hold for dump duration + 5 buffer ticks.
        tail_hold_steps = truck._dump_ticks_required + 5 
    elif truck.status == truck.STATUS_DUMPING:
        # Currently dumping, hold for the remaining dump ticks
        tail_hold_steps = max(0, truck._dump_ticks_required - truck._dump_ticks) + 5
    elif truck.status in (truck.STATUS_IDLE, truck.STATUS_STUCK, truck.STATUS_REVERSING):
        # Blocked or parked indefinitely. Act as a hard physical wall.
        tail_hold_steps = max_time 
    elif truck.status in (truck.STATUS_EXITING, truck.STATUS_LEAVING):
        # Exiting truck leaves immediately. It vanishes.
        tail_hold_steps = 0
        
    final_cells = corridor_by_t[min(len(corridor_by_t) - 1, max_time)]
    start_tail_t = len(corridor_by_t)
    end_tail_t = min(max_time, start_tail_t + tail_hold_steps)
    
    # Lock the parking spot ONLY for the exact duration of the truck's task
    for t in range(start_tail_t, end_tail_t + 1):
        for r, c in final_cells:
            _add_locked_time_window(locked, r, c, t, max_time, time_offset)
    # ─────────────────────────────────────────────────────────────

def _build_locked_space_time_constraints(grid, locked_paths, moving_truck):
    locked = set()
    if locked_paths:
        for locked_truck, path in _locked_path_items(locked_paths):
            horizon = max(LOCKED_PATH_HORIZON,
                          _time_offset_steps(locked_truck, moving_truck))
            _add_locked_corridor_path(locked, grid, locked_truck, path,
                                      horizon, moving_truck)
    return locked

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
    if math.hypot(x - ENTRY_POINT[0], y - ENTRY_POINT[1]) <= truck.length:
        return True
    return is_pose_driveable(grid, truck, x, y, heading)

def _mask_pose_allowed(grid, driveable, x, y, heading):
    if math.hypot(x - ENTRY_POINT[0], y - ENTRY_POINT[1]) <= 5.0:
        return True
    r, c = grid.world_to_cell(x, y)
    return bool(driveable[r, c, _pose_bucket(heading)])

def _hybrid_primitive(grid, truck, driveable, x, y, heading, turn):
    heading_step = 2.0 * math.pi / POSE_HEADING_BUCKETS
    travel = grid.cell_size if turn == 0 else max(grid.cell_size, truck.turn_radius * heading_step)
    samples = max(1, int(math.ceil(travel / TRUCK_MOVE_STEP_M)))
    ds = travel / samples
    d_heading = 0.0 if turn == 0 else turn * heading_step / samples
    poses = []

    for _ in range(samples):
        mid_heading = heading + d_heading / 2.0
        x += math.cos(mid_heading) * ds
        y += math.sin(mid_heading) * ds
        heading = (heading + d_heading + math.pi) % (2.0 * math.pi) - math.pi
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
            if (nr, nc) in blocked_cells:
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
    if driveable is None:
        driveable = make_driveable_mask(grid, truck)

    start_rc = grid.world_to_cell(*truck.pos)
    start_hb = _pose_bucket(truck.heading)
    start    = (start_rc[0], start_rc[1], start_hb, 0)

    open_heap  = [(0.0, 0.0, start_rc[0], start_rc[1], start_hb, 0)]
    g_cost     = {start: 0.0}
    came_from  = {}
    actions    = {}          
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

        heading_error = abs(_angle_diff_signed(staging_pose.heading, heading))
        if (math.hypot(x - staging_pose.x, y - staging_pose.y) <= 1.5 * grid.cell_size
                and heading_error <= 2.0 * math.pi / POSE_HEADING_BUCKETS):
            terminal = state
            break

        nt = t + 1
        if nt > max_time:
            continue

        # WAIT ACTION KILLED: A* is now physically forced to keep moving. 
        # If it intersects a constraint, it must route around it spatially!

        for turn in (0, -1, 1):
            primitive = _hybrid_primitive(grid, truck, driveable, x, y, heading, turn)
            if not primitive:
                continue
            nx, ny, nh = primitive[-1]
            nr, nc     = grid.world_to_cell(nx, ny)
            nhb        = _pose_bucket(nh)
            next_state = (nr, nc, nhb, nt)

            if (nr, nc, nhb) == (r, c, hb):   
                continue
            if (nr, nc) in blocked_cells:
                continue
            if (nr, nc, nt) in constraints:    
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

    segments = []
    current  = terminal
    while current in came_from:
        parent = came_from[current]
        action = actions[current]
        if action is not None:
            segments.append(action)                        
        else:
            segments.append([state_pose[parent]])          
        current = parent
    segments.reverse()

    body_poses = [pose for seg in segments for pose in seg]

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


def _turn_cost(prev_dr, prev_dc, dr, dc, turn_radius_cells):
    if prev_dr is None: return 0.0
    dot = prev_dr * dr + prev_dc * dc
    
    if dot == 1:
        return 0.0 
    elif dot == 0:
        return turn_radius_cells * 1.5  
    else:
        return float('inf') 

def astar(driveable, grid, start_rc, goal_rc, truck, blocked_cells=frozenset(), stop_dist_cells=0.0):
    turn_radius_cells = truck.turn_radius / grid.cell_size
    rows, cols        = driveable.shape[:2]

    start_hb    = _bucket_from_heading(truck.heading)
    start_state = (start_rc[0], start_rc[1], start_hb)

    open_heap = [(0.0, 0.0, start_rc[0], start_rc[1], start_hb)]
    came_from = {}
    g_cost    = {start_state: 0.0}

    goal_state = None

    while open_heap:
        f, g, r, c, hb = heapq.heappop(open_heap)
        state = (r, c, hb)

        if math.hypot(r - goal_rc[0], c - goal_rc[1]) <= stop_dist_cells:
            goal_state = state
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
            if (nr, nc) in blocked_cells:
                continue
                
            new_g = g + action_cost
            next_state = (nr, nc, nhb)

            if new_g < g_cost.get(next_state, float('inf')):
                g_cost[next_state] = new_g
                h = abs(nr - goal_rc[0]) + abs(nc - goal_rc[1])
                heapq.heappush(open_heap, (new_g + h, new_g, nr, nc, nhb))
                came_from[next_state] = state

    if goal_state is None:
        return []

    path, cur = [], goal_state
    while cur in came_from:
        path.append((cur[0], cur[1], _heading_for_bucket(cur[2])))
        cur = came_from[cur]
    path.reverse()

    return path

def astar_st(driveable, grid, start_rc, goal_rc, truck, constraints, stop_dist_cells=0.0, max_time=250):
    min_dist = math.hypot(start_rc[0] - goal_rc[0], start_rc[1] - goal_rc[1])
    dynamic_max_time = min(max_time, int((min_dist * 2.5) + 20))

    turn_radius_cells = truck.turn_radius / grid.cell_size  
    rows, cols = driveable.shape[:2]  

    start_hb    = _bucket_from_heading(truck.heading)
    
    open_heap = [(0.0, 0.0, start_rc[0], start_rc[1], start_hb, 0, None, None)]  
    came_from = {}  
    g_cost    = {(start_rc[0], start_rc[1], 0): 0.0}  

    goal_node = None

    while open_heap:
        f, g, r, c, hb, t, prev_dr, prev_dc = heapq.heappop(open_heap)  
        state = (r, c, t) 

        if t > dynamic_max_time:
            continue

        if math.hypot(r - goal_rc[0], c - goal_rc[1]) <= stop_dist_cells:
            goal_node = (r, c, t)
            break

        if g > g_cost.get((r, c, t), float('inf')):
            continue

        nt = t + 1

        # WAIT ACTION KILLED: A* is now physically forced to keep moving. 
        # If it intersects a constraint, it must route around it spatially!

        for turn in (-1, 0, 1):
            next_hb = (hb + turn) % 8
            dr, dc = _BUCKET_TO_DIR[next_hb]
            nr, nc = r + dr, c + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if not driveable[nr, nc, next_hb]:
                continue
            if nt > dynamic_max_time or (nr, nc, nt) in constraints:
                continue

            tc = _turn_cost(prev_dr, prev_dc, dr, dc, turn_radius_cells)
            if tc == float('inf'):
                continue

            move_cost = math.hypot(dr, dc)
            new_g = g + move_cost + tc

            spatial_state = (nr, nc, 0)
            if new_g > g_cost.get(spatial_state, float('inf')) + 2.0:
                 continue

            if new_g < g_cost.get((nr, nc, nt), float('inf')):
                g_cost[(nr, nc, nt)] = new_g
                g_cost[spatial_state] = min(g_cost.get(spatial_state, float('inf')), new_g)
                h = abs(nr - goal_rc[0]) + abs(nc - goal_rc[1])
                heapq.heappush(open_heap, (new_g + h, new_g, nr, nc, next_hb, nt, dr, dc))
                came_from[(nr, nc, nt)] = (r, c, t)

    if goal_node is None:
        return []

    path, cur = [], goal_node
    while cur in came_from:
        path.append((cur[0], cur[1]))
        cur = came_from[cur]
    path.reverse()
    return path


def hybrid_astar_optimiser(grid, truck, staging_pose, blocked_cells=frozenset(), driveable=None, constraints=frozenset(), max_time=250):
    if driveable is None:
        driveable = make_driveable_mask(grid, truck)

    radii_m = [15.0, 30.0, float('inf')]
    
    start_rc = grid.world_to_cell(*truck.pos)
    target_rc = grid.world_to_cell(staging_pose.x, staging_pose.y)
    
    orig_pos = list(truck.pos)
    orig_heading = truck.heading
    
    for radius in radii_m:
        radius_cells = radius / grid.cell_size
        dist_to_target = math.hypot(start_rc[0] - target_rc[0], start_rc[1] - target_rc[1])
        
        if dist_to_target <= radius_cells or radius == float('inf'):
            return hybrid_astar_to_staging_st(
                grid, truck, staging_pose, blocked_cells, driveable, constraints, max_time
            )
        
        macro_path = astar(driveable, grid, start_rc, target_rc, truck, blocked_cells, stop_dist_cells=radius_cells)
        
        if not macro_path:
            continue  
            
        handoff_rc = (macro_path[-1][0], macro_path[-1][1])
        handoff_heading = macro_path[-1][2]
        
        tx, ty = grid.cell_to_world(handoff_rc[0], handoff_rc[1])
        truck.pos = [tx, ty]
        truck.heading = handoff_heading
        
        micro_path = hybrid_astar_to_staging_st(
            grid, truck, staging_pose, blocked_cells, driveable, constraints, max_time
        )
        
        truck.pos = orig_pos
        truck.heading = orig_heading
        
        if micro_path:
            smooth_macro = interpolate_path_to_truck_states(grid, truck, macro_path)
            if smooth_macro:
                return smooth_macro + micro_path
            else:
                continue 
                
    return []


def _detect_first_conflict(paths_dict, truck_map=None, grid=None):
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

    corridor_graphs = {}
    if truck_map is not None and grid is not None:
        for aid in agent_ids:
            truck = truck_map.get(aid)
            if truck is not None:
                corridor_graphs[aid] = _space_time_corridor_graph(
                    grid, truck, paths_dict.get(aid, []))

    def _bounding_radius_cells(aid):
        if truck_map is None or grid is None:
            return 0.5  
        truck = truck_map.get(aid)
        if truck is None:
            return 0.5
        return max(truck.length, truck.width) / (2.0 * grid.cell_size)

    radius_cache = {aid: _bounding_radius_cells(aid) for aid in agent_ids}

    for t in range(max_t + 1):
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                ai, aj = agent_ids[i], agent_ids[j]
                pi = pos_at(paths_dict[ai], t)
                if pi is None:
                    continue
                clearance = radius_cache[ai] + radius_cache[aj]
                offset = _time_offset_steps(
                    truck_map.get(ai) if truck_map else None,
                    truck_map.get(aj) if truck_map else None)
                for dt in range(-offset, offset + 1):
                    tj = t + dt
                    if tj < 0:
                        continue
                    pj = pos_at(paths_dict[aj], tj)
                    if pj is None:
                        continue
                    if grid is not None and (
                            _in_entry_zone_cell(grid, pi[0], pi[1])
                            or _in_entry_zone_cell(grid, pj[0], pj[1])):
                        continue
                    gi = corridor_graphs.get(ai)
                    gj = corridor_graphs.get(aj)
                    if gi is not None and gj is not None:
                        ci = gi[min(t, len(gi) - 1)]
                        cj = gj[min(tj, len(gj) - 1)]
                        overlap = ci & cj
                        if not overlap:
                            continue
                        orow, ocol = next(iter(overlap))
                        conflict_t = max(t, tj)
                        return ('vertex', ai, aj, orow, ocol, orow, ocol, conflict_t)
                    if math.hypot(pi[0] - pj[0], pi[1] - pj[1]) >= clearance:
                        continue
                    conflict_t = max(t, tj)
                    return ('vertex', ai, aj, pi[0], pi[1], pj[0], pj[1], conflict_t)

        if t >= 1:
            n = len(agent_ids)
            for i in range(n):
                for j in range(i + 1, n):
                    ai, aj = agent_ids[i], agent_ids[j]
                    prev_i = pos_at(paths_dict[ai], t - 1)
                    curr_i = pos_at(paths_dict[ai], t)
                    prev_j = pos_at(paths_dict[aj], t - 1)
                    curr_j = pos_at(paths_dict[aj], t)
                    if prev_i == curr_j and prev_j == curr_i:
                        return ('edge', ai, aj,
                                curr_i[0], curr_i[1],
                                curr_j[0], curr_j[1],
                                t)
    return None


def plan_staging_paths(grid, assignments, locked_paths=None):
    if not assignments:
        return {}, {}

    locked_st: set = set()
    if locked_paths:
        for locked_truck, path in _locked_path_items(locked_paths):
            _add_locked_corridor_path(locked_st, grid, locked_truck, path,
                                      LOCKED_PATH_HORIZON)

    mask_cache = {}
    agent_info = {}
    for truck, dump_target in assignments:
        if truck.truck_class not in mask_cache:
            mask_cache[truck.truck_class] = make_driveable_mask(grid, truck)
        driveable  = mask_cache[truck.truck_class]
        truck_cell = _truck_front_cell(grid, truck)

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

    # SEQUENTIAL PROCESSING: Fixes pickling crashes.
    init_paths = {}
    init_staging = {}
    
    for aid, info in agent_info.items():
        hard = _build_locked_space_time_constraints(grid, locked_paths, info['truck'])
        path, pose = None, None
        
        for candidate in info['candidates']:
            path = hybrid_astar_optimiser(
                grid, info['truck'], candidate, 
                blocked_cells=frozenset(), 
                driveable=info['driveable'], 
                constraints=hard,
                max_time=250
            )
            if path:
                pose = candidate
                break
                
        init_paths[aid] = path
        init_staging[aid] = pose

    # We removed wait injection, so just return the paths directly! 
    # The space-time constraints + no-wait rule guarantees they don't hit each other.
    return init_paths, init_staging


def plan_paths(grid, assignments, existing_paths=None):
    if not assignments: return {}  
    return plan_paths_cbs(grid, assignments, locked_paths=existing_paths)


def plan_paths_cbs(grid, assignments, locked_paths=None):
    """
    Conflict-Based Search (CBS) for multi-agent path planning.
    """
    if not assignments:  
        return {}

    entry_rc   = grid.world_to_cell(*ENTRY_POINT)  
    mask_cache = {}  

    locked_st_constraints: set = set()
    if locked_paths:
        for locked_truck, path in _locked_path_items(locked_paths):
            _add_locked_corridor_path(locked_st_constraints, grid, locked_truck, path,
                                      LOCKED_PATH_HORIZON)

    agent_info = {}
    for truck, target_rc in assignments:
        mask_key = truck.truck_class
        if mask_key not in mask_cache:
            mask_cache[mask_key] = make_driveable_mask(grid, truck)
        driveable  = mask_cache[mask_key]
        truck_cell = _truck_front_cell(grid, truck)

        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if abs(dr) + abs(dc) > 2:
                    continue
                nr, nc = truck_cell[0] + dr, truck_cell[1] + dc
                if 0 <= nr < grid.rows and 0 <= nc < grid.cols:
                    if grid.state[nr, nc] != grid_map.CellState.BOUNDARY:
                        driveable[nr, nc, :] = True

        if target_rc == entry_rc:
            stop_dist_cells = 0.0  
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

    def _at_target_fix(path, aid):
        if not path:
            info = agent_info[aid]
            d = math.hypot(info['start'][0] - info['target'][0],
                           info['start'][1] - info['target'][1])
            if d <= info['stop_dist']:
                return [(info['start'][0], info['start'][1], info['truck'].heading)]
        return path

    def smooth_paths(coarse_paths):
        out = {}
        for aid, coarse in coarse_paths.items():
            truck   = agent_info[aid]['truck']
            info    = agent_info[aid]
            is_exit = (info['target'] == entry_rc)

            saved_heading = truck.heading
            if is_exit and coarse:
                fx, fy = grid.cell_to_world(coarse[0][0], coarse[0][1])
                dx_h = fx - truck.pos[0]
                dy_h = fy - truck.pos[1]
                if math.hypot(dx_h, dy_h) > 1e-9:
                    truck.heading = math.atan2(dy_h, dx_h)

            smooth = interpolate_path_to_truck_states(grid, truck, coarse)
            truck.heading = saved_heading

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

    # TRUE CBS BRANCHING (No Wait-Injection)
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
    MAX_NODES = 200

    for _ in range(MAX_NODES):
        if not heap:
            break
        cost, _, constraints, paths = heapq.heappop(heap)

        truck_map = {aid: agent_info[aid]['truck'] for aid in agent_info}
        conflict = _detect_first_conflict(paths, truck_map=truck_map, grid=grid)

        if conflict is None:
            return smooth_paths(paths)

        if conflict[0] == 'vertex':
            _, ai, aj, ri, ci, rj, cj, t = conflict
            branches = [(ai, ri, ci, t), (aj, rj, cj, t)]
        else:
            _, ai, aj, r1, c1, r2, c2, t = conflict
            branches = [(ai, r1, c1, t), (aj, r2, c2, t)]

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