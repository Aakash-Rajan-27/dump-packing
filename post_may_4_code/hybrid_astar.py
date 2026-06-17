# hybrid_astar.py
# ─────────────────────────────────────────────────────────────
# Hybrid A* planner for staging:
#   • Pose-bucket utilities (_pose_bucket, _pose_heading, etc.)
#   • _hybrid_primitive() — one forward bicycle arc
#   • hybrid_astar_to_staging()    — spatial-only version
#   • hybrid_astar_to_staging_st() — space-time (CBS) version
#
# Collision model: every candidate pose is checked against the
# full oriented truck footprint using _rect_overlap_2d / SAT —
# no single-cell occupancy logic remains.
# ─────────────────────────────────────────────────────────────

import heapq
import math
import grid_map
from path_utils import _angle_diff_signed
from filters import is_pose_driveable, make_driveable_mask
from config import (ENTRY_POINT, POSE_HEADING_BUCKETS, TRUCK_MOVE_STEP_M,
                    ASTAR_WAIT_COST, ASTAR_MAX_TIME, PATH_BUFFER_M)
from staging import score_staging_candidates
from conflict_detect import _rect_overlap_2d, _footprints_conflict


def _pose_bucket(heading):
    step = 2.0 * math.pi / POSE_HEADING_BUCKETS
    return int(round(heading / step)) % POSE_HEADING_BUCKETS


def _pose_heading(bucket):
    return bucket * 2.0 * math.pi / POSE_HEADING_BUCKETS


def _pose_allowed(grid, truck, x, y, heading):
    if math.hypot(x - ENTRY_POINT[0], y - ENTRY_POINT[1]) <= truck.length:
        return True
    return is_pose_driveable(grid, truck, x, y, heading)


def _mask_pose_allowed(grid, driveable, x, y, heading, grace_dist=5.0):
    if math.hypot(x - ENTRY_POINT[0], y - ENTRY_POINT[1]) <= grace_dist:
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
    grace = truck.length + 2.0
    poses = []

    for step in range(samples):
        mid_heading = heading + d_heading / 2.0
        x += math.cos(mid_heading) * ds
        y += math.sin(mid_heading) * ds
        heading = (heading + d_heading + math.pi) % (2.0 * math.pi) - math.pi
        if step % 3 == 0 or step == samples - 1:
            if not _mask_pose_allowed(grid, driveable, x, y, heading, grace_dist=grace):
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


def _footprint_blocked(truck, x, y, heading, blocked_footprints, buffer=PATH_BUFFER_M):
    """Return True if truck body at (x,y,heading) comes within `buffer` metres
    of any blocked footprint (Minkowski expansion on the checking truck)."""
    if not blocked_footprints:
        return False
    hl = truck.length / 2.0 + buffer
    hw = truck.width  / 2.0 + buffer
    for fx, fy, fh, fhl, fhw in blocked_footprints:
        if _rect_overlap_2d(x, y, heading, hl, hw, fx, fy, fh, fhl, fhw):
            return True
    return False


def hybrid_astar_to_staging(grid, truck, staging_pose, blocked_footprints=(), driveable=None):
    """Plan forward bicycle arcs to an outward-facing staging pose.

    blocked_footprints: sequence of (cx, cy, heading, half_length, half_width) —
      truck body footprints that must never be overlapped.  Replaces the old
      blocked_cells frozenset; all collision checks now use SAT rectangle overlap.
    """
    if driveable is None:
        driveable = make_driveable_mask(grid, truck)

    hl = truck.length / 2.0
    hw = truck.width / 2.0

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
            # Full-footprint blocked check — replaces (nr, nc) in blocked_cells.
            if _footprint_blocked(truck, nx, ny, nh, blocked_footprints):
                continue
            nr, nc = grid.world_to_cell(nx, ny)
            next_state = (nr, nc, _pose_bucket(nh))
            if next_state == state:
                continue
            turn_cost = 0.0 if turn == 0 else 0.01 * truck.turn_radius * (2.0 * math.pi / POSE_HEADING_BUCKETS)
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


def hybrid_astar_to_staging_st(grid, truck, staging_pose, blocked_footprints=(),
                                driveable=None, constraints=None, max_time=ASTAR_MAX_TIME):
    """Space-time variant of hybrid_astar_to_staging.

    State: (r, c, heading_bucket, t).
    constraints: dict {t: [(cx, cy, heading, half_length, half_width), ...]}
      — forbidden footprints at specific timesteps (CBS + locked paths).
      Every candidate pose is checked against the full oriented truck rectangle
      using SAT (_rect_overlap_2d).  No (r, c, t) cell-occupancy logic remains.
    Wait action included so the planner can hold position to let another truck pass.
    """
    if constraints is None:
        constraints = {}
    if driveable is None:
        driveable = make_driveable_mask(grid, truck)

    hl = truck.length / 2.0
    hw = truck.width / 2.0

    start_rc = grid.world_to_cell(*truck.pos)
    start_hb = _pose_bucket(truck.heading)
    start    = (start_rc[0], start_rc[1], start_hb, 0)

    # ── EARLY CONFLICT CHECK (start of planning) ─────────────────────────────────
    constraint_ts = sorted(constraints.keys()) if constraints else []
    if constraints:
        print(f"[HYBRID_ST] Truck {truck.id}: planning to staging pose "
              f"({staging_pose.x:.1f},{staging_pose.y:.1f}) h={staging_pose.heading:.2f}, "
              f"fp_constraints at t={constraint_ts}")
        sx, sy = truck.pos
        sh = truck.heading
        for check_t in constraint_ts[:3]:
            if _footprints_conflict(sx, sy, sh, hl, hw, check_t, constraints):
                print(f"[HYBRID_ST] Truck {truck.id}: RECTANGLE CONFLICT at start pos "
                      f"({sx:.1f},{sy:.1f}) heading={sh:.2f} at t={check_t}")
                break
        else:
            print(f"[HYBRID_ST] Truck {truck.id}: no rectangle conflict at start position")
    else:
        print(f"[HYBRID_ST] Truck {truck.id}: planning to staging pose "
              f"({staging_pose.x:.1f},{staging_pose.y:.1f}) h={staging_pose.heading:.2f}, "
              f"no constraints")
    # ─────────────────────────────────────────────────────────────────────────────

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

        # ── WAIT ─────────────────────────────────────────────────────────────────
        # Check full footprint at current world pose against constraints at nt.
        _wait_conflict = _footprints_conflict(x, y, heading, hl, hw, nt, constraints)
        if _wait_conflict:
            print(f"[HYBRID_ST] Truck {truck.id}: WAIT conflict at "
                  f"({x:.1f},{y:.1f}) heading={heading:.2f} t={nt}")
        else:
            wait_state = (r, c, hb, nt)
            wait_g = g + ASTAR_WAIT_COST
            if wait_g < g_cost.get(wait_state, float('inf')):
                g_cost[wait_state]     = wait_g
                came_from[wait_state]  = state
                actions[wait_state]    = None
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

            # Full-footprint blocked check against statically blocked footprints.
            if _footprint_blocked(truck, nx, ny, nh, blocked_footprints):
                continue

            # Full-footprint constraint check — replaces (nr, nc, nt) in constraints.
            if _footprints_conflict(nx, ny, nh, hl, hw, nt, constraints):
                print(f"[HYBRID_ST] Truck {truck.id}: MOVE conflict → "
                      f"({nx:.1f},{ny:.1f}) heading={nh:.2f} t={nt}")
                continue

            nr, nc = grid.world_to_cell(nx, ny)
            nhb    = _pose_bucket(nh)
            next_state = (nr, nc, nhb, nt)

            if (nr, nc, nhb) == (r, c, hb):
                continue

            turn_cost = (0.0 if turn == 0
                         else 0.05 * truck.turn_radius * (2.0 * math.pi / POSE_HEADING_BUCKETS))
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
