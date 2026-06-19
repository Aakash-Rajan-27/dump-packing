# hybrid_astar.py
# ─────────────────────────────────────────────────────────────
# Hybrid A* planner for staging:
#   • Pose-bucket utilities (_pose_bucket, _pose_heading, etc.)
#   • _hybrid_primitive() — one forward bicycle arc
#   • hybrid_astar_to_staging()    — spatial-only version
#   • hybrid_astar_to_staging_st() — space-time (CBS) version
# ─────────────────────────────────────────────────────────────

import heapq
import math
import grid_map
from path_utils import _angle_diff_signed
from filters import is_pose_driveable, make_driveable_mask
from config import (ENTRY_POINT, POSE_HEADING_BUCKETS, TRUCK_MOVE_STEP_M,
                    ASTAR_WAIT_COST, ASTAR_MAX_TIME)
from staging import score_staging_candidates


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


def _mask_pose_allowed(grid, driveable, x, y, heading, grace_dist=5.0):
    # Grace zone: skip the mask check near the entry so trucks can cross the gate.
    # Callers that know the truck length pass (truck.length + 2.0) so large trucks
    # can straddle the boundary during entry/exit without being blocked.
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
    # Grace zone is one full truck length + 2 m so any size truck can cross the gate.
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
                                driveable=None, constraints=frozenset(), max_time=ASTAR_MAX_TIME):
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
