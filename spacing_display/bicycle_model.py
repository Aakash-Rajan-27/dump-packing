# bicycle_model.py
# ─────────────────────────────────────────────────────────────
# Bicycle-model smooth path interpolation.
# Takes a coarse cell-level A* path and emits rear-axle world
# poses that respect the truck's minimum turning radius.
# ─────────────────────────────────────────────────────────────

import math
from path_utils import _truck_inside_boundary, _truck_front_world, _truck_rear_pose, _angle_diff_signed
from config import (TRUCK_MOVE_STEP_M, TURN_REFINEMENT_ITERATIONS,
                    TURN_LOOKAHEAD_RADIUS_FACTOR, TURN_PATH_TOLERANCE_M,
                    TURN_MAX_SMOOTH_STEPS)


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
