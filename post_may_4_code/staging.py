from dataclasses import dataclass
import math

from config import (
    REVERSE_DUMP_CLOSE_ENOUGH_M,
    REVERSE_DUMP_STEP_M,
    STAGING_DISTANCE_WEIGHT,
    STAGING_EXTRA_DISTANCE_M,
    STAGING_HEADING_WEIGHT,
    STAGING_NUM_ANGLES,
    _TAN_REPOSE,
)
from filters import is_pose_driveable


@dataclass(frozen=True)
class StagingPose:
    x: float
    y: float
    heading: float
    score: float


def required_dump_clearance_m(truck):
    pile_radius = (3.0 * truck.payload_volume_m3 / (math.pi * _TAN_REPOSE)) ** (1.0 / 3.0)
    return max(3.0, pile_radius + 1.0)


def reverse_segment_poses(grid, truck, dump_target, staging_pose):
    dump_x, dump_y = grid.cell_to_world(*dump_target)
    clearance = required_dump_clearance_m(truck)
    x, y = staging_pose.x, staging_pose.y
    poses = []

    for _ in range(200):
        rear_x = x - math.cos(staging_pose.heading) * truck.length / 2.0
        rear_y = y - math.sin(staging_pose.heading) * truck.length / 2.0
        if math.hypot(rear_x - dump_x, rear_y - dump_y) <= clearance + REVERSE_DUMP_CLOSE_ENOUGH_M:
            return poses
        x -= math.cos(staging_pose.heading) * REVERSE_DUMP_STEP_M
        y -= math.sin(staging_pose.heading) * REVERSE_DUMP_STEP_M
        if not is_pose_driveable(grid, truck, x, y, staging_pose.heading):
            return None
        poses.append((x, y, staging_pose.heading))
    return None


def score_staging_candidates(grid, truck, dump_target):
    dump_x, dump_y = grid.cell_to_world(*dump_target)
    clearance = required_dump_clearance_m(truck)
    staging_radius = clearance + truck.length / 2.0 + STAGING_EXTRA_DISTANCE_M
    candidates = []

    for index in range(STAGING_NUM_ANGLES):
        heading = index * 2.0 * math.pi / STAGING_NUM_ANGLES
        x = dump_x + math.cos(heading) * staging_radius
        y = dump_y + math.sin(heading) * staging_radius
        if not is_pose_driveable(grid, truck, x, y, heading):
            continue
        candidate = StagingPose(x, y, heading, 0.0)
        reverse_poses = reverse_segment_poses(grid, truck, dump_target, candidate)
        if reverse_poses is None:
            continue

        distance = math.hypot(x - truck.pos[0], y - truck.pos[1])
        heading_delta = abs((heading - truck.heading + math.pi) % (2.0 * math.pi) - math.pi)
        score = STAGING_DISTANCE_WEIGHT * distance + STAGING_HEADING_WEIGHT * heading_delta
        candidates.append(StagingPose(x, y, heading, score))

    return sorted(candidates, key=lambda candidate: candidate.score)
