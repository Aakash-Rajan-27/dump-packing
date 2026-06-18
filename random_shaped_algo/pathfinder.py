# pathfinder.py
# ─────────────────────────────────────────────────────────────
# Thin re-export shim — all logic has been split into focused
# submodules.  Any existing `from pathfinder import X` continues
# to work without changes to callers.
#
# Module map:
#   path_utils     — direction tables, bucket helpers, cell/path
#                    helpers, _truck_inside_boundary, _corridor_cell_set
#   bicycle_model  — smooth bicycle-model path interpolation
#   deadlock       — reverse retreat, yield maneuver, escape_and_replan_exit
#   hybrid_astar   — pose-bucket utils, hybrid A* (spatial + ST)
#   astar_core     — astar(), astar_st(), plan_paths()
#   conflict_detect— SAT helpers, _detect_first_conflict()
#   staging_paths  — plan_staging_paths()
#   cbs_planner    — plan_paths_cbs()
# ─────────────────────────────────────────────────────────────

from path_utils import (
    _DIR_TO_BUCKET, _BUCKET_TO_DIR, _BUCKET_TO_HEADING,
    _heading_bucket, _angle_diff_signed, _bucket_from_heading, _heading_for_bucket,
    _state_cell, _path_cells, _dedup_path_cells,
    _truck_front_cell, _truck_rear_pose, _truck_front_world,
    _truck_inside_boundary,
    _corridor_cell_set,
)

from bicycle_model import (
    _steering_angle, _turn_radius, _refine_front_target,
    _polyline_lengths, _sample_polyline, _closest_polyline_distance,
    interpolate_path_to_truck_states,
)

from deadlock import (
    generate_reverse_retreat,
    generate_yield_maneuver,
    escape_and_replan_exit,
)

from hybrid_astar import (
    _pose_bucket, _pose_heading, _pose_allowed, _mask_pose_allowed,
    _hybrid_primitive, _route_exactly_driveable,
    hybrid_astar_to_staging,
    hybrid_astar_to_staging_st,
)

from astar_core import (
    _turn_cost,
    astar,
    astar_st,
    plan_paths,
)

from conflict_detect import (
    _rect_overlap_2d,
    _infer_heading_at_t,
    _detect_first_conflict,
    _detect_physical_conflict,
    _keeper_forbidden_states,
)

from staging_paths import plan_staging_paths

from cbs_planner import plan_paths_cbs
