# conflict_detect.py
# ─────────────────────────────────────────────────────────────
# Conflict detection and footprint-constraint helpers:
#   • _rect_overlap_2d()         — SAT oriented-rectangle test
#   • _infer_heading_at_t()      — heading from cell-path direction
#   • _detect_first_conflict()   — earliest vertex or edge conflict
#   • _footprints_conflict()     — single footprint vs constraint dict
#   • _merge_fp_constraints()    — combine two {t:[fp,...]} dicts
#   • _build_locked_footprints() — convert locked_paths to {t:[fp,...]}
# ─────────────────────────────────────────────────────────────

import math
from path_utils import _corridor_cell_set
from config import ENTRY_POINT, ENTRY_CORRIDOR_CELLS, PATH_BUFFER_M


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
    # Padded timestep: truck has stopped at the final cell.  Both forward/backward
    # candidates collapse to `end` (skipped by the t1==t0 guard), so the old loop
    # always returned 0.0 — wrong heading → wrong SAT rectangle → false conflicts.
    # Fix: use the direction of the last actual step as the parked heading.
    if t > end:
        if end > 0:
            dr = path[end][0] - path[end - 1][0]
            dc = path[end][1] - path[end - 1][1]
            if dr or dc:
                return math.atan2(dr, dc)
        return 0.0
    for t1 in (min(t + 1, end), min(max(t - 1, 0), end)):
        if t1 == t0:
            continue
        dr = path[t1][0] - path[t0][0] if t1 > t0 else path[t0][0] - path[t1][0]
        dc = path[t1][1] - path[t0][1] if t1 > t0 else path[t0][1] - path[t1][1]
        if dr or dc:
            return math.atan2(dr, dc)
    return 0.0


def _footprints_conflict(cx, cy, heading, hl, hw, t, fp_constraints, buffer=PATH_BUFFER_M):
    """Return True if the truck footprint (cx,cy,heading,hl,hw) comes within
    `buffer` metres of any locked footprint at timestep t.

    The checking truck's half-dimensions are expanded by `buffer` (Minkowski sum)
    before the SAT test, so the check rejects any position where the real gap
    between the two rectangles would be less than `buffer`.

    fp_constraints is a dict {t: [(cx,cy,h,hl,hw),...]}."""
    ehl = hl + buffer
    ehw = hw + buffer
    for fp_cx, fp_cy, fp_h, fp_hl, fp_hw in fp_constraints.get(t, ()):
        if _rect_overlap_2d(cx, cy, heading, ehl, ehw, fp_cx, fp_cy, fp_h, fp_hl, fp_hw):
            return True
    return False


def _merge_fp_constraints(a, b):
    """Merge two {t: [footprint_tuples]} dicts into one (non-destructive)."""
    if not a:
        return b
    if not b:
        return a
    merged = {t: list(fps) for t, fps in a.items()}
    for t, fps in b.items():
        if t in merged:
            merged[t] = merged[t] + fps
        else:
            merged[t] = list(fps)
    return merged


def _build_locked_footprints(locked_paths, grid):
    """Convert a locked_paths dict to a per-timestep footprint dict.

    locked_paths values must be 4-tuples produced by _make_locked_entry():
      (body_center_poses, tail_ticks, half_length, half_width)
    where body_center_poses is a list of (cx, cy, heading) body-centre world coords.

    Returns {t: [(cx, cy, heading, hl, hw), ...]} sampled at one footprint per
    grid.cell_size of travel along each locked path, so that timestep t here
    matches timestep t in the space-time planners (which also advance t by one
    cell-width of travel per planning step).
    """
    result = {}

    for path_entry in (locked_paths or {}).values():
        if len(path_entry) != 4:
            continue
        raw_path, tail_ticks, hl, hw = path_entry

        if not raw_path:
            continue

        def _add(t, cx, cy, h):
            result.setdefault(t, []).append((float(cx), float(cy), float(h),
                                             float(hl), float(hw)))

        # Emit first footprint at t=0 (truck's current position).
        t = 0
        prev_x, prev_y = float(raw_path[0][0]), float(raw_path[0][1])
        _add(0, prev_x, prev_y, float(raw_path[0][2]))
        accum = 0.0

        for wp in raw_path[1:]:
            cx, cy, h = float(wp[0]), float(wp[1]), float(wp[2])
            dist = math.hypot(cx - prev_x, cy - prev_y)
            accum += dist
            # Emit one footprint per cell_size of travel to match planner timesteps.
            while accum >= grid.cell_size:
                accum -= grid.cell_size
                t += 1
                _add(t, cx, cy, h)
            prev_x, prev_y = cx, cy

        # Tail: hold the final pose for tail_ticks more planner timesteps.
        if tail_ticks > 0:
            fx, fy, fh = float(raw_path[-1][0]), float(raw_path[-1][1]), float(raw_path[-1][2])
            for dt in range(tail_ticks):
                _add(t + dt + 1, fx, fy, fh)

    return result


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
        return (path[min(t, len(path) - 1)][0], path[min(t, len(path) - 1)][1])

    entry_rc_local = grid.world_to_cell(*ENTRY_POINT) if grid is not None else None

    for t in range(max_t + 1):

        # ── VERTEX CONFLICT CHECK (exact oriented-rectangle SAT) ─────────────────
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
                        # Expand ti's footprint by PATH_BUFFER_M (Minkowski sum) so
                        # CBS treats paths within buffer distance as a conflict.
                        if not _rect_overlap_2d(wxi, wyi, hi,
                                                ti.length / 2 + PATH_BUFFER_M,
                                                ti.width  / 2 + PATH_BUFFER_M,
                                                wxj, wyj, hj,
                                                tj.length / 2, tj.width / 2):
                            continue
                        # SAT says rectangles touch — but if the planned path
                        # corridors share no cells, the paths never actually cross
                        # and the overlap is a geometry artefact.  Corridors are
                        # the planning guarantee; trust them and skip the conflict.
                        ci = _corridor_cell_set(grid, ti.id)
                        cj = _corridor_cell_set(grid, tj.id)
                        if ci and cj and ci.isdisjoint(cj):
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
                                if not _rect_overlap_2d(mid_ix, mid_iy, hi,
                                                        ti.length / 2 + PATH_BUFFER_M,
                                                        ti.width  / 2 + PATH_BUFFER_M,
                                                        mid_jx, mid_jy, hj,
                                                        tj.length / 2, tj.width / 2):
                                    continue
                        print(f"[CONFLICT] EDGE SWAP: trucks {ai} and {aj} swapped at t={t} "
                              f"— {prev_i}↔{prev_j}")
                        return ('edge', ai, aj,
                                curr_i[0], curr_i[1],
                                curr_j[0], curr_j[1],
                                t)

    return None
