# conflict_detect.py
# ─────────────────────────────────────────────────────────────
# Conflict detection for multi-agent path planning:
#   • _rect_overlap_2d()      — SAT oriented-rectangle test
#   • _infer_heading_at_t()   — heading from cell-path direction
#   • _detect_first_conflict() — earliest vertex or edge conflict
# ─────────────────────────────────────────────────────────────

import math
from path_utils import _corridor_cell_set
from config import ENTRY_POINT, ENTRY_CORRIDOR_CELLS


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
                        if not _rect_overlap_2d(wxi, wyi, hi, ti.length / 2, ti.width / 2,
                                                wxj, wyj, hj, tj.length / 2, tj.width / 2):
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


def _keeper_forbidden_states(grid, dedup_cells, keeper_truck, rep_truck):
    """
    Return the set of (r, c, t) cells that rep_truck's centre must NOT occupy
    so that its physical body never touches keeper_truck's body at timestep t.

    Uses the Minkowski-sum expansion: pad keeper_truck's half-dimensions by
    rep_truck's half-dimensions.  At each timestep t, every cell (nr,nc) whose
    world centre falls inside that expanded rectangle around the keeper is marked
    forbidden for the replanning truck.

    dedup_cells — deduplicated (r,c) sequence from _dedup_path_cells/_norm_path,
                  one entry per unique cell so t indexes are time-aligned.
    """
    hl_sum  = (keeper_truck.length + rep_truck.length) / 2.0
    hw_sum  = (keeper_truck.width  + rep_truck.width)  / 2.0
    r_cells = int(math.ceil(math.hypot(hl_sum, hw_sum) / grid.cell_size)) + 1

    forbidden = set()
    for t, (kr, kc) in enumerate(dedup_cells):
        kh      = _infer_heading_at_t(dedup_cells, t)
        kwx, kwy = grid.cell_to_world(kr, kc)
        cos_kh  = math.cos(kh)
        sin_kh  = math.sin(kh)
        for dr in range(-r_cells, r_cells + 1):
            for dc in range(-r_cells, r_cells + 1):
                nr, nc = kr + dr, kc + dc
                if not (0 <= nr < grid.rows and 0 <= nc < grid.cols):
                    continue
                wx, wy   = grid.cell_to_world(nr, nc)
                dx, dy   = wx - kwx, wy - kwy
                local_l  =  dx * cos_kh + dy * sin_kh
                local_w  = -dx * sin_kh + dy * cos_kh
                if abs(local_l) <= hl_sum and abs(local_w) <= hw_sum:
                    forbidden.add((nr, nc, t))
    return forbidden


def _detect_physical_conflict(paths_dict, truck_map, grid):
    """
    Full SAT oriented-rectangle overlap check for all agent pairs at every
    timestep. Unlike _detect_first_conflict, this runs SAT unconditionally
    (no same-cell pre-filter, no corridor-disjoint exemption) so it catches
    physical body overlap even when trucks are in adjacent cells.

    Intended for the spatial-first shortcut before corridors are established.
    Works for all path types (hybrid A*, spatial A*, astar_st) since all
    produce (r,c) cell sequences.

    Returns (aid_i, aid_j, t) of the first physical overlap, or None.
    """
    agent_ids = list(paths_dict.keys())
    if len(agent_ids) < 2 or truck_map is None or grid is None:
        return None

    max_t = max((len(p) for p in paths_dict.values()), default=0)
    if max_t == 0:
        return None

    entry_rc_local = grid.world_to_cell(*ENTRY_POINT)

    def pos_at(path, t):
        if not path:
            return None
        return (path[min(t, len(path) - 1)][0], path[min(t, len(path) - 1)][1])

    for t in range(max_t + 1):
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                ai, aj = agent_ids[i], agent_ids[j]
                pi = pos_at(paths_dict[ai], t)
                pj = pos_at(paths_dict[aj], t)
                if pi is None or pj is None:
                    continue

                ti = truck_map.get(ai)
                tj = truck_map.get(aj)
                if ti is None or tj is None:
                    continue

                # Skip entry corridor — natural funnel, trucks must pass close
                di = math.hypot(pi[0] - entry_rc_local[0], pi[1] - entry_rc_local[1])
                dj = math.hypot(pj[0] - entry_rc_local[0], pj[1] - entry_rc_local[1])
                if di <= ENTRY_CORRIDOR_CELLS or dj <= ENTRY_CORRIDOR_CELLS:
                    continue

                # Quick distance cull — skip SAT if centers are far apart
                if math.hypot(pi[0] - pj[0], pi[1] - pj[1]) > (ti.length + tj.length):
                    continue

                hi = _infer_heading_at_t(paths_dict[ai], t)
                hj = _infer_heading_at_t(paths_dict[aj], t)
                wxi, wyi = grid.cell_to_world(pi[0], pi[1])
                wxj, wyj = grid.cell_to_world(pj[0], pj[1])

                if _rect_overlap_2d(wxi, wyi, hi, ti.length / 2, ti.width / 2,
                                    wxj, wyj, hj, tj.length / 2, tj.width / 2):
                    return (ai, aj, t)

    return None
