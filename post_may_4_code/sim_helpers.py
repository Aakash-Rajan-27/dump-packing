# sim_helpers.py
# ─────────────────────────────────────────────────────────────
# Simulation-level helpers used by run_simulation():
#   • initialise_half_full_dump() — seed initial terrain
#   • _corridor_cells()           — expand path to corridor set
#   • build_fleet()               — create all Truck objects
#   • _try_inplace_replan()       — resolve a 2-truck conflict
#                                   via CBS without forcing idle
# ─────────────────────────────────────────────────────────────

import math
import random
import numpy as np
import shapely.geometry

import grid_map
from truck import Truck
from pathfinder import (plan_staging_paths, plan_paths_cbs, _path_cells, _make_locked_entry)
from config import (ENTRY_POINT, FLEET_COMPOSITION)


def initialise_pre_filled_dump(grid):
    """Procedural terrain: clustered Gaussian-cone piles targeting
    ~60 % pack density and ~5 m average nearest-neighbour pile spacing.

    Strategy
    --------
    1. Place pile centres using a three-phase clustered scheme:
       anchor points (well-separated) → satellite piles (close around each
       anchor) → isolated piles (scattered elsewhere).  This produces the
       organic clustering visible in the reference image.
    2. Each pile gets a Gaussian-cone height profile with random radius and
       peak height, so neighbouring piles merge into larger mounds naturally.
    3. A binary search finds the global scale factor that makes
       mean(clip(z, 0, TARGET))/TARGET ≈ 60 % before the relaxation pass.
    4. After relax_all() a quick correction scales z_height if the measured
       pack_pct is still outside ±2 %.
    5. The cycle repeats (up to 6 passes) adjusting n_anchors to steer
       average nearest-neighbour spacing toward 5 m ± 0.5 m.

    Returns (pile_world_coords, avg_nn_spacing_m).
    """
    from config import TARGET_PILE_HEIGHT
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(42)

    TARGET_PACK    = 0.60
    TARGET_SPACING = 5.0   # m  average nearest-neighbour
    PACK_TOL       = 0.02
    SPACING_TOL    = 0.5

    cs = grid.cell_size
    ox, oy = grid.origin

    valid_mask = (
        (grid.state != grid_map.CellState.BOUNDARY) &
        (grid.state != grid_map.CellState.PROTECTED) &
        (grid.state != grid_map.CellState.OBSTACLE)
    )

    # World-coordinate centres of every grid cell (precomputed, vectorised)
    row_y = (oy + np.arange(grid.rows) * cs + cs / 2).astype(np.float64)
    col_x = (ox + np.arange(grid.cols) * cs + cs / 2).astype(np.float64)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _nn_mean(pts_arr):
        """Mean nearest-neighbour distance via KD-tree (O(N log N))."""
        if len(pts_arr) < 2:
            return float('inf')
        tree = cKDTree(pts_arr)
        d, _ = tree.query(pts_arr, k=2)   # k=2: self + nearest neighbour
        return float(np.mean(d[:, 1]))

    def _place_centers(n_anchors):
        """Three-phase clustered placement; returns (N,2) float64 array."""
        xmin = ox + 4.0;  xmax = ox + grid.cols * cs - 4.0
        ymin = oy + 4.0;  ymax = oy + grid.rows * cs - 4.0

        centers = []   # list of [x, y]

        # Phase 1 — anchor points (minimum separation ~10 m)
        anchors = []
        anchor_min_sep = 10.0
        for _ in range(n_anchors * 80):
            if len(anchors) >= n_anchors:
                break
            x = float(rng.uniform(xmin, xmax))
            y = float(rng.uniform(ymin, ymax))
            ci = int((x - ox) / cs);  ri = int((y - oy) / cs)
            if not (0 <= ri < grid.rows and 0 <= ci < grid.cols):
                continue
            if not valid_mask[ri, ci]:
                continue
            if anchors:
                aa = np.array(anchors)
                if np.min(np.hypot(aa[:, 0] - x, aa[:, 1] - y)) < anchor_min_sep:
                    continue
            anchors.append([x, y])

        # Phase 2 — satellite piles around each anchor (2-5 per anchor)
        sat_min_sep = 2.5
        for ax, ay in anchors:
            n_sats    = int(rng.integers(2, 6))
            cluster_r = float(rng.uniform(4.0, 8.0))
            in_cluster = 0
            for _ in range(n_sats * 30):
                if in_cluster >= n_sats:
                    break
                angle = float(rng.uniform(0.0, 2.0 * math.pi))
                dist  = float(rng.uniform(2.0, cluster_r))
                x = ax + dist * math.cos(angle)
                y = ay + dist * math.sin(angle)
                ci = int((x - ox) / cs);  ri = int((y - oy) / cs)
                if not (0 <= ri < grid.rows and 0 <= ci < grid.cols):
                    continue
                if not valid_mask[ri, ci]:
                    continue
                if centers:
                    cc = np.array(centers)
                    if np.min(np.hypot(cc[:, 0] - x, cc[:, 1] - y)) < sat_min_sep:
                        continue
                centers.append([x, y])
                in_cluster += 1

        # Phase 3 — isolated piles scattered elsewhere (min 3 m from all others)
        n_isolated  = int(rng.integers(10, 18))
        iso_min_sep = 3.0
        iso_added   = 0
        for _ in range(n_isolated * 50):
            if iso_added >= n_isolated:
                break
            x = float(rng.uniform(xmin, xmax))
            y = float(rng.uniform(ymin, ymax))
            ci = int((x - ox) / cs);  ri = int((y - oy) / cs)
            if not (0 <= ri < grid.rows and 0 <= ci < grid.cols):
                continue
            if not valid_mask[ri, ci]:
                continue
            if centers:
                cc = np.array(centers)
                if np.min(np.hypot(cc[:, 0] - x, cc[:, 1] - y)) < iso_min_sep:
                    continue
            centers.append([x, y])
            iso_added += 1

        return np.array(centers, dtype=np.float64) if centers else np.empty((0, 2))

    def _apply_piles(centers_arr):
        """Superimpose Gaussian-cone elevations; returns (rows, cols) float64."""
        N      = len(centers_arr)
        radii  = rng.uniform(3.0, 8.0,  N)
        peak_h = rng.uniform(0.5, 1.25, N) * TARGET_PILE_HEIGHT

        z = np.zeros((grid.rows, grid.cols), dtype=np.float64)
        for i in range(N):
            cx, cy  = centers_arr[i]
            R       = radii[i]
            ph      = peak_h[i]
            sig2    = (R * 0.45) ** 2
            cutoff2 = (2.5 * R) ** 2
            extent  = int(math.ceil(2.5 * R / cs))

            cr     = int((cy - oy) / cs);  cc_col = int((cx - ox) / cs)
            rmin   = max(0, cr - extent);   rmax   = min(grid.rows, cr + extent + 1)
            cmin   = max(0, cc_col - extent); cmax  = min(grid.cols, cc_col + extent + 1)

            wy    = row_y[rmin:rmax, np.newaxis]   # (H, 1)
            wx    = col_x[np.newaxis, cmin:cmax]   # (1, W)
            dist2 = (wx - cx) ** 2 + (wy - cy) ** 2
            h     = ph * np.exp(-dist2 / (2.0 * sig2))
            h[dist2 > cutoff2] = 0.0
            z[rmin:rmax, cmin:cmax] += h

        return z

    def _pack_scale(z_raw):
        """Binary-search global multiplier so pack_pct(z_raw * s) ≈ TARGET_PACK."""
        z_v  = z_raw[valid_mask]
        s_lo, s_hi = 1e-3, 20.0
        for _ in range(50):
            s    = (s_lo + s_hi) * 0.5
            pack = np.mean(np.clip(z_v * s, 0.0, TARGET_PILE_HEIGHT)) / TARGET_PILE_HEIGHT
            if pack < TARGET_PACK:
                s_lo = s
            else:
                s_hi = s
        return (s_lo + s_hi) * 0.5

    # ── iterative outer loop: adjust n_anchors to hit spacing target ─────────
    # More anchors → more total piles → smaller average NN spacing.
    n_anchors   = 18   # starting guess (≈70 total piles → ~5 m NN spacing)
    centers_arr = np.empty((0, 2))
    avg_spacing = float('inf')

    for _attempt in range(6):
        c_arr = _place_centers(n_anchors)
        if len(c_arr) < 3:
            n_anchors = max(5, n_anchors - 2)
            continue

        sp = _nn_mean(c_arr)

        # Steer n_anchors toward TARGET_SPACING and save best result
        centers_arr = c_arr
        avg_spacing = sp

        if sp > TARGET_SPACING + SPACING_TOL:
            n_anchors = min(50, n_anchors + 3)   # too sparse → add anchors
        elif sp < TARGET_SPACING - SPACING_TOL:
            n_anchors = max(5,  n_anchors - 2)   # too dense  → remove anchors
        else:
            break   # within tolerance

    # ── build elevation map, scale to pack target, relax ─────────────────────
    z_accum = _apply_piles(centers_arr)
    scale   = _pack_scale(z_accum)
    z_accum *= scale

    # Write scaled heights to grid; slight overflow allowed for relax to settle
    grid.z_height[:] = 0.0
    grid.z_height[valid_mask] = np.clip(
        z_accum[valid_mask], 0.0, TARGET_PILE_HEIGHT * 1.15
    ).astype(np.float32)

    grid.relax_all()   # enforces angle-of-repose, updates cell states + pheromone

    # Post-relax correction: if density drifted outside ±PACK_TOL, rescale
    pack = grid.pack_pct()
    if abs(pack - TARGET_PACK) > PACK_TOL:
        adj = TARGET_PACK / max(pack, 1e-6)
        grid.z_height = np.clip(
            grid.z_height * adj, 0.0, TARGET_PILE_HEIGHT * 0.95
        ).astype(np.float32)
        for r in range(grid.rows):
            for c in range(grid.cols):
                grid._update_state(r, c)
        grid.pheromone[grid.z_height > 0] = 0.0

    avg_spacing = float(_nn_mean(centers_arr))
    pack_final  = grid.pack_pct()

    print(f"Pre-fill: {len(centers_arr)} Gaussian-cone piles, "
          f"pack = {100 * pack_final:.1f}%, "
          f"avg NN spacing = {avg_spacing:.2f} m")

    return [(float(x), float(y)) for x, y in centers_arr], avg_spacing


def _corridor_cells(grid, path, truck):
    half_w = truck.width / 2.0 / grid.cell_size
    half   = int(math.ceil(half_w))
    result = set()
    for r, c in _path_cells(grid, path):
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                if math.hypot(dr, dc) <= half_w:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid.rows and 0 <= nc < grid.cols:
                        result.add((nr, nc))
    return result


def build_fleet():
    from config import TRUCK_CLASSES
    trucks     = []
    truck_id   = 0
    queue_slot = 0
    for cls in ('small', 'medium', 'large'):
        for _ in range(FLEET_COMPOSITION.get(cls, 0)):
            truck_len = TRUCK_CLASSES[cls]['length_m']
            offset    = (queue_slot + 1) * truck_len * 1.5
            home      = (ENTRY_POINT[0], ENTRY_POINT[1] - offset)
            if truck_id == 0:
                trucks.append(Truck(truck_id, cls, home_pos=home))
            else:
                trucks.append(Truck(truck_id, cls, start_pos=home,
                                    waiting=True, home_pos=home))
            queue_slot += 1
            truck_id   += 1
    return trucks


def _try_inplace_replan(ta, tb, grid, entry_rc, all_trucks=None):
    """Resolve a conflict between two trucks via CBS without forcing idle.
    Stays in Thread 1 (rare, fast-path — cooldown limits frequency).

    all_trucks: full fleet list.  Used to lock third-party trucks so the
    replanned paths for ta/tb cannot conflict with anyone else.
    """
    both_nav  = (ta.status == ta.STATUS_NAVIGATING and tb.status == tb.STATUS_NAVIGATING)
    both_exit = (ta.status == ta.STATUS_EXITING    and tb.status == tb.STATUS_EXITING)
    a_nav     = ta.status == ta.STATUS_NAVIGATING

    def _third_locks(excluded_ids):
        """Build locked_paths for all active trucks not in excluded_ids."""
        locked = {}
        for ot in (all_trucks or []):
            if ot.id in excluded_ids:
                continue
            if ot.status == ot.STATUS_NAVIGATING and ot.path:
                locked[ot.id] = _make_locked_entry(
                    ot, ot.path, ot._dump_ticks_required + 2, grid)
            elif ot.status == ot.STATUS_EXITING and ot._exit_path:
                locked[ot.id] = _make_locked_entry(ot, ot._exit_path, 0, grid)
            elif ot.status in (ot.STATUS_DUMPING, ot.STATUS_REVERSING):
                rem = max(0, ot._dump_ticks_required - ot._dump_ticks)
                locked[ot.id] = _make_locked_entry(ot, [], rem + 2)
        return locked

    if both_nav:
        if not ta.dump_target or not tb.dump_target:
            return False
        locked = _third_locks({ta.id, tb.id})
        ta.clear_all_corridors(grid)
        tb.clear_all_corridors(grid)
        new_paths, new_staging = plan_staging_paths(
            grid, [(ta, ta.dump_target), (tb, tb.dump_target)], locked_paths=locked)
        np_a = new_paths.get(ta.id, [])
        np_b = new_paths.get(tb.id, [])
        if np_a and np_b:
            ta.path = np_a;  ta.stop_target = ta.path[-1];  ta.staging_pose = new_staging.get(ta.id)
            tb.path = np_b;  tb.stop_target = tb.path[-1];  tb.staging_pose = new_staging.get(tb.id)
            ta._mark_dump_corridor(grid, ta.path)
            tb._mark_dump_corridor(grid, tb.path)
            ta._conflict_cooldown = 5;  tb._conflict_cooldown = 5
            print(f"[CBS-REPLAN] T{ta.id}+T{tb.id} nav conflict resolved in-place.")
            return True
        return False

    if both_exit:
        locked = _third_locks({ta.id, tb.id})
        ta._clear_exit_corridor(grid)
        tb._clear_exit_corridor(grid)
        new_paths = plan_paths_cbs(grid, [(ta, entry_rc), (tb, entry_rc)], locked_paths=locked)
        np_a = new_paths.get(ta.id, [])
        np_b = new_paths.get(tb.id, [])
        if np_a and np_b:
            ta._exit_path = np_a;  tb._exit_path = np_b
            ta._mark_exit_corridor(grid, np_a)
            tb._mark_exit_corridor(grid, np_b)
            ta._conflict_cooldown = 5;  tb._conflict_cooldown = 5
            print(f"[CBS-REPLAN] T{ta.id}+T{tb.id} exit conflict resolved in-place.")
            return True
        return False

    nav_t  = ta if a_nav else tb
    exit_t = tb if a_nav else ta

    if nav_t.path:
        # Lock nav_t's path plus all unrelated trucks; replan exit_t around everything.
        locked = _third_locks({exit_t.id})
        locked[nav_t.id] = _make_locked_entry(
            nav_t, nav_t.path, nav_t._dump_ticks_required + 2, grid)
        exit_t._clear_exit_corridor(grid)
        ep = plan_paths_cbs(grid, [(exit_t, entry_rc)], locked_paths=locked)
        np_exit = ep.get(exit_t.id, [])
        if np_exit:
            exit_t._exit_path = np_exit
            exit_t._mark_exit_corridor(grid, np_exit)
            exit_t._conflict_cooldown = 5
            print(f"[CBS-REPLAN] T{exit_t.id} exit replanned around T{nav_t.id}.")
            return True

    if exit_t._exit_path and nav_t.dump_target:
        # Lock exit_t's path plus all unrelated trucks; replan nav_t around everything.
        locked = _third_locks({nav_t.id})
        locked[exit_t.id] = _make_locked_entry(exit_t, exit_t._exit_path, 0, grid)
        nav_t.clear_all_corridors(grid)
        np2, ns2 = plan_staging_paths(grid, [(nav_t, nav_t.dump_target)],
                                      locked_paths=locked)
        np_nav = np2.get(nav_t.id, [])
        if np_nav:
            nav_t.path = np_nav;  nav_t.stop_target = nav_t.path[-1]
            nav_t.staging_pose = ns2.get(nav_t.id)
            nav_t._mark_dump_corridor(grid, nav_t.path)
            nav_t._conflict_cooldown = 5
            print(f"[CBS-REPLAN] T{nav_t.id} nav replanned around T{exit_t.id}.")
            return True

    return False
