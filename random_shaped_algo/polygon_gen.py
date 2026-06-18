# polygon_gen.py
# ─────────────────────────────────────────────────────────────
# Generates a random non-self-intersecting dump-site polygon.
#
# Shape structure (bottom → top):
#   • Entry section  — wide base, ENTRY_POINT is an exact vertex
#   • Narrow neck    — 1.5–2.5× largest truck width; random offset
#   • Main dump area — random dome, 60-100 m wide
#
# Guarantees:
#   • Polygon is always valid (shapely-checked, buffer(0) repaired)
#   • Neck ≥ 1.5× largest truck width, so trucks can always pass
#   • Entry section width ~60-90 m — similar feel to the original
# ─────────────────────────────────────────────────────────────

import math
import random
import shapely.geometry

from config import ENTRY_POINT, TRUCK_CLASSES


def generate_random_polygon(seed=None):
    """
    Return a list of (x, y) world-coordinate vertices.

    Vertex order is counterclockwise:
      ENTRY_POINT vertex → right entry corner → right neck wall
      → main-area arc (right → top → left) → left neck wall
      → left entry corner  (polygon closes back to ENTRY_POINT)

    seed: int for reproducible shapes; None = different each run.
    """
    rng = random.Random(seed)
    ex, ey = ENTRY_POINT

    max_truck_w = max(cls['width_m'] for cls in TRUCK_CLASSES.values())  # 9.8 m

    for attempt in range(60):
        # ── Entry section (bottom) ────────────────────────────────────────
        # Width similar to original polygon bottom (~90 m).
        entry_w = rng.uniform(60, 92)
        half_ew = entry_w / 2.0

        # ── Narrow neck ────────────────────────────────────────────────────
        # Must be wider than the largest truck, but clearly narrower than entry.
        neck_w  = rng.uniform(max_truck_w * 1.5, max_truck_w * 2.4)
        half_nw = neck_w / 2.0

        # Neck center x: constrained so neck fits inside entry width.
        max_off = (half_ew - half_nw - 4)          # 4 m clearance to entry walls
        if max_off < 0:                             # shouldn't happen with params above
            continue
        neck_cx = ex + rng.uniform(-max_off, max_off)

        # Neck height above entry base: how far up the bottleneck sits.
        neck_y = ey + rng.uniform(18, 40)

        # ── Main dump area (above neck) ────────────────────────────────────
        main_h  = rng.uniform(50, 90)
        main_w  = rng.uniform(60, 105)

        # Main area center x can drift from neck center (creates L-shape feel).
        max_main_drift = min(20, (main_w / 2.0 - half_nw) * 0.6)
        main_cx = neck_cx + rng.uniform(-max_main_drift, max_main_drift)

        # Main area radii (half-extents) for the polar arc.
        r_min = min(main_w, main_h) * 0.38
        r_max = max(main_w, main_h) * 0.58

        main_cy = neck_y + main_h * 0.5   # vertical centre of main area

        # ── Generate main-area arc via polar sort on [0, π] ───────────────
        # Angles 0 → π map right → top → left.
        # One random angle per equal sector → sorted → no self-crossing.
        n_main = rng.randint(5, 10)
        step   = math.pi / n_main
        angles = sorted(rng.uniform(i * step, (i + 1) * step)
                        for i in range(n_main))

        # Scale x and y radii separately to get the right aspect ratio.
        rx = (main_w / 2.0)
        ry = (main_h / 2.0)
        main_verts = [(main_cx + rng.uniform(r_min, r_max) / r_max * rx * math.cos(a),
                       main_cy + rng.uniform(r_min, r_max) / r_max * ry * math.sin(a))
                      for a in angles]

        # ── Assemble polygon CCW ───────────────────────────────────────────
        verts = (
            [(ex, ey),                              # ← ENTRY VERTEX
             (ex + half_ew, ey),                   # entry right corner
             (neck_cx + half_nw, neck_y)]           # neck right wall
            + main_verts                            # main area arc
            + [(neck_cx - half_nw, neck_y),         # neck left wall
               (ex - half_ew, ey)]                  # entry left corner
        )

        poly = shapely.geometry.Polygon(verts)
        if not poly.is_valid:
            poly = poly.buffer(0)          # repair minor self-intersections
        if poly.is_empty or poly.area < 2500:
            continue

        coords = list(poly.exterior.coords)
        return coords[:-1]                 # drop repeated closing vertex

    # ── Deterministic fallback ─────────────────────────────────────────────
    # Wide base → narrow neck → large main area.  Always valid.
    hw  = 44.0
    hnw = max_truck_w * 1.6
    return [
        (ex,      ey),
        (ex + hw, ey),
        (ex + hnw, ey + 30),
        (ex + hw,  ey + 30 + 70),
        (ex - hw,  ey + 30 + 70),
        (ex - hnw, ey + 30),
        (ex - hw,  ey),
    ]
