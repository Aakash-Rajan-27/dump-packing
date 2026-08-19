# metrics.py
# ─────────────────────────────────────────────────────────────
# Run instrumentation — Phase 0 of the neural-LNS build.
#
# Two streams, both written to ONE .jsonl file, distinguished by
# the 'kind' field:
#   • kind='tick'  — one row per tick (fill/pack/spacing/counters)
#   • kind=<event> — one row per discrete event (intrusion, replan,
#                    deadlock, headlock, cbs_fallback, path_done…)
#
# Nothing here changes simulation behaviour: MetricsSink only
# observes.  A disabled sink (path=None) is a cheap no-op so the
# instrumented code paths cost nothing in normal runs.
#
# NOTE: main.py already uses a local variable named `metrics` for
# the renderer HUD dict, so main.py imports this module as
# `metrics as metrics_sink` to avoid shadowing.
# ─────────────────────────────────────────────────────────────

import io
import json
import math
import os
import time

import numpy as np


# Fields that are inherently non-deterministic (wall-clock timings).
# Stripped before computing a run signature so two same-seed runs can
# be compared for behavioural determinism.
_TIMING_FIELDS = ('latency_s', 'wall_clock_s', 'solver_s')


def _json_default(o):
    """numpy scalars/arrays are not JSON-serialisable by default."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


class MetricsSink:
    """Append-only .jsonl writer for tick rows and event rows.

    Construct with path=None to disable entirely (all record_* calls
    become no-ops), which is the default for interactive runs.
    """

    def __init__(self, path=None, run_id=None, config_snapshot=None,
                 flush_every=200):
        self.path        = path
        self.enabled     = path is not None
        self.run_id      = run_id or f"run_{int(time.time())}"
        self.flush_every = flush_every

        self._buf   = []
        self._fh    = None
        self._t0    = time.perf_counter()

        # Live counters — also mirrored into tick rows so the tick
        # stream is self-contained without needing event replay.
        self.counters = {
            'intrusions_opened':   0,
            'intrusions_closed':   0,
            'sustained_intrusions': 0,
            'collision_blocks':    0,
            'conflicts_detected':  0,
            'replans_attempted':   0,
            'replans_succeeded':   0,
            'cbs_fallbacks':       0,
            'deadlocks':           0,
            'headlocks':           0,
            'stuck_exits':         0,
            'force_idles':         0,
        }

        if not self.enabled:
            return

        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        self._fh = io.open(path, 'w', encoding='utf-8')
        self._write({'kind': 'run_start',
                     'run_id': self.run_id,
                     'config': config_snapshot or {}})

    # ── internals ────────────────────────────────────────────────────────────

    def _write(self, row):
        if not self.enabled:
            return
        self._buf.append(json.dumps(row, default=_json_default,
                                    sort_keys=True))
        if len(self._buf) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.enabled or not self._buf:
            return
        self._fh.write('\n'.join(self._buf) + '\n')
        self._fh.flush()
        self._buf.clear()

    def close(self, final=None):
        if not self.enabled:
            return
        self._write({'kind': 'run_end',
                     'run_id': self.run_id,
                     'wall_clock_s': time.perf_counter() - self._t0,
                     'counters': dict(self.counters),
                     **(final or {})})
        self.flush()
        self._fh.close()
        self._fh = None
        self.enabled = False

    # ── tick stream ──────────────────────────────────────────────────────────

    def record_tick(self, tick, fill_pct, pack_pct, active_trucks,
                    active_intrusions=0, avg_spacing=None, idle=0,
                    in_flight=0):
        self._write({
            'kind':              'tick',
            'tick':              tick,
            'fill_pct':          fill_pct,
            'pack_pct':          pack_pct,
            'active_trucks':     active_trucks,
            'idle_trucks':       idle,
            'in_flight':         in_flight,
            'active_intrusions': active_intrusions,
            'avg_spacing':       avg_spacing,
            'c_intrusions':      self.counters['intrusions_closed'],
            'c_replans':         self.counters['replans_attempted'],
            'c_deadlocks':       self.counters['deadlocks'],
        })

    # ── event stream ─────────────────────────────────────────────────────────

    def record_intrusion(self, tick, a, b, start_tick, duration_ticks,
                         in_entry_corridor):
        """One CLOSED intrusion (bodies overlapped, then separated)."""
        self.counters['intrusions_closed'] += 1
        if duration_ticks >= 2:
            self.counters['sustained_intrusions'] += 1
        self._write({
            'kind':               'intrusion',
            'tick':               tick,
            'truck_a_id':         a.id,
            'truck_a_class':      a.truck_class,
            'truck_b_id':         b.id,
            'truck_b_class':      b.truck_class,
            'overlap_start_tick': start_tick,
            'overlap_end_tick':   tick,
            'duration_ticks':     duration_ticks,
            'sustained':          duration_ticks >= 2,
            'truck_a_pos':        [round(a.pos[0], 3), round(a.pos[1], 3)],
            'truck_a_heading':    round(a.heading, 4),
            'truck_b_pos':        [round(b.pos[0], 3), round(b.pos[1], 3)],
            'truck_b_heading':    round(b.heading, 4),
            'in_entry_corridor':  in_entry_corridor,
        })

    def record_collision_block(self, tick, truck_id, other_id):
        """The hard-block guard fired (bypass OFF) — a would-be intrusion
        that was prevented by rolling the move back."""
        self.counters['collision_blocks'] += 1
        self._write({'kind': 'collision_block', 'tick': tick,
                     'truck_id': truck_id, 'other_id': other_id})

    def record_conflict_detected(self, tick, a_id, b_id, conflict_t, kind):
        self.counters['conflicts_detected'] += 1
        self._write({'kind': 'conflict_detected', 'tick': tick,
                     'truck_a_id': a_id, 'truck_b_id': b_id,
                     'conflict_t': conflict_t, 'conflict_kind': kind})

    def record_replan(self, tick, subset_ids, mode, latency_s, success,
                      cost_before=None, cost_after=None):
        self.counters['replans_attempted'] += 1
        if success:
            self.counters['replans_succeeded'] += 1
        self._write({'kind': 'replan', 'tick': tick,
                     'subset_ids': sorted(subset_ids), 'mode': mode,
                     'latency_s': latency_s, 'success': success,
                     'cost_before': cost_before, 'cost_after': cost_after})

    def record_cbs_fallback(self, tick, subset_ids, nodes_expanded):
        """CBS hit CBS_MAX_NODES and returned a best-effort node — the
        returned paths may still contain the conflict it could not
        resolve.  Leading indicator of downstream intrusions."""
        self.counters['cbs_fallbacks'] += 1
        self._write({'kind': 'cbs_fallback', 'tick': tick,
                     'subset_ids': sorted(subset_ids),
                     'nodes_expanded': nodes_expanded})

    def record_deadlock(self, tick, a_id, b_id, resolution, steps):
        self.counters['deadlocks'] += 1
        self._write({'kind': 'deadlock', 'tick': tick,
                     'truck_a_id': a_id, 'truck_b_id': b_id,
                     'resolution': resolution, 'steps': steps})

    def record_headlock(self, tick, a_id, b_id, reverser_id, steps):
        self.counters['headlocks'] += 1
        self._write({'kind': 'headlock', 'tick': tick,
                     'truck_a_id': a_id, 'truck_b_id': b_id,
                     'reverser_id': reverser_id, 'steps': steps})

    def record_stuck_exit(self, tick, truck_id):
        self.counters['stuck_exits'] += 1
        self._write({'kind': 'stuck_exit', 'tick': tick,
                     'truck_id': truck_id})

    def record_force_idle(self, tick, truck_id):
        """A NAVIGATING truck was force-reset to IDLE (path abandoned)."""
        self.counters['force_idles'] += 1
        self._write({'kind': 'force_idle', 'tick': tick,
                     'truck_id': truck_id})

    def record_path_completed(self, tick, truck_id, driven_m, straight_m):
        """Path efficiency: driven distance vs straight-line distance."""
        ratio = (driven_m / straight_m) if straight_m > 1e-6 else None
        self._write({'kind': 'path_done', 'tick': tick,
                     'truck_id': truck_id, 'driven_m': driven_m,
                     'straight_m': straight_m, 'efficiency_ratio': ratio})

    def record_dump(self, tick, truck_id, cell, nearest_neighbour_m):
        """One completed dump + its nearest-neighbour spacing."""
        self._write({'kind': 'dump', 'tick': tick, 'truck_id': truck_id,
                     'cell': list(cell) if cell else None,
                     'nn_spacing_m': nearest_neighbour_m})


class IntrusionTracker:
    """Tracks open truck-truck body overlaps across ticks so each
    overlap is logged ONCE with a duration, not once per tick.

    An intrusion is keyed by the unordered truck pair.  It opens on
    the first tick the two footprints overlap and closes on the first
    tick they no longer do.
    """

    def __init__(self, sink):
        self.sink  = sink
        self._open = {}      # frozenset({id_a, id_b}) -> dict

    def update(self, tick, overlapping_pairs, truck_map, entry_xy,
               corridor_r):
        """overlapping_pairs: iterable of (id_a, id_b) currently overlapping."""
        seen = set()
        for a_id, b_id in overlapping_pairs:
            key = frozenset((a_id, b_id))
            seen.add(key)
            if key not in self._open:
                a = truck_map.get(a_id)
                b = truck_map.get(b_id)
                in_corr = False
                if a and b and entry_xy is not None:
                    ex, ey = entry_xy
                    in_corr = (math.hypot(a.pos[0] - ex, a.pos[1] - ey) <= corridor_r
                               or math.hypot(b.pos[0] - ex, b.pos[1] - ey) <= corridor_r)
                self._open[key] = {'start': tick, 'in_corridor': in_corr}
                self.sink.counters['intrusions_opened'] += 1

        # Close any intrusion whose pair no longer overlaps.
        for key in list(self._open):
            if key in seen:
                continue
            rec = self._open.pop(key)
            a_id, b_id = sorted(key)
            a, b = truck_map.get(a_id), truck_map.get(b_id)
            if a is not None and b is not None:
                self.sink.record_intrusion(
                    tick, a, b, rec['start'], tick - rec['start'],
                    rec['in_corridor'])

    def active_count(self):
        return len(self._open)

    def close_all(self, tick, truck_map):
        """Flush still-open intrusions at end of run."""
        for key in list(self._open):
            rec = self._open.pop(key)
            a_id, b_id = sorted(key)
            a, b = truck_map.get(a_id), truck_map.get(b_id)
            if a is not None and b is not None:
                self.sink.record_intrusion(
                    tick, a, b, rec['start'], max(1, tick - rec['start']),
                    rec['in_corridor'])


# ── Offline analysis ─────────────────────────────────────────────────────────

def load(path):
    """Read a .jsonl run log into a list of row dicts."""
    rows = []
    with io.open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _pct(values, p):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(path):
    """Aggregate one run log into the KPIs used for LNS ablations."""
    rows   = load(path)
    ticks  = [r for r in rows if r.get('kind') == 'tick']
    intr   = [r for r in rows if r.get('kind') == 'intrusion']
    replan = [r for r in rows if r.get('kind') == 'replan']
    dumps  = [r for r in rows if r.get('kind') == 'dump']
    paths  = [r for r in rows if r.get('kind') == 'path_done']
    end    = next((r for r in rows if r.get('kind') == 'run_end'), {})

    n_ticks   = len(ticks)
    sustained = [r for r in intr if r.get('sustained')]
    durations = [r['duration_ticks'] for r in intr if r.get('duration_ticks')]

    # Clean fraction — share of ticks with zero active intrusions.
    clean = sum(1 for r in ticks if not r.get('active_intrusions'))

    spacings = [r['nn_spacing_m'] for r in dumps
                if r.get('nn_spacing_m') is not None]
    ratios   = [r['efficiency_ratio'] for r in paths
                if r.get('efficiency_ratio') is not None]
    lat      = [r['latency_s'] for r in replan
                if r.get('latency_s') is not None]

    counters = end.get('counters', {})

    def _per_1k(n):
        return (n / n_ticks * 1000.0) if n_ticks else None

    return {
        'run_id':                 end.get('run_id'),
        'total_ticks':            n_ticks,
        'wall_clock_s':           end.get('wall_clock_s'),

        # ── primary collision KPIs ──
        'intrusions':             len(intr),
        'intrusions_sustained':   len(sustained),
        'intrusions_momentary':   len(intr) - len(sustained),
        'intrusions_per_1k_ticks': _per_1k(len(intr)),
        'clean_fraction':         (clean / n_ticks) if n_ticks else None,
        'intrusion_dur_mean':     (sum(durations) / len(durations)) if durations else None,
        'intrusion_dur_p90':      _pct(durations, 90),
        'collision_blocks':       counters.get('collision_blocks'),

        # ── coordination effort ──
        'conflicts_detected':     counters.get('conflicts_detected'),
        'replans_attempted':      counters.get('replans_attempted'),
        'replans_succeeded':      counters.get('replans_succeeded'),
        'replan_latency_mean':    (sum(lat) / len(lat)) if lat else None,
        'replan_latency_p95':     _pct(lat, 95),
        'cbs_fallbacks':          counters.get('cbs_fallbacks'),
        'deadlocks':              counters.get('deadlocks'),
        'headlocks':              counters.get('headlocks'),
        'force_idles':            counters.get('force_idles'),
        'stuck_exits':            counters.get('stuck_exits'),

        # ── packing quality ──
        'final_fill_pct':         ticks[-1]['fill_pct'] if ticks else None,
        'final_pack_pct':         ticks[-1]['pack_pct'] if ticks else None,
        'dumps':                  len(dumps),
        'spacing_mean':           (sum(spacings) / len(spacings)) if spacings else None,
        'spacing_p50':            _pct(spacings, 50),
        'spacing_p90':            _pct(spacings, 90),

        # ── planner quality ──
        'path_efficiency_mean':   (sum(ratios) / len(ratios)) if ratios else None,
    }


def signature(path):
    """Behavioural fingerprint of a run, with wall-clock timing fields
    stripped.  Two same-seed runs must produce the SAME signature —
    this is the determinism check (raw byte-compare fails because
    latency measurements legitimately vary run to run)."""
    import hashlib
    h = hashlib.sha256()
    for row in load(path):
        if row.get('kind') in ('run_start', 'run_end'):
            continue
        clean = {k: v for k, v in row.items() if k not in _TIMING_FIELDS}
        h.update(json.dumps(clean, sort_keys=True,
                            default=_json_default).encode('utf-8'))
    return h.hexdigest()


if __name__ == '__main__':
    import argparse
    _ap = argparse.ArgumentParser(description='Summarize a metrics run log.')
    _ap.add_argument('path')
    _ap.add_argument('--signature', action='store_true',
                     help='Print the determinism signature instead.')
    _a = _ap.parse_args()
    if _a.signature:
        print(signature(_a.path))
    else:
        for k, v in summarize(_a.path).items():
            print(f"{k:26s} {v}")
