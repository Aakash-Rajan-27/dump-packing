# Neural-LNS Integration Design Doc

Status: pre-implementation design. No code has been written yet. This
document is the thing to review/argue with before Phase 0 starts.

## 1. Why this exists

`final_code_with_LNS/` is a copy of `final_code/` (the approved
CBS-based MAPF sim). The goal is to add a **neural-guided Large
Neighborhood Search (LNS)** outer loop, adapting ideas from
*"Neural Neighborhood Search for Multi-Agent Path Finding"*
(Yan & Wu, ICLR 2024), to:

1. reduce planner computation time, and
2. reduce agent-agent collisions,

without discarding the existing CBS/space-time-A* solver, which
already works and is footprint-aware (oriented-rectangle SAT
collision checks, not naive cell occupancy).

### 1.1 How each goal is actually achieved (be precise about this)

These two goals are **not** achieved by the same mechanism, and
conflating them overclaims what LNS does. Stated honestly:

**Computation (direct, this is what LNS is for).** Re-solving a
k-truck subset is vastly cheaper than re-solving the whole fleet, and
a learned scorer means only the top-M most promising subsets get an
expensive CBS call at all. This is the paper's actual contribution
and it transfers here directly.

**Collisions (indirect — requires deliberate mechanisms, not a free
side effect).** CBS is already collision-free *by construction*:
whatever subset it is handed, its constraint tree guarantees the
returned paths don't conflict with each other or with anything
locked. So "solve smaller subsets" does **not** make output paths
more collision-free than "solve everything at once" — both are sound
within their subproblem. In the paper's own benchmark there is no
collision metric at all; LNS optimizes *sum-of-delay* on an
already-feasible solution. Any collision reduction in *this*
codebase therefore has to come from three codebase-specific effects,
each of which must be built deliberately:

1. **Wider detection net (coverage, not solver quality).** Today,
   replanning only fires once `_detect_first_conflict` already flags
   a specific pair as about to collide (`main.py:440-465`) — a
   narrow, late, reactive trigger. A periodic LNS pass over the
   *whole* active fleet with wider subsets catches conflict-prone
   configurations earlier, before they reach the physical SAT guard
   at `main.py:375-415`. More trucks considered, more often, earlier.
2. **Closing the stale-snapshot race (load-bearing, see Phase 3).**
   The planner solves against a frozen snapshot (`_make_grid_snapshot`);
   by the time a result is applied, live state may have moved. Two
   independently-solved, each-individually-conflict-free subproblems
   can clash once both are applied. **LNS does not fix this for free** —
   it requires an explicit re-verify-against-live-state check before
   accepting any subproblem result. This is a named requirement of
   Phase 3, not an incidental verification step.
3. **Suppressing the deadlock/headlock escape path (the likely real
   culprit).** The reverse-and-arc / yield maneuvers in `deadlock.py`
   are hand-written heuristics that are **not** CBS- or SAT-verified
   against every other truck — they are the most plausible source of
   genuine physical body overlaps. LNS reducing congestion and delay
   means fewer trucks wedge into contended states, which means fewer
   deadlock triggers, which means fewer opportunities for that
   unverified escape logic to cause a real intrusion. Real effect,
   but strictly second-order: LNS optimizes delay, and delay
   reduction happens to suppress the conditions that fire the
   actually-risky code path.

**The falsifiable prediction**, therefore, is: intrusion count under
bypass mode should fall mainly via mechanisms (1) and (3), and the
Phase 7 ablation should show *deadlock/headlock maneuver counts*
falling alongside intrusion counts. If intrusions drop but deadlock
maneuvers don't, the causal story above is wrong and worth
re-examining rather than claiming the win.

## 2. Framing decision

**CBS stays as the low-level solver. LNS is added as an outer loop
that decides *which subset of trucks* to re-run CBS on, and *when*.**
This mirrors the paper exactly — MAPF-LNS/Neural-LNS never
reimplements the low-level solver; it wraps it. Concretely:

- Today, conflict resolution is **reactive and pairwise only**:
  `main.py` detects a future conflict between two trucks each tick
  (`main.py:440-465`) and calls `_try_inplace_replan(ta, tb, ...)`
  (`sim_helpers.py:68`), which locks every *other* truck's path as a
  hard constraint (`_third_locks`, `sim_helpers.py:79-93`) and
  re-invokes `plan_paths_cbs` / `plan_staging_paths` for just that
  pair.
- That "CBS-with-everyone-else-locked" call **is already the
  subproblem solver** the paper calls `Solver(P(Sⁱ, α))`. What's
  missing is: (a) a version of it that isn't hardcoded to exactly 2
  trucks, (b) a loop that considers subsets beyond "the pair currently
  colliding," (c) a way to score candidate subsets before spending
  CBS time on them, and (d) numbers to know if any of this helped.

This means **no MAPF solver gets rewritten**. Every phase below is
additive: new modules plus small, mechanical edits to
`sim_helpers.py`, `main.py`, `planning_worker.py`, `config.py`. The
existing deadlock/headlock escape-maneuver code (`deadlock.py`,
`main.py` ~467-673) is explicitly **out of scope and untouched** — it
handles physically-stuck trucks, a failure mode LNS/CBS reasoning
doesn't reach.

## 3. Gaps in the current codebase (things that must be built, not reused)

| Gap | Detail |
|---|---|
| No metrics/instrumentation | Only `print()` statements (`[CONFLICT]`, `[DEADLOCK]`, `[CBS-REPLAN]`). Nothing is persisted — no collision counts, replan latency, or delay stats survive a run. Any "reduced collisions" claim is currently unfalsifiable. |
| No true collision/intrusion metric | `main.py:375-415` already hard-blocks truck-body overlap (rolls the move back the instant SAT detects overlap), so today there is no way to observe the *underlying* conflict rate the planner produces — every near-miss is invisibly absorbed by a rollback before it can be counted. Need a bypass mode (§5, Phase 0) to make this measurable. |
| No LNS of any kind | `grep -rn "LNS\|large neighborhood"` across the repo is empty. |
| Subset-of-2 hardcoding | `_try_inplace_replan` and `_third_locks` are written for exactly `ta, tb`, not an arbitrary subset. |
| No parallelism across cores | Only Python `threading` (`planning_worker.py`, one daemon thread, GIL-bound). Zero use of the 32 logical CPUs for planning. |
| No ML infra | No `torch`/`sklearn` installed, no `requirements.txt` at all. `.venv` currently has only `numpy`, `scipy`, `shapely`, `pygame`, `Pillow`. |
| No determinism guarantee | `initialise_half_full_dump` (`sim_helpers.py`) calls `random.choice` without a seeded RNG — A/B comparisons need reproducible runs. |
| Default fleet is tiny | `FLEET_COMPOSITION` (`config.py:52-56`) defaults to 3 small trucks, 0 medium, 0 large. LNS needs enough concurrent agents to have neighborhoods worth searching. |

## 4. Assumptions being made (flag if any of these are wrong)

1. **CBS/`plan_paths_cbs` + `plan_staging_paths` remain correct and untouched.** LNS only changes *which* trucks get handed to them and *when*.
2. **`solve_subproblem` calls are side-effect-free (propose, don't apply).** This is required so we can evaluate multiple candidate subsets and keep only the best without mutating shared grid/truck state J times. Verified achievable from reading `cbs_planner.py` — it takes `grid` + `assignments` + `locked_paths` and returns new paths without mutating the passed-in trucks.
3. **The pheromone field (`grid.pheromone`, `grid_map.py`) is a usable congestion proxy** for the "Intersection-local" subset-construction heuristic, since it's suppressed by truck traffic (`TRAIL_STRENGTH`, `grid_map.py:265-266`). Not verified by actually running the sim yet — will confirm in Phase 2.
4. **`Truck` and `GridMap` snapshot objects are pickle-safe** for `ProcessPoolExecutor` (Phase 4 only). Not yet verified — no locks/pygame-surface references were spotted while reading `truck.py`/`grid_map.py`, but this needs a real `pickle.dumps()` spike before committing to multiprocessing, since Windows uses spawn (re-imports every module, stricter than fork). **If this fails, the fallback is `ThreadPoolExecutor`** (still GIL-bound during CBS/A*, but avoids the pickling risk entirely) — Phase 4 will spike this first and pick a lane before writing the real implementation.
5. **A linear/handcrafted-features scoring model (not a deep 3D-conv+attention net) is the right starting point.** The paper's deep architecture assumes hundreds of agents on a discrete occupancy grid with a GPU. This fleet defaults to single-digit truck counts with continuous footprint-based collision — the paper's own results show the "Linear" baseline is competitive with the deep model in several settings, and it needs no GPU, no big training-data budget, and is trivially fast at inference. A PyTorch graduation path (Phase 6b) is explicitly optional, gated on the linear model actually hitting a ceiling.
6. **Data collection budget is scaled down ~1000x from the paper.** The paper collects up to 9000 seeds over 10-50 GPU-hours. Given this fleet size and CPU-only hardware, the plan targets ~50-100 headless seeds (~100k-300k labeled subproblem rows), achievable in a few hours wall-clock.
7. **Reactive pairwise replanning stays live as a fallback** even after the LNS loop is added (defense in depth, and it gives a clean baseline to A/B against via a `--lns-mode {off,reactive_only,unguided,guided}` flag).

## 5. Phase-by-phase plan

### Phase 0 — Metrics & instrumentation (prerequisite)

This phase absorbed a teammate's separate metrics-handoff doc — the
full inventory below is now the Phase 0 scope, not just LNS-specific
counters, since collisions/coordination-effort/packing-quality/
scalability all need to be measured through the same pipeline to make
apples-to-apples ablations possible later (§ "Ablation & evaluation
strategy").

**Metric inventory** (status = whether it exists today in any form):

| # | Metric | Measures | Status today |
|---|---|---|---|
| 1 | Avg. nearest-neighbour dump spacing | Packing quality | Tracked live (`_avg_spacing`), not logged over time |
| 2 | Spacing distribution (median/std/p90) | Worst-case packing, not just average | Not implemented |
| 3 | `fill_pct()` / `pack_pct()` over time | Density convergence | Tracked live, not logged |
| 4 | Ticks to reach target fill % | Throughput | Not implemented |
| 5 | Path efficiency (driven ÷ straight-line distance) | Planner quality | Not implemented |
| 6 | **Truck-truck intrusions** (see below) | Multi-agent coordination failure rate | **Not measurable at all today — hard-blocked before it can be counted** |
| 7 | CBS replans / freeze-recoveries / deadlock maneuvers per run | Coordination *effort* | Prints to console only, not counted |
| 8 | Planning latency vs. truck count (CBS nodes expanded, wall-clock ms) | Scalability | Not implemented |
| 9 | CBS fallback-to-best-effort rate (hit `CBS_MAX_NODES`) | Leading indicator of intrusions | Not implemented |

- New `metrics.py`: `MetricsSink` class (imported into `main.py` under
  the alias `metrics_sink` — **`main.py:417` already has a local
  variable literally named `metrics`** for the HUD dict, so a plain
  `import metrics` would shadow/confuse; use `import metrics as
  metrics_sink` or rename the HUD dict, decide during implementation
  but flag now so it isn't a mid-Phase-0 surprise).
  - Tick-level writer: `record_tick(tick, fill_pct, pack_pct, avg_spacing, active_trucks, cbs_replans_this_tick, active_intrusions_this_tick)`.
  - Event-level writer (single `.jsonl`, `event_type` column distinguishes them): `record_conflict_detected`, `record_replan` (latency, success), `record_deadlock`, `record_headlock`, `record_cbs_fallback` (hit `CBS_MAX_NODES`), `record_intrusion` (see below), `record_path_completed` (driven vs. straight-line distance).
  - `summarize(path) -> dict`: collision/intrusion count (momentary vs. sustained split), intrusions per 1000 ticks, **clean fraction** (fraction of ticks with zero active intrusions fleet-wide — the single most important commit-to-commit KPI per the handoff doc), replan latency percentiles, CBS fallback rate, stuck-tick (livelock proxy) count, sum-of-delay, spacing median/std/p90, ticks-to-target-fill.
- Edits: `main.py` (call sites at the existing `[CONFLICT]`/`[DEADLOCK]`/`[HEADLOCK]`/`[CBS-REPLAN]` prints, plus `--seed`/`--metrics-out` CLI flags and seeding `random`/`np.random` at startup), `sim_helpers.py` (`_try_inplace_replan` returns richer success/latency info instead of a bare bool), `astar_core.py`/`hybrid_astar.py` call sites (log straight-line vs driven distance once a truck completes a path).
- **Verify:** same-seed run twice → byte-identical `.jsonl` (proves determinism, required for every later A/B).

#### Truck-truck intrusion metric & collision-bypass mode

The current hard safety net (`main.py:375-415`) detects SAT
rectangle overlap after every `truck.step()` and **immediately rolls
the move back** (`collided` branch). This is good for the live demo
(no visible pass-through) but means the *true* underlying conflict
rate the planner would otherwise produce is invisible — every
near-collision is silently absorbed before it can be measured. To
actually quantify "did the LNS changes reduce collisions," we need a
mode where overlaps are allowed to happen and are *counted* rather
than *prevented*.

- **Config flag** (`config.py`): `ALLOW_COLLISION_BYPASS = False`
  (off by default — see below) plus `INTRUSION_LOG_PATH`.
- **Hook**: `main.py:375-415`, the existing `collided` block, gets an
  `if ALLOW_COLLISION_BYPASS:` branch — when true, skip the
  pos/heading/path rollback entirely (let the two trucks' bodies
  actually overlap and both continue moving on their committed
  paths), but still call `metrics_sink.record_intrusion(...)`. This
  reuses the exact SAT check and truck-pair loop already present at
  that line — no new collision test is written, only the *response*
  to a positive changes.
- **Definition**: an intrusion = a tick where two trucks' oriented
  rectangles genuinely overlap (same SAT test as line 394,
  `_rect_overlap_2d`), **including inside the entry corridor** — the
  corridor exemption at `main.py:389-393` exists for planning-time
  conflict tolerance, not for physical-body truth, so the intrusion
  check does *not* inherit that exemption (log `in_entry_corridor` as
  a field instead, so it can be filtered post-hoc rather than
  silently excluded).
- **Momentary vs. sustained**: track open intrusions in a dict keyed
  by `frozenset({truck_a.id, truck_b.id})`; an intrusion still open
  next tick is the same event (increment duration), one that clears
  is closed and logged with `duration_ticks`. Sustained (duration ≥
  2 ticks) is flagged separately — per the handoff doc, this is the
  strongest single signal of a genuine planning gap vs. a harmless
  near-miss that resolves itself through ordinary motion.
- **Log row** (event-level, `event_type='intrusion'`): `tick,
  truck_a_id, truck_a_class, truck_b_id, truck_b_class,
  overlap_start_tick, overlap_end_tick, duration_ticks, truck_a_pos,
  truck_a_heading, truck_b_pos, truck_b_heading, in_entry_corridor`.
- **Renderer** (`renderer.py`, optional/nice-to-have, not required
  for the metrics pipeline itself): flash overlapping trucks red for
  the duration of an open intrusion when bypass mode is on, so it's
  visible during interactive debugging, not just in the log.
- **Default stays `False`** for any interactive/demo run — the visible
  hard-block behavior is unchanged unless the flag is explicitly
  flipped on. **Bypass mode is strictly a benchmarking/ablation
  setting**, turned on only by `eval_harness.py` (Phase 7) runs, never
  by default. This makes the change purely additive to `main.py`'s
  existing behavior, consistent with the rest of this plan's
  additive-only approach (§7 non-goals).

Why this matters for the LNS work specifically: **the intrusion count
under bypass mode is the actual "reduce collisions" success metric**
the project's real goal (per the earlier /plan session) is scored
against — sum-of-delay and wall-clock are secondary. Collision-guard
rollback events under the *default* (bypass-off) mode are also worth
logging as their own counter (how often the hard-block fires at all)
since a lower rollback-event rate under LNS is itself a weaker but
still valid signal, usable even for the demo-mode default config.

### Phase 1 — Generalize the subproblem solver
- New `subproblem.py`: `build_third_party_locks(grid, subset_ids, all_trucks)` (lifted/generalized from `_third_locks`), `solve_subproblem(grid, subset, all_trucks, entry_rc) -> SubproblemResult` (dataclass: success, new_paths, new_staging, cost_before, cost_after, latency_s) — handles arbitrary subset size, partitions into nav/exit/mixed the way `_try_inplace_replan`'s branches already do, treats dumping/reversing trucks caught in a subset as un-touchable.
- Edit: `sim_helpers.py::_try_inplace_replan` becomes a thin wrapper calling `solve_subproblem` on `{ta.id, tb.id}` — this is a pure refactor, so it must reproduce today's behavior exactly.
- **Verify:** same-seed run before/after refactor → identical replan counts/latencies/outcomes in `metrics.jsonl`. Plus manual calls on 1-truck and 3-truck subsets to confirm no crash outside the historical 2-truck path.

### Phase 2 — Subset construction heuristics
- New `neighborhood.py`: `build_conflict_graph`, `sample_uniform`, `sample_agent_local` (random-walk from a delayed/conflicting truck), `sample_intersection_local` (random-walk from a low-pheromone/high-congestion cell), `construct_candidates(trucks, grid, J, rng, mix)`. All take an explicit seeded `rng`.
- No integration yet — standalone, tested against a scripted synthetic fleet.

### Phase 3 — Unguided LNS loop (single-process)
- New `lns_loop.py::run_lns_pass(...)`: build J candidates (Phase 2), solve each via `solve_subproblem` (Phase 1), accept the best cost-improving, still-conflict-free subset (order-independent "best of batch" rule, chosen specifically so Phase 4's parallelism doesn't change results), log via `metrics.py`.
- **Re-verify-before-accept (load-bearing correctness requirement, per §1.1 mechanism 2).** A subproblem is solved against a *snapshot*; live truck/grid state may have advanced before the result is applied. Before any accept, re-run `_detect_first_conflict` on the proposed paths **against the current live state of all non-subset trucks** — not against the snapshot they were planned on — and reject the result if it now conflicts. Without this step, LNS can actively *introduce* intrusions (two separately-valid proposals clashing on application), which would invert the collision goal. This is a correctness mechanism, not a test assertion.
- New config: `LNS_J`, `LNS_SUBSET_SIZE_RANGE`, `LNS_PASS_PERIOD_TICKS`, `LNS_MAX_ACCEPTS_PER_PASS`, `LNS_BUDGET_S` in `config.py`.
- Edit: `main.py` gains a periodic LNS pass call (after the reactive block, before deadlock resolution), gated by `--lns-mode`.
- **Verify (first real experiment):** same-seed A/B, `reactive_only` vs `unguided` — collision count must not increase (assert via `_detect_first_conflict` after every accept, don't just trust the lock), sum-of-delay should drop.

### Phase 4 — Parallelize across cores
- Spike first: `pickle.dumps()` a live `Truck`/`GridMap` snapshot to settle the `ProcessPoolExecutor` vs `ThreadPoolExecutor` question (see Assumption 4).
- `planning_worker.py` gains an `'lns_pass'` task type; candidate construction + dispatch happens there, reusing the existing snapshot pattern (`_make_grid_snapshot`, `_make_truck_snapshot`); a module-level pool (sized `cpu_count()-2`) evaluates candidates, best accepted result(s) pushed to `result_q`.
- `main.py` pushes `'lns_pass'` tasks on the periodic cadence and applies accepted results from `result_q` the same way the existing `'idle'/'exit'/'gate'` results are applied.
- **Verify:** pool-size-1 output ≈ Phase 3 single-process output (same seed); then scale up and confirm near-linear wall-clock reduction with no render-thread stutter.

### Phase 5 — Data collection mode
- `--collect-data <path>` flag; logs `(features, cost_before - cost_after)` for **every** evaluated candidate (not just the winner — negatives are needed for ranking).
- Feature extraction shared with Phase 6's scorer (see below), computed cheaply without solving.
- **Verify:** row counts match `passes × J`; spot-check a few rows against `metrics.jsonl`/console output for consistency.

### Phase 6 — Linear scoring model
- New `lns_scoring.py` (name avoids collision with existing `scoring.py`): `extract_features` (subset size, stagnation/cooldown stats, mean pairwise distance, local pheromone, exit/nav mix, truck-class footprint, current remaining path length, conflict-graph density within the subset, local driveable-cell headroom), `train_linear_model` (numpy/sklearn Ridge or pairwise-hinge, matching the paper's ranking framing), `LNSScorer.score(...)`.
- New `requirements.txt` (numpy, scipy, shapely, pygame, pillow, scikit-learn) — first one in the repo.
- Integration point: inside `planning_worker.py`'s `'lns_pass'` handler — score all J candidates cheaply, only actually `solve_subproblem` (expensive CBS call) on the top-M ranked. This is the paper's actual compute-saving mechanism.
- **Verify:** offline — Spearman correlation between predicted and true improvement on held-out seeds. Online — A/B `guided` vs `unguided` vs `reactive_only`: guided should match/beat unguided's collision/delay improvement at lower wall-clock (fewer CBS calls per pass).
- **Phase 6b (optional, only if linear underperforms):** small PyTorch MLP over the same handcrafted features, trained on the RTX 5070, inference still cheap enough for CPU. Not the paper's 3D-conv+attention architecture — that assumes a discrete-grid representation over hundreds of agents that doesn't fit this footprint-continuous small fleet.

### Phase 7 — Evaluation harness
- New `eval_harness.py`: runs `run_simulation(headless=True, seed=s, lns_mode=m, allow_collision_bypass=True, ...)` across a fixed seed list × `{reactive_only, unguided, guided}` (× optionally truck-count, per the scalability sweep below) in parallel subprocesses, aggregates `metrics_sink.summarize()` into a comparison table.
- Bypass mode (Phase 0) is turned **on** for every harness run — this is exactly the "internal benchmarking" use case the flag exists for; the harness never runs with the demo/hard-block default.
- **Verify harness itself first:** `reactive_only` vs `reactive_only` (same seeds) should show ~zero delta before trusting cross-mode deltas.

### Ablation & evaluation strategy (spans Phase 0 and Phase 7)

The point of Phase 0's metrics is to make every later phase's "did
this help" question answerable with numbers, not renderer-watching.
Concretely:

- **Primary KPI**: intrusions per 1000 ticks and clean fraction
  (bypass mode on), compared across `lns_mode ∈
  {reactive_only, unguided, guided}`. This is the direct "reduced
  collisions" evidence.
- **Secondary KPIs**: sum-of-delay, wall-clock runtime, replan
  latency percentiles, CBS fallback-to-best-effort rate (a rising
  fallback rate under a given mode is a leading indicator its
  intrusion count will also rise — worth plotting together).
- **Scalability sweep**: repeat the `lns_mode` comparison at multiple
  `FLEET_COMPOSITION` sizes (e.g. 3 → 15 trucks, fixed mix, then a
  fixed count with all-small/all-large/mixed composition) — this
  directly tests whether Unguided/Guided LNS's advantage over
  reactive-pairwise-only *grows* with fleet size, which is the
  regime where the paper's own results show the largest gains and
  is the most convincing evidence for this project's specific claim
  (reduced computation *and* reduced collisions at scale).
- **Statistical rigor**: fix a seed list, run **N ≥ 10 seeds per
  (mode, fleet-size) cell**, report mean ± std for every KPI — a
  single run's numbers have real spread (MCTS rollouts and initial
  terrain seeding are both randomized), so one lucky/unlucky seed
  must never be reported as if it were representative.
- **Reporting**: `eval_harness.py` should emit a table shaped like:

  | lns_mode | fleet size | intrusions/1000 ticks | clean fraction | sum-of-delay | wall-clock (s) | CBS fallback rate |
  |---|---|---|---|---|---|---|
  | reactive_only | 3 | | | | | |
  | unguided | 3 | | | | | |
  | guided | 3 | | | | | |
  | reactive_only | 9 | | | | | |
  | … | | | | | | |

  Left blank here deliberately — filled in only once Phase 7 actually
  runs, never with placeholder numbers.

## 6. Dependency graph

```
Phase 0 (metrics) ──┬─→ Phase 1 → Phase 2 → Phase 3 (single-proc unguided)
                     │                              │
                     │                              ▼
                     │                       Phase 4 (parallelize)
                     │                        │            │
                     │                        ▼            ▼
                     │                 Phase 5 (data)       │
                     │                        │             │
                     │                        ▼             │
                     │                 Phase 6 (linear scorer) ─┘
                     └──────────────────────────────────────────→ Phase 7 (eval harness)
```

Phase 0 gates everything (all later phases are validated through its
metrics). Phases 1-2 are independent of each other. Phase 4 must not
start until Phase 3's algorithm is validated single-process — don't
parallelize before correctness is proven. Phase 7 is last.

## 7. Explicit non-goals

- Not touching `astar_core.py`, `conflict_detect.py`, `bicycle_model.py`, `hybrid_astar.py`, or the SAT footprint collision logic.
- Not touching `deadlock.py` or the deadlock/headlock resolution blocks in `main.py`.
- Not attempting the paper's full 3D-conv + intra-path-attention architecture as a first cut.
- Not modifying `final_code/` — all work happens in `final_code_with_LNS/` only.

## 8. Open items to confirm before Phase 0 code lands

- Fleet size to actually test with — default `FLEET_COMPOSITION` (3 small trucks) may be too small to produce meaningful LNS signal; recommend also testing at a bumped-up mixed fleet (8-12 trucks) once Phase 7 exists, and specifically running the scalability sweep (3→15 trucks) once Phase 7 lands.
- Whether metrics `.jsonl`/CSV output should live inside `final_code_with_LNS/runs/` (gitignored) or outside the repo entirely.
- Confirm `ALLOW_COLLISION_BYPASS` default of `False` for anything interactive/demo, `True` only inside `eval_harness.py` — this is assumed, not yet confirmed with the team member who authored the metrics handoff doc.
- The metrics handoff doc's baselines-for-comparison idea (staggered spot-point grid, heuristic-only-no-MCTS) are packing-algorithm baselines, orthogonal to this plan's `lns_mode` baselines — worth keeping both comparison axes distinct in `eval_harness.py`'s output rather than conflating "which packing heuristic" with "which multi-agent coordination mode."
