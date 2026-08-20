# Dump-Packing Rebuild — Build Handoff (checkpoint 2026-06-14, reverse-dock session)

Resume point for the rebuild toward: max packing density, full realism (no
collisions / boundary breaches / driving over piles / turn-radius breaches),
smooth 30+ FPS, configurable fleet, dump-spacing comparison vs staffed ops.

**Work only in `post_may_4_code/`.** Other folders are stale forks.
Branch `aakashr`. Commit/push ONLY when explicitly asked. Commit messages end with
`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## How to run / test
```
# headless, deterministic, fast (USE THIS for logic iteration):
# run FROM post_may_4_code/ ; venv python lives in the PARENT dir (../.venv):
SYNC_PLAN=1 ../.venv/Scripts/python.exe main.py --headless --ticks 800 --seed 42 --report r.json
# windowed (real time, FPS cap, +/- speed keys):
../.venv/Scripts/python.exe main.py
# flags: --headless --ticks N --seed N --report path.json --screenshot p.png
# env toggles: SYNC_PLAN=1 (inline planning, no threads), DUMP_TRACE=1 (status + staging/exit debug)
```
- ALWAYS use `../.venv/Scripts/python.exe` (venv is in the PARENT of post_may_4_code; system/Bash `python` is 3.14, no pygame).
- ALWAYS use `SYNC_PLAN=1` for logic runs (threaded planner is GIL-starved; see task #3).
- `validation.py` (Validator) prints a per-episode VALIDATION REPORT — the scoreboard.
- Kill stray runs: `Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where {$_.CommandLine -like '*main.py*'} | Stop-Process`.

## THROUGHPUT COLLAPSE — FIXED (2026-06-14)
The old headline (fleet stops after ~3 dumps, freeze=2) is RESOLVED by the gate
**directional mutex** in main.py: hold an inbound release while ANY truck is
EXITING/LEAVING (was a racy radius/short-path test that let an exiter starting
~250 m out slip through and meet the inbound truck nose-to-nose in the throat —
the `[HEADLOCK]` livelock). Now sustained: 1200 ticks seed 42 → **12 dumps,
freeze 0, teleport 0, idle 0.0** (trucks are capacity-limited, never stalled).
2400-tick run confirmed the same. This change is UNCOMMITTED (commit when asked).

## THE HEADLINE PROBLEM NOW: over_pile (start here)
With throughput fixed, `over_pile` is the dominant violation: 1200 ticks seed 42
baseline = **26 events / 4934 violating-substeps** (vs boundary 685, collision 96).
`Fix #1` (below) trims it to 22 with no regression; the rest is a DEEP problem.

ROOT CAUSE (proven by a DUMP_TRACE diagnostic that logged truck→own-dump distance
on each over_pile flag — see git history of validation.py if you re-add it):
- **`plan_reverse_dock`'s goal is too loose.** Goal = `rear_dist <= goal_d`
  (goal_d = required_dump_clearance_m + REVERSE_DUMP_CLOSE_ENOUGH_M ≈ 4.74 m for
  the small truck). This accepts ANY pose with the rear within 4.74 m — INCLUDING
  the rear overshooting PAST the dump. Measured: the truck parks with its body
  centre only **0.5–2.6 m from its own dump point**, i.e. the rear ends up ~1.8 m
  on the FAR side of the dump. The truck **straddles its own dump and builds the
  pile under its body** → the z≈0.8–2.4 m `DUMPING` over_pile (the largest mode).
- **Stale exit/haul paths** → the `EXITING` over_pile (z≈1.4 m, the most frequent
  by count). The planner avoids piles known AT PLAN TIME, but other trucks dump
  fresh mounds onto the route before this truck traverses it. The plan is fine;
  reality changed.
- **Neighbour-pile merge** contributes a minor share (mean dump spacing 5.47 m <
  6.5 m two-pile merge diameter), so some docks sit on a neighbour's pile.

WHAT WAS TRIED AND FAILED (do NOT repeat blindly — all regress freeze; the
freeze=0 baseline is a delicately tuned traffic equilibrium):
1. Hard dump-spacing keep-out in get_raw_candidates → dumps 12→15 but **freeze
   0→2**, over_pile unchanged. Reverted.
2. Live drive-over guard in the movement loop (revert any step that climbs higher
   onto a pile + clear path to replan) → **total livelock** (dumps 12→2, trucks
   frozen ~536 s ON their own fresh pile right after dumping; the guard traps a
   truck against the mound it just built). Reverted.
3. `plan_reverse_dock` greedy truncation (stop reverse at first dense pose
   reaching goal_d) → made docking WORSE (body centre 2.6 m → 0.5–2.0 m; greedy
   "first crossing" picks worse terminals than the cost-ordered search) + freeze
   0→2. Reverted.

THE PROPER FIX (next session): rework `plan_reverse_dock`'s GOAL CONDITION, not
the search mechanics. Require the rear to stop in a NEAR-SIDE shell at ~goal_d
AND the nose to point away from the dump (`dot(heading_vec, dump - pos) < 0`), so
the dump lands BEHIND the rear and the whole footprint stays on the near side off
the pile. NOTE a realism judgement call for the user: a real rear-dump truck's
tailgate IS right at the pile it's making, so some `DUMPING` over-footprint is
physically inherent — consider whether the validator should exempt the rear
region (or a small radius) during the DUMPING state, OR offset dump_target a few
metres behind the tailgate. Decide that before chasing DUMPING over_pile to 0.
For the EXITING staleness mode, a freeze-SAFE supervisor-level exit replan (every
planning cycle, around piles, never a per-frame revert) is the likely path.

## What's DONE this session (reverse-dock rebuild)
- **Kinematic bicycle model** (truck.py `_drive_arc`): proper front/rear axle,
  `delta_max = atan2(L, turn_radius*1.05)`, rear-axle arc integration, heading
  recompute, pos = rear + L/2·heading. Clean motion, curvature ~0.
- **Lookahead + sub-stepping** (truck.py `_advance_along_path`): first waypoint
  ≥ `_lookahead_m` scanned over `path[:40]`, sub-step `min(TRUCK_MOVE_STEP_M, budget)`,
  `if moved < 1e-2: break`. Killed the oscillation around the corridor.
- **Proper reverse-dock planner** (pathfinder.py):
  - `_dock_primitive(grid, truck, driveable, x, y, heading, turn, gear, arc)` —
    one bicycle primitive, forward (gear+1) or reverse (gear−1), returns dense
    body-centre poses or None.
  - `plan_reverse_dock(grid, truck, dump_point, driveable=None, max_nodes=3000)` —
    hybrid A* over (r,c,heading_bucket,gear) with forward+reverse primitives,
    goal = rear axle within `required_dump_clearance_m(truck)+REVERSE_DUMP_CLOSE_ENOUGH_M`
    of dump, gear-change penalty `1.5*turn_radius`. Returns dense
    `(x,y,heading,gear)` body-centre waypoints or [].
- **truck.py dock tracking**: `_dock_path`/`_pre_dock_path` fields threaded through
  `preload_dump_path`/`set_path`/`cancel_preload`; ENTERING→NAVIGATING copies
  `_pre_dock_path`→`_dock_path`; NAVIGATING arrival→REVERSING; REVERSING tracks
  the dense dock poses at `reverse_mps*dt` (snap within budget) → DUMPING.
- **staging.py** `score_staging_candidates` rewritten to handoff scoring (circular
  poses at `dump+R·[cosθ,sinθ]`, heading faces away, angle_penalty around π/4,
  validity via is_pose_driveable + reverse segment). Mostly superseded by
  plan_reverse_dock now.
- **main.py**: `_plan_rear_dump_approach` (cell-A* haul to dump cell, then
  `plan_reverse_dock` from haul end → `(path, dock)`); workers return dock;
  result handlers thread `dock_path=` into set_path/preload_dump_path. Gate
  single-lane traffic control: `_exit_hold` + `_entry_busy`.
- **Single gate** kept (user decision) with entry/exit hold radius rules.

## Earlier-session wins still in place
- validation.py per-frame SAT collision/boundary/over-pile/curvature/teleport/freeze.
- Headless+CLI, real dt time model (Cat 777/785/793 speeds), vectorized terrain blit.
- Polygon scaled ~270 m, CELL_SIZE 2.0. Boundary setback on dump targets
  (filters.py) + WEIGHTS_EARLY rebalance — fixed the old "1 dump then stall".
- Remote-dump bug fixed (smooth_paths emits 2-tuples).

## config.py knobs added this session
`CONCURRENT_SPREAD_M=45`, `DUMP_SPACING_M=5`, `EXIT_HOLD_RADIUS_M=30`,
`ENTRY_RELEASE_LEN_MULT=1.2`, `STAGING_MAX_CANDIDATES=6`, `STAGING_NUM_ANGLES 24→12`.
User config: turn radii small=10/med=15/large=33; fleet 3 small; TARGET_PILE_HEIGHT=5.0.

## REMAINING TASKS (priority order)
1. **[DONE] Sustained throughput** — fixed by the gate directional mutex (see
   THROUGHPUT COLLAPSE — FIXED). 12 dumps/1200t, freeze 0. Stretch goal (>>28
   dumps + full pit fill) is now a throughput-tuning problem, not a livelock.
2. **[CRITICAL] over_pile** — see THE HEADLINE PROBLEM NOW. `Fix #1` (kept, in
   pathfinder.py plan_paths_cbs): the bulldozer override no longer forces
   currently-piled cells (z>MAX_DRIVEOVER) driveable, so the exit A* can't be
   routed over a mound. Trims over_pile 26→22, freeze still 0. The rest needs the
   plan_reverse_dock goal-condition rework (near-side shell + nose-away) and a
   freeze-safe exit replan; do NOT re-try the 3 reverted approaches.
3. **Renderer corridors must match the actual smooth path** the truck drives
   (interpolate the rendered corridor the same way movement is interpolated).
4. **boundary/collision** — exit/LEAVING path quality near the gate (685/96
   substeps): trucks clip the polygon edge (y≈265, x≈5) and overlap while leaving.
5. **Cleanup**: remove DUMP_TRACE-gated debug prints ([ARRIVE]/[SETPATH]/[GUARD]);
   delete dev artifacts (live*.png, before/after*.png, r.json, rep.json,
   rep_*.json, _render_test.py); drop now-unused imports in main.py
   (score_staging_candidates / STAGING_MAX_CANDIDATES / REAR_DUMP_FLAG).
6. **Process pool** (multiprocessing) to replace GIL-starved planner thread so
   non-SYNC realtime mode can plan. 32 cores available; skip GPU.
7. Algorithm overlays; fleet-composition validator passes. (Drive-over-pile +
   DUMP_SPACING enforcement folded into task #2 above.)

## Testing note
1200 ticks SYNC headless is ~250 s wall (the dump cone relaxation double-loop
dominates). Use `--ticks 600` for fast over_pile iteration; always seed 42 to
compare against rep_base.json (regenerate it: it is over_pile 26 / freeze 0 /
12 dumps at 1200t). Watch `freeze` like a hawk — every over_pile attempt so far
has traded it away.

## Approved acceptance criteria & decisions
See memory: dump-packing-project-contract, dump-packing-diagnosis, aakash-hardware
under `~/.claude/projects/C--Users-aakra-dump-packing/memory/`.
