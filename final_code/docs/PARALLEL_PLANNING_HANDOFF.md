# Parallel Planning — Implementation Handoff

## Codebase
`C:\Users\aakra\dump-packing\post_may_4_code\`  
Entry point: `main.py` → `run_simulation()`  
Key files: `main.py`, `truck.py`, `pathfinder.py`, `staging.py`

---

## What to Build

Move dump-path planning off the main tick loop into a background thread pool so trucks that are navigating/dumping/exiting are never paused while pathfinding runs for idle trucks.

---

## Current Behaviour (what to replace)

In `main.py` around line 487, the **idle planning block**:
```python
idle_trucks = [t for t in trucks if t.is_idle() and not t._pre_path]
if idle_trucks:
    # ... MCTS + CBS for ALL idle trucks in one synchronous call ...
    paths, staging_poses = plan_staging_paths(grid, assignments_all, ...)
    for truck, dump_point in assignments_all:
        truck.set_path(...)
```
This blocks the entire tick — all trucks freeze while planning runs.

---

## What to Implement

### 1. Thread pool + two queues

At the top of `run_simulation()`:
```python
from concurrent.futures import ThreadPoolExecutor
import queue

executor = ThreadPoolExecutor(max_workers=4)
plan_results = queue.Queue()   # worker → main thread
```

### 2. Add flag to Truck (`truck.py`)

In `Truck.__init__`, add:
```python
self._planning_in_flight = False   # True while a background plan job is running
```
Reset to `False` in `set_path()` and whenever the truck is force-idled.

### 3. Planning job function (pure, operates on grid snapshot)

Define this function (can live at module level in `main.py`):
```python
def _plan_job(truck_id, truck_class, truck_pos, truck_heading,
              grid_snapshot, assignments, locked_paths):
    # assignments = [(truck_stub, dump_point), ...]
    # grid_snapshot is a detached copy — safe to read from worker thread
    paths, staging = plan_staging_paths(grid_snapshot, assignments,
                                        locked_paths=locked_paths)
    return truck_id, paths, staging
```
The worker posts `(truck_id, paths, staging)` into `plan_results`.

### 4. Grid snapshot at submission time

When submitting a job for a newly-idle truck:
```python
import copy
grid_snap = copy.copy(grid)          # shallow copy of the GridMap object
grid_snap.state = grid.state.copy()  # deep-copy the numpy state array
grid_snap.height = grid.height.copy()
# copy any other mutable arrays (pheromone, trail, etc.)
```
Workers **only read** `grid_snap` — never write to it. The live `grid` is only written by the main thread.

### 5. Replace the idle planning block

```python
# ── SUBMIT planning jobs for newly-idle trucks ──────────────────
for t in [t for t in trucks if t.is_idle() and not t._pre_path
          and not t._planning_in_flight]:
    # --- same candidate search as today ---
    # get_raw_candidates, score_candidates, is_accessible, mcts_select_dump_points, assign
    # (copy the existing logic from the current idle block, but per-truck)
    if not assignments_for_this_truck:
        continue
    locked = {ot.id: ... for ot in trucks ...}   # same locked-path logic as today
    grid_snap = _snapshot_grid(grid)
    t._planning_in_flight = True
    future = executor.submit(_plan_job, t.id, ..., grid_snap,
                             assignments_for_this_truck, locked)
    future.add_done_callback(lambda f: plan_results.put(f.result()))

# ── APPLY finished plans (non-blocking) ─────────────────────────
while not plan_results.empty():
    truck_id, paths, staging = plan_results.get_nowait()
    t = next((t for t in trucks if t.id == truck_id), None)
    if t is None or not t.is_idle():
        continue   # truck moved on (got force-idled etc.) — discard
    truck_path = paths.get(t.id, [])
    dump_point = t._pending_dump_target   # stored on the truck at submit time
    t._planning_in_flight = False
    t.set_path(truck_path, dump_point, grid,
               staging_pose=staging.get(t.id))
```

You'll also need `t._pending_dump_target` — set it on the truck when you call `assign()` at submission time (just before the `executor.submit` call), so the main thread knows which cell to pass to `set_path` when the result arrives.

### 6. Thread safety rules (important)

- Workers **only read** the grid snapshot. Never pass `grid` (live) to a worker.
- `plan_results.put/get` is the only cross-thread communication.
- `truck._planning_in_flight` is set/read only on the main thread (inside the tick loop) — no lock needed.
- `plan_staging_paths` and `mcts_select_dump_points` must not touch any shared mutable state. Check `pathfinder.py` — if it mutates the grid (e.g. marks corridors temporarily), it must do so on `grid_snap`, not `grid`.

### 7. Shutdown

After the simulation loop ends:
```python
executor.shutdown(wait=False)
```

---

## What Does NOT Change

- Exit path planning (`needs_exit_path()` block, `plan_paths_cbs`) — leave synchronous for now; it's short and rare.
- Pre-planning for WAITING trucks (gate logic block) — leave as-is.
- All collision detection, deadlock/headlock resolution — unchanged.
- `Truck.set_path()`, `Truck.set_exit_path()` — unchanged.

---

## Acceptance Check

After implementation:
1. A truck that just went IDLE should call `executor.submit(...)` that tick and set `_planning_in_flight = True`.
2. All other trucks should continue moving that same tick (no freeze).
3. 1–2 ticks later, the result arrives in `plan_results` and the truck transitions to `NAVIGATING`.
4. If multiple trucks go IDLE simultaneously, they all get submitted the same tick and planned concurrently.
