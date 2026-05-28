# config.py

import math

CELL_SIZE = 1.0   # metres per cell (0.5 for finest resolution, 1.0 for speed)

POLYGON_BOUNDARY = [
    (0,  0),
    (90, 0),
    (90, 60),
    (60, 60),
    (60, 90),
    (0,  90),
]

ENTRY_POINT = (45.0, 0.0)

# ── Mixed Fleet ────────────────────────────────────────────
TRUCK_CLASSES = {
    'small': {
        'payload_t':              45.0,
        'width_m':                 5.5,
        'length_m':                8.7,
        'turn_radius_m':          22.0,
        'dump_ticks':                2,
        'pile_height_per_dump':    0.6,
        'colour':       (100, 220, 255),
        'label':               'S',
    },
    'medium': {
        'payload_t':             133.0,
        'width_m':                 7.1,
        'length_m':               11.6,
        'turn_radius_m':          27.0,
        'dump_ticks':                3,
        'pile_height_per_dump':    1.8,
        'colour':       (255, 200,  80),
        'label':               'M',
    },
    'large': {
        'payload_t':             227.0,
        'width_m':                 9.8,
        'length_m':               14.8,
        'turn_radius_m':          33.0,
        'dump_ticks':                4,
        'pile_height_per_dump':    3.2,
        'colour':       (255,  90,  90),
        'label':               'L',
    },
}

FLEET_COMPOSITION = {
    'small':  2,
    'medium': 0,
    'large':  0,
}

# ── Dump / Pile Physics ────────────────────────────────────
TARGET_PILE_HEIGHT = 5.0
ANGLE_OF_REPOSE    = 35.0
DRIVE_CLEARANCE_M  = 0.1
_TAN_REPOSE = math.tan(math.radians(ANGLE_OF_REPOSE))

# ── Hybrid Accessibility Toggle ────────────────────────────
# When fill_pct < this threshold, we score first, then BFS the top N.
# When fill_pct >= this threshold, we BFS everything first, then score.
CONFIG_MATERIAL_HEIGHT_THRESHOLD = 0.70

# ── Scoring filter sizes ───────────────────────────────────
SCORE_FILTER_SIZE    = int(round(24.0 / CELL_SIZE))
ENTRY_CORRIDOR_CELLS = max(1, int(round(3.0 / CELL_SIZE)))

# ── Pheromone ──────────────────────────────────────────────
PHEROMONE_DECAY = 0.85
# Spread sigma in CELL units — keep physical spread ~3m regardless of CELL_SIZE
# At CELL_SIZE=1.0: sigma=3.0 cells = 3m. At CELL_SIZE=0.5: sigma=6.0 cells = 3m
PHEROMONE_SPREAD_SIGMA = max(1.0, 3.0 / CELL_SIZE)
# Trail deposit: how strongly a truck step suppresses pheromone at its cell (0–1).
# Gaussian gradient falls off over TRAIL_RADIUS_M metres around the truck position.
TRAIL_STRENGTH  = 0.5
TRAIL_RADIUS_M  = 3.0

# ── MCTS ───────────────────────────────────────────────────
MCTS_SIMULATIONS = 200
MCTS_DEPTH       = 20

# ── Hungarian ──────────────────────────────────────────────
W_DISTANCE = 1.0
W_HEADING  = 0.4

# EARLY PHASE: Massive priority (0.4) on Entry Distance to push trucks to the back.
WEIGHTS_EARLY = (0, 0, 0, 30, 0, 0) 

# MID PHASE: Focus shifts to clustering and filling gaps.
WEIGHTS_MID   = (0.3, 0.2, 0.2, 0.1, 0.1, 0.1) 

# LATE PHASE: Pure focus on density and surface evenness. Distance doesn't matter anymore.
WEIGHTS_LATE  = (0.5, 0.0, 0.3, 0.1, 0.1, 0.0)

# ── Simulation ─────────────────────────────────────────────
TICK_DELAY     = 0.05   # seconds between rendered frames
# FIX: STEPS_PER_TICK was missing — trucks appeared frozen.
# Each step moves one cell (1m at CELL_SIZE=1.0). At STEPS_PER_TICK=3,
# trucks move 3m per frame which is clearly visible on screen.
# Increase to 5-8 for faster animation.
STEPS_PER_TICK = 3
PYGAME_SCALE   = None   # auto-computed in renderer