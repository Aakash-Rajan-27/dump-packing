# config.py
# ─────────────────────────────────────────────────────────────
# ALL magic numbers live here. Never hardcode numbers inside
# your classes — always import from here.
#
# Fleet now consists of 3 real Caterpillar mining truck classes:
#   Small  → Cat 773F  (45t  payload, 22m turning radius)
#   Medium → Cat 785D  (133t payload, 27m turning radius)
#   Large  → Cat 793F  (227t payload, 33m turning radius)
# ─────────────────────────────────────────────────────────────

# ── Grid ──────────────────────────────────────────────────
CELL_SIZE = 3.0   # metres per grid cell

# L-shaped paddock boundary (x, y) in metres
POLYGON_BOUNDARY = [
    (0,  0),
    (90, 0),
    (90, 60),
    (60, 60),
    (60, 90),
    (0,  90),
]

# Where trucks enter the polygon
ENTRY_POINT = (45.0, 0.0)

# ── Mixed Fleet Truck Definitions ──────────────────────────
# Each dict specifies one truck class based on real Cat specs.
# Source: LECTURA Specs / Cat official datasheets
#
# dump_radius_m     : material spreads this far from dump point (m)
# payload_t         : tonnes of material per trip
# width_m           : truck body width (over tyres)
# length_m          : truck overall length
# turn_radius_m     : minimum OUTSIDE turning radius
# dump_ticks        : ticks the dump action takes (larger = slower)
# pile_height_per_dump : how many metres ONE dump adds to cell height
#                        derived from payload / pile density / cell area

TRUCK_CLASSES = {
    'small': {
        # Cat 773F — light-duty mining haul truck
        # Real specs: 45.4t net load, 8.7m long, 5.53m wide, 22.1m outside turn radius
        'payload_t':              45.0,
        'width_m':                 5.5,
        'length_m':                8.7,
        'turn_radius_m':          22.0,
        'dump_radius_m':           3.5,   # smaller spread footprint
        'dump_ticks':                2,   # fast — light load, quick hoist
        'pile_height_per_dump':    0.6,   # metres added to cell z_height per dump
        'colour':       (100, 220, 255),  # light blue
        'label':               'S',
    },
    'medium': {
        # Cat 785D — mid-range mining haul truck
        # Real specs: 133t net load, 11.55m long, 7.06m wide, 27.4m outside turn radius
        'payload_t':             133.0,
        'width_m':                 7.1,
        'length_m':               11.6,
        'turn_radius_m':          27.0,
        'dump_radius_m':           5.0,   # medium spread footprint
        'dump_ticks':                3,   # standard duration
        'pile_height_per_dump':    1.8,   # about 3x a small truck
        'colour':       (255, 200,  80),  # amber
        'label':               'M',
    },
    'large': {
        # Cat 793F — large-format mining haul truck
        # Real specs: 226.8t net load, 14.8m long, 9.75m wide, 33m outside turn radius
        'payload_t':             227.0,
        'width_m':                 9.8,
        'length_m':               14.8,
        'turn_radius_m':          33.0,
        'dump_radius_m':           7.0,   # wide spread footprint
        'dump_ticks':                4,   # slow — heavy load, large hoist
        'pile_height_per_dump':    3.2,   # ~5x a small truck per dump
        'colour':       (255,  90,  90),  # red
        'label':               'L',
    },
}

# Fleet composition: 3 of each class (9 trucks total)
FLEET_COMPOSITION = {
    'small':  3,
    'medium': 3,
    'large':  3,
}

# ── Dump Pile Physics ──────────────────────────────────────
# Cells are PARTIAL until z_height reaches TARGET_PILE_HEIGHT.
# A PARTIAL cell can still receive more dumps from any truck.
# A FILLED cell is at or above TARGET_PILE_HEIGHT — no more dumps.
TARGET_PILE_HEIGHT  = 5.0   # metres — fully packed cell height
ANGLE_OF_REPOSE     = 35.0  # degrees — pile slope angle

# ── Pheromone ──────────────────────────────────────────────
PHEROMONE_DECAY         = 0.85
PHEROMONE_SPREAD_SIGMA  = 1.0

# ── MCTS ───────────────────────────────────────────────────
MCTS_SIMULATIONS = 200
MCTS_DEPTH       = 20

# ── Hungarian Assignment Weights ───────────────────────────
W_DISTANCE = 1.0
W_HEADING  = 0.4

# ── Scoring Heuristic Weights ──────────────────────────────
# Order: [density, coverage, lowspot, pheromone, boundary]
WEIGHTS_EARLY  = [0.2, 0.3, 0.1, 0.2, 0.2]   # fill < 30%
WEIGHTS_MID    = [0.3, 0.2, 0.2, 0.2, 0.1]   # fill 30–70%
WEIGHTS_LATE   = [0.4, 0.1, 0.3, 0.1, 0.1]   # fill > 70%

# ── Simulation ─────────────────────────────────────────────
TICK_DELAY   = 0.05
PYGAME_SCALE = 8
