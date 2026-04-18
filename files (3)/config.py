# config.py
# ─────────────────────────────────────────────────────────────
# ALL magic numbers live here. Never hardcode numbers inside
# your classes — always import from here. This way you can
# tune everything from one place.
# ─────────────────────────────────────────────────────────────

# How big each grid cell is in real-world metres.
# Smaller = more precise but slower. 3m is a good start.
CELL_SIZE = 3.0

# The dump polygon defined as (x, y) corner coordinates in metres.
# This is a simple L-shaped paddock — replace with your real shape.
# Go clockwise around the boundary.
POLYGON_BOUNDARY = [
    (0,  0),
    (90, 0),
    (90, 60),
    (60, 60),
    (60, 90),
    (0,  90),
]

# Where trucks enter the polygon (x, y) in metres.
ENTRY_POINT = (45.0, 0.0)

# ── Truck physical dimensions ──────────────────────────────
TRUCK_WIDTH  = 8.0   # metres side-to-side
TRUCK_LENGTH = 14.0  # metres front-to-back
MIN_TURN_RADIUS = 12.0  # minimum turning radius in metres
                         # (tighter turns than this are physically impossible)

# ── Dump pile physics ──────────────────────────────────────
DUMP_SPREAD_RADIUS = 4.0    # how far material spreads from dump point (metres)
TARGET_PILE_HEIGHT = 3.0   # target height of a full pile (metres)
ANGLE_OF_REPOSE    = 35.0  # degrees — slope of a pile edge (30–40 is real)

# ── Pheromone (implicit coordination between trucks) ───────
PHEROMONE_DECAY = 0.85  # each tick, pheromone multiplied by this
                          # 0.85 = 15% decay per tick. Lower = fades faster.
PHEROMONE_SPREAD_SIGMA = 1.0  # how far pheromone spreads to neighbour cells
                                # (gaussian blur sigma in cell units)

# ── MCTS settings ──────────────────────────────────────────
MCTS_SIMULATIONS = 200  # number of tree search simulations per planning call
                          # more = better decisions but slower. 200 is a good balance.
MCTS_DEPTH = 20         # how many future dump steps each simulation looks ahead

# ── Hungarian assignment weights ───────────────────────────
W_DISTANCE = 1.0   # how much straight-line distance matters in cost matrix
W_HEADING  = 0.4   # how much truck heading alignment matters (secondary)

# ── Scoring heuristic weights (adapt with fill %) ─────────
# These shift as the polygon fills up — see scoring.py for how they're used
WEIGHTS_EARLY  = [0.2, 0.3, 0.1, 0.2, 0.2]  # fill < 30%: spread out
WEIGHTS_MID    = [0.3, 0.2, 0.2, 0.2, 0.1]  # fill 30–70%: balance
WEIGHTS_LATE   = [0.4, 0.1, 0.3, 0.1, 0.1]  # fill > 70%: pack tight

# ── Simulation settings ────────────────────────────────────
NUM_TRUCKS   = 4       # how many trucks in the fleet
TICK_DELAY   = 0.05   # seconds between simulation ticks (controls animation speed)
PYGAME_SCALE = 8       # pixels per cell in the pygame window
                        # 8 means each 3m cell = 8×8 pixels on screen
