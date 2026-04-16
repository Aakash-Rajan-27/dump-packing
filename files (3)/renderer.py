# viz/renderer.py
# ─────────────────────────────────────────────────────────────
# THE PYGAME VISUALISER.
#
# What pygame does here:
# - Opens a window on your screen
# - Draws a coloured rectangle for every grid cell every frame
# - Draws each truck as a coloured dot with a direction arrow
# - Shows a metrics panel on the right (fill%, tick count, etc.)
# - Lets you close the window to stop the simulation
#
# Without this: you'd just see numbers in a terminal and have
# no idea if your algorithm is working correctly.
# With this: you watch trucks navigate and fill the polygon live.
# ─────────────────────────────────────────────────────────────

import pygame          # the visualisation library
import numpy as np

from grid_map import CellState


# ── Colour map for each cell state ────────────────────────
# RGB tuples — what colour each cell type is drawn in
CELL_COLOURS = {
    CellState.BOUNDARY:  (30,  30,  30),    # dark grey — outside polygon
    CellState.EMPTY:     (140, 190, 140),   # soft green — empty valid cell
    CellState.FILLED:    (160, 110,  60),   # brown — material dumped here
    CellState.RESERVED:  (230, 180,  50),   # yellow — truck is heading here
    CellState.PROTECTED: (80,  120, 200),   # blue — entry corridor
    CellState.OBSTACLE:  (200,  50,  50),   # red — blocked
}

# Truck colours (one per truck ID, cycles if more than 4 trucks)
TRUCK_COLOURS = [
    (255, 100, 100),   # truck 0: red
    (100, 200, 255),   # truck 1: blue
    (100, 255, 150),   # truck 2: green
    (255, 200,  80),   # truck 3: orange
]

# Background colour for the metrics panel
PANEL_BG = (20, 20, 30)


class Renderer:
    """
    Handles all pygame drawing.
    Call draw() every tick from main.py to update the display.
    """

    def __init__(self, grid, scale=8):
        """
        grid: GridMap object — tells us grid dimensions
        scale: pixels per cell. 8 means each 3m cell = 8×8 pixels on screen.
               Larger = bigger window, more detail. Smaller = more cells visible.
        """
        self.grid  = grid
        self.scale = scale   # pixels per cell

        # Calculate window size
        grid_pixel_w = grid.cols * scale   # width of the grid area in pixels
        grid_pixel_h = grid.rows * scale   # height of the grid area in pixels
        panel_w      = 220                  # width of the metrics panel on the right

        self.grid_w = grid_pixel_w
        self.grid_h = grid_pixel_h

        # Initialise pygame
        pygame.init()
        pygame.display.set_caption("Optimal Dump Packing Simulation")

        # Create the window: grid area + metrics panel side by side
        self.screen = pygame.display.set_mode((grid_pixel_w + panel_w, grid_pixel_h))

        # Font for drawing text in the metrics panel
        self.font_large = pygame.font.SysFont('monospace', 16, bold=True)
        self.font_small = pygame.font.SysFont('monospace', 13)

        # Surface for drawing the grid (we redraw this every frame)
        self.grid_surface = pygame.Surface((grid_pixel_w, grid_pixel_h))

    def draw(self, trucks=None, metrics=None):
        """
        Draw one frame. Call this every tick.

        trucks: list of Truck objects (can be None or empty)
        metrics: dict of {label: value} to display in the panel
                 e.g. {'fill%': '42.3', 'tick': 150, 'trucks': 4}
        """
        if trucks  is None: trucks  = []
        if metrics is None: metrics = {}

        # ── 1. Draw the grid cells ─────────────────────────
        self._draw_grid()

        # ── 2. Draw trucks on top of grid ─────────────────
        for truck in trucks:
            self._draw_truck(truck)

        # Blit (copy) the grid surface onto the main screen at (0,0)
        self.screen.blit(self.grid_surface, (0, 0))

        # ── 3. Draw the metrics panel ──────────────────────
        self._draw_panel(metrics, trucks)

        # ── 4. Update the display ──────────────────────────
        # pygame.display.flip() pushes everything we drew to the actual screen.
        # Nothing you draw is visible until you call this.
        pygame.display.flip()

    def _draw_grid(self):
        """Draw every cell as a coloured rectangle."""
        g = self.grid
        s = self.scale   # pixels per cell

        for r in range(g.rows):
            for c in range(g.cols):
                # Get the base colour for this cell's state
                state = g.state[r, c]
                colour = CELL_COLOURS.get(state, (50, 50, 50))

                # Tint filled cells by height — taller = darker brown
                if state == CellState.FILLED:
                    height_ratio = min(1.0, g.z_height[r, c] / 3.0)  # normalise 0–3m
                    # Darken the colour proportional to height
                    colour = tuple(int(v * (1.0 - 0.4 * height_ratio)) for v in colour)

                # Tint empty cells by pheromone — recently-used areas are slightly orange
                elif state == CellState.EMPTY:
                    ph = g.pheromone[r, c]  # 0 = recently used, 1 = untouched
                    # Blend toward orange-ish when pheromone is low
                    r_val = int(colour[0] + (1 - ph) * 60)
                    g_val = int(colour[1] - (1 - ph) * 30)
                    b_val = colour[2]
                    colour = (min(255, r_val), max(0, g_val), b_val)

                # Convert (row, col) to pixel coordinates
                # row → y axis (top of screen = row 0), col → x axis
                px = c * s      # x pixel position
                py = r * s      # y pixel position

                # Draw the rectangle (leave 1px gap between cells for grid lines)
                pygame.draw.rect(
                    self.grid_surface,
                    colour,
                    (px, py, s - 1, s - 1)   # x, y, width, height
                )

    def _draw_truck(self, truck):
        """Draw a single truck as a circle with a heading arrow."""
        s = self.scale

        # Convert truck's real-world position (metres) to pixel position
        world_x, world_y = truck.pos
        origin_x, origin_y = self.grid.origin

        # Pixel coords: (world - origin) / cell_size * scale
        px = int((world_x - origin_x) / self.grid.cell_size * s)
        py = int((world_y - origin_y) / self.grid.cell_size * s)

        # Pick colour based on truck ID (cycles through TRUCK_COLOURS)
        colour = TRUCK_COLOURS[truck.id % len(TRUCK_COLOURS)]

        # Draw truck body as a filled circle
        radius = max(3, s // 2)  # circle radius in pixels
        pygame.draw.circle(self.grid_surface, colour, (px, py), radius)

        # Draw heading arrow — shows which direction the truck is facing
        arrow_len = radius + 4  # arrow length in pixels
        arrow_end_x = int(px + arrow_len * np.cos(truck.heading))
        arrow_end_y = int(py + arrow_len * np.sin(truck.heading))
        pygame.draw.line(self.grid_surface, (255, 255, 255),
                         (px, py), (arrow_end_x, arrow_end_y), 2)
        # White line from truck centre in the direction it's heading

        # Draw truck ID number next to the truck
        label = self.font_small.render(f"T{truck.id}", True, (255, 255, 255))
        self.grid_surface.blit(label, (px + radius + 2, py - radius))

    def _draw_panel(self, metrics, trucks):
        """Draw the metrics panel on the right side of the window."""
        # Panel starts at x = grid width, spans full height
        panel_rect = pygame.Rect(self.grid_w, 0, 220, self.grid_h)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect)

        # Draw a vertical separator line
        pygame.draw.line(self.screen, (60, 60, 80),
                         (self.grid_w, 0), (self.grid_w, self.grid_h), 1)

        y = 16  # current y position for text, increments downward

        # Title
        title = self.font_large.render("Dump Packing Sim", True, (200, 200, 220))
        self.screen.blit(title, (self.grid_w + 10, y))
        y += 30

        # Separator line under title
        pygame.draw.line(self.screen, (60, 60, 80),
                         (self.grid_w + 10, y), (self.grid_w + 210, y), 1)
        y += 14

        # ── Metrics from the dict ──────────────────────────
        for key, val in metrics.items():
            label = self.font_small.render(f"{key}:", True, (140, 160, 180))
            value = self.font_small.render(str(val), True, (220, 220, 255))
            self.screen.blit(label, (self.grid_w + 10, y))
            self.screen.blit(value, (self.grid_w + 120, y))
            y += 20

        y += 10

        # ── Packing density progress bar ───────────────────
        fill = self.grid.fill_pct()  # 0.0 to 1.0
        bar_label = self.font_small.render(f"Fill: {fill*100:.1f}%", True, (180, 200, 180))
        self.screen.blit(bar_label, (self.grid_w + 10, y))
        y += 18

        # Background bar (empty)
        pygame.draw.rect(self.screen, (50, 60, 50),
                         (self.grid_w + 10, y, 200, 14))
        # Filled bar (proportional to fill%)
        pygame.draw.rect(self.screen, (80, 180, 100),
                         (self.grid_w + 10, y, int(200 * fill), 14))
        y += 24

        # ── Target vs autonomous baseline ─────────────────
        y += 6
        target_label = self.font_small.render("Target:  < 5.0m spacing", True, (100, 200, 100))
        base_label   = self.font_small.render("Baseline: 7.38m spacing", True, (200, 120, 80))
        self.screen.blit(target_label, (self.grid_w + 10, y)); y += 18
        self.screen.blit(base_label,   (self.grid_w + 10, y)); y += 28

        # ── Per-truck status ───────────────────────────────
        status_title = self.font_small.render("Trucks:", True, (160, 160, 180))
        self.screen.blit(status_title, (self.grid_w + 10, y))
        y += 18

        for truck in trucks:
            colour = TRUCK_COLOURS[truck.id % len(TRUCK_COLOURS)]
            # Small coloured dot
            pygame.draw.circle(self.screen, colour, (self.grid_w + 18, y + 6), 5)
            # Status text
            status_text = self.font_small.render(
                f"T{truck.id}: {truck.status}", True, (180, 180, 200)
            )
            self.screen.blit(status_text, (self.grid_w + 28, y))
            y += 18

    def check_quit(self):
        """
        Call this every tick to check if the user closed the window.
        Returns True if they clicked X (you should then exit the sim loop).
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            # Also allow quitting with Escape key
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
        return False

    def close(self):
        """Shut down pygame cleanly."""
        pygame.quit()
