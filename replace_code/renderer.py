# renderer.py
# ─────────────────────────────────────────────────────────────
# Pygame visualiser. Updated for mixed fleet and partial piles:
#   - PARTIAL cells drawn in a distinct orange-brown colour
#     with brightness proportional to how full they are
#   - Truck circles sized proportional to truck class
#   - Panel shows per-truck class label (S/M/L) and status
# ─────────────────────────────────────────────────────────────

import pygame
import numpy as np
from grid_map import CellState
from config import TARGET_PILE_HEIGHT

CELL_COLOURS = {
    CellState.BOUNDARY:  ( 30,  30,  30),   # dark grey
    CellState.EMPTY:     (140, 190, 140),   # soft green
    CellState.PARTIAL:   (200, 140,  60),   # orange-brown (partial fill)
    CellState.FILLED:    (110,  70,  30),   # dark brown (full pile)
    CellState.RESERVED:  (230, 180,  50),   # yellow
    CellState.PROTECTED: ( 80, 120, 200),   # blue
    CellState.OBSTACLE:  (200,  50,  50),   # red
}

PANEL_BG = (20, 20, 30)


class Renderer:
    def __init__(self, grid, scale=8):
        self.grid  = grid
        self.scale = scale

        grid_pixel_w = grid.cols * scale
        grid_pixel_h = grid.rows * scale
        panel_w      = 240

        self.grid_w = grid_pixel_w
        self.grid_h = grid_pixel_h

        pygame.init()
        pygame.display.set_caption("Optimal Dump Packing — Mixed Fleet Simulation")
        self.screen       = pygame.display.set_mode(
            (grid_pixel_w + panel_w, grid_pixel_h))
        self.font_large   = pygame.font.SysFont('monospace', 16, bold=True)
        self.font_small   = pygame.font.SysFont('monospace', 13)
        self.grid_surface = pygame.Surface((grid_pixel_w, grid_pixel_h))

    def draw(self, trucks=None, metrics=None):
        if trucks  is None: trucks  = []
        if metrics is None: metrics = {}
        self._draw_grid()
        for truck in trucks:
            self._draw_truck(truck)
        self.screen.blit(self.grid_surface, (0, 0))
        self._draw_panel(metrics, trucks)
        pygame.display.flip()

    def _draw_grid(self):
        g = self.grid
        s = self.scale

        for r in range(g.rows):
            for c in range(g.cols):
                state  = g.state[r, c]
                colour = CELL_COLOURS.get(state, (50, 50, 50))

                if state == CellState.FILLED:
                    # Darker = higher pile
                    ratio  = min(1.0, g.z_height[r, c] / TARGET_PILE_HEIGHT)
                    colour = tuple(int(v * (1.0 - 0.35 * ratio)) for v in colour)

                elif state == CellState.PARTIAL:
                    # Brightness proportional to partial fill fraction
                    ratio  = min(1.0, g.z_height[r, c] / TARGET_PILE_HEIGHT)
                    # Blend from green (empty) toward orange-brown (partial)
                    empty_c  = CELL_COLOURS[CellState.EMPTY]
                    partial_c = colour
                    colour = tuple(
                        int(empty_c[i] + ratio * (partial_c[i] - empty_c[i]))
                        for i in range(3)
                    )

                elif state == CellState.EMPTY:
                    ph    = g.pheromone[r, c]
                    r_val = min(255, int(colour[0] + (1 - ph) * 50))
                    g_val = max(0,   int(colour[1] - (1 - ph) * 25))
                    colour = (r_val, g_val, colour[2])

                pygame.draw.rect(
                    self.grid_surface, colour,
                    (c * s, r * s, s - 1, s - 1)
                )

    def _draw_truck(self, truck):
        s = self.scale
        ox, oy = self.grid.origin

        px = int((truck.pos[0] - ox) / self.grid.cell_size * s)
        py = int((truck.pos[1] - oy) / self.grid.cell_size * s)

        # Radius scales with truck class
        base_radius = max(3, s // 2)
        size_scale  = {'small': 0.8, 'medium': 1.1, 'large': 1.5}
        radius      = int(base_radius * size_scale.get(truck.truck_class, 1.0))

        # Truck body circle
        pygame.draw.circle(self.grid_surface, truck.colour, (px, py), radius)
        pygame.draw.circle(self.grid_surface, (255,255,255), (px, py), radius, 1)

        # Heading arrow
        arrow_end = (
            int(px + (radius + 4) * np.cos(truck.heading)),
            int(py + (radius + 4) * np.sin(truck.heading))
        )
        pygame.draw.line(self.grid_surface, (255,255,255), (px, py), arrow_end, 2)

        # Label: class letter + ID
        label = self.font_small.render(
            f"{truck.label}{truck.id}", True, (255,255,255))
        self.grid_surface.blit(label, (px + radius + 2, py - radius))

    def _draw_panel(self, metrics, trucks):
        panel_rect = pygame.Rect(self.grid_w, 0, 240, self.grid_h)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect)
        pygame.draw.line(self.screen, (60,60,80),
                         (self.grid_w, 0), (self.grid_w, self.grid_h), 1)

        y = 14
        title = self.font_large.render("Dump Packing — Mixed Fleet", True, (200,200,220))
        self.screen.blit(title, (self.grid_w + 8, y)); y += 28

        pygame.draw.line(self.screen, (60,60,80),
                         (self.grid_w+8, y), (self.grid_w+232, y), 1); y += 12

        for key, val in metrics.items():
            lbl = self.font_small.render(f"{key}:", True, (140,160,180))
            val_s = self.font_small.render(str(val), True, (220,220,255))
            self.screen.blit(lbl,   (self.grid_w + 8,   y))
            self.screen.blit(val_s, (self.grid_w + 130,  y))
            y += 18

        y += 6
        # Fill progress bar
        fill = self.grid.fill_pct()
        pack = self.grid.pack_pct()
        for label_str, val, col in [
            (f"Full cells: {fill*100:.1f}%", fill, (80,180,100)),
            (f"Pack density:{pack*100:.1f}%", pack, (100,160,220)),
        ]:
            lbl = self.font_small.render(label_str, True, (180,200,180))
            self.screen.blit(lbl, (self.grid_w + 8, y)); y += 16
            pygame.draw.rect(self.screen, (50,60,50),
                             (self.grid_w+8, y, 220, 10))
            pygame.draw.rect(self.screen, col,
                             (self.grid_w+8, y, int(220*val), 10))
            y += 18

        y += 8
        pygame.draw.line(self.screen, (60,60,80),
                         (self.grid_w+8, y), (self.grid_w+232, y), 1); y += 10

        # Per-truck status
        hdr = self.font_small.render("Fleet:", True, (160,160,180))
        self.screen.blit(hdr, (self.grid_w + 8, y)); y += 16

        for truck in trucks:
            pygame.draw.circle(self.screen, truck.colour,
                                (self.grid_w + 18, y + 6), 5)
            status_txt = self.font_small.render(
                f"{truck.label}{truck.id} [{truck.truck_class[:3]}]: {truck.status[:3]}",
                True, (180,180,200)
            )
            self.screen.blit(status_txt, (self.grid_w + 28, y))
            y += 16

        # Target reminders
        y += 8
        for txt, col in [
            ("Target:   <5.0m spacing", (100,200,100)),
            ("Baseline: 7.38m spacing", (200,120, 80)),
        ]:
            lbl = self.font_small.render(txt, True, col)
            self.screen.blit(lbl, (self.grid_w + 8, y)); y += 18

    def check_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
        return False

    def close(self):
        pygame.quit()
