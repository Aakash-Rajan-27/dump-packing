# renderer.py
# ─────────────────────────────────────────────────────────────
# FIX: Added surface wiping to prevent ghosting/trailing trucks.
# FIX: Reverted Pheromone rendering logic.
# ─────────────────────────────────────────────────────────────

import pygame
import numpy as np
import math
from grid_map import CellState
from config import TARGET_PILE_HEIGHT, _TAN_REPOSE

CELL_COLOURS = {
    CellState.BOUNDARY:      ( 30,  30,  30),
    CellState.EMPTY:         (140, 190, 140),
    CellState.PARTIAL:       (200, 140,  60),
    CellState.FILLED:        (110,  70,  30),
    CellState.RESERVED:      (230, 180,  50),
    CellState.PROTECTED:     ( 80, 120, 200),
    CellState.OBSTACLE:      (200,  50,  50),
    CellState.PATH_RESERVED: (160,  90, 220),  # purple: active path corridor
}

PANEL_W  = 280
PANEL_BG = (20, 20, 30)

_COL_EMPTY   = np.array([140, 190, 140], dtype=float)
_COL_PARTIAL = np.array([200, 140,  60], dtype=float)
_COL_FILLED  = np.array([110,  70,  30], dtype=float)


def _height_colour(z_norm):
    if z_norm <= 0.5:
        t = z_norm * 2.0
        rgb = _COL_EMPTY + t * (_COL_PARTIAL - _COL_EMPTY)
    else:
        t = (z_norm - 0.5) * 2.0
        rgb = _COL_PARTIAL + t * (_COL_FILLED - _COL_PARTIAL)
    return tuple(int(v) for v in np.clip(rgb, 0, 255))


class Renderer:
    def __init__(self, grid, scale=None):
        self.grid = grid

        pygame.init()
        info = pygame.display.Info()
        screen_h = info.current_h
        screen_w = info.current_w

        if scale is None:
            self.scale = max(2, int(screen_h * 0.85) // grid.rows)
        else:
            self.scale = scale

        self.grid_w = grid.cols * self.scale
        self.grid_h = grid.rows * self.scale

        win_w = min(self.grid_w + PANEL_W, screen_w)
        win_h = min(self.grid_h, screen_h)

        pygame.display.set_caption("Optimal Dump Packing — Mixed Fleet")
        self.screen = pygame.display.set_mode(
            (win_w, win_h), pygame.RESIZABLE
        )

        self.font_large = pygame.font.SysFont('monospace', 16, bold=True)
        self.font_small = pygame.font.SysFont('monospace', 13)
        self.font_tiny  = pygame.font.SysFont('monospace', 11)

        self.grid_surface = pygame.Surface((self.grid_w, self.grid_h))
        self.m_per_px = grid.cell_size / self.scale

    def world_to_px(self, x, y):
        ox, oy = self.grid.origin
        px = int((x - ox) / self.m_per_px)
        py = int((y - oy) / self.m_per_px)
        return px, py

    def metres_to_px(self, metres):
        return max(1, int(metres / self.m_per_px))

    def draw(self, trucks=None, metrics=None):
        if trucks  is None: trucks  = []
        if metrics is None: metrics = {}
        self._draw_grid()
        self._draw_paths(trucks)       # planned-path overlay (under trucks)
        self._draw_dump_radii(trucks)
        for truck in trucks:
            if not getattr(truck, "on_grid", True):
                continue
            self._draw_truck(truck)
        self.screen.blit(self.grid_surface, (0, 0))
        self._draw_panel(metrics, trucks)
        pygame.display.flip()

    def _draw_paths(self, trucks):
        """
        Draw each truck's remaining planned path as a light polyline.
        NAVIGATING trucks show dump path; EXITING trucks show exit path.
        Sampled every STRIDE waypoints so it's O(path_len / STRIDE) — cheap.
        No alpha surfaces: just a pastel version of the truck's own colour.
        Does NOT touch grid state or any simulation data.
        """
        STRIDE = 8   # sample 1 point per STRIDE waypoints (~25 pts for a 200-wp path)

        for truck in trucks:
            if not getattr(truck, "on_grid", True):
                continue
            if truck.status == truck.STATUS_NAVIGATING and truck.path:
                waypoints = truck.path
            elif truck.status == truck.STATUS_EXITING and truck._exit_path:
                waypoints = truck._exit_path
            else:
                continue

            if not waypoints:
                continue

            # Pastel: blend 35% truck colour + 65% white so it's clearly lighter
            r, g, b = truck.colour
            light = (
                min(255, int(r * 0.35 + 255 * 0.65)),
                min(255, int(g * 0.35 + 255 * 0.65)),
                min(255, int(b * 0.35 + 255 * 0.65)),
            )

            # Build sampled pixel-coord list
            half_len = truck.length / 2.0
            px_points = []

            indices = list(range(0, len(waypoints), STRIDE))
            if indices[-1] != len(waypoints) - 1:
                indices.append(len(waypoints) - 1)   # always include last point

            for i in indices:
                wp = waypoints[i]
                if len(wp) == 3 and not isinstance(wp[0], (int, np.integer)):
                    # smooth pose: (rear_x, rear_y, heading)
                    rx, ry, heading = wp
                    bx = rx + math.cos(heading) * half_len
                    by = ry + math.sin(heading) * half_len
                else:
                    # cell waypoint: (r, c)
                    bx, by = self.grid.cell_to_world(int(wp[0]), int(wp[1]))
                px_points.append(self.world_to_px(bx, by))

            if len(px_points) < 2:
                continue

            # Thin connecting line
            pygame.draw.lines(self.grid_surface, light, False, px_points, 2)

    def _draw_grid(self):
        # THE FIX: Wipe the canvas clean every frame to stop ghost trails
        self.grid_surface.fill((0, 0, 0))

        g = self.grid
        s = self.scale

        z_norm = np.clip(g.z_height / TARGET_PILE_HEIGHT, 0.0, 1.0)

        for r in range(g.rows):
            for c in range(g.cols):
                state = g.state[r, c]

                if state == CellState.BOUNDARY:
                    colour = CELL_COLOURS[CellState.BOUNDARY]
                elif state in (CellState.PARTIAL, CellState.FILLED):
                    colour = _height_colour(z_norm[r, c])
                elif state == CellState.RESERVED:
                    colour = CELL_COLOURS[CellState.EMPTY]
                else:
                    colour = CELL_COLOURS.get(state, (50, 50, 50))

                # Pheromone trail tint: cyan-blue overlay, intensity = 1 - pheromone.
                # Visible on all non-boundary cells; fades out as pheromone recovers to 1.
                if state != CellState.BOUNDARY:
                    trail = 1.0 - g.pheromone[r, c]
                    if trail > 0.01:
                        colour = (
                            max(0,   int(colour[0] - trail * 40)),
                            max(0,   int(colour[1] - trail * 20)),
                            min(255, int(colour[2] + trail * 80)),
                        )

                pygame.draw.rect(
                    self.grid_surface, colour,
                    (c * s, r * s, max(1, s - 1), max(1, s - 1))
                )

    def _draw_dump_radii(self, trucks):
        for truck in trucks:
            if truck.status != truck.STATUS_DUMPING:
                continue
            px, py    = self.world_to_px(*truck.pos)
            cone_r_px = self.metres_to_px(truck.pile_height_per_dump / _TAN_REPOSE)
            surf = pygame.Surface(
                (cone_r_px * 2 + 2, cone_r_px * 2 + 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*truck.colour, 80),
                               (cone_r_px+1, cone_r_px+1), cone_r_px)
            pygame.draw.circle(surf, (*truck.colour, 180),
                               (cone_r_px+1, cone_r_px+1), cone_r_px, 2)
            self.grid_surface.blit(surf, (px-cone_r_px-1, py-cone_r_px-1))

    def _draw_truck(self, truck):
        w_px_real = self.metres_to_px(truck.width)
        l_px_real = self.metres_to_px(truck.length)

        max_dim = max(20, self.grid_h // 12)
        scale_factor = min(1.0, max_dim / max(w_px_real, l_px_real))
        w_px = max(6,  int(w_px_real * scale_factor))
        l_px = max(10, int(l_px_real * scale_factor))

        truck_surf = pygame.Surface((l_px, w_px), pygame.SRCALPHA)

        pygame.draw.rect(truck_surf, truck.colour,
                         (0, 0, l_px, w_px), border_radius=2)
        pygame.draw.rect(truck_surf, (255, 255, 255),
                         (0, 0, l_px, w_px), 1, border_radius=2)
                         
        cab_w = max(3, l_px // 4)
        cab_colour = tuple(max(0, v - 60) for v in truck.colour)
        pygame.draw.rect(truck_surf, cab_colour,
                         (l_px - cab_w, 0, cab_w, w_px), border_radius=1)
                         
        pygame.draw.line(truck_surf, (255, 255, 255),
                         (0, w_px // 2), (l_px, w_px // 2), 1)

        heading_deg = math.degrees(truck.heading)
        rotated = pygame.transform.rotate(truck_surf, -heading_deg)

        px, py = self.world_to_px(*truck.pos)
        rect = rotated.get_rect(center=(px, py))
        self.grid_surface.blit(rotated, rect)

        status_colours = {
            truck.STATUS_IDLE:       (120, 120, 120),
            truck.STATUS_NAVIGATING: ( 80, 220,  80),
            truck.STATUS_REVERSING:  (240, 200,  50),
            truck.STATUS_DUMPING:    (240,  60,  60),
        }
        ring_col = status_colours.get(truck.status, (180, 180, 180))
        ring_r   = max(4, w_px // 2 + 2)
        pygame.draw.circle(self.grid_surface, ring_col, (px, py), ring_r, 2)

        arrow_len = max(8, l_px // 2 + 4)
        tip_x = int(px + arrow_len * math.cos(truck.heading))
        tip_y = int(py + arrow_len * math.sin(truck.heading))
        pygame.draw.line(self.grid_surface, (255, 255, 255),
                         (px, py), (tip_x, tip_y), 2)

        head_len = max(4, arrow_len // 3)
        for angle_offset in (math.pi * 5/6, -math.pi * 5/6):
            hx = int(tip_x + head_len * math.cos(truck.heading + angle_offset))
            hy = int(tip_y + head_len * math.sin(truck.heading + angle_offset))
            pygame.draw.line(self.grid_surface, (255, 255, 255),
                             (tip_x, tip_y), (hx, hy), 2)

        label = self.font_tiny.render(
            f"{truck.label}{truck.id}", True, (255, 255, 255))
        self.grid_surface.blit(label, (px - label.get_width() // 2,
                                       py - ring_r - 12))

    def _draw_panel(self, metrics, trucks):
        panel_x = self.grid_w
        panel_rect = pygame.Rect(panel_x, 0, PANEL_W, self.grid_h)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect)
        pygame.draw.line(self.screen, (60,60,80),
                         (panel_x,0),(panel_x,self.grid_h),1)

        y = 14
        title = self.font_large.render(
            "Dump Packing — Mixed Fleet", True, (200,200,220))
        self.screen.blit(title, (panel_x+8, y)); y += 28

        pygame.draw.line(self.screen, (60,60,80),
                         (panel_x+8,y),(panel_x+PANEL_W-8,y),1); y += 12

        for key, val in metrics.items():
            lbl   = self.font_small.render(f"{key}:", True, (140,160,180))
            val_s = self.font_small.render(str(val), True, (220,220,255))
            self.screen.blit(lbl,   (panel_x+8,  y))
            self.screen.blit(val_s, (panel_x+130, y))
            y += 18

        y += 6
        fill = self.grid.fill_pct()
        pack = self.grid.pack_pct()
        for label_str, val, col in [
            (f"Full cells: {fill*100:.1f}%", fill, (80,180,100)),
            (f"Pack density:{pack*100:.1f}%", pack, (100,160,220)),
        ]:
            lbl = self.font_small.render(label_str, True, (180,200,180))
            self.screen.blit(lbl,(panel_x+8,y)); y += 16
            pygame.draw.rect(self.screen,(50,60,50),(panel_x+8,y,PANEL_W-16,10))
            pygame.draw.rect(self.screen,col,(panel_x+8,y,int((PANEL_W-16)*val),10))
            y += 18

        y += 8
        pygame.draw.line(self.screen,(60,60,80),
                         (panel_x+8,y),(panel_x+PANEL_W-8,y),1); y += 10

        hdr = self.font_small.render("Fleet:", True, (160,160,180))
        self.screen.blit(hdr,(panel_x+8,y)); y += 16

        for truck in trucks:
            pygame.draw.rect(self.screen, truck.colour,
                             (panel_x+8, y+2, 10, 10))
            status_txt = self.font_small.render(
                f"{truck.label}{truck.id} [{truck.truck_class[:3]}]: {truck.status[:3]}",
                True, (180,180,200))
            self.screen.blit(status_txt, (panel_x+22,y))
            y += 16

        y += 8
        for txt, col in [
            ("Target:   <5.0m spacing", (100,200,100)),
            ("Baseline: 7.38m spacing", (200,120, 80)),
        ]:
            lbl = self.font_small.render(txt, True, col)
            self.screen.blit(lbl,(panel_x+8,y)); y += 18

        y += 8
        pygame.draw.line(self.screen,(60,60,80),
                         (panel_x+8,y),(panel_x+PANEL_W-8,y),1); y += 10
        legend_title = self.font_small.render("Legend:", True,(160,160,180))
        self.screen.blit(legend_title,(panel_x+8,y)); y += 16

        for state, name in [
            (CellState.EMPTY,         "Empty"),
            (CellState.PARTIAL,       "Partial pile"),
            (CellState.FILLED,        "Full pile"),
            (CellState.PROTECTED,     "Entry corridor"),
            (CellState.PATH_RESERVED, "Path corridor"),
        ]:
            col = CELL_COLOURS[state]
            pygame.draw.rect(self.screen, col,(panel_x+8,y+2,10,10))
            lbl = self.font_tiny.render(name, True,(180,180,200))
            self.screen.blit(lbl,(panel_x+22,y)); y += 16

    def check_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
            if event.type == pygame.VIDEORESIZE:
                new_scale = max(2, event.h // self.grid.rows)
                if new_scale != self.scale:
                    self.scale        = new_scale
                    self.grid_w       = self.grid.cols * self.scale
                    self.grid_h       = self.grid.rows * self.scale
                    self.m_per_px     = self.grid.cell_size / self.scale
                    self.grid_surface = pygame.Surface((self.grid_w, self.grid_h))
        return False

    def close(self):
        pygame.quit()
