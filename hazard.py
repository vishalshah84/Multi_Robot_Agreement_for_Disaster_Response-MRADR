# hazard.py
import random
import math
import pygame

from config import (
    FPS,
    HAZARD_COOLDOWN_MIN,
    HAZARD_COOLDOWN_MAX,
    SENSE_RADIUS,
    ROBOT_RADIUS,
    ARRIVAL_RADIUS_FACTOR,
)

font_small = None


def init_fonts():
    global font_small
    if font_small is None:
        font_small = pygame.font.SysFont(None, 18)


def point_rect_distance(px, py, rect: pygame.Rect) -> float:
    """Shortest distance from a point to a rectangle (0 if inside)."""
    rx = min(max(px, rect.left), rect.right)
    ry = min(max(py, rect.top), rect.bottom)
    dx = px - rx
    dy = py - ry
    return math.hypot(dx, dy)


class HazardManager:
    def __init__(self, buildings):
        self.buildings = buildings
        self.active = False
        self.building_index = None
        self.timer_until_next = self._random_cooldown()
        self.id_counter = 0

        # timing / metrics
        self.start_time = None
        self.detect_time = None
        self.consensus_time = None
        self.aggregated_time = None

    def _random_cooldown(self):
        sec = random.uniform(HAZARD_COOLDOWN_MIN, HAZARD_COOLDOWN_MAX)
        return int(sec * FPS)

    def _spawn_hazard(self, sim_time):
        self.active = True
        self.id_counter += 1
        self.building_index = random.randrange(len(self.buildings))
        self.timer_until_next = None
        self.start_time = sim_time
        self.detect_time = None
        self.consensus_time = None
        self.aggregated_time = None
        print(f"[INFO] Hazard #{self.id_counter} at "
              f"{self.buildings[self.building_index]['name']}")

    def _clear_hazard(self):
        print(f"[INFO] Hazard {self.id_counter} cleared.")
        self.active = False
        self.building_index = None
        self.timer_until_next = self._random_cooldown()
        self.start_time = None

    def update(self, robots, sim_time, consensus_done):
        """
        - Spawns and clears hazards.
        - Local detection: robots near the *building footprint* learn about it.
        - Updates timing markers for metrics.
        """
        # No active hazard → count down to next one
        if not self.active:
            self.timer_until_next -= 1
            if self.timer_until_next <= 0:
                self._spawn_hazard(sim_time)
            return

        # There is an active hazard
        b = self.buildings[self.building_index]
        rect = b["rect"]

        # Local detection: distance to rectangle, not just center
        for r in robots:
            if not r.has_info:
                d = point_rect_distance(r.x, r.y, rect)
                if d < SENSE_RADIUS:
                    r.has_info = True
                    # one–hot belief on this building
                    n = len(self.buildings)
                    r.belief = [0.0] * n
                    r.belief[self.building_index] = 1.0
                    if self.detect_time is None:
                        self.detect_time = sim_time

        # Consensus time
        if consensus_done and self.consensus_time is None:
            self.consensus_time = sim_time

        # Aggregation complete? (only robots that know about the hazard count)
        informed = [r for r in robots if r.has_info]
        if informed and all(r.has_arrived() for r in informed):
            if self.aggregated_time is None:
                self.aggregated_time = sim_time
            self._clear_hazard()

    def draw(self, surface):
        if not self.active or self.building_index is None:
            return
        init_fonts()
        b = self.buildings[self.building_index]
        rect = b["rect"]
        cx, cy = rect.center

        t = pygame.time.get_ticks() / 250.0
        pulse = 6 * math.sin(t)
        radius = 28 + pulse

        pygame.draw.circle(surface, (230, 60, 60),
                           (int(cx), int(cy)), int(radius), 3)
        label = font_small.render(f"Hazard #{self.id_counter}", True,
                                  (255, 255, 255))
        surface.blit(label, (cx - 45, cy - 40))
