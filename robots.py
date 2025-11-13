# robots.py
import math
import random
import pygame

from config import (
    CITY_MARGIN, ROBOT_RADIUS, PATROL_SPEED, RESPONSE_SPEED,
    ARRIVAL_RADIUS_FACTOR, COMM_RADIUS
)
from city_map import is_in_building, random_road_position

font_small = None

def init_fonts():
    global font_small
    if font_small is None:
        font_small = pygame.font.SysFont(None, 18)

class Robot:
    def __init__(self, name, color, buildings, width, height):
        self.name = name
        self.color = color
        self.buildings = buildings
        self.width = width
        self.height = height

        self.x, self.y = random_road_position(buildings, CITY_MARGIN,
                                              width, height, ROBOT_RADIUS)

        ang = random.uniform(0, 2 * math.pi)
        self.vx = math.cos(ang) * PATROL_SPEED
        self.vy = math.sin(ang) * PATROL_SPEED

        # State: PATROL / CONSENSUS / RESPOND
        self.state = "PATROL"
        self.target_building_index = None

        # Consensus belief over buildings
        n = len(buildings)
        self.belief = [1.0 / n] * n
        self.has_info = False  # has heard about hazard

    def distance_to(self, x, y):
        dx = x - self.x
        dy = y - self.y
        return math.sqrt(dx*dx + dy*dy)

    def randomize_direction(self):
        ang = random.uniform(0, 2 * math.pi)
        self.vx = math.cos(ang) * PATROL_SPEED
        self.vy = math.sin(ang) * PATROL_SPEED

    def bounce_edges(self):
        if self.x - ROBOT_RADIUS < CITY_MARGIN:
            self.x = CITY_MARGIN + ROBOT_RADIUS
            self.vx *= -1
        if self.x + ROBOT_RADIUS > self.width - CITY_MARGIN:
            self.x = self.width - CITY_MARGIN - ROBOT_RADIUS
            self.vx *= -1
        if self.y - ROBOT_RADIUS < CITY_MARGIN:
            self.y = CITY_MARGIN + ROBOT_RADIUS
            self.vy *= -1
        if self.y + ROBOT_RADIUS > self.height - CITY_MARGIN:
            self.y = self.height - CITY_MARGIN - ROBOT_RADIUS
            self.vy *= -1

    def bounce_buildings(self):
        for b in self.buildings:
            if b["rect"].collidepoint(self.x, self.y):
                self.x -= self.vx
                self.y -= self.vy
                self.vx *= -1
                self.vy *= -1
                return

    def step_patrol(self):
        if random.random() < 0.03:
            da = random.uniform(-math.pi/4, math.pi/4)
            ang = math.atan2(self.vy, self.vx) + da
            self.vx = math.cos(ang) * PATROL_SPEED
            self.vy = math.sin(ang) * PATROL_SPEED

        self.x += self.vx
        self.y += self.vy
        self.bounce_edges()
        self.bounce_buildings()

    def step_respond(self):
        assert self.target_building_index is not None
        rect = self.buildings[self.target_building_index]["rect"]
        tx, ty = rect.center
        dx = tx - self.x
        dy = ty - self.y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > 1e-6:
            step = min(RESPONSE_SPEED, dist)
            self.x += dx/dist * step
            self.y += dy/dist * step
        self.bounce_edges()

    def has_arrived(self):
        if self.target_building_index is None:
            return False
        rect = self.buildings[self.target_building_index]["rect"]
        cx, cy = rect.center
        return self.distance_to(cx, cy) < ARRIVAL_RADIUS_FACTOR * ROBOT_RADIUS

    def update_state_machine(self, hazard_active, hazard_building_index,
                             consensus_done, global_hazard_known):
        """
        Simple logic:
          - If no hazard: PATROL, forget old target.
          - If hazard just started and robot has info but consensus not done: CONSENSUS.
          - Once consensus done: RESPOND to chosen building.
        """
        if not hazard_active:
            if self.state != "PATROL":
                # re-spawn onto roads & reset motion
                self.x, self.y = random_road_position(self.buildings, CITY_MARGIN,
                                                      self.width, self.height,
                                                      ROBOT_RADIUS)
                self.randomize_direction()
            self.state = "PATROL"
            self.target_building_index = None
            return

        # hazard active
        if global_hazard_known and consensus_done:
            # everyone agreed → go respond
            self.state = "RESPOND"
            self.target_building_index = hazard_building_index
        else:
            # hazard active but still in info/consensus phase
            if self.has_info:
                self.state = "CONSENSUS"
            else:
                self.state = "PATROL"  # still patrolling until informed

    def update_motion(self):
        if self.state == "PATROL" or self.state == "CONSENSUS":
            self.step_patrol()
        elif self.state == "RESPOND":
            self.step_respond()

    def draw(self, surface):
        init_fonts()
        pygame.draw.circle(surface, self.color,
                           (int(self.x), int(self.y)), ROBOT_RADIUS)
        label = font_small.render(self.name, True, (255, 255, 255))
        surface.blit(label, (int(self.x) - 10, int(self.y) - 24))

    def draw_comm_links(self, surface, robots):
        # Visualize comm graph
        for other in robots:
            if other is self:
                continue
            d = self.distance_to(other.x, other.y)
            if d < COMM_RADIUS:
                pygame.draw.line(surface, (80, 80, 120),
                                 (int(self.x), int(self.y)),
                                 (int(other.x), int(other.y)), 1)
