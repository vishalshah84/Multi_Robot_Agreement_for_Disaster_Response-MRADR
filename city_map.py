# city_map.py
import pygame
from config import WIDTH, HEIGHT, CITY_MARGIN

font_small = None

def init_fonts():
    global font_small
    if font_small is None:
        font_small = pygame.font.SysFont(None, 18)

def create_city_buildings():
    buildings = []

    def add_building(x, y, w, h, name, color):
        rect = pygame.Rect(x, y, w, h)
        buildings.append({"rect": rect, "name": name, "color": color})

    cx = WIDTH // 2
    cy = HEIGHT // 2

    # Downtown core
    add_building(cx - 200, cy - 160, 90, 140, "Tower A", (70, 90, 140))
    add_building(cx - 90,  cy - 180, 80, 160, "Tower B", (75, 100, 150))
    add_building(cx + 20,  cy - 150, 100, 190, "Finance Plaza", (60, 80, 130))
    add_building(cx + 140, cy - 140, 90, 120, "IT Hub", (65, 95, 145))

    # Residential area
    base_x = CITY_MARGIN + 40
    base_y = CITY_MARGIN + 60
    block_w = 90
    block_h = 70
    colors_res = (120, 100, 80)
    for r in range(2):
        for c in range(3):
            x = base_x + c * (block_w + 30)
            y = base_y + r * (block_h + 30)
            add_building(x, y, block_w, block_h,
                         f"House {chr(65 + r*3 + c)}", colors_res)

    # Shops
    shop_y = CITY_MARGIN + 40
    add_building(WIDTH - CITY_MARGIN - 240, shop_y,      80, 60, "Shop 1", (140, 110, 80))
    add_building(WIDTH - CITY_MARGIN - 150, shop_y + 10, 70, 60, "Shop 2", (150, 120, 90))
    add_building(WIDTH - CITY_MARGIN - 330, shop_y + 20, 70, 55, "Cafe",   (170, 130, 90))

    # Emergency services
    add_building(WIDTH - CITY_MARGIN - 230, HEIGHT - CITY_MARGIN - 200,
                 110, 80, "Hospital", (190, 80, 80))
    add_building(WIDTH - CITY_MARGIN - 360, HEIGHT - CITY_MARGIN - 180,
                 90, 70, "Fire Station", (200, 100, 70))
    add_building(WIDTH - CITY_MARGIN - 230, HEIGHT - CITY_MARGIN - 300,
                 110, 70, "Police HQ", (70, 110, 180))

    # Mall & Warehouse
    add_building(cx - 260, HEIGHT - CITY_MARGIN - 140,
                 180, 90, "Mall", (150, 70, 100))
    add_building(cx + 180, HEIGHT - CITY_MARGIN - 130,
                 160, 80, "Warehouse", (110, 80, 80))

    return buildings

def is_in_building(x, y, buildings):
    point = (x, y)
    for b in buildings:
        if b["rect"].collidepoint(point):
            return True
    return False

def random_road_position(buildings, margin, width, height, robot_radius):
    import random
    while True:
        x = random.uniform(margin, width - margin)
        y = random.uniform(margin, height - margin)
        if not is_in_building(x, y, buildings):
            return x, y

def draw_city(surface, buildings):
    from config import WIDTH, HEIGHT, CITY_MARGIN
    init_fonts()
    surface.fill((15, 18, 28))

    # Outer border
    pygame.draw.rect(surface, (180, 180, 180),
                     pygame.Rect(CITY_MARGIN, CITY_MARGIN,
                                 WIDTH - 2 * CITY_MARGIN,
                                 HEIGHT - 2 * CITY_MARGIN), 3)

    # Main roads
    main_v = pygame.Rect(WIDTH//2 - 40, CITY_MARGIN,
                         80, HEIGHT - 2*CITY_MARGIN)
    pygame.draw.rect(surface, (60, 60, 70), main_v)

    main_h = pygame.Rect(CITY_MARGIN, HEIGHT//2 - 35,
                         WIDTH - 2*CITY_MARGIN, 70)
    pygame.draw.rect(surface, (60, 60, 70), main_h)

    # Cross streets
    for y in [CITY_MARGIN + 140, HEIGHT - CITY_MARGIN - 140]:
        street = pygame.Rect(CITY_MARGIN, y - 20,
                             WIDTH - 2 * CITY_MARGIN, 40)
        pygame.draw.rect(surface, (50, 50, 60), street)
    for x in [CITY_MARGIN + 180, WIDTH - CITY_MARGIN - 180]:
        street = pygame.Rect(x - 20, CITY_MARGIN,
                             40, HEIGHT - 2 * CITY_MARGIN)
        pygame.draw.rect(surface, (50, 50, 60), street)

    # Park
    park_rect = pygame.Rect(CITY_MARGIN + 30,
                            CITY_MARGIN + 260, 180, 120)
    pygame.draw.rect(surface, (40, 100, 60), park_rect)
    park_label = font_small.render("Park", True, (230, 255, 230))
    surface.blit(park_label, (park_rect.x + 8, park_rect.y + 8))

    # River
    river_rect = pygame.Rect(CITY_MARGIN,
                             HEIGHT - CITY_MARGIN - 60,
                             WIDTH - 2 * CITY_MARGIN, 40)
    pygame.draw.rect(surface, (40, 70, 130), river_rect)

    # Buildings
    for b in buildings:
        rect = b["rect"]
        pygame.draw.rect(surface, b["color"], rect)
        pygame.draw.rect(surface, (20, 20, 35), rect, 2)
        name = b["name"]
        txt = font_small.render(name, True, (255, 255, 255))
        surface.blit(txt, (rect.x + 4, rect.y + 4))
