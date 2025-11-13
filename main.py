# main.py
import pygame
from config import WIDTH, HEIGHT, FPS, NUM_ROBOTS
from city_map import create_city_buildings, draw_city
from robots import Robot
from hazard import HazardManager
from consensus import build_neighbor_graph, step_average_consensus
from metrics import MetricsLogger

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Multi-Robot City Hazard Response with Consensus")
clock = pygame.time.Clock()
font_big = pygame.font.SysFont(None, 26)

def main():
    buildings = create_city_buildings()
    robots = []
    colors = [
        (255, 90, 90),
        (90, 200, 255),
        (120, 255, 120),
        (255, 200, 90),
        (200, 120, 255),
        (255, 140, 190),
        (140, 255, 200),
        (200, 200, 255),
        (255, 255, 140),
        (160, 220, 255),
    ]
    for i in range(NUM_ROBOTS):
        robots.append(Robot(f"R{i+1}", colors[i % len(colors)],
                            buildings, WIDTH, HEIGHT))

    hazard_mgr = HazardManager(buildings)
    metrics = MetricsLogger()

    sim_time = 0.0  # seconds

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        sim_time += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Neighbor graph + consensus
        neighbors = build_neighbor_graph(robots)
        consensus_done, global_known, agreed_index = step_average_consensus(
            robots, neighbors, hazard_mgr.active
        )

        # If consensus says hazard at a different building than true → this is also interesting
        # But for now we assume consensus converges to true building (because initial info is correct).

        # Update hazard (local detection + metrics timings)
        hazard_mgr.update(robots, sim_time, consensus_done)

        # State machine for each robot
        for r in robots:
            # if consensus_done and agreed_index is not None, we treat that as global building index
            hb_index = hazard_mgr.building_index
            # We *force* hazard target = true building; if you want to show failure cases,
            # you can switch to agreed_index.
            r.update_state_machine(
                hazard_mgr.active,
                hb_index,
                consensus_done,
                global_known
            )
            r.update_motion()

        # If a hazard just got cleared & we have times, log metrics
        if not hazard_mgr.active and hazard_mgr.aggregated_time is not None:
            metrics.log_hazard(hazard_mgr, buildings)
            # reset to avoid double logging
            hazard_mgr.aggregated_time = None

        # DRAW
        draw_city(screen, buildings)

        # Draw communication edges first (under robots)
        for r in robots:
            r.draw_comm_links(screen, robots)

        # Draw robots
        for r in robots:
            r.draw(screen)

        # Draw hazard marker
        hazard_mgr.draw(screen)

        # HUD
        hud_lines = []
        if hazard_mgr.active and hazard_mgr.building_index is not None:
            bname = buildings[hazard_mgr.building_index]["name"]
            hud_lines.append(f"Hazard #{hazard_mgr.id_counter} at {bname}")
            hud_lines.append("Phase: detection + consensus + response")
        else:
            hud_lines.append("No active hazard: robots patrolling.")
            if hazard_mgr.timer_until_next is not None:
                t_left = hazard_mgr.timer_until_next / FPS
                hud_lines.append(f"Next hazard in ~{t_left:4.1f}s")

        y = 10
        for line in hud_lines:
            txt = font_big.render(line, True, (240, 240, 240))
            screen.blit(txt, (20, y))
            y += 28

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
