# robot.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from environment import (
    WorldConfig,
    Site,
    clip_to_world,
    sense_sites,
)


@dataclass
class RobotConfig:
    """Robot physical + sensing/comm parameters."""
    radius: float = 0.1

    # sensing and communication
    sensor_range: float = 1.5     # for detecting victims / false positives
    comm_range: float = 3.0       # for consensus graph

    # motion
    v_search: float = 0.8         # speed during phase 1
    v_nav: float = 1.0            # speed during phase 2
    max_omega: float = 3.0        # max turning rate (rad/s)


@dataclass
class Robot:
    id: int
    x: float
    y: float
    theta: float

    mode: str = "phase1"
    assigned_victim: Optional[int] = None
    slot_angle: Optional[float] = None   # <--- ADD THIS LINE

    info_state: float = 0.0
    info_history: List[float] = field(default_factory=list)

    belief_victims: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=float)
    )
    has_seen_victims: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=bool)
    )

    pos_history_phase1: List[np.ndarray] = field(default_factory=list)
    pos_history_phase2: List[np.ndarray] = field(default_factory=list)

    def pose(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    # ---------------- Motion ---------------- #

    def step_unicycle(self,
                      v: float,
                      omega: float,
                      dt: float,
                      world_cfg: WorldConfig) -> None:
        """
        Simple unicycle update in continuous world, with clipping
        back into the box. Logging is handled by the phase code.
        """
        max_omega = np.inf  # could use RobotConfig.max_omega if passed in
        omega = float(np.clip(omega, -max_omega, max_omega))

        # integrate
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += omega * dt

        # wrap heading to [-pi, pi]
        self.theta = (self.theta + np.pi) % (2.0 * np.pi) - np.pi

        # keep inside world
        p = clip_to_world(self.pose(), world_cfg)
        self.x, self.y = float(p[0]), float(p[1])

    # ---------------- Sensing ---------------- #

    def sense_environment(self,
                          victims: List[Site],
                          false_sites: List[Site],
                          r_cfg: RobotConfig) -> List[Site]:
        """
        Return list of all sites within sensor range.
        Uses environment.sense_sites under the hood.
        """
        hits = sense_sites(self.pose(), victims, false_sites, r_cfg.sensor_range)
        return hits


# ------------- Helper to create initial swarm ------------- #



# def init_robots(
#     n_robots: int,
#     world_cfg,
#     robot_cfg,
#     rng_seed: int = None,
# ) -> List[Robot]:
#     """
#     Initialise robots along all 4 edges of the world.

#     Tries to read world bounds from:
#       - world_cfg.x_min, x_max, y_min, y_max   (if they exist)
#       - otherwise world_cfg.xlim, ylim (tuples)
#       - otherwise world_cfg.world_size (0 .. world_size)
#       - otherwise falls back to [0, 10] × [0, 10]
#     """
#     def get_bounds(axis: str):
#         # 1) x_min/x_max, y_min/y_max
#         a_min = f"{axis}_min"
#         a_max = f"{axis}_max"
#         if hasattr(world_cfg, a_min) and hasattr(world_cfg, a_max):
#             return getattr(world_cfg, a_min), getattr(world_cfg, a_max)

#         # 2) xlim / ylim
#         lim_name = f"{axis}lim"
#         if hasattr(world_cfg, lim_name):
#             lo, hi = getattr(world_cfg, lim_name)
#             return float(lo), float(hi)

#         # 3) world_size -> [0, world_size]
#         if hasattr(world_cfg, "world_size"):
#             size = float(getattr(world_cfg, "world_size"))
#             return 0.0, size

#         # 4) default fallback
#         return 0.0, 10.0

#     x_min, x_max = get_bounds("x")
#     y_min, y_max = get_bounds("y")

#     rng = np.random.default_rng(rng_seed)
#     robots: List[Robot] = []

#     # how many robots per edge
#     base = n_robots // 4
#     rem = n_robots % 4
#     counts = [base] * 4
#     for i in range(rem):
#         counts[i] += 1

#     # small offset so they are just inside the border
#     dx = 0.15
#     dy = 0.15

#     robot_id = 1

#     # 1) left edge: x = x_min + dx, y random, heading roughly +x
#     for _ in range(counts[0]):
#         x = x_min + dx
#         y = rng.uniform(y_min + dy, y_max - dy)
#         theta = rng.normal(0.0, 0.25)
#         robots.append(Robot(id=robot_id, x=x, y=y, theta=theta))
#         robot_id += 1

#     # 2) right edge: x = x_max - dx, heading roughly -x
#     for _ in range(counts[1]):
#         x = x_max - dx
#         y = rng.uniform(y_min + dy, y_max - dy)
#         theta = np.pi + rng.normal(0.0, 0.25)
#         robots.append(Robot(id=robot_id, x=x, y=y, theta=theta))
#         robot_id += 1

#     # 3) bottom edge: y = y_min + dy, heading roughly +y
#     for _ in range(counts[2]):
#         x = rng.uniform(x_min + dx, x_max - dx)
#         y = y_min + dy
#         theta = np.pi / 2 + rng.normal(0.0, 0.25)
#         robots.append(Robot(id=robot_id, x=x, y=y, theta=theta))
#         robot_id += 1

#     # 4) top edge: y = y_max - dy, heading roughly -y
#     for _ in range(counts[3]):
#         x = rng.uniform(x_min + dx, x_max - dx)
#         y = y_max - dy
#         theta = -np.pi / 2 + rng.normal(0.0, 0.25)
#         robots.append(Robot(id=robot_id, x=x, y=y, theta=theta))
#         robot_id += 1

#     return robots




def init_robots(n_robots: int,
                world_cfg: WorldConfig,
                r_cfg: RobotConfig,
                rng_seed: Optional[int] = None) -> List[Robot]:
    """
    Spawn robots along the left side of the world with random y and heading.
    Belief arrays are initialized later (in phase1_markov) once we know n_victims.
    """
    rng = np.random.default_rng(rng_seed)
    robots: List[Robot] = []

    for i in range(n_robots):
        # small offset from left wall
        x0 = 0.5
        y0 = rng.uniform(1.0, world_cfg.height - 1.0)

        # heading roughly pointing into the world (towards +x)
        theta0 = rng.uniform(-np.pi / 4, np.pi / 4)

        r = Robot(
            id=i,
            x=x0,
            y=y0,
            theta=theta0,
            mode="phase1",
        )
        robots.append(r)

    return robots
