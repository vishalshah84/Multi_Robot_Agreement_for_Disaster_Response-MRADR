# environment.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class WorldConfig:
    """Continuous 2D world + site configuration."""
    width: float = 10.0
    height: float = 10.0

    # how far from the border we are allowed to place sites
    margin: float = 1.0

    n_victims: int = 2
    n_false_positives: int = 10

    # minimum distance between any two sites
    min_site_separation: float = 0.7

    # optional seed for reproducibility
    rng_seed: Optional[int] = None


@dataclass
class Site:
    """A fixed point in the world: either a victim or a false positive."""
    x: float
    y: float
    is_victim: bool
    id: int  # index inside victims or false positives

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


def _sample_point(cfg: WorldConfig, rng: np.random.Generator) -> np.ndarray:
    """Sample a random point inside the world, away from the outer margin."""
    x = rng.uniform(cfg.margin, cfg.width - cfg.margin)
    y = rng.uniform(cfg.margin, cfg.height - cfg.margin)
    return np.array([x, y], dtype=float)


def _far_from_existing(p: np.ndarray,
                       existing: List[Site],
                       min_dist: float) -> bool:
    """Check if p is at least min_dist away from every existing site."""
    for s in existing:
        if np.linalg.norm(p - s.pos) < min_dist:
            return False
    return True


def build_environment(cfg: WorldConfig) -> Tuple[List[Site], List[Site]]:
    """
    Create victim and false-positive sites in a continuous 2D world.
    Returns:
        victims:      list of Site objects with is_victim=True
        false_sites:  list of Site objects with is_victim=False
    """
    rng = np.random.default_rng(cfg.rng_seed)

    victims: List[Site] = []
    false_sites: List[Site] = []

    # 1) Place victims
    while len(victims) < cfg.n_victims:
        p = _sample_point(cfg, rng)
        if _far_from_existing(p, victims, cfg.min_site_separation):
            vid = len(victims)
            victims.append(Site(x=p[0], y=p[1], is_victim=True, id=vid))

    # 2) Place false positives (must be far from victims AND other FPs)
    while len(false_sites) < cfg.n_false_positives:
        p = _sample_point(cfg, rng)
        if _far_from_existing(p, victims + false_sites, cfg.min_site_separation):
            fid = len(false_sites)
            false_sites.append(Site(x=p[0], y=p[1], is_victim=False, id=fid))

    return victims, false_sites


def clip_to_world(pos: np.ndarray, cfg: WorldConfig,
                  eps: float = 1e-3) -> np.ndarray:
    """
    Keep a position inside the world box [0,W] x [0,H].
    Slight epsilon so we never sit exactly on the wall.
    """
    x = float(np.clip(pos[0], eps, cfg.width - eps))
    y = float(np.clip(pos[1], eps, cfg.height - eps))
    return np.array([x, y], dtype=float)


def distance_to_walls(pos: np.ndarray, cfg: WorldConfig) -> np.ndarray:
    """
    Return distances to the four walls:
    [left, right, bottom, top].
    """
    x, y = float(pos[0]), float(pos[1])
    d_left = x
    d_right = cfg.width - x
    d_bottom = y
    d_top = cfg.height - y
    return np.array([d_left, d_right, d_bottom, d_top], dtype=float)


def sense_sites(pos: np.ndarray,
                victims: List[Site],
                false_sites: List[Site],
                sensor_range: float) -> List[Site]:
    """
    Return all sites (victims + false positives) that are within
    sensor_range of the given position.
    """
    hits: List[Site] = []
    for s in victims + false_sites:
        if np.linalg.norm(pos - s.pos) <= sensor_range:
            hits.append(s)
    return hits
