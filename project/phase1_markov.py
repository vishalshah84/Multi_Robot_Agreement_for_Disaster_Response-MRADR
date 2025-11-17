# phase1_markov.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd

from environment import WorldConfig, Site, distance_to_walls
from robot import Robot, RobotConfig


@dataclass
class Phase1Config:
    """Configuration for random walk + noisy, confused consensus."""
    dt: float = 0.1
    n_steps: int = 1000

    # random walk
    omega_noise_std: float = 1.0
    wall_margin: float = 0.5

    # consensus params
    alpha: float = 0.2          # consensus step size
    anchor_gain_true: float = 0.5   # gain for robots that truly saw victim
    pos_tol: float = 0.1        # tolerance on std(x), std(y)
    min_steps_before_check: int = 50

    # sensing / measurement noise
    meas_noise_std_victim: float = 0.05   # noise on true victim measurement
    meas_noise_std_fp: float = 0.1        # noise when confusing false positive
    fp_confusion_prob: float = 0.2        # P(confuse FP as victim when only FP is seen)
    fp_update_gain: float = 0.1           # how strongly FP confusion pulls belief


@dataclass
class Phase1Result:
    consensus_reached: bool
    steps_run: int
    consensus_positions: np.ndarray  # shape (n_victims, 2)
    detection_steps: List[Optional[int]]  # first true detection per victim


def _enforce_robot_spacing(robots: List[Robot], min_dist: float = 0.2) -> None:
    """Push robots apart slightly if they are too close (visual only)."""
    n = len(robots)
    for i in range(n):
        for j in range(i + 1, n):
            ri = robots[i]
            rj = robots[j]

            pi = ri.pose()
            pj = rj.pose()
            dvec = pi - pj
            dist = np.linalg.norm(dvec)

            if dist < 1e-6:
                angle = np.random.uniform(0, 2 * np.pi)
                dvec = np.array([np.cos(angle), np.sin(angle)])
                dist = 1.0

            if dist < min_dist:
                overlap = min_dist - dist
                dunit = dvec / dist
                shift = 0.5 * overlap
                ri.x += dunit[0] * shift
                ri.y += dunit[1] * shift
                rj.x -= dunit[0] * shift


def run_phase1(
    world_cfg: WorldConfig,
    robot_cfg: RobotConfig,
    phase_cfg: Phase1Config,
    victims: List[Site],
    false_sites: List[Site],
    robots: List[Robot],
    rng_seed: Optional[int] = None,
) -> Phase1Result:
    """
    Phase 1:
      - Continuous random walk with wall avoidance
      - Noisy measurements of true victims
      - Occasional confusion with false positives
      - Consensus on victim positions still converges to true locations
    """
    rng = np.random.default_rng(rng_seed)
    n_robots = len(robots)
    n_victims = len(victims)

    # === Initialise beliefs and logs ===
    init_belief = np.array(
        [world_cfg.width / 2.0, world_cfg.height / 2.0], dtype=float
    )
    for r in robots:
        r.belief_victims = np.tile(init_belief, (n_victims, 1))
        r.has_seen_victims = np.zeros(n_victims, dtype=bool)
        r.pos_history_phase1.clear()
        r.pos_history_phase2.clear()  # empty for now

    detection_steps: List[Optional[int]] = [None] * n_victims
    consensus_reached = False
    steps_run = phase_cfg.n_steps

    for k in range(phase_cfg.n_steps):
        t = k * phase_cfg.dt  # not used but kept for clarity

        # === 1) Random-walk motion + wall avoidance ===
        for r in robots:
            v = robot_cfg.v_search
            omega = rng.normal(0.0, phase_cfg.omega_noise_std)

            d_walls = distance_to_walls(r.pose(), world_cfg)
            left, right, bottom, top = d_walls

            if left < phase_cfg.wall_margin:
                omega += 2.0
            if right < phase_cfg.wall_margin:
                omega -= 2.0
            if bottom < phase_cfg.wall_margin:
                omega += 2.0 * np.sign(np.cos(r.theta))
            if top < phase_cfg.wall_margin:
                omega -= 2.0 * np.sign(np.cos(r.theta))

            r.step_unicycle(v=v, omega=omega, dt=phase_cfg.dt, world_cfg=world_cfg)

            # log Phase-1 position
            r.pos_history_phase1.append(r.pose().copy())

        _enforce_robot_spacing(robots, min_dist=2 * robot_cfg.radius)

        # === 2) Sensing with noisy measurements + FP confusion ===
        for r in robots:
            pos = r.pose()

            # Which victims are inside sensor range?
            victims_in_range = []
            for vi, v in enumerate(victims):
                if np.linalg.norm(pos - v.pos) <= robot_cfg.sensor_range:
                    victims_in_range.append(vi)

            # Which false positives are inside range?
            fps_in_range = []
            for fi, fp in enumerate(false_sites):
                if np.linalg.norm(pos - fp.pos) <= robot_cfg.sensor_range:
                    fps_in_range.append(fi)

            # --- Case A: true victim(s) visible ---
            if victims_in_range:
                for vi in victims_in_range:
                    v = victims[vi]

                    # first detection time
                    if detection_steps[vi] is None:
                        detection_steps[vi] = k

                    r.has_seen_victims[vi] = True

                    # noisy measurement of victim location
                    noise = rng.normal(
                        0.0,
                        phase_cfg.meas_noise_std_victim,
                        size=2,
                    )
                    z = v.pos + noise  # measurement

                    r.belief_victims[vi, :] = z

            # --- Case B: no victim, but false positives nearby ---
            elif fps_in_range:
                if rng.random() < phase_cfg.fp_confusion_prob:
                    # pick nearest FP
                    dists = [
                        np.linalg.norm(pos - false_sites[fi].pos)
                        for fi in fps_in_range
                    ]
                    fi_sel = fps_in_range[int(np.argmin(dists))]
                    fp_pos = false_sites[fi_sel].pos

                    # choose a victim index to mis-assign (nearest victim to this FP)
                    v_dists = [np.linalg.norm(fp_pos - v.pos) for v in victims]
                    vi = int(np.argmin(v_dists))

                    # noisy "fake" measurement
                    noise = rng.normal(
                        0.0,
                        phase_cfg.meas_noise_std_fp,
                        size=2,
                    )
                    z_fp = fp_pos + noise

                    # weakly pull belief toward wrong location
                    r.belief_victims[vi, :] = (
                        (1.0 - phase_cfg.fp_update_gain) * r.belief_victims[vi, :]
                        + phase_cfg.fp_update_gain * z_fp
                    )
            # else: nothing sensed, no update

        # === 3) Build communication graph ===
        positions = np.stack([r.pose() for r in robots], axis=0)
        A = np.zeros((n_robots, n_robots), dtype=float)
        for i in range(n_robots):
            for j in range(i + 1, n_robots):
                d = np.linalg.norm(positions[i] - positions[j])
                if d <= robot_cfg.comm_range:
                    A[i, j] = 1.0
                    A[j, i] = 1.0
            A[i, i] = 1.0  # self-loop

        # === 4) Position consensus for each victim ===
        for vi, v in enumerate(victims):
            if detection_steps[vi] is None:
                continue

            beliefs = np.stack(
                [r.belief_victims[vi] for r in robots], axis=0
            )  # (N,2)
            has_seen = np.array(
                [r.has_seen_victims[vi] for r in robots], dtype=bool
            )

            new_beliefs = beliefs.copy()
            for i in range(n_robots):
                neighbors = np.where(A[i] > 0)[0]
                if len(neighbors) > 0:
                    neighbor_mean = np.mean(beliefs[neighbors], axis=0)
                    new = beliefs[i] + phase_cfg.alpha * (
                        neighbor_mean - beliefs[i]
                    )

                    if has_seen[i]:
                        new = (
                            (1.0 - phase_cfg.anchor_gain_true) * new
                            + phase_cfg.anchor_gain_true * v.pos
                        )

                    new_beliefs[i] = new

            for i, r in enumerate(robots):
                r.belief_victims[vi, :] = new_beliefs[i]

        # === 5) Check consensus convergence ===
        if k >= phase_cfg.min_steps_before_check:
            all_good = True
            any_detected = False
            for vi in range(n_victims):
                if detection_steps[vi] is None:
                    all_good = False
                    continue

                any_detected = True
                beliefs = np.stack(
                    [r.belief_victims[vi] for r in robots], axis=0
                )
                std_xy = np.std(beliefs, axis=0)
                if np.any(std_xy > phase_cfg.pos_tol):
                    all_good = False

            if any_detected and all_good:
                consensus_reached = True
                steps_run = k + 1
                break

    consensus_positions = np.zeros((n_victims, 2), dtype=float)
    for vi in range(n_victims):
        beliefs = np.stack(
            [r.belief_victims[vi] for r in robots], axis=0
        )
        consensus_positions[vi, :] = np.mean(beliefs, axis=0)

    return Phase1Result(
        consensus_reached=consensus_reached,
        steps_run=steps_run,
        consensus_positions=consensus_positions,
        detection_steps=detection_steps,
    )


def save_phase1_results_to_excel(
    victims: List[Site],
    false_sites: List[Site],
    result: Phase1Result,
    filename: str = "phase1_results.xlsx",
) -> None:
    rows_v = []
    for vi, v in enumerate(victims):
        cons_x, cons_y = result.consensus_positions[vi]
        true_x, true_y = v.x, v.y
        err_x = cons_x - true_x
        err_y = cons_y - true_y
        err_euclid = float(np.linalg.norm([err_x, err_y]))
        rows_v.append(
            {
                "victim_id": vi + 1,
                "true_x": true_x,
                "true_y": true_y,
                "consensus_x": cons_x,
                "consensus_y": cons_y,
                "error_x": err_x,
                "error_y": err_y,
                "euclidean_error": err_euclid,
                "first_detection_step": result.detection_steps[vi],
            }
        )

    df_v = pd.DataFrame(rows_v)

    rows_fp = []
    for fp in false_sites:
        rows_fp.append(
            {
                "false_id": fp.id + 1,
                "x": fp.x,
                "y": fp.y,
            }
        )
    df_fp = pd.DataFrame(rows_fp)

    with pd.ExcelWriter(filename) as writer:
        df_v.to_excel(writer, sheet_name="victims", index=False)
        df_fp.to_excel(writer, sheet_name="false_positives", index=False)

    print(f"[phase1] Saved Phase-1 consensus results to {filename}")
