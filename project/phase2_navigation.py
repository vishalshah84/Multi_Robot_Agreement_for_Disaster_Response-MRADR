from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np
import pandas as pd

from environment import WorldConfig, Site, distance_to_walls
from robot import Robot, RobotConfig


@dataclass
class Phase2Config:
    dt: float = 0.1
    n_steps: int = 20000

    # swarm neighbour forces
    k_neighbor_attr: float = 0.25
    k_neighbor_rep: float = 1.0
    neighbor_range: float = 2.0
    neighbor_rep_dist: float = 0.7

    # local attraction / repulsion around sites (victims + FPs)
    influence_radius: float = 0.8      # outer radius where site attracts
    safe_radius: float = 0.15          # inner “hard core” repulsion

    k_victim_local: float = 1.2        # attraction strength to victims
    k_fp_local: float = 0.4            # weaker attraction to false positives
    k_site_rep: float = 2.0            # inner repulsion strength (both)

    # heading / wall repulsion
    k_omega: float = 2.5
    wall_margin: float = 0.5
    omega_noise_std: float = 0.2

    # ring formation
    robots_per_victim: int = 6
    robots_per_fp: int = 4
    ring_radius: float = 0.7

    # position consensus (for victims)
    alpha: float = 0.2
    anchor_gain_true: float = 0.7
    pos_tol: float = 0.05
    min_steps_before_check: int = 80

    # sensing / measurement noise (kept for compatibility)
    meas_noise_std_victim: float = 0.03
    meas_noise_std_fp: float = 0.08
    fp_confusion_prob: float = 0.1
    fp_update_gain: float = 0.05

    # signal-strength model for victim vs FP classification
    victim_signal_mean: float = 1.0
    fp_signal_mean: float = 0.35
    signal_noise_std: float = 0.1
    signal_consensus_gain: float = 0.3
    anchor_gain_signal: float = 0.5
    signal_threshold: float = 0.7     # between victim and FP means

    # small global drift so main swarm keeps sweeping the map
    global_drift_x: float = 1.0

    # how long to keep simulating after consensus (for nice animation)
    extra_time_after_consensus: float = 20.0

@dataclass
class Phase2Result:
    consensus_reached: bool
    steps_run: int
    consensus_positions: np.ndarray


def _wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


# ---------- generic K-nearest sub-swarm assignment (victims + FPs) ----------

def _assign_ring_team_knn(
    robots: List[Robot],
    center: np.ndarray,
    team_size: int,
    site_type: str,         # "victim" or "fp"
    site_index: int,
) -> None:
    """
    Assign a sub-swarm around a site using K-nearest neighbours.

    - Use only robots that are not already assigned to any site.
    - Sort by distance to site centre.
    - Take up to 'team_size' robots.
    - Give each one an evenly spaced slot angle on a ring.
    """
    # robots that are free (no victim and no FP assignment)
    free_indices = [
        i for i, r in enumerate(robots)
        if getattr(r, "assigned_victim", None) is None
        and getattr(r, "assigned_fp", None) is None
    ]
    if not free_indices or team_size <= 0:
        return

    dists = []
    for i in free_indices:
        p = robots[i].pose()
        dists.append((np.linalg.norm(p - center), i))
    dists.sort(key=lambda x: x[0])

    actual_team_size = min(team_size, len(dists))
    if actual_team_size == 0:
        return

    team_indices = [idx for _, idx in dists[:actual_team_size]]

    for slot_idx, rid in enumerate(team_indices):
        r = robots[rid]
        if site_type == "victim":
            r.assigned_victim = site_index
        elif site_type == "fp":
            r.assigned_fp = site_index
        else:
            raise ValueError(f"Unknown site_type: {site_type}")
        # evenly spaced angles on the ring
        r.slot_angle = 2.0 * np.pi * slot_idx / actual_team_size


# ---------- main Phase 2 loop ----------

def run_phase2(
    world_cfg: WorldConfig,
    robot_cfg: RobotConfig,
    phase_cfg: Phase2Config,
    victims: List[Site],
    false_sites: List[Site],
    robots: List[Robot],
    initial_estimates: np.ndarray,
    rng_seed: Optional[int] = None,
) -> Phase2Result:

    rng = np.random.default_rng(rng_seed)
    n_robots = len(robots)
    n_victims = len(victims)
    n_fps = len(false_sites)

    logs: List[dict] = []

    # ---------- init robots ----------
    for r in robots:
        r.mode = "phase2"
        r.pos_history_phase2.clear()
        r.belief_victims = initial_estimates.copy()
        r.has_seen_victims = np.zeros(n_victims, dtype=bool)
        r.has_seen_fp = np.zeros(n_fps, dtype=bool)

        r.assigned_victim = None
        r.assigned_fp = None
        r.slot_angle = None

        r.victim_signal = np.zeros(n_victims, dtype=float)
        r.fp_signal = np.zeros(n_fps, dtype=float)

        r.logged_victim_detection = np.zeros(n_victims, dtype=bool)
        r.logged_fp_detection = np.zeros(n_fps, dtype=bool)

    victim_has_team: Dict[int, bool] = {vi: False for vi in range(n_victims)}
    detector_robot_v: Dict[int, Optional[int]] = {vi: None for vi in range(n_victims)}

    fp_has_team: Dict[int, bool] = {fi: False for fi in range(n_fps)}
    detector_robot_fp: Dict[int, Optional[int]] = {fi: None for fi in range(n_fps)}
    fp_resolved: Dict[int, bool] = {fi: False for fi in range(n_fps)}

    consensus_reached = False
    consensus_step: Optional[int] = None
    steps_run = 0

    max_steps = phase_cfg.n_steps
    grace_steps = int(phase_cfg.extra_time_after_consensus / phase_cfg.dt)

    # zig-zag: right → down → left → down → repeat
    horizontal_len = 600      # steps to sweep horizontally
    drop_len = 50            # steps to move down a bit
    total_segment = horizontal_len + drop_len

    k = 0
    while True:
        t = k * phase_cfg.dt

        # ---------- 1) sensing ----------
        for r in robots:
            pos = r.pose()

            # victims
            for vi, v in enumerate(victims):
                if np.linalg.norm(pos - v.pos) <= robot_cfg.sensor_range:
                    r.has_seen_victims[vi] = True

                    if not r.logged_victim_detection[vi]:
                        r.logged_victim_detection[vi] = True
                        logs.append(
                            dict(
                                step=k,
                                time=t,
                                event="detect_victim",
                                robot_id=r.id,
                                victim_index=vi,
                                x=pos[0],
                                y=pos[1],
                            )
                        )

                    noise_pos = rng.normal(
                        0.0, phase_cfg.meas_noise_std_victim, size=2
                    )
                    r.belief_victims[vi] = v.pos + noise_pos

                    noise_sig = rng.normal(0.0, phase_cfg.signal_noise_std)
                    r.victim_signal[vi] = phase_cfg.victim_signal_mean + noise_sig

                    if detector_robot_v[vi] is None:
                        detector_robot_v[vi] = r.id

            # false positives
            for fi, fp in enumerate(false_sites):
                if fp_resolved[fi]:
                    continue
                if np.linalg.norm(pos - fp.pos) <= robot_cfg.sensor_range:
                    r.has_seen_fp[fi] = True

                    if not r.logged_fp_detection[fi]:
                        r.logged_fp_detection[fi] = True
                        logs.append(
                            dict(
                                step=k,
                                time=t,
                                event="detect_fp",
                                robot_id=r.id,
                                fp_index=fi,
                                x=pos[0],
                                y=pos[1],
                            )
                        )

                    noise_sig = rng.normal(0.0, phase_cfg.signal_noise_std)
                    r.fp_signal[fi] = phase_cfg.fp_signal_mean + noise_sig

                    if detector_robot_fp[fi] is None:
                        detector_robot_fp[fi] = r.id

        # ---------- 2) form teams ----------
        for vi, v in enumerate(victims):
            if not victim_has_team[vi] and detector_robot_v[vi] is not None:
                victim_has_team[vi] = True
                _assign_ring_team_knn(
                    robots, v.pos, phase_cfg.robots_per_victim, "victim", vi
                )

        for fi, fp in enumerate(false_sites):
            if fp_resolved[fi]:
                continue
            if not fp_has_team[fi] and detector_robot_fp[fi] is not None:
                fp_has_team[fi] = True
                _assign_ring_team_knn(
                    robots, fp.pos, phase_cfg.robots_per_fp, "fp", fi
                )

        # ---------- 3) zig-zag drift decision ----------
        positions = np.stack([r.pose() for r in robots], axis=0)

        idx_in_cycle = k % (2 * total_segment)
        if idx_in_cycle < horizontal_len:
            zig_phase = 0          # right
        elif idx_in_cycle < total_segment:
            zig_phase = 1          # down
        elif idx_in_cycle < total_segment + horizontal_len:
            zig_phase = 2          # left
        else:
            zig_phase = 3          # down

        speed = phase_cfg.global_drift_x
        if zig_phase == 0:
            drift_vec = np.array([speed, 0.0])
        elif zig_phase == 1:
            drift_vec = np.array([0.0, -speed])
        elif zig_phase == 2:
            drift_vec = np.array([-speed, 0.0])
        else:
            drift_vec = np.array([0.0, -speed])

        # ---------- 4) move robots ----------
        for i, r in enumerate(robots):
            pos = positions[i]

            if r.assigned_victim is not None or r.assigned_fp is not None:
                # team robot → ring control
                if r.assigned_victim is not None:
                    centre = victims[r.assigned_victim].pos
                else:
                    centre = false_sites[r.assigned_fp].pos

                diff_c = centre - pos
                dist_c = np.linalg.norm(diff_c)

                ring_target = centre + phase_cfg.ring_radius * np.array(
                    [np.cos(r.slot_angle), np.sin(r.slot_angle)]
                )

                if dist_c > 1.5 * phase_cfg.ring_radius:
                    target = centre
                else:
                    target = ring_target

                diff = target - pos
                heading = np.arctan2(diff[1], diff[0])
                err = _wrap_to_pi(heading - r.theta)
                omega = phase_cfg.k_omega * err
                v = robot_cfg.v_nav * max(0.0, np.cos(err))

            else:
                # free robot
                total_force = np.zeros(2)

                # neighbours
                for j in range(n_robots):
                    if j == i:
                        continue
                    diff = positions[j] - pos
                    dist = np.linalg.norm(diff) + 1e-6
                    if dist <= phase_cfg.neighbor_range:
                        total_force += phase_cfg.k_neighbor_attr * diff
                    if dist <= phase_cfg.neighbor_rep_dist:
                        total_force += phase_cfg.k_neighbor_rep * (-diff) / (dist**2)

                # victims potentials
                for vi, v_site in enumerate(victims):
                    diff = v_site.pos - pos
                    dist = np.linalg.norm(diff) + 1e-6
                    if dist <= phase_cfg.influence_radius:
                        if victim_has_team[vi]:
                            total_force += (
                                2.0 * phase_cfg.k_site_rep
                                * (pos - v_site.pos)
                                / (dist**2)
                            )
                        else:
                            if dist >= phase_cfg.safe_radius:
                                total_force += (
                                    phase_cfg.k_victim_local * diff / (dist**2)
                                )
                            else:
                                total_force += (
                                    phase_cfg.k_site_rep
                                    * (pos - v_site.pos)
                                    / (dist**2)
                                )

                # FP potentials
                for fi, fp_site in enumerate(false_sites):
                    diff = fp_site.pos - pos
                    dist = np.linalg.norm(diff) + 1e-6
                    if dist <= phase_cfg.influence_radius:
                        if fp_resolved[fi] or fp_has_team[fi]:
                            if dist < phase_cfg.safe_radius:
                                total_force += (
                                    phase_cfg.k_site_rep
                                    * (pos - fp_site.pos)
                                    / (dist**2)
                                )
                        else:
                            if dist >= phase_cfg.safe_radius:
                                total_force += (
                                    phase_cfg.k_fp_local * diff / (dist**2)
                                )
                            else:
                                total_force += (
                                    phase_cfg.k_site_rep
                                    * (pos - fp_site.pos)
                                    / (dist**2)
                                )

                # zig-zag drift
                total_force += drift_vec

                # convert force to velocity
                normF = np.linalg.norm(total_force)
                if normF < 1e-6:
                    omega = rng.normal(0.0, phase_cfg.omega_noise_std)
                    v = 0.6 * robot_cfg.v_nav
                else:
                    heading = np.arctan2(total_force[1], total_force[0])
                    err = _wrap_to_pi(heading - r.theta)
                    omega = phase_cfg.k_omega * err
                    v = robot_cfg.v_nav * max(0.0, np.cos(err))

            # walls
            left, right, bottom, top = distance_to_walls(pos, world_cfg)
            if left < phase_cfg.wall_margin:
                omega += 2.0
            if right < phase_cfg.wall_margin:
                omega -= 2.0
            if bottom < phase_cfg.wall_margin:
                omega += 2.0 * np.sign(np.cos(r.theta))
            if top < phase_cfg.wall_margin:
                omega -= 2.0 * np.sign(np.cos(r.theta))

            # step robot
            r.step_unicycle(v=v, omega=omega, dt=phase_cfg.dt, world_cfg=world_cfg)
            r.pos_history_phase2.append(r.pose().copy())

        # ---------- 5) communication graph ----------
        positions = np.stack([r.pose() for r in robots], axis=0)
        A = np.zeros((n_robots, n_robots), dtype=float)
        for i in range(n_robots):
            for j in range(i + 1, n_robots):
                if np.linalg.norm(positions[i] - positions[j]) <= robot_cfg.comm_range:
                    A[i, j] = A[j, i] = 1.0
            A[i, i] = 1.0

        # ---------- 6) victim consensus ----------
        for vi in range(n_victims):
            team = [idx for idx, r in enumerate(robots) if r.assigned_victim == vi]
            if not team:
                continue

            beliefs = np.stack(
                [robots[i].belief_victims[vi] for i in team], axis=0
            )
            seen = np.array(
                [robots[i].has_seen_victims[vi] for i in team], dtype=bool
            )
            signals = np.array(
                [robots[i].victim_signal[vi] for i in team], dtype=float
            )

            newB = beliefs.copy()
            newS = signals.copy()

            for l, ig in enumerate(team):
                neigh_global = [j for j in team if A[ig, j] > 0]
                if not neigh_global:
                    continue
                neigh_local = [team.index(jg) for jg in neigh_global]

                mpos = beliefs[neigh_local].mean(axis=0)
                pos_new = beliefs[l] + phase_cfg.alpha * (mpos - beliefs[l])
                if seen[l]:
                    pos_new = (
                        (1.0 - phase_cfg.anchor_gain_true) * pos_new
                        + phase_cfg.anchor_gain_true * victims[vi].pos
                    )
                newB[l] = pos_new

                msig = signals[neigh_local].mean()
                sig_new = signals[l] + phase_cfg.signal_consensus_gain * (
                    msig - signals[l]
                )
                if seen[l]:
                    sig_new = (
                        (1.0 - phase_cfg.anchor_gain_signal) * sig_new
                        + phase_cfg.anchor_gain_signal * phase_cfg.victim_signal_mean
                    )
                newS[l] = sig_new

            for l, ig in enumerate(team):
                robots[ig].belief_victims[vi] = newB[l]
                robots[ig].victim_signal[vi] = newS[l]

        # ---------- 7) FP consensus + classify ----------
        for fi in range(n_fps):
            if fp_resolved[fi]:
                continue
            team = [idx for idx, r in enumerate(robots) if r.assigned_fp == fi]
            if not team:
                continue

            signals = np.array(
                [robots[i].fp_signal[fi] for i in team], dtype=float
            )
            seen = np.array(
                [robots[i].has_seen_fp[fi] for i in team], dtype=bool
            )
            newS = signals.copy()

            for l, ig in enumerate(team):
                neigh_global = [j for j in team if A[ig, j] > 0]
                if not neigh_global:
                    continue
                neigh_local = [team.index(jg) for jg in neigh_global]

                msig = signals[neigh_local].mean()
                sig_new = signals[l] + phase_cfg.signal_consensus_gain * (
                    msig - signals[l]
                )
                if seen[l]:
                    sig_new = (
                        (1.0 - phase_cfg.anchor_gain_signal) * sig_new
                        + phase_cfg.anchor_gain_signal * phase_cfg.fp_signal_mean
                    )
                newS[l] = sig_new

            for l, ig in enumerate(team):
                robots[ig].fp_signal[fi] = newS[l]

            mean_sig = float(newS.mean())
            if mean_sig < phase_cfg.signal_threshold:
                fp_resolved[fi] = True
                logs.append(
                    dict(
                        step=k,
                        time=t,
                        event="fp_resolved",
                        fp_index=fi,
                        robots=";".join(f"R{robots[i].id}" for i in team),
                        mean_signal=mean_sig,
                    )
                )
                for ig in team:
                    robots[ig].assigned_fp = None
                    robots[ig].slot_angle = None

        # ---------- 8) check global victim consensus ----------
        if k >= phase_cfg.min_steps_before_check:
            all_good = True
            for vi in range(n_victims):
                team = [idx for idx, r in enumerate(robots) if r.assigned_victim == vi]
                if len(team) < 2:
                    all_good = False
                    break
                beliefs = np.stack(
                    [robots[i].belief_victims[vi] for i in team], axis=0
                )
                if np.any(np.std(beliefs, axis=0) > phase_cfg.pos_tol):
                    all_good = False
                    break

            if all_good and consensus_step is None:
                consensus_reached = True
                consensus_step = k
                logs.append(
                    dict(
                        step=k,
                        time=t,
                        event="victim_consensus_reached",
                    )
                )

        # ---------- 9) advance time & stopping logic ----------
        k += 1
        steps_run = k

        if consensus_step is not None:
            if k >= consensus_step + grace_steps:
                break
        elif k >= max_steps:
            break

    # ---------- 10) final consensus positions ----------
    consensus_positions = np.zeros((n_victims, 2), dtype=float)
    for vi in range(n_victims):
        team = [idx for idx, r in enumerate(robots) if r.assigned_victim == vi]
        if team:
            beliefs = np.stack(
                [robots[i].belief_victims[vi] for i in team], axis=0
            )
        else:
            beliefs = np.stack(
                [r.belief_victims[vi] for r in robots], axis=0
            )
        consensus_positions[vi] = beliefs.mean(axis=0)

    if logs:
        pd.DataFrame(logs).to_csv("phase2_log.csv", index=False)

    return Phase2Result(
        consensus_reached=consensus_reached,
        steps_run=steps_run,
        consensus_positions=consensus_positions,
    )


def save_phase2_results_to_excel(
    victims: List[Site],
    result: Phase2Result,
    filename: str = "phase2_results.xlsx",
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
            }
        )

    df_v = pd.DataFrame(rows_v)
    df_v.to_excel(filename, index=False)
    print(f"[phase2] Saved Phase-2 consensus results to {filename}")