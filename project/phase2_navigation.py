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
    n_steps: int = 3000

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
    """
    Phase 2 behaviour (navigation + consensus):

    • Main swarm: neighbour attraction/repulsion + walls + small global drift.
    • Victims and FPs both act as attractive sites (FP weaker).
    • When a site (victim or FP) is detected, form a K-nearest sub-swarm.
    • Sub-swarm:
        - rendezvous at site centre,
        - then form a ring around it,
        - run position consensus (victims) and signal-strength consensus (victims + FPs).
    • For false positives: if agreed signal strength < threshold, robots release and
      rejoin main swarm; the FP is marked “resolved” and ignored afterwards.
    """
    rng = np.random.default_rng(rng_seed)
    n_robots = len(robots)
    n_victims = len(victims)
    n_fps = len(false_sites)

    # logging
    logs: List[dict] = []

    # init state for Phase 2 (does not touch Phase 1 code/settings)
    for r in robots:
        r.mode = "phase2"
        r.pos_history_phase2.clear()
        r.belief_victims = initial_estimates.copy()
        r.has_seen_victims = np.zeros(n_victims, dtype=bool)
        r.assigned_victim = None          # sub-swarm around a true victim
        r.assigned_fp = None              # sub-swarm around a false positive
        r.slot_angle = None

        # signal-strength memories (for consensus)
        r.victim_signal = np.zeros(n_victims, dtype=float)
        r.fp_signal = np.zeros(n_fps, dtype=float)
        r.has_seen_fp = np.zeros(n_fps, dtype=bool)

        # for logging first detections
        r.logged_victim_detection = np.zeros(n_victims, dtype=bool)
        r.logged_fp_detection = np.zeros(n_fps, dtype=bool)

    # per-site flags
    victim_has_team: Dict[int, bool] = {vi: False for vi in range(n_victims)}
    detector_robot_v: Dict[int, Optional[int]] = {vi: None for vi in range(n_victims)}

    fp_has_team: Dict[int, bool] = {fi: False for fi in range(n_fps)}
    detector_robot_fp: Dict[int, Optional[int]] = {fi: None for fi in range(n_fps)}
    fp_resolved: Dict[int, bool] = {fi: False for fi in range(n_fps)}

    consensus_reached = False
    steps_run = 0

    # very large safety cap, just in case something goes wrong
    max_steps_cap = phase_cfg.n_steps

    k = 0
    while True:
        t = k * phase_cfg.dt  # not used but available for logging

        # ---- 1) Sensing: update beliefs + signal strengths ----
        for r in robots:
            pos = r.pose()

            victims_in_range = []
            for vi, v in enumerate(victims):
                if np.linalg.norm(pos - v.pos) <= robot_cfg.sensor_range:
                    victims_in_range.append(vi)

            fps_in_range = []
            for fi, fp in enumerate(false_sites):
                if fp_resolved[fi]:
                    continue  # already classified and ignored
                if np.linalg.norm(pos - fp.pos) <= robot_cfg.sensor_range:
                    fps_in_range.append(fi)

            # true victim(s) nearby: update position + strong signal
            for vi in victims_in_range:
                v = victims[vi]
                r.has_seen_victims[vi] = True

                # log first time this robot detects this victim
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

                # noisy position measurement
                noise_pos = rng.normal(
                    0.0, phase_cfg.meas_noise_std_victim, size=2
                )
                z_pos = v.pos + noise_pos
                r.belief_victims[vi, :] = z_pos

                # signal strength measurement
                noise_sig = rng.normal(0.0, phase_cfg.signal_noise_std)
                z_sig = phase_cfg.victim_signal_mean + noise_sig
                r.victim_signal[vi] = z_sig

                if detector_robot_v[vi] is None:
                    detector_robot_v[vi] = r.id

            # FP(s) nearby: weaker signal around that site
            for fi in fps_in_range:
                fp = false_sites[fi]
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
                z_sig = phase_cfg.fp_signal_mean + noise_sig
                r.fp_signal[fi] = z_sig

                if detector_robot_fp[fi] is None:
                    detector_robot_fp[fi] = r.id

        # ---- 2) Create sub-swarms via K-nearest (victims + FPs) ----
        # victims
        for vi, v in enumerate(victims):
            if victim_has_team[vi]:
                continue
            if detector_robot_v[vi] is not None:
                victim_has_team[vi] = True
                _assign_ring_team_knn(
                    robots=robots,
                    center=v.pos,
                    team_size=phase_cfg.robots_per_victim,
                    site_type="victim",
                    site_index=vi,
                )

        # false positives
        for fi, fp in enumerate(false_sites):
            if fp_resolved[fi]:
                continue
            if fp_has_team[fi]:
                continue
            if detector_robot_fp[fi] is not None:
                fp_has_team[fi] = True
                _assign_ring_team_knn(
                    robots=robots,
                    center=fp.pos,
                    team_size=phase_cfg.robots_per_fp,
                    site_type="fp",
                    site_index=fi,
                )

        # ---- 3) Navigation step ----
        positions = np.stack([r.pose() for r in robots], axis=0)

        for i, r in enumerate(robots):
            pos = positions[i]

            # 3a) robots assigned to a site (victim or FP): rendezvous + ring
            if r.assigned_victim is not None or r.assigned_fp is not None:
                if r.assigned_victim is not None:
                    centre = victims[r.assigned_victim].pos
                else:
                    centre = false_sites[r.assigned_fp].pos

                # vector to site centre
                to_c = centre - pos
                dist_c = np.linalg.norm(to_c) + 1e-6

                ring_target = centre + phase_cfg.ring_radius * np.array(
                    [np.cos(r.slot_angle), np.sin(r.slot_angle)]
                )

                # First go to centre, then move out to ring slot
                if dist_c > 1.5 * phase_cfg.ring_radius:
                    target = centre
                else:
                    target = ring_target

                diff = target - pos
                angle_to_target = np.arctan2(diff[1], diff[0])
                angle_err = _wrap_to_pi(angle_to_target - r.theta)

                omega = phase_cfg.k_omega * angle_err
                omega += rng.normal(0.0, phase_cfg.omega_noise_std * 0.5)
                v = robot_cfg.v_nav * max(0.0, np.cos(angle_err))

            # 3b) free robots -> swarm forces + site potential + global drift
            else:
                total_force = np.zeros(2)

                # neighbour cohesion + repulsion
                for j in range(n_robots):
                    if j == i:
                        continue
                    diff = positions[j] - pos
                    dist = np.linalg.norm(diff) + 1e-6

                    if dist <= phase_cfg.neighbor_range:
                        total_force += phase_cfg.k_neighbor_attr * diff
                    if dist <= phase_cfg.neighbor_rep_dist:
                        total_force += phase_cfg.k_neighbor_rep * (-diff) / (dist**2)

                # victims: donut potential field, but free robots must NOT disturb teams
                for vi, v_site in enumerate(victims):
                    diff = v_site.pos - pos
                    dist = np.linalg.norm(diff) + 1e-6

                    if dist <= phase_cfg.influence_radius:

                        if victim_has_team[vi]:
                            # victim already has a team:
                            # free robots MUST NOT approach -> strong outward push

                            # (1) strong outward repulsion
                            total_force += (
                                2.0 * phase_cfg.k_site_rep * (pos - v_site.pos) / (dist**2)
                            )

                            # (2) soft repel zone to push swarm around the victim
                            if dist < 2.0 * phase_cfg.ring_radius:
                                total_force += 0.5 * (pos - v_site.pos)

                        else:
                            # victim has NO team yet: all robots allowed to be attracted
                            if dist >= phase_cfg.safe_radius:
                                total_force += (
                                    phase_cfg.k_victim_local * diff / (dist**2)
                                )
                            else:
                                total_force += (
                                    phase_cfg.k_site_rep * (pos - v_site.pos) / (dist**2)
                                )


                # false positives: same shape but weaker attraction
                for fi, fp_site in enumerate(false_sites):
                    diff = fp_site.pos - pos
                    dist = np.linalg.norm(diff) + 1e-6

                    if dist <= phase_cfg.influence_radius:
                        if fp_resolved[fi] or fp_has_team[fi]:
                            # resolved or already has team:
                            # no outer attraction for free robots; only core repulsion
                            if dist < phase_cfg.safe_radius:
                                total_force += (
                                    phase_cfg.k_site_rep
                                    * (pos - fp_site.pos)
                                    / (dist**2)
                                )
                        else:
                            # unresolved FP with no team yet
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

                # small global drift so swarm sweeps the environment
                total_force += np.array([phase_cfg.global_drift_x, 0.0])

                # choose heading and speed
                normF = np.linalg.norm(total_force)

                if normF < 1e-6:
                    omega = rng.normal(0.0, phase_cfg.omega_noise_std)
                    v = 0.6 * robot_cfg.v_nav
                else:
                    heading = np.arctan2(total_force[1], total_force[0])
                    angle_err = _wrap_to_pi(heading - r.theta)
                    omega = phase_cfg.k_omega * angle_err
                    omega += rng.normal(0.0, phase_cfg.omega_noise_std)
                    v = robot_cfg.v_nav * max(0.0, np.cos(angle_err))

            # wall / border avoidance (turn away from walls)
            d_walls = distance_to_walls(pos, world_cfg)
            left, right, bottom, top = d_walls

            if left < phase_cfg.wall_margin:
                omega += 2.0
            if right < phase_cfg.wall_margin:
                omega -= 2.0
            if bottom < phase_cfg.wall_margin:
                omega += 2.0 * np.sign(np.cos(r.theta))
            if top < phase_cfg.wall_margin:
                omega -= 2.0 * np.sign(np.cos(r.theta))

            # apply unicycle step
            r.step_unicycle(v=v, omega=omega, dt=phase_cfg.dt, world_cfg=world_cfg)
            r.pos_history_phase2.append(r.pose().copy())

        # ---- 4) Communication graph (for consensus) ----
        positions = np.stack([r.pose() for r in robots], axis=0)
        A = np.zeros((n_robots, n_robots), dtype=float)
        for i in range(n_robots):
            for j in range(i + 1, n_robots):
                d = np.linalg.norm(positions[i] - positions[j])
                if d <= robot_cfg.comm_range:
                    A[i, j] = 1.0
                    A[j, i] = 1.0
            A[i, i] = 1.0  # self-loop

        # ---- 5) Consensus inside victim teams (position + signal) ----
        for vi, v_site in enumerate(victims):
            team_indices = [
                idx for idx, r in enumerate(robots)
                if r.assigned_victim == vi
            ]
            m = len(team_indices)
            if m == 0:
                continue

            beliefs = np.stack(
                [robots[i].belief_victims[vi] for i in team_indices], axis=0
            )
            has_seen = np.array(
                [robots[i].has_seen_victims[vi] for i in team_indices],
                dtype=bool,
            )
            signals = np.array(
                [robots[i].victim_signal[vi] for i in team_indices],
                dtype=float,
            )

            new_beliefs = beliefs.copy()
            new_signals = signals.copy()

            for l in range(m):
                i_global = team_indices[l]
                neighbors_global = [
                    j for j in team_indices if A[i_global, j] > 0
                ]
                if not neighbors_global:
                    continue
                neighbor_local = [team_indices.index(jg) for jg in neighbors_global]

                # position consensus
                neighbor_mean_pos = np.mean(beliefs[neighbor_local], axis=0)
                new_pos = beliefs[l] + phase_cfg.alpha * (
                    neighbor_mean_pos - beliefs[l]
                )
                if has_seen[l]:
                    new_pos = (
                        (1.0 - phase_cfg.anchor_gain_true) * new_pos
                        + phase_cfg.anchor_gain_true * v_site.pos
                    )
                new_beliefs[l] = new_pos

                # signal-strength consensus
                neighbor_mean_sig = np.mean(signals[neighbor_local])
                new_sig = signals[l] + phase_cfg.signal_consensus_gain * (
                    neighbor_mean_sig - signals[l]
                )
                if has_seen[l]:
                    new_sig = (
                        (1.0 - phase_cfg.anchor_gain_signal) * new_sig
                        + phase_cfg.anchor_gain_signal * phase_cfg.victim_signal_mean
                    )
                new_signals[l] = new_sig

            for l, i_global in enumerate(team_indices):
                robots[i_global].belief_victims[vi, :] = new_beliefs[l]
                robots[i_global].victim_signal[vi] = new_signals[l]

        # ---- 6) Consensus inside FP teams (signal only) + leave if weak ----
        for fi, fp_site in enumerate(false_sites):
            if fp_resolved[fi]:
                continue

            team_indices = [
                idx for idx, r in enumerate(robots)
                if r.assigned_fp == fi
            ]
            m = len(team_indices)
            if m == 0:
                continue

            signals = np.array(
                [robots[i].fp_signal[fi] for i in team_indices],
                dtype=float,
            )
            has_seen = np.array(
                [robots[i].has_seen_fp[fi] for i in team_indices],
                dtype=bool,
            )

            new_signals = signals.copy()

            for l in range(m):
                i_global = team_indices[l]
                neighbors_global = [
                    j for j in team_indices if A[i_global, j] > 0
                ]
                if not neighbors_global:
                    continue
                neighbor_local = [team_indices.index(jg) for jg in neighbors_global]

                neighbor_mean_sig = np.mean(signals[neighbor_local])
                new_sig = signals[l] + phase_cfg.signal_consensus_gain * (
                    neighbor_mean_sig - signals[l]
                )
                if has_seen[l]:
                    new_sig = (
                        (1.0 - phase_cfg.anchor_gain_signal) * new_sig
                        + phase_cfg.anchor_gain_signal * phase_cfg.fp_signal_mean
                    )
                new_signals[l] = new_sig

            for l, i_global in enumerate(team_indices):
                robots[i_global].fp_signal[fi] = new_signals[l]

            # classification: if mean signal below threshold, FP confirmed -> release team
            mean_sig = float(np.mean(new_signals))
            if mean_sig < phase_cfg.signal_threshold:
                fp_resolved[fi] = True
                logs.append(
                    dict(
                        step=k,
                        time=t,
                        event="fp_resolved",
                        fp_index=fi,
                        robots=";".join(f"R{robots[i].id}" for i in team_indices),
                        mean_signal=mean_sig,
                    )
                )
                for i_global in team_indices:
                    robots[i_global].assigned_fp = None
                    robots[i_global].slot_angle = None

        # ---- 7) Check victim consensus (position) ----
        if k >= phase_cfg.min_steps_before_check:
            all_good = True
            for vi in range(n_victims):
                team_indices = [
                    idx for idx, r in enumerate(robots)
                    if r.assigned_victim == vi
                ]
                if len(team_indices) < 2:
                    all_good = False
                    break

                beliefs = np.stack(
                    [robots[i].belief_victims[vi] for i in team_indices],
                    axis=0,
                )
                std_xy = np.std(beliefs, axis=0)
                if np.any(std_xy > phase_cfg.pos_tol):
                    all_good = False
                    break

            if all_good:
                consensus_reached = True
                steps_run = k + 1
                logs.append(
                    dict(
                        step=k,
                        time=t,
                        event="victim_consensus_reached",
                    )
                )
                break

        # ---- 8) advance time + safety cap ----
        k += 1
        steps_run = k

        if k >= max_steps_cap and not consensus_reached:
            print(
                f"[phase2] WARNING: safety cap reached without consensus "
                f"({k} steps, {k * phase_cfg.dt:.1f} s)."
            )
            break

    # after loop
    if consensus_reached:
        print(
            f"[phase2] Reached consensus in {steps_run} steps "
            f"({steps_run * phase_cfg.dt:.1f} s)."
        )
    else:
        print(
            f"[phase2] Finished without consensus after {steps_run} steps "
            f"({steps_run * phase_cfg.dt:.1f} s)."
        )

    # ---- 9) Final consensus positions (victims only) ----
    consensus_positions = np.zeros((n_victims, 2), dtype=float)
    for vi in range(n_victims):
        team_indices = [
            idx for idx, r in enumerate(robots)
            if r.assigned_victim == vi
        ]
        if team_indices:
            beliefs = np.stack(
                [robots[i].belief_victims[vi] for i in team_indices], axis=0
            )
        else:
            beliefs = np.stack(
                [r.belief_victims[vi] for r in robots], axis=0
            )
        consensus_positions[vi, :] = np.mean(beliefs, axis=0)

    # ---- 10) Save log file ----
    if len(logs) > 0:
        df_log = pd.DataFrame(logs)
        df_log.to_csv("phase2_log.csv", index=False)
        print("[phase2] Saved log to phase2_log.csv")

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
