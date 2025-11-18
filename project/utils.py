# utils.py
"""
Phase-2 analysis plots for the MRADR project.

This module generates the specific figures you listed for the RESULTS:
- (E2) Inter-site distance statistics
- (E3) Robot initial distribution
- (N1) Swarm centroid trajectory
- (N2) Coverage heatmap of the environment
- (N4) Robot–robot distance distribution
- (G1) Degree distribution and connectivity over time
- (G2) Algebraic connectivity lambda_2(L) over time
- (F1) Robots assigned to each victim/FP over time (step-like, from logs)
- (F2) Team formation delay (detection time)
- (C1) Belief mean error (final, per victim)
- (S1) Signal convergence for victim teams (final mean per team)
- (S4) Misallocation vs correct allocation ratio (final)
- Combined potential field heatmap of victims and FPs

Call run_phase2_analysis(...) once after Phase 2 finishes.
It will create all plots in a folder (default: "phase_2_analysis_plots").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from environment import WorldConfig, Site
from robot import Robot, RobotConfig
from phase2_navigation import Phase2Config, Phase2Result
from mpl_toolkits.mplot3d import Axes3D  # <- add this line


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _stack_positions_phase2(robots: List[Robot]) -> np.ndarray:
    """Stack Robot.pos_history_phase2 into (T, N, 2).

    T = max length of histories
    N = number of robots
    """
    if not robots:
        raise ValueError("No robots provided")

    n_robots = len(robots)
    max_len = max(len(r.pos_history_phase2) for r in robots)
    if max_len == 0:
        raise ValueError("pos_history_phase2 is empty. Did you run Phase 2?")

    arr = np.full((max_len, n_robots, 2), np.nan, dtype=float)
    for j, r in enumerate(robots):
        for k, p in enumerate(r.pos_history_phase2):
            arr[k, j, 0] = p[0]
            arr[k, j, 1] = p[1]
    return arr


def _time_vector(T: int, dt: float) -> np.ndarray:
    return np.arange(T, dtype=float) * dt


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    """Compute pairwise distances for a set of points (M, 2)."""
    m = points.shape[0]
    if m < 2:
        return np.array([], dtype=float)
    dists = []
    for i in range(m):
        for j in range(i + 1, m):
            dists.append(float(np.linalg.norm(points[i] - points[j])))
    return np.array(dists, dtype=float)


def _build_comm_graph(pts: np.ndarray, comm_range: float) -> np.ndarray:
    """Return adjacency matrix A for an undirected graph.

    pts: (M, 2) array of positions (NaNs removed beforehand).
    """
    m = pts.shape[0]
    A = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(i + 1, m):
            if np.linalg.norm(pts[i] - pts[j]) <= comm_range:
                A[i, j] = A[j, i] = 1.0
    return A


def _num_components(adj: np.ndarray) -> int:
    """Number of connected components in an undirected graph."""
    m = adj.shape[0]
    if m == 0:
        return 0
    seen = np.zeros(m, dtype=bool)
    n_comp = 0
    for i in range(m):
        if seen[i]:
            continue
        n_comp += 1
        stack = [i]
        seen[i] = True
        while stack:
            u = stack.pop()
            for v in range(m):
                if not seen[v] and adj[u, v] > 0:
                    seen[v] = True
                    stack.append(v)
    return n_comp


def _algebraic_connectivity(adj: np.ndarray) -> float:
    """Second-smallest eigenvalue of the graph Laplacian (lambda_2)."""
    m = adj.shape[0]
    if m <= 1:
        return 0.0
    deg = np.diag(adj.sum(axis=1))
    L = deg - adj
    vals = np.linalg.eigvalsh(L)
    vals_sorted = np.sort(vals)
    if len(vals_sorted) < 2:
        return 0.0
    return float(vals_sorted[1])


# ---------------------------------------------------------------------------
# (E2) Inter-site distance statistics
# ---------------------------------------------------------------------------

def plot_inter_site_distances(
    victims: List[Site],
    false_sites: List[Site],
    out_dir: str,
    fname: str = "E2_inter_site_distances.png",
) -> None:
    out_dir = _ensure_dir(out_dir)

    v_pts = np.array([v.pos for v in victims], dtype=float) if victims else np.zeros((0, 2))
    f_pts = np.array([f.pos for f in false_sites], dtype=float) if false_sites else np.zeros((0, 2))

    d_vv = _pairwise_distances(v_pts)
    d_ff = _pairwise_distances(f_pts)

    # victim–FP distances (all pairs)
    if v_pts.size > 0 and f_pts.size > 0:
        d_vf = []
        for pv in v_pts:
            for pf in f_pts:
                d_vf.append(float(np.linalg.norm(pv - pf)))
        d_vf = np.array(d_vf, dtype=float)
    else:
        d_vf = np.array([], dtype=float)

    # Build padded 2D array so matplotlib doesn't complain about ragged data
    data_arrays = []
    labels = []

    if d_vv.size:
        data_arrays.append(d_vv)
        labels.append("victim–victim")
    if d_ff.size:
        data_arrays.append(d_ff)
        labels.append("FP–FP")
    if d_vf.size:
        data_arrays.append(d_vf)
        labels.append("victim–FP")

    if not data_arrays:
        print("[analysis] No sites to plot inter-site distances.")
        return

    max_len = max(len(a) for a in data_arrays)
    k = len(data_arrays)
    padded = np.full((max_len, k), np.nan, dtype=float)
    for i, arr in enumerate(data_arrays):
        padded[: len(arr), i] = arr

    fig, ax = plt.subplots()
    ax.boxplot(padded, labels=labels)
    ax.set_ylabel("distance")
    ax.set_title("(E2) Inter-site distance statistics")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# (E3) Robot initial distribution (Phase 2)
# ---------------------------------------------------------------------------

def plot_initial_robot_distribution(
    positions: np.ndarray,
    out_dir: str,
    fname: str = "E3_initial_robot_distribution.png",
) -> None:
    """Histogram of initial y-positions at the start of Phase 2."""
    out_dir = _ensure_dir(out_dir)
    if positions.shape[0] == 0:
        print("[analysis] No positions for initial distribution.")
        return

    y0 = positions[0, :, 1]
    y0 = y0[~np.isnan(y0)]

    fig, ax = plt.subplots()
    ax.hist(y0, bins=20)
    ax.set_xlabel("initial y")
    ax.set_ylabel("count")
    ax.set_title("(E3) Robot initial y-distribution (start of Phase 2)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# (N1) Swarm centroid trajectory  &  (N2) coverage heatmap
# ---------------------------------------------------------------------------

def plot_swarm_centroid_trajectory(
    world_cfg: WorldConfig,
    positions: np.ndarray,
    phase_cfg: Phase2Config,
    out_dir: str,
    fname: str = "N1_swarm_centroid_trajectory.png",
) -> None:
    out_dir = _ensure_dir(out_dir)
    T = positions.shape[0]
    centroid = np.nanmean(positions, axis=1)  # (T, 2)

    fig, ax = plt.subplots()
    ax.set_xlim(0, world_cfg.width)
    ax.set_ylim(0, world_cfg.height)
    ax.set_aspect("equal")

    ax.plot(centroid[:, 0], centroid[:, 1], "-")
    ax.scatter(centroid[0, 0], centroid[0, 1], marker="o", label="start")
    ax.scatter(centroid[-1, 0], centroid[-1, 1], marker="s", label="end")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("(N1) Swarm centroid trajectory (Phase 2)")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=200)
    plt.close(fig)


def plot_coverage_heatmap(
    world_cfg: WorldConfig,
    positions: np.ndarray,
    out_dir: str,
    fname: str = "N2_coverage_heatmap.png",
    n_bins: int = 40,
) -> None:
    out_dir = _ensure_dir(out_dir)
    xs = positions[..., 0].ravel()
    ys = positions[..., 1].ravel()
    mask = ~np.isnan(xs) & ~np.isnan(ys)
    xs = xs[mask]
    ys = ys[mask]

    if xs.size == 0:
        print("[analysis] No positions for coverage heatmap.")
        return

    H, xedges, yedges = np.histogram2d(
        xs, ys,
        bins=n_bins,
        range=[[0.0, world_cfg.width], [0.0, world_cfg.height]],
    )

    fig, ax = plt.subplots()
    im = ax.imshow(
        H.T,
        origin="lower",
        extent=[0.0, world_cfg.width, 0.0, world_cfg.height],
        aspect="equal",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("(N2) Coverage heatmap (Phase 2)")
    fig.colorbar(im, ax=ax, label="visit count")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# (N4) Robot–robot distance distribution (final snapshot)
# ---------------------------------------------------------------------------

def plot_robot_distance_distribution(
    positions: np.ndarray,
    out_dir: str,
    fname: str = "N4_robot_robot_distance_distribution.png",
) -> None:
    out_dir = _ensure_dir(out_dir)
    if positions.shape[0] == 0:
        print("[analysis] No positions for robot distance distribution.")
        return

    final = positions[-1]   # (N, 2)
    mask = ~np.isnan(final[:, 0]) & ~np.isnan(final[:, 1])
    pts = final[mask]

    dists = _pairwise_distances(pts)
    if dists.size == 0:
        print("[analysis] Not enough robots for pairwise distances.")
        return

    fig, ax = plt.subplots()
    ax.hist(dists, bins=30)
    ax.set_xlabel("distance")
    ax.set_ylabel("count")
    ax.set_title("(N4) Robot–robot distance distribution (final)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# (G1) Degree distribution & connectivity  +  (G2) lambda_2(L)
# ---------------------------------------------------------------------------

def plot_comm_graph_stats(
    positions: np.ndarray,
    robot_cfg: RobotConfig,
    phase_cfg: Phase2Config,
    out_dir: str,
    fname: str = "G1_G2_comm_graph_stats.png",
    step_stride: int = 10,
) -> None:
    """Plot average degree, #components, and lambda_2(L) over time."""
    out_dir = _ensure_dir(out_dir)

    T, N, _ = positions.shape
    steps = np.arange(0, T, step_stride, dtype=int)

    avg_deg = []
    min_deg = []
    n_comp = []
    lam2 = []

    for k in steps:
        pts = positions[k]
        mask = ~np.isnan(pts[:, 0]) & ~np.isnan(pts[:, 1])
        pts = pts[mask]
        if pts.shape[0] == 0:
            avg_deg.append(np.nan)
            min_deg.append(np.nan)
            n_comp.append(np.nan)
            lam2.append(np.nan)
            continue

        A = _build_comm_graph(pts, robot_cfg.comm_range)
        deg = A.sum(axis=1)
        avg_deg.append(float(deg.mean()))
        min_deg.append(float(deg.min()) if deg.size else np.nan)
        n_comp.append(float(_num_components(A)))
        lam2.append(_algebraic_connectivity(A))

    t_vec = _time_vector(len(steps), phase_cfg.dt * step_stride)

    fig, ax1 = plt.subplots()
    ax1.plot(t_vec, avg_deg, label="avg degree")
    ax1.plot(t_vec, min_deg, label="min degree", linestyle="--")
    ax1.set_xlabel("time")
    ax1.set_ylabel("degree")

    ax2 = ax1.twinx()
    ax2.plot(t_vec, n_comp, label="#components", linestyle=":")
    ax2.plot(t_vec, lam2, label="lambda_2(L)", linestyle="-.")
    ax2.set_ylabel("connectivity metrics")

    ax1.set_title("(G1,G2) Communication graph stats")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# (F1) Robots assigned to each victim/FP over time
# (F2) Team formation delay
# ---------------------------------------------------------------------------

def _load_phase2_log(log_csv: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(log_csv):
        print(f"[analysis] No {log_csv} found; skipping log-based plots.")
        return None
    try:
        return pd.read_csv(log_csv)
    except Exception as e:
        print(f"[analysis] Could not read {log_csv}: {e}")
        return None


def plot_team_sizes_over_time(
    df_log: pd.DataFrame,
    phase_cfg: Phase2Config,
    out_dir: str,
    robots_per_victim: int,
    robots_per_fp: int,
    n_victims: int,
    n_fps: int,
    fname: str = "F1_team_sizes_over_time.png",
) -> None:
    """Use detection + resolution events to build step-like team sizes.

    Approximation:
    - victim team: 0 until first detect_victim event for that index,
      then robots_per_victim robots afterwards.
    - FP team: 0 until first detect_fp, then robots_per_fp until fp_resolved.
    """
    out_dir = _ensure_dir(out_dir)

    # Collect relevant times
    times = set([0.0])
    for _, row in df_log.iterrows():
        if row.get("event") in ["detect_victim", "detect_fp", "fp_resolved"]:
            times.add(float(row.get("time", 0.0)))
    if len(times) == 1:
        print("[analysis] Log has no detect/resolve events for teams.")
        return

    times = sorted(times)
    # detection / resolution lookups
    det_v = {vi: None for vi in range(n_victims)}
    det_f = {fi: None for fi in range(n_fps)}
    res_f = {fi: None for fi in range(n_fps)}

    for _, row in df_log.iterrows():
        ev = row.get("event")
        t = float(row.get("time", 0.0))
        if ev == "detect_victim":
            vi = int(row.get("victim_index", -1))
            if 0 <= vi < n_victims and det_v[vi] is None:
                det_v[vi] = t
        elif ev == "detect_fp":
            fi = int(row.get("fp_index", -1))
            if 0 <= fi < n_fps and det_f[fi] is None:
                det_f[fi] = t
        elif ev == "fp_resolved":
            fi = int(row.get("fp_index", -1))
            if 0 <= fi < n_fps and res_f[fi] is None:
                res_f[fi] = t

    team_v = []
    team_f = []

    for t in times:
        c_v = 0
        c_f = 0
        for vi in range(n_victims):
            if det_v[vi] is not None and t >= det_v[vi]:
                c_v += robots_per_victim
        for fi in range(n_fps):
            if det_f[fi] is not None and t >= det_f[fi]:
                if res_f[fi] is None or t < res_f[fi]:
                    c_f += robots_per_fp
        team_v.append(c_v)
        team_f.append(c_f)

    fig, ax = plt.subplots()
    ax.step(times, team_v, where="post", label="victim teams")
    ax.step(times, team_f, where="post", label="FP teams")
    ax.set_xlabel("time")
    ax.set_ylabel("#robots in teams")
    ax.set_title("(F1) Robots assigned to victim/FP teams over time (approx)")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=200)
    plt.close(fig)


def plot_team_formation_delay(
    df_log: pd.DataFrame,
    out_dir: str,
    n_victims: int,
    n_fps: int,
    fname: str = "F2_team_formation_delay.png",
) -> None:
    """Bar chart of detection times per victim and FP.

    Since team formation happens immediately after detection in the code,
    we use detection time as formation time.
    """
    out_dir = _ensure_dir(out_dir)

    det_v = {vi: None for vi in range(n_victims)}
    det_f = {fi: None for fi in range(n_fps)}

    for _, row in df_log.iterrows():
        ev = row.get("event")
        t = float(row.get("time", 0.0))
        if ev == "detect_victim":
            vi = int(row.get("victim_index", -1))
            if 0 <= vi < n_victims and det_v[vi] is None:
                det_v[vi] = t
        elif ev == "detect_fp":
            fi = int(row.get("fp_index", -1))
            if 0 <= fi < n_fps and det_f[fi] is None:
                det_f[fi] = t

    labels = []
    times_v = []
    times_f = []

    for vi in range(n_victims):
        if det_v[vi] is not None:
            labels.append(f"Victim {vi+1}")
            times_v.append(det_v[vi])
    for fi in range(n_fps):
        if det_f[fi] is not None:
            labels.append(f"FP {fi+1}")
            times_f.append(det_f[fi])

    if not labels:
        print("[analysis] No detections for formation delay plot.")
        return

    x_v = np.arange(len(times_v))
    x_f = np.arange(len(times_f)) + len(times_v)

    fig, ax = plt.subplots()
    if times_v:
        ax.bar(x_v, times_v, label="victims")
    if times_f:
        ax.bar(x_f, times_f, label="FPs")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("time to first detection / team formation")
    ax.set_title("(F2) Team formation delay (approx)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# (C1) Belief mean error (final)  &  (S1) signal convergence (final)
# (S4) misallocation ratio
# ---------------------------------------------------------------------------

def plot_belief_mean_error_final(
    victims: List[Site],
    robots: List[Robot],
    out_dir: str,
    fname: str = "C1_belief_mean_error_final.png",
) -> None:
    """Final mean position error for each victim team.

    For each victim:
    - take robots whose assigned_victim == that index
    - average their belief_victims[vi]
    - compute ||mean - true||.
    """
    out_dir = _ensure_dir(out_dir)

    errors = []
    labels = []

    for vi, v in enumerate(victims):
        team_beliefs = []
        for r in robots:
            if getattr(r, "assigned_victim", None) == vi:
                if hasattr(r, "belief_victims") and r.belief_victims.shape[0] > vi:
                    team_beliefs.append(r.belief_victims[vi])
        if not team_beliefs:
            continue
        team_beliefs = np.vstack(team_beliefs)
        mean_belief = team_beliefs.mean(axis=0)
        err = float(np.linalg.norm(mean_belief - v.pos))
        errors.append(err)
        labels.append(f"Victim {vi+1}")

    if not errors:
        print("[analysis] No victim teams for belief error plot.")
        return

    fig, ax = plt.subplots()
    ax.bar(np.arange(len(errors)), errors)
    ax.set_xticks(np.arange(len(errors)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("||mean belief - true||")
    ax.set_title("(C1) Belief mean error (final per victim)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=200)
    plt.close(fig)


def plot_signal_convergence_final(
    victims: List[Site],
    robots: List[Robot],
    phase_cfg: Phase2Config,
    out_dir: str,
    fname: str = "S1_signal_convergence_final.png",
) -> None:
    """Final mean signal for each victim team vs model mean."""
    out_dir = _ensure_dir(out_dir)

    means = []
    labels = []

    for vi, v in enumerate(victims):
        team_sigs = []
        for r in robots:
            if getattr(r, "assigned_victim", None) == vi:
                if hasattr(r, "victim_signal") and len(r.victim_signal) > vi:
                    team_sigs.append(r.victim_signal[vi])
        if not team_sigs:
            continue
        team_sigs = np.array(team_sigs, dtype=float)
        means.append(float(team_sigs.mean()))
        labels.append(f"Victim {vi+1}")

    if not means:
        print("[analysis] No victim teams for signal convergence plot.")
        return

    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, label="final mean signal")
    ax.axhline(phase_cfg.victim_signal_mean, color="r", linestyle="--", label="model victim mean")
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("signal value")
    ax.set_title("(S1) Signal convergence for victim teams (final)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=200)
    plt.close(fig)


def plot_misallocation_ratio(
    robots: List[Robot],
    out_dir: str,
    fname: str = "S4_misallocation_ratio.png",
) -> None:
    """Plot final allocation of robots to victim teams, FP teams, or free.

    Misallocation ratio = robots in FP teams / (robots in victim + FP teams).
    """
    out_dir = _ensure_dir(out_dir)

    n_victim = 0
    n_fp = 0
    n_free = 0

    for r in robots:
        av = getattr(r, "assigned_victim", None)
        af = getattr(r, "assigned_fp", None)
        if av is not None:
            n_victim += 1
        elif af is not None:
            n_fp += 1
        else:
            n_free += 1

    total_team = n_victim + n_fp
    if total_team > 0:
        mis_ratio = n_fp / total_team
    else:
        mis_ratio = 0.0

    fig, ax = plt.subplots()
    ax.bar(["victim teams", "FP teams", "free"], [n_victim, n_fp, n_free])
    ax.set_ylabel("#robots")
    ax.set_title(f"(S4) Misallocation ratio = {mis_ratio:.2f}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Combined potential field heatmap of victims and FPs
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Combined potential field: victims = tall donut hills, FPs = shorter ones
# ---------------------------------------------------------------------------

def _victim_local_potential(
    pos: np.ndarray,
    center: np.ndarray,
    phase_cfg: Phase2Config,
) -> float:
    """
    Donut-shaped radial potential for a victim.

    V(r) = tall Gaussian bump - sharp central Gaussian
    -> looks like a hill with a hollow top (repulsive core).
    """
    diff = pos - center
    r = float(np.linalg.norm(diff))

    # Outer "attractive" hill
    sigma_outer = 1.0 * phase_cfg.influence_radius
    k_outer = phase_cfg.k_victim_local

    # Inner repulsive core (sharper and stronger near r = 0)
    sigma_inner = 0.4 * phase_cfg.safe_radius
    k_inner = 1.5 * phase_cfg.k_victim_local

    if sigma_outer <= 0 or sigma_inner <= 0:
        return 0.0

    V_outer = k_outer * np.exp(-(r ** 2) / (2.0 * sigma_outer ** 2))
    V_inner = k_inner * np.exp(-(r ** 2) / (2.0 * sigma_inner ** 2))

    # Hill with hollow center: big outer bump minus sharp inner bump
    V = V_outer - V_inner

    # Shift up so everything is non-negative (only for nicer plotting)
    return V + k_inner


def _fp_local_potential(
    pos: np.ndarray,
    center: np.ndarray,
    phase_cfg: Phase2Config,
) -> float:
    """
    Donut potential for a false positive.

    Same shape as victim, but with lower amplitude,
    so FP hills are shorter.
    """
    diff = pos - center
    r = float(np.linalg.norm(diff))

    # Outer hill: weaker than victims
    sigma_outer = 0.7 * phase_cfg.influence_radius
    k_outer = 0.5 * phase_cfg.k_victim_local  # shorter hill

    # Inner repulsive core: also weaker
    sigma_inner = 0.4 * phase_cfg.safe_radius
    k_inner = 0.75 * phase_cfg.k_victim_local

    if sigma_outer <= 0 or sigma_inner <= 0:
        return 0.0

    V_outer = k_outer * np.exp(-(r ** 2) / (2.0 * sigma_outer ** 2))
    V_inner = k_inner * np.exp(-(r ** 2) / (2.0 * sigma_inner ** 2))

    V = V_outer - V_inner
    return V + k_inner



def plot_combined_potential_field(
    world_cfg: WorldConfig,
    victims: List[Site],
    false_sites: List[Site],
    phase_cfg: Phase2Config,
    out_dir: str,
    fname: str = "combined_potential_field.png",
    grid_size: int = 80,
) -> None:
    """
    3D surface of combined potential from all victims and FPs.

    - Victims: tall donut hills (attractive with repulsive hollow center)
    - FPs: shorter donut hills
    """
    out_dir = _ensure_dir(out_dir)

    xs = np.linspace(0.0, world_cfg.width, grid_size)
    ys = np.linspace(0.0, world_cfg.height, grid_size)
    X, Y = np.meshgrid(xs, ys)

    V = np.zeros_like(X)

    for i in range(grid_size):
        for j in range(grid_size):
            p = np.array([X[i, j], Y[i, j]], dtype=float)
            v_total = 0.0
            for v in victims:
                v_total += _victim_local_potential(p, v.pos, phase_cfg)
            for fp in false_sites:
                v_total += _fp_local_potential(p, fp.pos, phase_cfg)
            V[i, j] = v_total

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X,
        Y,
        V,
        rstride=1,
        cstride=1,
        linewidth=0.1,
        antialiased=True,
    )

    # Mark victims and FPs on the surface
    if victims:
        vx = [v.x for v in victims]
        vy = [v.y for v in victims]
        vz = []
        for v in victims:
            vz.append(_victim_local_potential(np.array(v.pos), np.array(v.pos), phase_cfg))
        ax.scatter(vx, vy, vz, marker="*", s=60, label="victims")

    if false_sites:
        fx = [f.x for f in false_sites]
        fy = [f.y for f in false_sites]
        fz = []
        for fp in false_sites:
            fz.append(_fp_local_potential(np.array(fp.pos), np.array(fp.pos), phase_cfg))
        ax.scatter(fx, fy, fz, marker="x", s=40, label="false pos")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("V(x,y)")
    ax.set_title("Combined potential field (victims = tall donut hills, FPs = shorter)")

    # Put legend slightly above the surface
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=220)
    plt.close(fig)



# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def run_phase2_analysis(
    world_cfg: WorldConfig,
    robot_cfg: RobotConfig,
    phase_cfg: Phase2Config,
    victims: List[Site],
    false_sites: List[Site],
    robots: List[Robot],
    phase2_result: Phase2Result,
    out_dir: str = "phase_2_analysis_plots",
    log_csv: str = "phase2_log.csv",
) -> None:
    """Generate all Phase-2 analysis plots you requested."""
    out_dir = _ensure_dir(out_dir)

    # Positions over time
    try:
        positions = _stack_positions_phase2(robots)
    except ValueError as e:
        print(f"[analysis] {e}")
        return

    n_victims = len(victims)
    n_fps = len(false_sites)

    # ---- Geometry / environment ----
    plot_inter_site_distances(victims, false_sites, out_dir)
    plot_initial_robot_distribution(positions, out_dir)

    # ---- Navigation & coverage ----
    plot_swarm_centroid_trajectory(world_cfg, positions, phase_cfg, out_dir)
    plot_coverage_heatmap(world_cfg, positions, out_dir)
    plot_robot_distance_distribution(positions, out_dir)

    # ---- Graph connectivity ----
    plot_comm_graph_stats(positions, robot_cfg, phase_cfg, out_dir)

    # ---- Team formation (log-based) ----
    df_log = _load_phase2_log(log_csv)
    if df_log is not None:
        plot_team_sizes_over_time(
            df_log,
            phase_cfg,
            out_dir,
            robots_per_victim=phase_cfg.robots_per_victim,
            robots_per_fp=phase_cfg.robots_per_fp,
            n_victims=n_victims,
            n_fps=n_fps,
        )
        plot_team_formation_delay(df_log, out_dir, n_victims, n_fps)

    # ---- Belief / signal and allocation (final) ----
    plot_belief_mean_error_final(victims, robots, out_dir)
    plot_signal_convergence_final(victims, robots, phase_cfg, out_dir)
    plot_misallocation_ratio(robots, out_dir)

    # ---- Combined potential field ----
    plot_combined_potential_field(world_cfg, victims, false_sites, phase_cfg, out_dir)
