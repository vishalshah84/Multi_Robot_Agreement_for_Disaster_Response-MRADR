# animation.py

from __future__ import annotations
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from environment import WorldConfig, Site
from robot import Robot, RobotConfig


def _draw_static_world(ax, world_cfg: WorldConfig,
                       victims: List[Site],
                       false_sites: List[Site]):
    ax.set_xlim(0, world_cfg.width)
    ax.set_ylim(0, world_cfg.height)
    ax.set_aspect("equal")

    # victims
    for v in victims:
        ax.plot(v.x, v.y, marker="*", color="red", markersize=12)
        ax.text(v.x + 0.1, v.y + 0.1, f"Victim {v.id+1}", color="red", fontsize=8)

    # false positives
    for fs in false_sites:
        ax.scatter(fs.x, fs.y, c="blue", s=40)
        ax.text(fs.x + 0.1, fs.y + 0.1, f"FP {fs.id+1}", color="blue", fontsize=7)

    ax.set_xticks(range(0, int(world_cfg.width) + 1))
    ax.set_yticks(range(0, int(world_cfg.height) + 1))
    ax.grid(True, linestyle="--", alpha=0.3)


def make_phase1_video_from_logs(
    world_cfg: WorldConfig,
    victims: List[Site],
    false_sites: List[Site],
    robots: List[Robot],
    dt: float,
    steps: int,
    out_file: str = "phase1_random_walk.mp4",
) -> None:
    paths = []
    for r in robots:
        arr = np.stack(r.pos_history_phase1[:steps], axis=0)
        paths.append(arr)

    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_static_world(ax, world_cfg, victims, false_sites)

    title_text = ax.set_title("Phase 1: Random Walk + Consensus", fontsize=10)

    robot_artists = []
    robot_labels = []
    for i, r in enumerate(robots):
        start = paths[i][0]
        point = ax.scatter(start[0], start[1], c="green", s=30, zorder=3)
        label = ax.text(
            start[0] + 0.1, start[1] + 0.1, f"R{r.id+1}",
            color="green", fontsize=7
        )
        robot_artists.append(point)
        robot_labels.append(label)

    def step(frame_idx):
        t = frame_idx * dt
        for i, (artist, label) in enumerate(zip(robot_artists, robot_labels)):
            pos = paths[i][frame_idx]
            artist.set_offsets([pos[0], pos[1]])
            label.set_position((pos[0] + 0.1, pos[1] + 0.1))
        title_text.set_text(f"Phase 1: Random Walk + Consensus — t = {t:.1f} s")
        return robot_artists + robot_labels + [title_text]

    anim = animation.FuncAnimation(
        fig,
        step,
        frames=steps,
        interval=dt * 1000,
        blit=False,
    )

    try:
        writer = animation.FFMpegWriter(fps=int(1 / dt))
        anim.save(out_file, writer=writer)
        print(f"[animation] Saved Phase-1 MP4 to {out_file}")
    except Exception as e:
        print("[animation] Could not save MP4, falling back to GIF.")
        print("  Error:", e)
        gif_name = out_file.replace(".mp4", ".gif")
        anim.save(gif_name, writer="pillow", fps=int(1 / dt))
        print(f"[animation] Saved Phase-1 GIF to {gif_name}")

    plt.close(fig)


def make_phase2_video_from_logs(
    world_cfg: WorldConfig,
    victims: List[Site],
    false_sites: List[Site],
    robots: List[Robot],
    dt: float,
    steps: int,
    out_file: str = "phase2_navigation.mp4",
) -> None:
    paths = []
    for r in robots:
        arr = np.stack(r.pos_history_phase2[:steps], axis=0)
        paths.append(arr)

    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_static_world(ax, world_cfg, victims, false_sites)

    title_text = ax.set_title("Phase 2: Navigation + Consensus", fontsize=10)

    robot_artists = []
    robot_labels = []
    for i, r in enumerate(robots):
        start = paths[i][0]
        point = ax.scatter(start[0], start[1], c="green", s=30, zorder=3)
        label = ax.text(
            start[0] + 0.1, start[1] + 0.1, f"R{r.id+1}",
            color="green", fontsize=7
        )
        robot_artists.append(point)
        robot_labels.append(label)

    def step(frame_idx):
        t = frame_idx * dt
        for i, (artist, label) in enumerate(zip(robot_artists, robot_labels)):
            pos = paths[i][frame_idx]
            artist.set_offsets([pos[0], pos[1]])
            label.set_position((pos[0] + 0.1, pos[1] + 0.1))
        title_text.set_text(f"Phase 2: Navigation + Consensus — t = {t:.1f} s")
        return robot_artists + robot_labels + [title_text]

    anim = animation.FuncAnimation(
        fig,
        step,
        frames=steps,
        interval=dt * 1000,
        blit=False,
    )

    try:
        writer = animation.FFMpegWriter(fps=int(1 / dt))
        anim.save(out_file, writer=writer)
        print(f"[animation] Saved Phase-2 MP4 to {out_file}")
    except Exception as e:
        print("[animation] Could not save MP4, falling back to GIF.")
        print("  Error:", e)
        gif_name = out_file.replace(".mp4", ".gif")
        anim.save(gif_name, writer="pillow", fps=int(1 / dt))
        print(f"[animation] Saved Phase-2 GIF to {gif_name}")

    plt.close(fig)


def make_combined_video_from_logs(
    world_cfg: WorldConfig,
    victims: List[Site],
    false_sites: List[Site],
    robots: List[Robot],
    dt1: float,
    steps1: int,
    dt2: float,
    steps2: int,
    out_file: str = "combined_phase1_phase2.mp4",
) -> None:
    """
    Combined video: Phase 1 followed by Phase 2.
    """
    paths1 = []
    paths2 = []
    for r in robots:
        p1 = np.stack(r.pos_history_phase1[:steps1], axis=0)
        p2 = np.stack(r.pos_history_phase2[:steps2], axis=0)
        paths1.append(p1)
        paths2.append(p2)

    total_frames = steps1 + steps2
    # use single dt (assume dt1 == dt2)
    dt = dt1

    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_static_world(ax, world_cfg, victims, false_sites)

    title_text = ax.set_title("Combined Phase 1 + Phase 2", fontsize=10)

    robot_artists = []
    robot_labels = []
    for i, r in enumerate(robots):
        start = paths1[i][0]
        point = ax.scatter(start[0], start[1], c="green", s=30, zorder=3)
        label = ax.text(
            start[0] + 0.1, start[1] + 0.1, f"R{r.id+1}",
            color="green", fontsize=7
        )
        robot_artists.append(point)
        robot_labels.append(label)

    def step(frame_idx):
        if frame_idx < steps1:
            phase = "Phase 1"
            t = frame_idx * dt
            for i, (artist, label) in enumerate(zip(robot_artists, robot_labels)):
                pos = paths1[i][frame_idx]
                artist.set_offsets([pos[0], pos[1]])
                label.set_position((pos[0] + 0.1, pos[1] + 0.1))
        else:
            phase = "Phase 2"
            j = frame_idx - steps1
            t = j * dt
            for i, (artist, label) in enumerate(zip(robot_artists, robot_labels)):
                pos = paths2[i][j]
                artist.set_offsets([pos[0], pos[1]])
                label.set_position((pos[0] + 0.1, pos[1] + 0.1))

        title_text.set_text(f"{phase} — t = {t:.1f} s")
        return robot_artists + robot_labels + [title_text]

    anim = animation.FuncAnimation(
        fig,
        step,
        frames=total_frames,
        interval=dt * 1000,
        blit=False,
    )

    try:
        writer = animation.FFMpegWriter(fps=int(1 / dt))
        anim.save(out_file, writer=writer)
        print(f"[animation] Saved combined MP4 to {out_file}")
    except Exception as e:
        print("[animation] Could not save MP4, falling back to GIF.")
        print("  Error:", e)
        gif_name = out_file.replace(".mp4", ".gif")
        anim.save(gif_name, writer="pillow", fps=int(1 / dt))
        print(f"[animation] Saved combined GIF to {gif_name}")

    plt.close(fig)
