import argparse
import copy

from environment import WorldConfig, build_environment
from robot import RobotConfig, init_robots
from phase1_markov import Phase1Config, run_phase1, save_phase1_results_to_excel
from phase2_navigation import Phase2Config, run_phase2, save_phase2_results_to_excel
from animation import (
    make_phase1_video_from_logs,
    make_phase2_video_from_logs,
    make_combined_video_from_logs,
)
from utils import run_phase2_analysis

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robots", type=int, default=20)
    parser.add_argument("--victims", type=int, default=2)
    parser.add_argument("--false", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--phase1_video", type=str, default="phase1_random_walk.mp4")
    parser.add_argument("--phase2_video", type=str, default="phase2_navigation.mp4")
    parser.add_argument(
        "--combined_video",
        type=str,
        default="combined_phase1_phase2.mp4",
    )

    parser.add_argument("--phase1_excel", type=str, default="phase1_results.xlsx")
    parser.add_argument("--phase2_excel", type=str, default="phase2_results.xlsx")

    parser.add_argument(
        "--phase2_mode",
        choices=["continue", "reset"],
        default="continue",
        help=(
            "continue: Phase 2 starts from end of Phase 1; "
            "reset: robots go back to initial positions before Phase 2."
        ),
    )

    args = parser.parse_args()

    # 1) world + sites
    world_cfg = WorldConfig(
        n_victims=args.victims,
        n_false_positives=args.false,
        rng_seed=args.seed,
    )
    victims, false_sites = build_environment(world_cfg)

    # 2) robots (initial positions)
    r_cfg = RobotConfig()
    robots = init_robots(args.robots, world_cfg, r_cfg, rng_seed=args.seed)
    initial_robots = [copy.deepcopy(r) for r in robots]

    # 3) Phase 1: random walk + consensus
    p1_cfg = Phase1Config()
    res1 = run_phase1(
        world_cfg=world_cfg,
        robot_cfg=r_cfg,
        phase_cfg=p1_cfg,
        victims=victims,
        false_sites=false_sites,
        robots=robots,
        rng_seed=args.seed,
    )

    print("[main] Phase 1 consensus reached:", res1.consensus_reached)
    print("[main] Phase 1 steps run:", res1.steps_run)

    save_phase1_results_to_excel(
        victims=victims,
        false_sites=false_sites,
        result=res1,
        filename=args.phase1_excel,
    )

    make_phase1_video_from_logs(
        world_cfg=world_cfg,
        victims=victims,
        false_sites=false_sites,
        robots=robots,
        dt=p1_cfg.dt,
        steps=res1.steps_run,
        out_file=args.phase1_video,
    )

    # 4) Phase 2: navigation + consensus refinement
    p2_cfg = Phase2Config()

    # choose how Phase 2 starts
    if args.phase2_mode == "continue":
        # start from end of Phase 1 (same objects)
        robots_phase2 = robots
    else:  # "reset"
        # restart robots from initial positions
        robots_phase2 = [copy.deepcopy(r) for r in initial_robots]

    res2 = run_phase2(
        world_cfg=world_cfg,
        robot_cfg=r_cfg,
        phase_cfg=p2_cfg,
        victims=victims,
        false_sites=false_sites,
        robots=robots_phase2,
        initial_estimates=res1.consensus_positions,
        rng_seed=args.seed,
    )

    print("[main] Phase 2 consensus reached:", res2.consensus_reached)
    print("[main] Phase 2 steps run:", res2.steps_run)

    save_phase2_results_to_excel(
        victims=victims,
        result=res2,
        filename=args.phase2_excel,
    )


    # ---- Phase-2 analysis plots (NEW) ----
    run_phase2_analysis(
        world_cfg=world_cfg,
        robot_cfg=r_cfg,
        phase_cfg=p2_cfg,
        victims=victims,
        false_sites=false_sites,
        robots=robots_phase2,      # or robots, depending on your mode
        phase2_result=res2,
        out_dir="phase_2_analysis_plots",
        log_csv="phase2_log.csv",
    )


    # If we used a separate robots_phase2 list (reset mode),
    # copy its phase-2 histories back into 'robots'
    if args.phase2_mode == "reset":
        for r_base, r2 in zip(robots, robots_phase2):
            r_base.pos_history_phase2 = list(r2.pos_history_phase2)

    # Phase-2 video uses the robots that actually ran Phase 2
    make_phase2_video_from_logs(
        world_cfg=world_cfg,
        victims=victims,
        false_sites=false_sites,
        robots=robots_phase2,
        dt=p2_cfg.dt,
        steps=res2.steps_run,
        out_file=args.phase2_video,
    )

    # 5) Combined video (robots now have phase1 and, in reset mode, copied phase2)
    make_combined_video_from_logs(
        world_cfg=world_cfg,
        victims=victims,
        false_sites=false_sites,
        robots=robots,
        dt1=p1_cfg.dt,
        steps1=res1.steps_run,
        dt2=p2_cfg.dt,
        steps2=res2.steps_run,
        out_file=args.combined_video,
    )


if __name__ == "__main__":
    main()
