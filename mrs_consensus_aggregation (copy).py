#!/usr/bin/env python3
"""
Multi-robot victim search, consensus, and aggregation demo.

- 4 differential-drive robots (unicycle model)
- 2D environment with:
    * 1 true victim: (CO2, Heat, Water)
    * Multiple false positives: any subset of {CO2, Heat, Water} except the triple
- Robots start from left edge, explore with random motion.
- When any robot senses all three signals, it "confirms" the victim.
- Robots share an estimate of victim position via distributed consensus.
- Once consensus converges, robots switch to an aggregation controller
  and move to the victim while avoiding collisions.
- The run is animated and saved as an MP4.
- Extra plots show consensus convergence and distance to victim.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass, field
import sys
import csv
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ===========================
# Parameters
# ===========================
WORLD_SIZE = 10.0  # world is [0, WORLD_SIZE] x [0, WORLD_SIZE]
N_ROBOTS = int(sys.argv[1]) if len(sys.argv) > 1 else 4

DT = 0.05          # time step [s]
N_STEPS = 800      # number of simulation steps

# Robot motion
V_SEARCH = 0.8     # linear speed during search
V_AGG = 1.0        # linear speed during aggregation
K_OMEGA = 2.5      # heading controller gain
K_REP = 1.5        # collision avoidance gain
ROBOT_RADIUS = 0.1
SAFE_DIST = 0.8    # desired min spacing during aggregation
SITE_SAFE_DIST = 0.3   # how close robots are allowed to get to any sign/victim
K_SITE_REP = 2.0       # strength of repulsion from sites
ROBOT_CLEAR_DIST = 2 * ROBOT_RADIUS * 1.1   # small safety margin
SIGN_MIN_DIST = 0.5   # min distance between any two signs/victim (in world units)



# Sensing and communication
SENSOR_RANGE = 1.2        # distance at which robot can sense a site
COMM_RANGE = 4.0          # robots closer than this can communicate
CONS_ALPHA = 0.2          # consensus step size
ANCHOR_GAIN = 0.5         # how strongly a robot that saw victim trusts its own measurement

# Consensus / mode switching
CONS_TOL = 0.1            # max std dev of estimates to declare consensus (per coordinate)
MIN_DETECT_STEPS = 50     # give some time for wandering before we expect detection

# Environment
N_FALSE_POS = int(sys.argv[2]) if len(sys.argv) > 2 else 6          # number of false positive sites

# Output
VIDEO_FILENAME = "mrs_victim_consensus.mp4"

# Flags / bookkeeping
consensus_reached = False     # becomes True once we switch to aggregation
found_robot_index = None      # index (0-based) of robot that first saw the victim


# ===========================
# Environment objects
# ===========================

@dataclass
class Site:
    """A location with some subset of {CO2, Heat, Water}."""
    x: float
    y: float
    co2: bool
    heat: bool
    water: bool
    is_victim: bool = False

    @property
    def vector(self):
        return np.array([self.x, self.y])


# All non-empty combinations except full triple will be false positives
FALSE_PATTERNS = [
    (True, False, False),   # CO2 only
    (False, True, False),   # Heat only
    (False, False, True),   # Water only
    (True, True, False),    # CO2+Heat
    (True, False, True),    # CO2+Water
    (False, True, True),    # Heat+Water
]

# Colors for each combination of (CO2, Heat, Water)
SIGN_COLORS = {
    (True,  False, False): "grey",        # CO2
    (False, True,  False): "red",         # Heat
    (False, False, True):  "blue",        # Water
    (True,  True,  False): "orange",      # Heat + CO2
    (False, True,  True):  "purple",      # Water + Heat
    (True,  False, True):  "deepskyblue"  # Water + CO2 (light blue)
}


def get_site_color(site):
    """Return the face color for a given sign site."""
    key = (site.co2, site.heat, site.water)
    return SIGN_COLORS.get(key, "black")



# ===========================
# Robot
# ===========================

@dataclass
class Robot:
    x: float
    y: float
    theta: float
    state: str = "search"       # "search" or "aggregate"
    has_seen_victim: bool = False

    belief: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    belief_history: list = field(default_factory=list)
    dist_to_victim_history: list = field(default_factory=list)

    def pose_vector(self):
        return np.array([self.x, self.y])

    def step_search(self, dt, sites):
        """Random exploration with wall avoidance (unicycle model)."""
        # small random turn
        omega = np.random.normal(0.0, 1.0)

        # Bounce off walls by turning away
        margin = 0.5
        if self.x < margin:
            omega += 2.0
        if self.x > WORLD_SIZE - margin:
            omega -= 2.0
        if self.y < margin:
            omega += 2.0 * np.sign(np.cos(self.theta))
        if self.y > WORLD_SIZE - margin:
            omega -= 2.0 * np.sign(np.cos(self.theta))

        v = V_SEARCH

        # Update unicycle
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += omega * dt

        # keep inside box
        self.x = np.clip(self.x, 0.0, WORLD_SIZE)
        self.y = np.clip(self.y, 0.0, WORLD_SIZE)
        # keep distance from signs and victim
        self.push_off_sites(sites)

        # Repulsion from signs/victim so robot doesn't sit on top of them
        for site in sites:
            dvec = self.pose_vector() - site.vector
            dist = np.linalg.norm(dvec)
            if 1e-3 < dist < SITE_SAFE_DIST:
                away_angle = np.arctan2(dvec[1], dvec[0])
                aerr = wrap_to_pi(away_angle - self.theta)
                omega += K_SITE_REP * (SITE_SAFE_DIST - dist) * np.sign(aerr)


    def step_aggregate(self, dt, target, neighbors, sites):
        """
        Move toward consensus target while avoiding nearby robots.
        neighbors: list of other Robot objects (for avoidance only).
        """
        # Desired heading to target
        dx, dy = target[0] - self.x, target[1] - self.y
        angle_target = np.arctan2(dy, dx)
        angle_err = wrap_to_pi(angle_target - self.theta)

        # base control
        omega = K_OMEGA * angle_err
        v = V_AGG * np.cos(angle_err)  # slow if heading error is large
        v = max(0.0, v)

        # simple repulsive term from close neighbors
        for nb in neighbors:
            dvec = self.pose_vector() - nb.pose_vector()
            dist = np.linalg.norm(dvec)
            if 1e-3 < dist < SAFE_DIST:
                away_angle = np.arctan2(dvec[1], dvec[0])
                aerr = wrap_to_pi(away_angle - self.theta)
                omega += K_REP * (SAFE_DIST - dist) * np.sign(aerr)

                # repulsion from signs and victim
        for site in sites:
            dvec = self.pose_vector() - site.vector
            dist = np.linalg.norm(dvec)
            if 1e-3 < dist < SITE_SAFE_DIST:
                away_angle = np.arctan2(dvec[1], dvec[0])
                aerr = wrap_to_pi(away_angle - self.theta)
                omega += K_SITE_REP * (SITE_SAFE_DIST - dist) * np.sign(aerr)


        # update pose
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += omega * dt

        # keep inside box
        self.x = np.clip(self.x, 0.0, WORLD_SIZE)
        self.y = np.clip(self.y, 0.0, WORLD_SIZE)
        # keep distance from signs and victim
        self.push_off_sites(sites)


    def push_off_sites(self, sites):
        """
        If the robot is too close to any site (false positive or victim),
        push it out to a safe distance so it does not overlap the icon.
        """
        min_clearance = SITE_SAFE_DIST + ROBOT_RADIUS

        for site in sites:
            dvec = self.pose_vector() - site.vector
            dist = np.linalg.norm(dvec)

            if dist < min_clearance:
                # If exactly on top, choose a random direction
                if dist < 1e-6:
                    angle = np.random.uniform(0, 2 * np.pi)
                    dvec = np.array([np.cos(angle), np.sin(angle)])
                    dist = 1.0

                # project robot to boundary of safe disk
                dunit = dvec / dist
                self.x = site.x + dunit[0] * min_clearance
                self.y = site.y + dunit[1] * min_clearance


# ===========================
# Helper functions
# ===========================

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def random_site(is_victim=False):
    """Sample a random site away from the boundaries."""
    margin = 1.0
    x = np.random.uniform(margin, WORLD_SIZE - margin)
    y = np.random.uniform(margin, WORLD_SIZE - margin)
    if is_victim:
        return Site(x, y, True, True, True, is_victim=True)
    else:
        co2, heat, water = FALSE_PATTERNS[np.random.randint(len(FALSE_PATTERNS))]
        return Site(x, y, co2, heat, water, is_victim=False)


def build_environment():
    victim = random_site(is_victim=True)

    false_sites = []
    while len(false_sites) < N_FALSE_POS:
        candidate = random_site(is_victim=False)

        # Check distance from victim
        ok = np.linalg.norm(candidate.vector - victim.vector) >= SIGN_MIN_DIST

        # Check distance from all existing false sites
        if ok:
            for fs in false_sites:
                if np.linalg.norm(candidate.vector - fs.vector) < SIGN_MIN_DIST:
                    ok = False
                    break

        if ok:
            false_sites.append(candidate)

    return victim, false_sites



def init_robots(victim):
    """Start robots along left edge with random y."""
    robots = []
    for _ in range(N_ROBOTS):
        x0 = 0.5
        y0 = np.random.uniform(1.0, WORLD_SIZE - 1.0)
        theta0 = np.random.uniform(-np.pi / 4, np.pi / 4)
        r = Robot(x0, y0, theta0)
        # initial belief: unknown -> center of world
        r.belief = np.array([WORLD_SIZE / 2, WORLD_SIZE / 2], dtype=float)
        robots.append(r)
    return robots


def sense_site(robot, site):
    """Return (co2, heat, water) sensed from this site (if within range)."""
    dist = np.linalg.norm(robot.pose_vector() - site.vector)
    if dist <= SENSOR_RANGE:
        return site.co2, site.heat, site.water, True
    return False, False, False, False


def compute_comm_graph(robots):
    """Return adjacency matrix based on COMM_RANGE."""
    n = len(robots)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(robots[i].pose_vector() - robots[j].pose_vector())
            if d <= COMM_RANGE:
                A[i, j] = 1.0
                A[j, i] = 1.0
    # self-loops
    for i in range(n):
        A[i, i] = 1.0
    return A


def metropolis_weights(A):
    """
    Build Metropolis weight matrix W from adjacency A.
    A_ij > 0 means i and j communicate. A has self-loops.
    """
    n = A.shape[0]
    W = np.zeros_like(A, dtype=float)
    for i in range(n):
        deg_i = np.sum(A[i, :] > 0) - 1  # exclude self
        for j in range(n):
            if i == j:
                continue
            if A[i, j] > 0:
                deg_j = np.sum(A[j, :] > 0) - 1
                W[i, j] = 1.0 / (1.0 + max(deg_i, deg_j))
        W[i, i] = 1.0 - np.sum(W[i, :])
    return W


def enforce_robot_clearance(robots):
    """
    Push robots apart if they are too close to each other,
    so their circles do not overlap in the plot.
    """
    min_clear = ROBOT_CLEAR_DIST

    n = len(robots)
    for i in range(n):
        for j in range(i + 1, n):
            ri = robots[i]
            rj = robots[j]

            dvec = ri.pose_vector() - rj.pose_vector()
            dist = np.linalg.norm(dvec)

            if dist < min_clear:
                # if they are on top of each other, pick random direction
                if dist < 1e-6:
                    angle = np.random.uniform(0, 2 * np.pi)
                    dvec = np.array([np.cos(angle), np.sin(angle)])
                    dist = 1.0

                # move each robot half the needed distance along opposite directions
                dunit = dvec / dist
                overlap = min_clear - dist
                shift = 0.5 * overlap

                ri.x += dunit[0] * shift
                ri.y += dunit[1] * shift
                rj.x -= dunit[0] * shift
                rj.y -= dunit[1] * shift

# ===========================
# Simulation and Animation
# ===========================

# NOTE: no fixed numpy seed here -> each run is different
victim, false_sites = build_environment()
robots = init_robots(victim)

# Logs
consensus_std_history = []
mean_dist_history = []
time_history = []
mode_history = []          # new: log search/aggregate

# Set up figure for animation
plt.rcParams["animation.ffmpeg_path"] = "ffmpeg"  # assumes ffmpeg in PATH
fig, ax = plt.subplots(figsize=(6, 6))

# Adjust the main axes to leave space on top for text
plt.subplots_adjust(top=0.78)
# Dynamic title (status line) above the axes (normal suptitle position)
status_text = fig.suptitle("", fontsize=10, fontweight="bold", y=0.98)



ax.set_xlim(0, WORLD_SIZE)
ax.set_ylim(0, WORLD_SIZE)
ax.set_aspect("equal")


# Graphic elements
robot_patches = []
label_texts = []
for i in range(N_ROBOTS):
    car = plt.Circle((0, 0), ROBOT_RADIUS, fc="tab:blue", ec="k", zorder=3)
    ax.add_patch(car)   # <<< add this line
    label = ax.text(0, 0, f"{i+1}", fontsize=8,
                    ha="center", va="bottom", color="black")
    robot_patches.append(car)
    label_texts.append(label)

# Draw victim and false positives (static)
victim_body, = ax.plot(victim.x, victim.y, marker="*", markersize=12,
                       color="red", label="Victim")
ax.text(victim.x + 0.2, victim.y + 0.2, "Victim", color="red", fontsize=9)

for fs in false_sites:
    color = get_site_color(fs)
    ax.scatter(fs.x, fs.y, marker="s", s=50,
               facecolor=color, edgecolor="k", zorder=2)
    # no text labels on the map – only color



# sensor labels near victim
ax.text(victim.x - 0.3, victim.y - 0.5, "CO₂", color="green", fontsize=8)
ax.text(victim.x, victim.y - 0.8, "Heat", color="orange", fontsize=8)
ax.text(victim.x + 0.3, victim.y - 0.5, "Water", color="blue", fontsize=8)

legend_elements = [
    Patch(facecolor="blue",       edgecolor="k", label="Water"),
    Patch(facecolor="grey",       edgecolor="k", label="CO₂"),
    Patch(facecolor="red",        edgecolor="k", label="Heat"),
    Patch(facecolor="orange",     edgecolor="k", label="Heat + CO₂"),
    Patch(facecolor="purple",     edgecolor="k", label="Water + Heat"),
    Patch(facecolor="deepskyblue", edgecolor="k", label="Water + CO₂"),
    Line2D([0], [0], marker="*", color="red", linestyle="None",
           markersize=10, label="Victim")
]

fig.legend(handles=legend_elements,
           loc="upper center",
           bbox_to_anchor=(0.5, 0.94),
           ncol=4)




def init_anim():
    for car, label, rob in zip(robot_patches, label_texts, robots):
        car.center = (rob.x, rob.y)
        label.set_position((rob.x, rob.y + ROBOT_RADIUS + 0.1))
    return robot_patches + label_texts + [status_text]


def step_simulation(k):
    """Run one simulation step: motion, sensing, consensus, mode switching."""
    global consensus_reached, found_robot_index
    t = k * DT

    # 1) Motion update (search or aggregate)
    victim_estimates = []
    sites_all = [victim] + false_sites

    for i, r in enumerate(robots):
        if r.state == "search":
            r.step_search(DT, sites_all)
        else:
            target = r.belief.copy()
            others = [robots[j] for j in range(len(robots)) if j != i]
            r.step_aggregate(DT, target, others, sites_all)

    # Enforce minimum spacing so robot circles do not overlap
    enforce_robot_clearance(robots)

    # 2) Sensing
    any_seen = False
    for idx, r in enumerate(robots):
        co2, heat, water, seen = sense_site(r, victim)
        if seen and co2 and heat and water:
            r.has_seen_victim = True
            any_seen = True
            r.belief = victim.vector.copy()

            # Record which robot saw the victim first
            if found_robot_index is None:
                found_robot_index = idx  # 0-based index

        for fs in false_sites:
            sense_site(r, fs)

    # 3) Communication graph and consensus if at least one detection
    if any_seen or any(r.has_seen_victim for r in robots):
        A = compute_comm_graph(robots)
        W = metropolis_weights(A)
        beliefs = np.array([r.belief for r in robots])  # shape (N,2)

        new_beliefs = beliefs.copy()
        for i, r in enumerate(robots):
            neighbor_avg = W[i, :].reshape(1, -1) @ beliefs
            neighbor_avg = neighbor_avg.ravel()
            new = beliefs[i] + CONS_ALPHA * (neighbor_avg - beliefs[i])
            if r.has_seen_victim:
                new = (1 - ANCHOR_GAIN) * new + ANCHOR_GAIN * victim.vector
            new_beliefs[i] = new

        for i, r in enumerate(robots):
            r.belief = new_beliefs[i].copy()
            victim_estimates.append(r.belief)

        victim_estimates = np.array(victim_estimates)
        std = np.std(victim_estimates, axis=0)
    else:
        std = np.array([np.nan, np.nan])

    # 4) Mode switching
    if any(r.has_seen_victim for r in robots) and k > MIN_DETECT_STEPS:
        if not np.any(np.isnan(std)) and np.all(std < CONS_TOL):
            # All robots adopt the same consensus point as their belief
            beliefs_now = np.array([r.belief for r in robots])
            consensus_point = np.mean(beliefs_now, axis=0)

            for r in robots:
                r.state = "aggregate"
                r.belief = consensus_point.copy()
            consensus_reached = True

    # 5) logging
    dists = []
    for r in robots:
        r.belief_history.append(r.belief.copy())
        d = np.linalg.norm(r.pose_vector() - victim.vector)
        r.dist_to_victim_history.append(d)
        dists.append(d)

    mean_dist = float(np.mean(dists))
    consensus_std_history.append(float(np.linalg.norm(std)))
    mean_dist_history.append(mean_dist)
    time_history.append(t)
    mode_history.append(robots[0].state)


def update_anim(frame):
    # Advance simulation one step
    step_simulation(frame)

    # Draw robots
    for car, label, r in zip(robot_patches, label_texts, robots):
        car.center = (r.x, r.y)
        label.set_position((r.x, r.y + ROBOT_RADIUS + 0.1))

    # ----- Update status line above the legend -----
    if not any(r.has_seen_victim for r in robots):
        # nobody has seen victim yet
        msg = "Phase: Exploring / Searching"
    elif not consensus_reached:
        # victim seen by at least one robot but not yet in aggregation mode
        msg = "Phase: Consensus (sharing and fusing victim estimates)"
    else:
        # consensus reached, we are aggregating
        fx, fy = victim.x, victim.y
        if found_robot_index is not None:
            msg = (f"Consensus reached at ({fx:.2f}, {fy:.2f}) "
                   f"found by robot {found_robot_index + 1} — Aggregating")
        else:
            msg = (f"Consensus reached at ({fx:.2f}, {fy:.2f}) — Aggregating")
    status_text.set_text(msg)
    # ----------------------------------------------

    return robot_patches + label_texts + [status_text]



# Create animation
anim = animation.FuncAnimation(
    fig,
    update_anim,
    init_func=init_anim,
    frames=N_STEPS,
    interval=DT * 1000,
    blit=False
)

# Save MP4 (with graceful fallback)
try:
    writer = animation.FFMpegWriter(fps=int(1 / DT))
    anim.save(VIDEO_FILENAME, writer=writer)
    print(f"Saved animation to {VIDEO_FILENAME}")
except Exception as e:
    print("Could not save MP4 (FFmpeg problem). Saving GIF instead.")
    print("Error:", e)
    anim.save("mrs_victim_consensus.gif", writer="pillow", fps=int(1 / DT))

plt.close(fig)

# ===========================
# Extra plots: consensus & aggregation
# ===========================
time_arr = np.array(time_history)

# 1) Consensus plot: each robot's x-estimate of victim position vs time
plt.figure(figsize=(6, 4))
for i, r in enumerate(robots):
    bh = np.array(r.belief_history)
    plt.plot(time_arr, bh[:, 0], label=f"Robot {i+1}")
plt.axhline(victim.x, color="k", linestyle="--", label="True victim x")
plt.xlabel("Time [s]")
plt.ylabel("Estimated victim x")
plt.title("Consensus on victim x-position")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("consensus_x.png")
plt.close()

# 2) Aggregation plot: mean distance to victim vs time
plt.figure(figsize=(6, 4))
plt.plot(time_arr, mean_dist_history, "b-")
plt.axhline(0.5, color="r", linestyle="--", label="0.5 m")
plt.xlabel("Time [s]")
plt.ylabel("Mean distance to victim")
plt.title("Robots aggregating to victim")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("aggregation_distance.png")
plt.close()

print("Saved plots: consensus_x.png, aggregation_distance.png")


with open("run_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time", "mode", "consensus_std", "mean_dist_to_victim"])
    for t, m, cs, md in zip(time_history, mode_history,
                            consensus_std_history, mean_dist_history):
        writer.writerow([t, m, cs, md])

print("Saved log: run_log.csv")
