# consensus.py
import math
from config import COMM_RADIUS, CONSENSUS_THRESHOLD


def build_neighbor_graph(robots):
    """
    neighbors[i] = [j1, j2, ...] where distance(i, j) < COMM_RADIUS
    """
    neighbors = [[] for _ in robots]
    for i, r1 in enumerate(robots):
        for j in range(i + 1, len(robots)):
            r2 = robots[j]
            d = math.hypot(r1.x - r2.x, r1.y - r2.y)
            if d < COMM_RADIUS:
                neighbors[i].append(j)
                neighbors[j].append(i)
    return neighbors


def step_average_consensus(robots, neighbors, hazard_active):
    """
    Consensus with *stubborn informed robots*:

    - Informed robots keep a one–hot belief on their preferred building
      (they act as "anchors").
    - Uninformed robots average beliefs with neighbors, gradually
      converging to the anchored belief.
    - When everyone has the same argmax building with probability above
      CONSENSUS_THRESHOLD, we say consensus is done.

    Returns:
        consensus_done (bool),
        global_hazard_known (bool),
        agreed_index (int or None)
    """
    if not robots:
        return False, False, None

    n_buildings = len(robots[0].belief)

    if not hazard_active:
        # reset between hazards
        for r in robots:
            r.has_info = False
            r.belief = [1.0 / n_buildings] * n_buildings
        return False, False, None

    global_known = any(r.has_info for r in robots)

    # --- first, compute neighbor-averaged beliefs for everyone ---
    new_beliefs = [None] * len(robots)

    for i, r in enumerate(robots):
        neigh = neighbors[i]
        participants = [i] + neigh

        avg = [0.0] * n_buildings
        for idx in participants:
            rb = robots[idx].belief
            for k in range(n_buildings):
                avg[k] += rb[k]
        denom = float(len(participants))
        avg = [v / denom for v in avg]

        # if robot was uninformed but any neighbor is informed,
        # mark it as informed from now on
        if not r.has_info and any(robots[idx].has_info for idx in participants):
            r.has_info = True

        new_beliefs[i] = avg

    # --- "stubborn" update: informed robots snap back to one–hot ---
    for i, r in enumerate(robots):
        if r.has_info:
            # preferred building = current argmax
            k = max(range(n_buildings), key=lambda ix: robots[i].belief[ix])
            one_hot = [0.0] * n_buildings
            one_hot[k] = 1.0
            new_beliefs[i] = one_hot

    # assign beliefs
    for i, r in enumerate(robots):
        r.belief = new_beliefs[i]

    # --- check consensus ---
    argmax_list = []
    maxvals = []
    for r in robots:
        k = max(range(n_buildings), key=lambda ix: r.belief[ix])
        argmax_list.append(k)
        maxvals.append(r.belief[k])

    same_argmax = len(set(argmax_list)) == 1
    good_conf = all(v >= CONSENSUS_THRESHOLD for v in maxvals)

    consensus_done = same_argmax and good_conf
    agreed_index = argmax_list[0] if same_argmax else None

    return consensus_done, global_known, agreed_index
