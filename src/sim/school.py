"""Shoaling simulation for neon tetras. Pure numpy; runs inside or outside Blender.

There is ONE shoal for the whole film. It is simulated once, from the moment the
water arrives to the last frame, and every scape poses its fish from that same
run. So when one scape dissolves into the next the fish are in the same place in
both pictures and simply carry on along their path.

Neons do not stream like a marine school. They hang in a loose group in open
water, each fish hovering, adjusting, and now and then darting, while the group
as a whole drifts around the tank. The model is a small boids system
(separation, alignment, a shared roaming target) plus per-fish speed moods and
occasional darts, with soft walls, a floor, and cylinders to keep fish out of
the hardscape.

What changes over the film is the water the shoal may use. Each scape module
declares it as a literal near the top of the file:

  SWIM = {
      "box": (xmin, xmax, ymin, ymax, zmin, zmax),   # open water the shoal roams, mm
      "floor": (front_mm, back_mm),                  # bed height at the front and back glass
      "obstacles": [(x, y, radius, top_z), ...],     # cylinders standing on the bed, or
  }                                                  # (x, y, radius, top_z, bottom_z) hanging in the water

Ahead of each transition the shoal eases into water that is open in both
scapes, and the next scape's obstacles are switched on before it appears, so no
fish is caught inside a rock when the picture changes.

Units are millimetres and seconds in the tank frame (X right, Y back, Z up).
The result is a pure function of the timeline and the SWIM literals (seeded),
so any frame of any render can be reproduced. Changing any scape's SWIM changes
the path from that scape onward: settle them all before the final render.
"""
import ast
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FPS = 30
SUBSTEPS = 4
VERSION = 18     # bump whenever the model changes, so cached runs are not reused
COUNT = 12       # fish in the film's shoal
SEED = 4
LEAD = 100       # frames before a scape appears that the shoal starts making room for it
RAMP = 60        # frames a change of water takes
DEFAULT_SWIM = {"box": (60, 240, 40, 170, 140, 225), "floor": (60, 110), "obstacles": []}
SLANT = 1.0      # most a heading may turn from side-on, as |dy| / |dx| (1.0 = 45 degrees)
PITCH = 0.21     # most a fish may nose up or down, as rise over run (about 12 degrees)
MIN_SHARED = (90.0, 60.0, 40.0)     # least width (x, y, z) worth calling water common to two scapes


def swim_spec(scape_id):
    """The SWIM literal of a scape module, read without importing it (the module needs Blender)."""
    path = SRC / "blender" / "scapes" / f"{scape_id}.py"
    spec = dict(DEFAULT_SWIM)
    if path.is_file():
        for node in ast.parse(path.read_text(encoding="utf-8-sig")).body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "SWIM" for t in node.targets):
                spec.update(ast.literal_eval(node.value))
                break
    return {"box": tuple(float(v) for v in spec["box"]), "floor": tuple(float(v) for v in spec["floor"]),
            "obstacles": [tuple(float(v) for v in o) for o in spec["obstacles"]]}


def film_plan():
    """Every scape in film order with the frames it is on screen and its SWIM."""
    import timeline
    return [dict(id=sid, first=a, last=b, **swim_spec(sid)) for sid, (a, b) in timeline.visible().items()]


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _shared(a, b):
    """Water open in both scapes: the overlap of their boxes, axis by axis. Where
    they barely overlap the shoal moves to the next scape's water early (the
    current scape's obstacles are still in force, so it stays out of the hardscape)."""
    box = []
    for k in range(3):
        lo, hi = max(a["box"][2 * k], b["box"][2 * k]), min(a["box"][2 * k + 1], b["box"][2 * k + 1])
        if hi - lo < MIN_SHARED[k]:
            lo, hi = b["box"][2 * k], b["box"][2 * k + 1]
        box += [lo, hi]
    return np.array(box + [max(a["floor"][0], b["floor"][0]), max(a["floor"][1], b["floor"][1])])


def environment(plan, frames):
    """For each frame: the roaming box and floor (F, 8), and each scape's obstacle weight (F, S)."""
    vec = lambda p: np.array(list(p["box"]) + list(p["floor"]))
    keys = [(-1.0e9, vec(plan[0]))]
    for a, b in zip(plan, plan[1:]):
        both = _shared(a, b)
        t0 = b["first"] - LEAD
        keys += [(t0, vec(a)), (t0 + RAMP, both), (a["last"], both), (a["last"] + RAMP, vec(b))]
    keys.append((1.0e9, keys[-1][1]))
    kf = np.array([k[0] for k in keys])
    kv = np.array([k[1] for k in keys])
    frames = np.asarray(frames, dtype=np.float64)
    i = np.clip(np.searchsorted(kf, frames, side="right") - 1, 0, len(kf) - 2)
    u = _smooth((frames - kf[i]) / np.maximum(kf[i + 1] - kf[i], 1e-9))[:, None]
    env = kv[i] * (1 - u) + kv[i + 1] * u
    wgt = np.zeros((len(frames), len(plan)))
    for s, p in enumerate(plan):
        on = np.ones(len(frames)) if s == 0 else _smooth((frames - (p["first"] - LEAD)) / RAMP)
        wgt[:, s] = on * (1.0 - _smooth((frames - p["last"]) / 20.0))
    return env, wgt


def simulate(plan, frame_start, frame_end, n=COUNT, seed=SEED,
             spread=(72.0, 45.0, 30.0), cruise=22.0, tour=(17.0, 11.0, 13.0), warmup_s=12.0):
    """Simulate `n` fish for frames [frame_start, frame_end) of the film.

    plan       film_plan(): scapes in order, each with first, last, box, floor, obstacles
    spread     how loosely the fish sit around the shoal's centre
    cruise     typical speed in mm/s
    tour       periods (s) of the shoal centre's wander in x, y, z
    Returns a dict of arrays with first axis = frame:
      pos (F,N,3) mm, fwd (F,N,3) unit, speed (F,N), phase (F,N) rad,
      amp (F,N) 0..1, bend (F,N) -1..1, roll (F,N) rad
    """
    rng = np.random.default_rng(seed)
    spread0 = np.array(spread)
    tour = np.array(tour)
    tour_phase = rng.uniform(0, 2 * math.pi, 3)

    warm = int(round(warmup_s * FPS))
    F = frame_end - frame_start
    env_frames = np.arange(frame_start - warm, frame_end + 1)
    ENV, WGT = environment(plan, env_frames)
    obs, owner = [], []
    for s, p in enumerate(plan):
        for o in p["obstacles"]:
            obs.append(tuple(o) + (-1.0e9,) * (5 - len(o)))       # no bottom given: it stands on the bed
            owner.append(s)
    obs = np.array(obs, dtype=np.float64).reshape(-1, 5)
    owner = np.array(owner, dtype=np.int64)
    hanging = obs[:, 4] > -1.0e8
    mid = np.where(hanging, (obs[:, 3] + obs[:, 4]) / 2, -1.0e9)

    def water(e):
        centre = np.array([(e[0] + e[1]) / 2, (e[2] + e[3]) / 2, (e[4] + e[5]) / 2])
        half = np.array([(e[1] - e[0]) / 2, (e[3] - e[2]) / 2, (e[5] - e[4]) / 2])
        # In a shallow box the group tightens so that it still has room to travel:
        # depth is what shows the tank is three-dimensional.
        sp = np.minimum(spread0, half * np.array([0.8, 0.36, 0.8]))
        return centre, np.maximum(half - sp * 0.9, 0.0), sp

    centre, roam, sp = water(ENV[0])
    # Each fish's place in the group. Real neons do not space themselves evenly:
    # they hang about in twos, threes and fours with gaps between, and the knots
    # slowly shift. So the fish belong to a few sub-groups, each with its own
    # centre that breathes in and out, and each fish keeps its own distance.
    n_knots = max(2, n // 4)
    knot_c = np.clip(rng.normal(0, 0.85, (n_knots, 3)), -1.25, 1.25)
    knot_of = rng.choice(n_knots, n, p=np.arange(n_knots, 0, -1) / np.arange(n_knots, 0, -1).sum())
    own = rng.normal(0, 0.22, (n, 3))
    knot_T = rng.uniform(19.0, 37.0, (n_knots, 3))
    knot_ph = rng.uniform(0, 2 * math.pi, (n_knots, 3))
    personal = rng.uniform(15.0, 46.0, n)                 # how near this fish lets a neighbour come, mm
    reach = (personal[:, None] + personal[None, :]) / 2
    unit = np.clip(knot_c[knot_of] + own, -1.6, 1.6)
    P = centre + unit * sp + rng.normal(0, 4, (n, 3))
    V = rng.normal(0, 6, (n, 3)) * np.array([1, 1, 0.3])
    H = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-6)
    phase = rng.uniform(0, 2 * math.pi, n)
    mood_ph = rng.uniform(0, 2 * math.pi, (n, 3))
    mood_T = rng.uniform(2.5, 7.0, (n, 3))
    wander_ph = rng.uniform(0, 2 * math.pi, (n, 3, 2))
    wander_T = rng.uniform(4.0, 11.0, (n, 3, 2))
    burst = np.zeros(n)
    lo = np.array([22.0, 22.0, 0.0])
    hi = np.array([278.0, 278.0, 250.0])
    HARD_LO = np.array([24.0, 18.0, 30.0])      # a fish's middle never comes nearer the glass than this
    HARD_HI = np.array([276.0, 282.0, 254.0])

    dt = 1.0 / (FPS * SUBSTEPS)
    out = {k: np.zeros((F, n) + s, dtype=np.float32) for k, s in
           (("pos", (3,)), ("fwd", (3,)), ("speed", ()), ("phase", ()), ("amp", ()), ("bend", ()), ("roll", ()))}
    yaw_rate = np.zeros(n)
    t = frame_start / FPS - warmup_s
    total_steps = (warm + F) * SUBSTEPS
    first_out = warm * SUBSTEPS
    for step in range(total_steps):
        e = ENV[step // SUBSTEPS]
        centre, roam, sp = water(e)

        # where the shoal as a whole wants to be
        T = centre + roam * np.sin(2 * math.pi * t / tour + tour_phase)
        wander = np.mean(np.sin(2 * math.pi * t / wander_T + wander_ph), axis=2) * sp * 0.45
        knots = knot_c * (1.0 + 0.45 * np.sin(2 * math.pi * t / knot_T + knot_ph))
        unit = np.clip(knots[knot_of] + own, -1.6, 1.6)
        goal = T + unit * sp + wander
        # A fish's own spot never lies outside the scape's open water or inside
        # its hardscape; otherwise it would spend the shot pressed against a wall
        # or a rock, straining toward a place it cannot reach.
        goal = np.clip(goal, e[[0, 2, 4]], e[[1, 3, 5]])
        w = WGT[step // SUBSTEPS][owner] if len(owner) else np.zeros(0)
        act = w > 0
        if act.any():
            o, wa = obs[act], w[act]
            r = goal[:, None, :2] - o[None, :, :2]
            d = np.linalg.norm(r, axis=2) + 1e-6
            gz = goal[:, 2][:, None]
            inz = (gz < o[None, :, 3] + 10.0) & (gz > o[None, :, 4] - 10.0)
            push = np.clip(o[None, :, 2] + 26.0 - d, 0, None) * inz * wa[None, :]
            goal[:, :2] += np.sum(r / d[..., None] * push[..., None], axis=1)
            goal = np.clip(goal, e[[0, 2, 4]], e[[1, 3, 5]])
        A = (goal - P) * np.array([0.55, 1.0, 0.55])       # follow the group's front-to-back wander closely

        # neighbours: keep apart, roughly agree on direction
        D = P[:, None, :] - P[None, :, :]
        dist = np.linalg.norm(D, axis=2) + np.eye(n) * 1e6
        near = dist < reach
        A += np.sum(np.where(near[..., None], D / dist[..., None] * (reach - dist)[..., None] * 12.0, 0.0), axis=1)
        nb = dist < 75.0
        cnt = np.maximum(nb.sum(axis=1, keepdims=True), 1)
        A += (np.sum(np.where(nb[..., None], V[None, :, :], 0.0), axis=1) / cnt - V) * 1.3 * (nb.any(axis=1, keepdims=True))

        # soft walls, surface and floor
        floor = e[6] + (e[7] - e[6]) * np.clip(P[:, 1], 0.0, 300.0) / 300.0
        lo_i = np.column_stack([np.full(n, lo[0]), np.full(n, lo[1]), floor + 18.0])
        m = np.array([32.0, 22.0, 32.0])           # fish come nearer the front and back glass than the sides
        A += np.clip(lo_i + m - P, 0, None) * 9.0
        A -= np.clip(P - (hi - m), 0, None) * 9.0

        # hardscape of every scape that is on screen or about to be
        if act.any():
            r = P[:, None, :2] - o[None, :, :2]
            d = np.linalg.norm(r, axis=2) + 1e-6
            z = P[:, 2][:, None]
            vz = np.clip(np.minimum(o[None, :, 3] + 16.0 - z, z - (o[None, :, 4] - 16.0)) / 16.0, 0.0, 1.0)
            pen = np.clip(o[None, :, 2] + 20.0 - d, 0, None) * vz * wa[None, :]
            A[:, :2] += np.sum(r / d[..., None] * pen[..., None], axis=1) * 14.0
            away = np.where(hanging[act][None, :] & (z < mid[act][None, :]), -1.0, 1.0)   # under a hanging piece: go down
            A[:, 2] += np.sum(pen * away, axis=1) * 3.0

        # each fish's own pace: idle, cruise, and the odd dart
        mood = 0.5 + 0.5 * np.mean(np.sin(2 * math.pi * t / mood_T + mood_ph), axis=1)
        darting = rng.random(n) < 0.07 * dt
        burst = burst * math.exp(-dt / 0.45) + darting * rng.uniform(45, 85, n)
        if darting.any():
            kick = rng.normal(0, 1, (n, 3)) * np.array([1, 1, 0.25])
            H = np.where(darting[:, None], H + kick * 0.6, H)
            H /= np.linalg.norm(H, axis=1, keepdims=True)
        want = cruise * (0.35 + 1.25 * mood) + burst
        speed = np.linalg.norm(V, axis=1)
        A += H * ((want - speed) * 3.0)[:, None]
        # Water drag on anything faster than the fish means to go, in whatever
        # direction: without it a fish shoved by a rock or a neighbour keeps the
        # speed and ricochets around the tank.
        A -= V / (speed[:, None] + 1e-6) * (np.clip(speed - (want * 1.4 + 12.0), 0, None) * 6.0)[:, None]

        A[:, 2] *= 0.35
        V = V + A * dt
        V[:, 2] *= (1.0 - 1.2 * dt)
        speed = np.linalg.norm(V, axis=1)
        V *= np.minimum(1.0, 110.0 / (speed + 1e-6))[:, None]
        P = P + V * dt
        # the glass and the surface are hard limits whatever the forces say
        Pc = np.clip(P, HARD_LO, HARD_HI)
        V = np.where(Pc != P, 0.0, V)
        P = Pc
        speed = np.linalg.norm(V, axis=1)

        # heading follows travel, but a hovering fish keeps facing where it was
        Hn = H + (V / (speed[:, None] + 1e-6) - H) * (np.minimum(1.0, speed / 20.0) * dt / 0.45)[:, None]
        # Tetras swim close to level: the pitch is capped (PITCH), measured against
        # the horizontal part of the heading so a hovering fish cannot tip on end.
        # Seen head-on or tail-on a tetra is just a dot, so headings never turn
        # more than SLANT allows away from side-on; fish still travel in depth, at a slant.
        Hn[:, 1] = np.clip(Hn[:, 1], -SLANT * np.abs(Hn[:, 0]) - 1e-3, SLANT * np.abs(Hn[:, 0]) + 1e-3)
        hxy = np.linalg.norm(Hn[:, :2], axis=1)
        flat = hxy < 0.2
        Hn[flat, :2] = H[flat, :2]
        hxy = np.maximum(np.linalg.norm(Hn[:, :2], axis=1), 1e-6)
        Hn[:, 2] = np.clip(Hn[:, 2], -PITCH * hxy, PITCH * hxy)
        Hn /= np.linalg.norm(Hn, axis=1, keepdims=True)
        yaw_now = (H[:, 0] * Hn[:, 1] - H[:, 1] * Hn[:, 0]) / dt
        yaw_rate += (yaw_now - yaw_rate) * min(1.0, dt / 0.12)
        H = Hn
        phase += 2 * math.pi * (2.2 + 0.085 * speed) * dt
        t += dt

        k = step - first_out
        if k >= 0 and k % SUBSTEPS == 0:
            f = k // SUBSTEPS
            out["pos"][f] = P
            out["fwd"][f] = H
            out["speed"][f] = speed
            out["phase"][f] = phase
            out["amp"][f] = np.clip(0.18 + speed / 75.0, 0.15, 1.0)
            out["bend"][f] = np.clip(yaw_rate / 2.6, -1.0, 1.0)
            out["roll"][f] = np.clip(yaw_rate * 0.07, -0.25, 0.25)
    out["frame_start"] = np.array(frame_start)
    return out


def film(cache_dir=None):
    """The film's shoal, cached on disk under a hash of everything it depends on."""
    import timeline
    plan = film_plan()
    first = min(p["first"] for p in plan)
    last = timeline._TL["total_frames"]
    sig = json.dumps([VERSION, COUNT, SEED, LEAD, RAMP, first, last, plan], sort_keys=True, default=str)
    if cache_dir is None:
        import paths
        cache_dir = paths.CACHE
    path = Path(cache_dir) / f"school_film_{hashlib.sha1(sig.encode()).hexdigest()[:12]}.npz"
    if path.is_file():
        return dict(np.load(path))
    data = simulate(plan, first, last)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return data


def report(data, camera_mm=(152.0, -555.0, 143.0)):
    """Numbers used by the QA checks."""
    pos = data["pos"]
    cam = np.array(camera_mm)
    d = np.linalg.norm(pos - cam, axis=2)
    depth_change = d.max(axis=0) - d.min(axis=0)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=2) * FPS
    D = np.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=3) + np.eye(pos.shape[1]) * 1e6
    return {
        "fish": int(pos.shape[1]), "frames": int(pos.shape[0]),
        "fish_with_5cm_depth_change": int((depth_change >= 50).sum()),
        "depth_change_mm_min_median_max": [round(float(v), 1) for v in (depth_change.min(), np.median(depth_change), depth_change.max())],
        "speed_mm_s_median_p95_max": [round(float(v), 1) for v in (np.median(step), np.percentile(step, 95), step.max())],
        "closest_pair_mm": round(float(D.min()), 1),
        "bounds_min": [round(float(v), 1) for v in pos.reshape(-1, 3).min(axis=0)],
        "bounds_max": [round(float(v), 1) for v in pos.reshape(-1, 3).max(axis=0)],
    }


def scape_report(data, plan):
    """Per scape, over the frames it is on screen: the shoal's numbers, and how
    often a fish centre sits inside one of that scape's own obstacle cylinders."""
    f0 = int(data["frame_start"])
    rows = {}
    for p in plan:
        pos = data["pos"][max(p["first"] - f0, 0):p["last"] + 1 - f0]
        rep = report({"pos": pos})
        inside = 0
        for o in p["obstacles"]:
            o = tuple(o) + (-1.0e9,) * (5 - len(o))
            d = np.linalg.norm(pos[..., :2] - np.array(o[:2]), axis=2)
            inside += int(((d < o[2]) & (pos[..., 2] < o[3]) & (pos[..., 2] > o[4])).sum())
        rep["samples_inside_own_obstacles"] = inside
        rows[p["id"]] = rep
    return rows


if __name__ == "__main__":
    import time
    t0 = time.time()
    plan = film_plan()
    data = film()
    print(f"film shoal: {data['pos'].shape[1]} fish, frames {int(data['frame_start'])}..{int(data['frame_start']) + data['pos'].shape[0] - 1}, {time.time() - t0:.1f}s")
    step = np.linalg.norm(np.diff(data["pos"], axis=0), axis=2) * FPS
    print(f"whole film: speed median {np.median(step):.1f} mm/s, max {step.max():.1f} mm/s (a jump would show as a spike here)")
    for sid, rep in scape_report(data, plan).items():
        print(f"{sid:14s} depth>=5cm {rep['fish_with_5cm_depth_change']:2d}/{rep['fish']}  speed p95 {rep['speed_mm_s_median_p95_max'][1]:5.1f}"
              f"  x {rep['bounds_min'][0]:5.0f}..{rep['bounds_max'][0]:5.0f}  y {rep['bounds_min'][1]:5.0f}..{rep['bounds_max'][1]:5.0f}"
              f"  z {rep['bounds_min'][2]:5.0f}..{rep['bounds_max'][2]:5.0f}  inside-obstacle samples {rep['samples_inside_own_obstacles']}")
