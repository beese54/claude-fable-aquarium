"""Shoaling simulation for neon tetras. Pure numpy; runs inside or outside Blender.

Neons do not stream like a marine school. They hang in a loose group in open
water, each fish hovering, adjusting, and now and then darting, while the group
as a whole drifts around the tank. The model here is a small boids system
(separation, alignment, a shared roaming target) plus per-fish speed moods and
occasional darts, with soft walls, a floor that follows the scape's ground, and
cylinders to keep fish out of the hardscape.

Units are millimetres and seconds in the tank frame (X right, Y back, Z up).
The result is a pure function of its arguments (seeded), so any frame of any
render can be reproduced.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

FPS = 30
SUBSTEPS = 4
VERSION = 7      # bump whenever the model changes, so cached runs are not reused


def _smooth_noise(t, phases, periods):
    """0..1 noise per fish from a few incommensurate sines. phases, periods: (N, K)."""
    return 0.5 + 0.5 * np.mean(np.sin(2 * math.pi * t / periods + phases), axis=1)


def simulate(n, frame_start, frame_end, box, seed=0, obstacles=(), ground=None,
             spread=(45.0, 45.0, 20.0), cruise=22.0, tour=(17.0, 10.0, 9.0), warmup_s=12.0):
    """Simulate `n` fish for frames [frame_start, frame_end).

    box        (xmin, xmax, ymin, ymax, zmin, zmax): the open water the shoal roams
    obstacles  cylinders (x, y, radius, top_z[, bottom_z]) fish must stay out of
    ground     function (x, y) -> floor height; fish stay 18 mm above it
    spread     how loosely the fish sit around the shoal's centre
    cruise     typical speed in mm/s
    tour       periods (s) of the shoal centre's wander in x, y, z
    Returns a dict of arrays with first axis = frame:
      pos (F,N,3) mm, fwd (F,N,3) unit, speed (F,N), phase (F,N) rad,
      amp (F,N) 0..1, bend (F,N) -1..1, roll (F,N) rad
    """
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax, zmin, zmax = box
    centre = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2])
    half = np.array([(xmax - xmin) / 2, (ymax - ymin) / 2, (zmax - zmin) / 2])
    spread = np.minimum(np.array(spread), half * 0.8)
    roam = np.maximum(half - spread * 0.9, 0.0)
    tour = np.array(tour)
    tour_phase = rng.uniform(0, 2 * math.pi, 3)

    offset = np.clip(rng.normal(0, 1, (n, 3)), -1.6, 1.6) * spread * 0.75     # no stragglers far outside the group
    P = centre + offset + rng.normal(0, 4, (n, 3))
    V = rng.normal(0, 6, (n, 3)) * np.array([1, 1, 0.3])
    H = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-6)
    phase = rng.uniform(0, 2 * math.pi, n)
    mood_ph = rng.uniform(0, 2 * math.pi, (n, 3))
    mood_T = rng.uniform(2.5, 7.0, (n, 3))
    wander_ph = rng.uniform(0, 2 * math.pi, (n, 3, 2))
    wander_T = rng.uniform(4.0, 11.0, (n, 3, 2))
    burst = np.zeros(n)
    # an obstacle is (x, y, radius, top_z) or, to let fish pass underneath, (x, y, radius, top_z, bottom_z)
    obstacles = np.array([tuple(o) + (-1.0e9,) * (5 - len(o)) for o in obstacles], dtype=np.float64).reshape(-1, 5)
    lo = np.array([22.0, 22.0, 0.0])
    hi = np.array([278.0, 278.0, 250.0])

    dt = 1.0 / (FPS * SUBSTEPS)
    F = frame_end - frame_start
    out = {k: np.zeros((F, n) + s, dtype=np.float32) for k, s in
           (("pos", (3,)), ("fwd", (3,)), ("speed", ()), ("phase", ()), ("amp", ()), ("bend", ()), ("roll", ()))}
    yaw_rate = np.zeros(n)
    t = frame_start / FPS - warmup_s
    total_steps = int(round((warmup_s + F / FPS) / dt))
    first_out = int(round(warmup_s / dt))
    for step in range(total_steps):
        # where the shoal as a whole wants to be
        T = centre + roam * np.sin(2 * math.pi * t / tour + tour_phase)
        wander = np.mean(np.sin(2 * math.pi * t / wander_T + wander_ph), axis=2) * spread * 0.45
        goal = T + offset + wander
        A = (goal - P) * np.array([0.55, 1.0, 0.55])       # follow the group's front-to-back wander closely

        # neighbours: keep apart, roughly agree on direction
        D = P[:, None, :] - P[None, :, :]
        dist = np.linalg.norm(D, axis=2) + np.eye(n) * 1e6
        near = dist < 36.0
        A += np.sum(np.where(near[..., None], D / dist[..., None] * (36.0 - dist[..., None]) * 14.0, 0.0), axis=1)
        nb = dist < 75.0
        cnt = np.maximum(nb.sum(axis=1, keepdims=True), 1)
        A += (np.sum(np.where(nb[..., None], V[None, :, :], 0.0), axis=1) / cnt - V) * 1.3 * (nb.any(axis=1, keepdims=True))

        # soft walls, surface and floor
        floor = np.full(n, 30.0) if ground is None else np.array([ground(min(max(float(x), 0.0), 300.0), min(max(float(y), 0.0), 300.0)) for x, y in P[:, :2]])
        lo_i = np.column_stack([np.full(n, lo[0]), np.full(n, lo[1]), floor + 18.0])
        m = 32.0
        A += np.clip(lo_i + m - P, 0, None) * 9.0
        A -= np.clip(P - (hi - m), 0, None) * 9.0

        # hardscape
        for ox, oy, orad, otop, obot in obstacles:
            r = P[:, :2] - np.array([ox, oy])
            d = np.linalg.norm(r, axis=1) + 1e-6
            pen = np.clip(orad + 20.0 - d, 0, None) * (P[:, 2] < otop + 16.0) * (P[:, 2] > obot - 16.0)
            A[:, :2] += (r / d[:, None]) * pen[:, None] * 14.0
            A[:, 2] += pen * 3.0 * np.where(P[:, 2] > (otop + obot) / 2, 1.0, -1.0) if obot > -1.0e8 else pen * 3.0

        # each fish's own pace: idle, cruise, and the odd dart
        mood = _smooth_noise(t, mood_ph, mood_T)
        darting = rng.random(n) < 0.07 * dt
        burst = burst * math.exp(-dt / 0.45) + darting * rng.uniform(45, 85, n)
        if darting.any():
            kick = rng.normal(0, 1, (n, 3)) * np.array([1, 1, 0.25])
            H = np.where(darting[:, None], H + kick * 0.6, H)
            H /= np.linalg.norm(H, axis=1, keepdims=True)
        want = cruise * (0.35 + 1.25 * mood) + burst
        speed = np.linalg.norm(V, axis=1)
        A += H * ((want - speed) * 3.0)[:, None]

        A[:, 2] *= 0.35
        V = V + A * dt
        V[:, 2] *= (1.0 - 1.2 * dt)
        speed = np.linalg.norm(V, axis=1)
        V *= np.minimum(1.0, 110.0 / (speed + 1e-6))[:, None]
        P = P + V * dt
        speed = np.linalg.norm(V, axis=1)

        # heading follows travel, but a hovering fish keeps facing where it was
        Hn = H + (V / (speed[:, None] + 1e-6) - H) * (np.minimum(1.0, speed / 20.0) * dt / 0.45)[:, None]
        # Tetras swim close to level: cap the pitch at about 20 degrees, measured
        # against the horizontal part of the heading so a hovering fish cannot tip on end.
        # Seen dead head-on or tail-on a tetra is just a dot, so headings stay at
        # least a third of the way toward side-on; fish still travel in depth, at a slant.
        Hn[:, 1] = np.clip(Hn[:, 1], -1.6 * np.abs(Hn[:, 0]) - 1e-3, 1.6 * np.abs(Hn[:, 0]) + 1e-3)
        hxy = np.linalg.norm(Hn[:, :2], axis=1)
        flat = hxy < 0.2
        Hn[flat, :2] = H[flat, :2]
        hxy = np.maximum(np.linalg.norm(Hn[:, :2], axis=1), 1e-6)
        Hn[:, 2] = np.clip(Hn[:, 2], -0.36 * hxy, 0.36 * hxy)
        Hn /= np.linalg.norm(Hn, axis=1, keepdims=True)
        yaw_now = (H[:, 0] * Hn[:, 1] - H[:, 1] * Hn[:, 0]) / dt
        yaw_rate += (yaw_now - yaw_rate) * min(1.0, dt / 0.12)
        H = Hn
        phase += 2 * math.pi * (2.2 + 0.085 * speed) * dt
        t += dt

        k = step - first_out
        if k >= 0 and k % SUBSTEPS == 0 and k // SUBSTEPS < F:
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


def cached(cache_dir, key, **kw):
    """simulate(**kw), cached on disk under a hash of the arguments.
    `ground` cannot be hashed, so `key` must change whenever the scape's ground does."""
    sig = {k: (list(v) if isinstance(v, (tuple, list)) else v) for k, v in kw.items() if k != "ground"}
    sig["obstacles"] = [list(map(float, o)) for o in kw.get("obstacles", ())]
    h = hashlib.sha1((f"{key}v{VERSION}" + json.dumps(sig, sort_keys=True, default=str)).encode()).hexdigest()[:12]
    path = Path(cache_dir) / f"school_{key}_{h}.npz"
    if path.is_file():
        return dict(np.load(path))
    data = simulate(**kw)
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


if __name__ == "__main__":
    data = simulate(12, 690, 930, box=(40, 262, 30, 250, 115, 232), seed=4,
                    obstacles=[(108, 176, 62, 250), (204, 132, 42, 190), (52, 120, 28, 150)],
                    ground=lambda x, y: 60 + 60 * (y / 300) ** 1.2)
    print(json.dumps(report(data), indent=1))
