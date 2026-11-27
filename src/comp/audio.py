"""Synthesise the film's ambient soundtrack entirely from code.

Every sound is generated here with numpy and scipy: no recorded samples, so
there is no licensing question. All event times are read from
config/timeline.json, so the sound follows the cut if the timeline changes.
Only the hand-made events in the opening footage (lamp fitting, switch) are
constants, because they are in the filmed plate rather than the timeline.

  python src/comp/audio.py [--no-report]

Writes <WORK>/audio/soundtrack.wav (48 kHz, 16-bit PCM, stereo, exactly
total_frames / fps long). Unless --no-report is given it also prints
per-section measurements and writes spectrogram PNGs next to the WAV.
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.interpolate import PchipInterpolator
from scipy.io import wavfile
from scipy.ndimage import minimum_filter1d

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
import paths  # noqa: E402

# --------------------------------------------------------------------------
# Constants that may be nudged by ear or to match the footage
# --------------------------------------------------------------------------
SR = 48000
SEED = 20261128

LIGHT_ON_S = 14.0                      # lamp switch, from the footage
HANDLING = [                           # (time s, kind) while the lamp is fitted
    (5.4, "tap"),
    (7.0, "tap"),
    (9.2, "tap"),
    (11.3, "clip"),
]

TARGET_LUFS = -16.0
TP_CEILING_DBTP = -1.5                 # 0.5 dB margin under the -1 dBTP limit
FINAL_FADE_S = 2.0
OUTRO_RESOLVE_S = 1.6                  # pad crossfade into the closing chord

# Levels before loudness normalisation (per-channel RMS dBFS unless noted).
# Normalisation shifts them all together, so only the differences matter.
ROOM_DB = -50.0
ROOM_HUM_DB = -63.0
TAP_PEAK_DB = -27.0
CLIP_PEAK_DB = -27.0
SWITCH_PEAK_DB = -24.0
SHIMMER_DB = -45.0
POUR_DB = -22.0
BED_DB = -25.0
PAD_DB = -35.0
PAD_AIR_DB = -42.0
WHOOSH_DB = -25.0
BUBBLE_DB = -11.5                      # peak of a loud 2 mm bubble

# Colour of each scape. Keyed by timeline id; unknown ids fall back by order.
#   root/ratios: the held open-fifth pad; pad_cut: pad brightness (Hz)
#   bed_cut: water-bed low-pass (Hz); fizz: faint distant bubbling
#   bub_rate: bubbles/s; bub_r: bubble radius range (mm); bub_cut: muffling
SCAPE_COLOURS = {
    "01_iwagumi":    dict(root=146.83, ratios=(1, 1.5, 2), pad_cut=900, pad_db=0.0,
                          bed_cut=700, bed_db=0.0, fizz=0.6,
                          bub_rate=0.65, bub_r=(1.0, 3.5), bub_cut=4500),
    "02_dutch":      dict(root=174.61, ratios=(1, 1.5, 2, 3), pad_cut=1300, pad_db=0.5,
                          bed_cut=850, bed_db=0.0, fizz=1.0,
                          bub_rate=1.1, bub_r=(0.8, 3.0), bub_cut=5000),
    "03_forest":     dict(root=130.81, ratios=(1, 1.5, 2, 3), pad_cut=750, pad_db=0.0,
                          bed_cut=650, bed_db=0.0, fizz=0.7,
                          bub_rate=0.8, bub_r=(1.0, 4.0), bub_cut=4000),
    "04_blackwater": dict(root=110.00, ratios=(1, 1.5, 2), pad_cut=380, pad_db=-1.0,
                          bed_cut=380, bed_db=-0.5, fizz=0.15,
                          bub_rate=0.55, bub_r=(1.8, 5.0), bub_cut=2200),
    "05_floating":   dict(root=196.00, ratios=(1, 1.5, 2.25, 3), pad_cut=2200, pad_db=0.0,
                          bed_cut=1000, bed_db=-0.5, fizz=0.8,
                          bub_rate=0.75, bub_r=(0.6, 2.5), bub_cut=6000),
}
# Closing chord: back home to the first scape's D, with a soft low octave.
OUTRO_COLOUR = dict(root=146.83, ratios=(0.5, 1, 1.5, 2), pad_cut=700, pad_db=0.0,
                    bed_cut=600, bed_db=0.0, fizz=0.4,
                    bub_rate=0.4, bub_r=(1.2, 3.5), bub_cut=3500)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def db2a(db):
    return 10.0 ** (np.asarray(db) / 20.0)


def a2db(a):
    return 20.0 * np.log10(np.maximum(a, 1e-12))


def smooth01(t, a, b):
    """0 before a, 1 after b, smoothstep in between (no corners, no clicks)."""
    u = np.clip((np.asarray(t, float) - a) / max(b - a, 1e-9), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def lin01(t, a, b):
    return np.clip((np.asarray(t, float) - a) / max(b - a, 1e-9), 0.0, 1.0)


def lp2(f, fc):
    """Magnitude of a 2nd-order low-pass; used as a spectral gain."""
    return 1.0 / np.sqrt(1.0 + (f / fc) ** 4)


def hp2(f, fc):
    r = (f / fc) ** 2
    return r / np.sqrt(1.0 + r * r)


def log_bump(f, fc, oct_sigma):
    """Gaussian band in log-frequency: a soft, natural-sounding band-pass."""
    return np.exp(-0.5 * (np.log2(np.maximum(f, 1.0) / fc) / oct_sigma) ** 2)


def pan_gains(p):
    """Equal-power pan, p in [-1, 1]."""
    th = (np.clip(p, -1, 1) + 1.0) * np.pi / 4.0
    return np.cos(th), np.sin(th)


def slow_noise(rng, t, spacing):
    """Smooth random curve (about unit std) through jittered control points.

    Jittered spacing and fresh random values mean the movement never repeats,
    unlike an LFO.
    """
    t = np.asarray(t, float)
    t0, t1 = float(t.min()) - 2 * spacing, float(t.max()) + 2 * spacing
    knots = [t0]
    while knots[-1] < t1:
        knots.append(knots[-1] + spacing * rng.uniform(0.6, 1.4))
    vals = rng.standard_normal(len(knots))
    return PchipInterpolator(np.array(knots), vals)(t)


def fade_edges(y, att_s, rel_s):
    """Raised-cosine attack and release so nothing starts or stops abruptly."""
    n = len(y)
    na = min(n, max(2, int(att_s * SR)))
    nr = min(n, max(2, int(rel_s * SR)))
    e = np.ones(n)
    e[:na] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(na) / na)
    e[n - nr:] *= 0.5 + 0.5 * np.cos(np.pi * (np.arange(nr) + 1) / nr)
    return y * e


def place(buf, y, t, pan=0.0, gain=1.0):
    """Mix mono y into stereo buf starting at time t (s)."""
    i0 = int(round(t * SR))
    if i0 >= buf.shape[1] or i0 + len(y) <= 0:
        return
    j0 = max(0, -i0)
    i0c = max(i0, 0)
    m = min(len(y) - j0, buf.shape[1] - i0c)
    gl, gr = pan_gains(pan)
    buf[0, i0c:i0c + m] += gain * gl * y[j0:j0 + m]
    buf[1, i0c:i0c + m] += gain * gr * y[j0:j0 + m]


def shaped_noise(rng, n, gain_fn, nper=4096):
    """White noise shaped by a time-varying spectral gain G(f, t) via STFT.

    One mechanism gives every noise layer (room, bed, pour, whoosh) its
    moving filter without per-sample IIR coefficient updates.
    """
    x = rng.standard_normal(n)
    nov = nper * 3 // 4
    f, tt, z = signal.stft(x, fs=SR, window="hann", nperseg=nper, noverlap=nov)
    z *= gain_fn(f[:, None], tt[None, :])
    _, y = signal.istft(z, fs=SR, window="hann", nperseg=nper, noverlap=nov)
    y = y[:n]
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-30))


# --------------------------------------------------------------------------
# Timeline
# --------------------------------------------------------------------------
class Plan:
    def __init__(self, tl):
        self.fps = tl["fps"]
        self.total = tl["total_frames"] / self.fps
        self.n = int(round(self.total * SR))
        segs = []
        for s in tl["segments"]:
            d = dict(s)
            d["t0"], d["t1"] = s["start"] / self.fps, s["end"] / self.fps
            d["mid"] = 0.5 * (d["t0"] + d["t1"])
            segs.append(d)
        self.segments = segs
        self.scapes = [s for s in segs if s["kind"] == "scape"]
        trans = [s for s in segs if s["kind"] == "transition"]
        self.awakening = next((s for s in trans if s["id"] == "awakening"), trans[0])
        self.transitions = [s for s in trans if s is not self.awakening]
        outro = next((s for s in segs if s["kind"] == "outro"), None)
        self.outro_t0 = outro["t0"] if outro else self.scapes[-1]["t1"]
        fallback = list(SCAPE_COLOURS.values())
        for i, s in enumerate(self.scapes):
            s["p"] = SCAPE_COLOURS.get(s["id"], fallback[i % len(fallback)])
        self.fade_t0 = self.total - FINAL_FADE_S

    def curve(self, key, t, log=False):
        """Per-scape parameter over time: flat inside a scape, smoothstepped
        across the gap (transition) between scapes, resolving in the outro."""
        t = np.asarray(t, float)
        tf = (lambda v: np.log(v)) if log else (lambda v: v)
        sc = self.scapes
        v = np.full(t.shape, tf(sc[0]["p"][key]), float)
        for a, b in zip(sc[:-1], sc[1:]):
            va, vb = tf(a["p"][key]), tf(b["p"][key])
            u = smooth01(t, a["t1"], b["t0"])
            v = np.where(t >= a["t1"], va + (vb - va) * u, v)
        va, vb = tf(sc[-1]["p"][key]), tf(OUTRO_COLOUR[key])
        u = smooth01(t, self.outro_t0, self.outro_t0 + OUTRO_RESOLVE_S)
        v = np.where(t >= self.outro_t0, va + (vb - va) * u, v)
        return np.exp(v) if log else v

    def master_env(self, t):
        return smooth01(t, 0.0, 0.02) * (0.5 + 0.5 * np.cos(np.pi * lin01(t, self.fade_t0, self.total)))


# --------------------------------------------------------------------------
# 1. Room tone (opening)
# --------------------------------------------------------------------------
def room_tone(plan, rng):
    aw = plan.awakening
    n = min(plan.n, int((aw["t1"] + 0.2) * SR))
    t = np.arange(n) / SR
    # Thins out as the water arrives: gone by the end of the awakening.
    env = smooth01(t, 0.0, 0.3) * (1.0 - smooth01(t, aw["t0"], aw["t1"]))
    out = np.zeros((2, plan.n))
    tk = t[::480]
    for ch in range(2):
        r = np.random.default_rng([SEED, 1, ch])

        def g(f, tt):
            # Soft broadband air: pinkish, rolled off at both ends like a phone mic.
            return hp2(f, 45.0) * lp2(f, 9000.0) * (np.maximum(f, 20.0) / 1000.0) ** -0.35
        y = shaped_noise(r, n, g)
        y *= db2a(ROOM_DB) / rms(y)
        y *= 1.0 + 0.12 * np.interp(t, tk, slow_noise(r, tk, 1.5))
        out[ch, :n] += y * env
    # Mains hum: one source, so nearly identical in both channels.
    hum = np.zeros(n)
    wob = 1.0 + 0.15 * np.interp(t, tk, slow_noise(rng, tk, 2.0))
    for k, a in zip((1, 2, 3, 4), (1.0, 0.55, 0.3, 0.12)):
        hum += a * np.sin(2 * np.pi * 50.0 * k * t + rng.uniform(0, 2 * np.pi))
    hum *= db2a(ROOM_HUM_DB) / rms(hum) * wob * env
    out[0, :n] += hum
    out[1, :n] += 0.97 * hum
    return out


# --------------------------------------------------------------------------
# 2-3. Handling, switch and shimmer (opening)
# --------------------------------------------------------------------------
def modal_hit(rng, modes, dur, noise_amp=0.5, noise_tau=0.0008, noise_hp=1500.0):
    """Struck object: a few exponentially decaying modes plus a contact tick."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for f, tau, a in modes:
        y += a * np.exp(-t / tau) * np.sin(2 * np.pi * f * t + rng.uniform(0, 0.4))
    if noise_amp:
        nz = rng.standard_normal(n) * np.exp(-t / noise_tau)
        nz = signal.sosfilt(signal.butter(2, noise_hp, "hp", fs=SR, output="sos"), nz)
        y += noise_amp * nz / (np.max(np.abs(nz)) + 1e-12) * np.max(np.abs(y))
    y = fade_edges(y, 0.0003, dur * 0.4)
    return y / np.max(np.abs(y))


def glass_tap(rng):
    j = lambda: rng.uniform(0.94, 1.06)  # noqa: E731  each tap lands somewhere new
    modes = [
        (760 * j(), 0.007, 0.9),       # plastic lamp body: short and woody
        (1650 * j(), 0.035, 1.0),      # glass panel modes ring a little longer
        (2950 * j(), 0.025, 0.55),
        (4350 * j(), 0.018, 0.35),
        (6100 * j(), 0.012, 0.2),
    ]
    return modal_hit(rng, modes, 0.16, noise_amp=0.6)


def clip_on(rng):
    """A spring clip: brief plastic scrape, a snap, then a smaller seat click."""
    n = int(0.32 * SR)
    y = np.zeros(n)
    sc_n = int(0.12 * SR)
    sc_t = np.arange(sc_n) / SR
    scrape = rng.standard_normal(sc_n)
    scrape = signal.sosfilt(signal.butter(2, [1800, 4200], "bp", fs=SR, output="sos"), scrape)
    scrape = fade_edges(scrape * (0.3 + sc_t / sc_t[-1]), 0.03, 0.01)
    y[:sc_n] += 0.12 * scrape / np.max(np.abs(scrape))
    snap = modal_hit(rng, [(2300, 0.006, 1.0), (3700, 0.005, 0.7), (5900, 0.004, 0.45)],
                     0.06, noise_amp=0.8, noise_tau=0.0006, noise_hp=2000)
    seat = modal_hit(rng, [(1950, 0.006, 1.0), (3150, 0.005, 0.6)], 0.05, noise_amp=0.6)
    i1 = sc_n
    y[i1:i1 + len(snap)] += snap
    i2 = i1 + int(0.028 * SR)
    y[i2:i2 + len(seat)] += 0.45 * seat
    return y / np.max(np.abs(y))


def switch_click(rng):
    """Small tactile switch: crisp press click (at t=0) and a quiet release."""
    press = modal_hit(rng, [(3400, 0.004, 1.0), (5200, 0.003, 0.7), (7600, 0.0025, 0.5),
                            (1250, 0.005, 0.3)], 0.04, noise_amp=1.0, noise_tau=0.0005,
                      noise_hp=2500)
    rel = modal_hit(rng, [(3000, 0.003, 1.0), (4700, 0.003, 0.6)], 0.03, noise_amp=0.8)
    y = np.zeros(int(0.2 * SR))
    y[:len(press)] += press
    i = int(0.115 * SR)
    y[i:i + len(rel)] += 0.35 * rel
    return y / np.max(np.abs(y))


def handling_and_switch(plan):
    out = np.zeros((2, plan.n))
    for k, (ts, kind) in enumerate(HANDLING):
        r = np.random.default_rng([SEED, 2, k])
        if kind == "clip":
            y = clip_on(r)
            # The snap is 0.12 s into the sound; align it to the listed time.
            place(out, y, ts - 0.12, pan=r.uniform(-0.1, 0.1), gain=db2a(CLIP_PEAK_DB))
        else:
            y = glass_tap(r)
            place(out, y, ts, pan=r.uniform(-0.12, 0.12),
                  gain=db2a(TAP_PEAK_DB + r.uniform(-2.5, 0.0)))
    r = np.random.default_rng([SEED, 3])
    place(out, switch_click(r), LIGHT_ON_S, pan=0.03, gain=db2a(SWITCH_PEAK_DB))
    return out


def light_shimmer(plan):
    """A very soft rising shimmer after the switch, as if the light had a sound.

    High partials of the first scape's open-fifth chord enter one after
    another from low to high and glide up slightly, so it reads as rising;
    a breath of bright filtered noise rises with them.
    """
    r = np.random.default_rng([SEED, 4])
    dur = min(4.4, plan.awakening["t1"] - LIGHT_ON_S)
    n = int(dur * SR)
    t = np.arange(n) / SR
    root = plan.scapes[0]["p"]["root"]
    ratios = [8, 12, 16, 18, 24, 27, 32]
    out = np.zeros((2, n))
    for k, rt in enumerate(ratios):
        on = 0.06 + 1.2 * k / (len(ratios) - 1) + r.uniform(0.0, 0.2)
        att = r.uniform(0.7, 1.0)
        tl = np.clip(t - on, 0, None)
        env = np.where(t < on, 0.0, np.sin(0.5 * np.pi * np.clip(tl / att, 0, 1)) ** 2)
        env *= np.exp(-np.clip(tl - att, 0, None) / 1.4)
        env *= 1.0 + 0.3 * slow_noise(r, t, 0.09)          # glitter, not a tremolo
        f = root * rt * 2 ** (0.5 / 12 * smooth01(t, on, dur))
        ph = 2 * np.pi * np.cumsum(f) / SR + r.uniform(0, 2 * np.pi)
        y = env * np.sin(ph) / (1.0 + 0.18 * k)
        gl, gr = pan_gains(r.uniform(-0.6, 0.6))
        out[0] += gl * y
        out[1] += gr * y
    rise = smooth01(t, 0.0, 3.0)

    for ch in range(2):
        rc = np.random.default_rng([SEED, 4, 10 + ch])
        sp = shaped_noise(rc, n, lambda f, tt: log_bump(f, 2500 * (7000 / 2500) ** smooth01(tt, 0, 3.0), 0.5))
        sp *= np.sin(0.5 * np.pi * lin01(t, 0, 1.3)) ** 2 * np.exp(-np.clip(t - 1.3, 0, None) / 1.2)
        out[ch] += 0.35 * sp / (rms(sp) + 1e-12) * rms(out[ch]) * (0.6 + 0.4 * rise)
    out[0] = fade_edges(out[0], 0.01, 0.8)
    out[1] = fade_edges(out[1], 0.01, 0.8)
    # Level is set at the shimmer's loudest half second.
    e = np.sqrt(np.mean(out ** 2, axis=0))
    win = int(0.5 * SR)
    peak = np.sqrt(np.max(np.convolve(e ** 2, np.ones(win) / win, "same")))
    out *= db2a(SHIMMER_DB) / peak
    full = np.zeros((2, plan.n))
    i0 = int(round((LIGHT_ON_S + 0.02) * SR))
    m = min(n, plan.n - i0)
    full[:, i0:i0 + m] = out[:, :m]
    return full


# --------------------------------------------------------------------------
# 4. Awakening: water arrives
# --------------------------------------------------------------------------
def pour(plan):
    aw = plan.awakening
    t0, dur = aw["t0"], aw["t1"] - aw["t0"]
    span = dur + 1.0
    n = int(span * SR)
    t = np.arange(n) / SR
    out = np.zeros((2, plan.n))
    u_peak = 0.42

    def env_u(u):
        # Swell to a peak, then a long settle into the bed.
        rise = np.sin(0.5 * np.pi * np.clip(u / u_peak, 0, 1)) ** 2
        fall = np.where(u > u_peak, np.exp(-((u - u_peak) / 0.45) ** 2 * 2.2), 1.0)
        return rise * fall * (1.0 - smooth01(u, 1.05, span / dur))

    for ch in range(2):
        r = np.random.default_rng([SEED, 5, ch])

        def g(f, tt):
            u = tt / dur
            # Bright rush at the peak, darkening gradually (over ~2 s) as the
            # tank fills and we go under.
            bright = smooth01(u, 0.0, u_peak) * (1.0 - smooth01(u, u_peak, 1.0))
            fc = 900 * (9000 / 900) ** bright
            # The air cavity above the water shrinks, so its resonance rises.
            res = 450 * (1700 / 450) ** smooth01(u, 0.1, 0.95)
            body = hp2(f, 60) * lp2(f, fc) * (np.maximum(f, 20) / 500) ** -0.3
            body = body * (1.0 + 1.6 * log_bump(f, res, 0.35))
            rumble = log_bump(f, 80, 0.8) * 1.8 * np.exp(-((u - 0.55) / 0.3) ** 2)
            return env_u(u) * body + rumble * (1.0 - smooth01(u, 1.05, span / dur))
        y = shaped_noise(r, n, g)
        # Turbulent flutter of falling water (random, not periodic).
        flutter = 1.0 + 0.25 * slow_noise(r, t, 0.06)
        y *= np.clip(flutter, 0.3, None)
        y = fade_edges(y, 0.05, 0.3)
        out[ch, int(t0 * SR):int(t0 * SR) + n] += y[: plan.n - int(t0 * SR)]
    # Level at the peak of the swell.
    pk = int((t0 + u_peak * dur) * SR)
    w = int(0.25 * SR)
    out *= db2a(POUR_DB) / rms(out[:, pk - w:pk + w])
    return out


# --------------------------------------------------------------------------
# 5. Underwater bed
# --------------------------------------------------------------------------
def underwater_bed(plan):
    aw = plan.awakening
    t0 = aw["t0"]
    i0 = int(t0 * SR)
    n = plan.n - i0
    dur = aw["t1"] - aw["t0"]
    out = np.zeros((2, plan.n))
    rs = np.random.default_rng([SEED, 6])
    tk = np.arange(0, n / SR + 1, 0.05)
    # Slow swells plus a medium-rate undulation (~1 s) so it moves like water.
    shared_cut = slow_noise(rs, tk, 4.0) + 0.45 * slow_noise(rs, tk, 1.1)
    shared_amp = slow_noise(rs, tk, 5.0) + 0.8 * slow_noise(rs, tk, 0.9)
    for ch in range(2):
        r = np.random.default_rng([SEED, 6, 10 + ch])
        ch_cut = slow_noise(r, tk, 3.0)
        ch_amp = slow_noise(r, tk, 3.5)
        fizz_mod = slow_noise(r, tk, 2.5)

        def g(f, tt):
            ta = tt + t0
            interp = lambda c: np.interp(tt, tk, c)  # noqa: E731
            env = smooth01(ta, t0 + 0.25 * dur, t0 + 0.9 * dur)
            settle = 1.0 + 2.0 * (1.0 - smooth01(ta, t0 + 0.4 * dur, aw["t1"] + 0.8))
            fc = plan.curve("bed_cut", ta, log=True) * settle * \
                2 ** (0.35 * interp(shared_cut) + 0.15 * interp(ch_cut))
            amp = env * db2a(plan.curve("bed_db", ta) + 0.9 * interp(shared_amp) + 0.6 * interp(ch_amp))
            # Weight sits at 80-600 Hz: sub-bass would only eat headroom and
            # vanish on small speakers.
            body = hp2(f, 55) * lp2(f, fc) * (np.maximum(f, 20) / 200) ** -0.2
            # Normalise each frame so a darker scape is muffled, not quieter.
            body = body / np.sqrt(np.mean(body ** 2, axis=0, keepdims=True))
            fz = plan.curve("fizz", ta) * 0.08 * log_bump(f, 3000, 0.7) * \
                np.clip(1.0 + 0.6 * interp(fizz_mod), 0.1, None)
            return amp * (body + fz)
        out[ch, i0:] = shaped_noise(r, n, g)
    ref = out[:, int(plan.scapes[0]["t0"] * SR):int(plan.scapes[-1]["t1"] * SR)]
    out *= db2a(BED_DB) / rms(ref)
    return out


# --------------------------------------------------------------------------
# 6. Pad
# --------------------------------------------------------------------------
PAD_SR = 16000  # pad has nothing above ~7 kHz; synthesise cheaply, upsample


def pad_sections(plan):
    """(colour, fade-in a, b, fade-out c, d) for each chord, in seconds."""
    aw = plan.awakening
    sc = plan.scapes
    secs = []
    for i, s in enumerate(sc):
        if i == 0:
            a, b = aw["t0"] + 0.4 * (aw["t1"] - aw["t0"]), s["t0"] + 1.0
        else:
            a, b = sc[i - 1]["t1"], s["t0"]
        if i + 1 < len(sc):
            c, d = s["t1"], sc[i + 1]["t0"]
        else:
            c, d = plan.outro_t0, plan.outro_t0 + OUTRO_RESOLVE_S
        secs.append((s["p"], a, b, c, d))
    secs.append((OUTRO_COLOUR, plan.outro_t0, plan.outro_t0 + OUTRO_RESOLVE_S,
                  plan.total + 1, plan.total + 2))
    return secs


def section_env(t, a, b, c, d):
    """Equal-power crossfade envelope (pads are uncorrelated, so power adds)."""
    fin = np.sin(0.5 * np.pi * lin01(t, a, b))
    fout = np.cos(0.5 * np.pi * lin01(t, c, d))
    return fin * fout


def pad(plan):
    n16 = int(np.ceil(plan.n / 3)) + 1
    t = np.arange(n16) / PAD_SR
    out = np.zeros((2, n16))
    for si, (p, a, b, c, d) in enumerate(pad_sections(plan)):
        i0, i1 = max(0, int(a * PAD_SR)), min(n16, int(d * PAD_SR) + 1)
        tt = t[i0:i1]
        env = section_env(tt, a, b, c, d)
        r0 = np.random.default_rng([SEED, 7, si])
        cut = p["pad_cut"] * 2 ** (0.45 * slow_noise(r0, tt, 3.5))  # slow filter movement
        sec = np.zeros((2, len(tt)))
        for ch in range(2):
            r = np.random.default_rng([SEED, 7, si, ch])
            for vi, ratio in enumerate(p["ratios"]):
                f0 = p["root"] * ratio
                vamp = db2a(2.0 * slow_noise(r, tt, 4.0)) / (1.0 + 0.4 * vi)
                # Three slightly detuned oscillators per voice, different in
                # each channel: slow beating gives life and stereo width.
                for cents in r.uniform(-7, 7, 3):
                    f = f0 * 2 ** (cents / 1200)
                    h = 1
                    while h * f < 7000 and h <= 8:
                        ah = h ** -1.4 * lp2(h * f, cut)
                        sec[ch] += vamp * ah * np.sin(2 * np.pi * h * f * tt + r.uniform(0, 2 * np.pi))
                        h += 1
        mid = (tt > b) & (tt < c)
        ref = sec[:, mid] if mid.sum() > PAD_SR else sec
        sec *= db2a(PAD_DB + p["pad_db"]) / rms(ref)
        out[:, i0:i1] += sec * env
    up = signal.resample_poly(out, 3, 1, axis=1)[:, :plan.n]
    return up


def pad_air(plan):
    """Breath of noise focused at the chord's partials: makes the pad airy."""
    secs = pad_sections(plan)
    i0 = int(secs[0][1] * SR)
    n = plan.n - i0
    out = np.zeros((2, plan.n))
    for ch in range(2):
        r = np.random.default_rng([SEED, 8, ch])
        tk = np.arange(0, n / SR + 1, 0.05)
        mv = slow_noise(r, tk, 3.0)

        def g(f, tt):
            ta = tt + i0 / SR
            G = np.zeros((f.shape[0], tt.shape[1]))
            for p, a, b, c, d in secs:
                env = section_env(ta, a, b, c, d)
                bumps = np.zeros_like(f)
                for vi, ratio in enumerate(p["ratios"]):
                    for h in (1, 2, 3):
                        fp = p["root"] * ratio * h
                        bw = max(0.03 * fp, 6.0)
                        bumps = bumps + np.exp(-0.5 * ((f - fp) / bw) ** 2) / (h * (1 + 0.4 * vi))
                G += env * bumps * lp2(f, p["pad_cut"] * 1.5)
            return G * db2a(1.5 * np.interp(tt, tk, mv))
        out[ch, i0:] = shaped_noise(r, n, g, nper=8192)
    ref = out[:, int(plan.scapes[0]["t0"] * SR):int(plan.scapes[-1]["t1"] * SR)]
    out *= db2a(PAD_AIR_DB) / rms(ref)
    return out


# --------------------------------------------------------------------------
# 7. Transition whooshes
# --------------------------------------------------------------------------
def whooshes(plan):
    out = np.zeros((2, plan.n))
    for k, tr in enumerate(plan.transitions):
        a, b = tr["t0"] - 0.35, tr["t1"] + 0.35         # bell spans the segment
        w0 = a - 0.1
        n = int((b - w0 + 0.1) * SR)
        t = np.arange(n) / SR + w0
        direction = 1 if k % 2 == 0 else -1
        seg = np.zeros((2, n))
        for ch in range(2):
            r = np.random.default_rng([SEED, 9, k, ch])

            def g(f, tt):
                ta = tt + w0
                u = lin01(ta, a, b)
                bell = np.sin(np.pi * u) ** 2
                fc = 320 * (3000 / 320) ** bell              # sweep up, then down
                air = 0.25 * hp2(f, 5000) * lp2(f, 12000) * bell
                return bell ** 1.5 * (log_bump(f, fc, 1.0) + air)
            y = shaped_noise(r, n, g, nper=2048)
            seg[ch] = y
        # A gentle drift across the image, alternating direction.
        u = lin01(t, a, b)
        pan = direction * 0.25 * (2 * smooth01(u, 0, 1) - 1)
        gl, gr = pan_gains(pan)
        seg[0] *= gl * np.sqrt(2)
        seg[1] *= gr * np.sqrt(2)
        seg[0] = fade_edges(seg[0], 0.05, 0.05)
        seg[1] = fade_edges(seg[1], 0.05, 0.05)
        mid = int((tr["mid"] - w0) * SR)
        w = int(0.15 * SR)
        seg *= db2a(WHOOSH_DB) / rms(seg[:, mid - w:mid + w])
        i0 = int(round(w0 * SR))
        m = min(n, plan.n - i0)
        out[:, i0:i0 + m] += seg[:, :m]
    return out


# --------------------------------------------------------------------------
# Bubbles (awakening cluster, transitions, underwater)
# --------------------------------------------------------------------------
def bubble(r_m, xi):
    """One bubble (van den Doel's model): damped sine at the Minnaert
    frequency (~3/r Hz) whose pitch rises as it decays. Smaller bubbles are
    higher and die faster."""
    f0 = 3.0 / r_m
    d = 0.043 * f0 + 0.0014 * f0 ** 1.5
    dur = min(6.9 / d, 0.35)
    n = int(dur * SR) + 1
    t = np.arange(n) / SR
    f = f0 * (1.0 + xi * d * t)
    y = np.exp(-d * t) * np.sin(2 * np.pi * np.cumsum(f) / SR)
    return fade_edges(y, min(0.0012, 0.4 / d), dur * 0.2)


def bubble_events(plan):
    """List of (t, radius m, xi, gain dB, pan)."""
    ev = []
    aw = plan.awakening
    # Awakening: a rising cluster, denser and smaller (higher) as it goes.
    r = np.random.default_rng([SEED, 10])
    a, b = aw["t0"] + 0.6, aw["t1"] + 0.4
    for _ in range(70):
        u = r.beta(2.2, 1.3)
        rad = 4.5e-3 * (0.9 / 4.5) ** u * np.exp(0.18 * r.standard_normal())
        ev.append((a + (b - a) * u, rad, r.uniform(0.06, 0.12), r.uniform(-12, -1),
                   r.uniform(-0.3, 0.3) * (1 + 2 * u)))
    # Transitions: a flurry of fine bubbles centred on the middle.
    for k, tr in enumerate(plan.transitions):
        r = np.random.default_rng([SEED, 11, k])
        sd = 0.16 * (tr["t1"] - tr["t0"])
        for _ in range(55):
            tt = np.clip(tr["mid"] + sd * r.standard_normal(), tr["t0"], tr["t1"])
            rad = np.clip(1.1e-3 * np.exp(0.3 * r.standard_normal()), 0.6e-3, 2.0e-3)
            ev.append((tt, rad, r.uniform(0.05, 0.12), r.uniform(-11, -1), r.uniform(-0.9, 0.9)))
    # Underwater: sparse singles plus occasional short runs. Inhomogeneous
    # Poisson (thinning) with a randomly wandering rate, so never a pattern.
    r = np.random.default_rng([SEED, 12])
    t_start, t_end = aw["t1"] + 0.3, plan.total - 0.6 * FINAL_FADE_S
    tk = np.arange(t_start - 5, t_end + 5, 0.1)
    mod = 2 ** (0.8 * slow_noise(r, tk, 3.0))
    rate = lambda tt: np.interp(tt, tk, mod) * plan.curve("bub_rate", tt)  # noqa: E731
    # Thinning needs a bound on the rate everywhere.
    lam_max = float(np.max(mod)) * max([s["p"]["bub_rate"] for s in plan.scapes]
                                       + [OUTRO_COLOUR["bub_rate"]]) * 1.01
    tt = t_start
    while True:
        tt += r.exponential(1.0 / lam_max)
        if tt > t_end:
            break
        if r.uniform() * lam_max > rate(tt):
            continue
        lo, hi = bubble_radius_range(plan, tt)
        rad = np.exp(r.uniform(np.log(lo), np.log(hi))) * 1e-3
        if r.uniform() < 0.12:                       # a small run of bubbles
            cnt = int(r.integers(3, 9))
            p = r.uniform(-0.8, 0.8)
            ts = tt
            for j in range(cnt):
                rj = rad * (0.93 ** j) * np.exp(0.12 * r.standard_normal())
                ev.append((ts, rj, r.uniform(0.06, 0.12), r.uniform(-10, -2) - 0.6 * j,
                           np.clip(p + 0.05 * j * r.choice([-1, 1]), -1, 1)))
                ts += r.gamma(2.0, 0.045)
            tt = ts
        else:
            ev.append((tt, rad, r.uniform(0.06, 0.12), r.uniform(-12, 0), r.uniform(-0.85, 0.85)))
    return ev


def bubble_radius_range(plan, tt):
    """Radius range (mm) of the scape playing at time tt."""
    for s in plan.scapes:
        if tt < s["t1"]:
            return s["p"]["bub_r"]
    return OUTRO_COLOUR["bub_r"]


def bubbles(plan):
    out = np.zeros((2, plan.n))
    ev = bubble_events(plan)
    for (tt, rad, xi, gdb, pan) in ev:
        f0 = 3.0 / rad
        # Bigger bubbles are louder; the water and glass muffle high ones.
        # (capped so the odd big bubble cannot dominate the peaks).
        g = db2a(BUBBLE_DB + gdb) * min((rad / 2e-3) ** 0.5, 1.35) * \
            lp2(f0, float(plan.curve("bub_cut", tt)))
        place(out, bubble(rad, xi), tt, pan=pan, gain=g)
    return out, ev


# --------------------------------------------------------------------------
# Mastering
# --------------------------------------------------------------------------
def integrated_lufs(x):
    """ITU-R BS.1770-4 integrated loudness (K-weighting, gated)."""
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a1 = [1.0, -1.69065929318241, 0.73248077421585]
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, -1.99004745483398, 0.99007225036621]
    y = signal.lfilter(b2, a2, signal.lfilter(b1, a1, x, axis=1), axis=1)
    blk, hop = int(0.4 * SR), int(0.1 * SR)
    cs = np.concatenate([np.zeros((2, 1)), np.cumsum(y * y, axis=1)], axis=1)
    st = np.arange(0, y.shape[1] - blk + 1, hop)
    z = (cs[:, st + blk] - cs[:, st]) / blk
    lk = -0.691 + 10 * np.log10(z.sum(0) + 1e-30)
    g = lk > -70
    l_abs = -0.691 + 10 * np.log10(z[:, g].mean(1).sum())
    g2 = g & (lk > l_abs - 10)
    return -0.691 + 10 * np.log10(z[:, g2].mean(1).sum())


def true_peak_per_sample(x):
    os_ = signal.resample_poly(x, 4, 1, axis=1)
    pk = np.abs(os_).max(axis=0)[: x.shape[1] * 4].reshape(-1, 4).max(axis=1)
    return np.maximum(pk, np.abs(x).max(axis=0))


def limit(x, ceiling_db):
    """Look-ahead peak limiter on oversampled peaks; a no-op if under the
    ceiling. Min-filter then a same-width smoothing window guarantees the
    gain at every peak is at or below what that peak needs."""
    thr = db2a(ceiling_db)
    pk = true_peak_per_sample(x)
    g = np.minimum(1.0, thr / np.maximum(pk, 1e-12))
    if g.min() >= 1.0:
        return x, 1.0, 0.0, 0.0
    L = int(0.005 * SR)
    g = minimum_filter1d(g, 2 * L + 1, mode="nearest")
    k = np.hanning(2 * L + 3)[1:-1]
    # Edge padding, not zeros, so the ends of the track are not ducked.
    g = np.convolve(np.pad(g, L, mode="edge"), k / k.sum(), mode="valid")
    return x * g, float(g.min()), float(np.mean(g < 0.999)), int(np.argmin(g)) / SR


def master(plan, mix):
    t = np.arange(plan.n) / SR
    env = plan.master_env(t)
    sos = signal.butter(2, 20.0, "hp", fs=SR, output="sos")
    x = signal.sosfiltfilt(sos, mix * env, axis=1)       # removes DC and sub-rumble
    x -= x.mean(axis=1, keepdims=True)
    x *= env                                            # exact silence at the end
    gain_db = 0.0
    for _ in range(4):                                  # gain, limit, re-measure
        y, gmin, frac, tmin = limit(x * db2a(gain_db), TP_CEILING_DBTP)
        err = TARGET_LUFS - integrated_lufs(y)
        if abs(err) < 0.05:
            break
        gain_db += err
    print(f"limiter: max reduction {-a2db(gmin):.2f} dB at {tmin:.2f} s, "
          f"active on {100 * frac:.3f}% of samples")
    return y, gain_db, env


def to_int16(x, env):
    """TPDF dither (seeded), faded with the master so the end is digital 0."""
    r = np.random.default_rng([SEED, 99])
    d = (r.uniform(-0.5, 0.5, x.shape) + r.uniform(-0.5, 0.5, x.shape)) * (env > 1e-4)
    q = np.round(x * 32767.0 + d)
    return np.clip(q, -32767, 32767).astype(np.int16).T.copy()


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------
def render(plan):
    rng = np.random.default_rng([SEED, 0])
    stems = {}
    stems["room"] = room_tone(plan, rng)
    stems["handling"] = handling_and_switch(plan)
    stems["shimmer"] = light_shimmer(plan)
    stems["pour"] = pour(plan)
    stems["bed"] = underwater_bed(plan)
    stems["pad"] = pad(plan)
    stems["pad_air"] = pad_air(plan)
    stems["whoosh"] = whooshes(plan)
    stems["bubbles"], events = bubbles(plan)
    mix = sum(stems.values())
    out, gain_db, env = master(plan, mix)
    return out, gain_db, env, stems, events


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def section_stats(x, a, b):
    s = x[:, int(a * SR):int(b * SR)]
    mono = s.mean(0)
    f, p = signal.welch(mono, fs=SR, nperseg=8192)
    cen = float((f * p).sum() / (p.sum() + 1e-30))
    corr = float(np.corrcoef(s[0], s[1])[0, 1])
    return 20 * np.log10(rms(s)), cen, corr


def band_env(x, lo, hi, smooth_s):
    """Smoothed energy in a band over time, from an STFT of the mono mix."""
    nper = 2048
    f, tt, z = signal.stft(x.mean(0), fs=SR, nperseg=nper, noverlap=nper * 3 // 4)
    sel = (f >= lo) & (f <= hi)
    e = (np.abs(z[sel]) ** 2).sum(0)
    w = max(1, int(smooth_s / (tt[1] - tt[0])))
    k = np.hanning(w + 2)[1:-1]
    return tt, np.convolve(e, k / k.sum(), "same")


def report(plan, out, gain_db, stems, events, wav_path):
    g = db2a(gain_db)
    print(f"\nnormalisation gain {gain_db:+.2f} dB; internal LUFS {integrated_lufs(out):.2f}; "
          f"internal true peak {a2db(true_peak_per_sample(out).max()):.2f} dBTP")
    print(f"\n{'section':16s} {'start':>6s} {'end':>6s} {'RMS dB':>7s} {'centroid':>9s} {'L/R corr':>8s}")
    rows = [("room tone only", 0.3, HANDLING[0][0] - 0.3)]
    rows += [(s["id"], s["t0"], s["t1"]) for s in plan.segments]
    res = {}
    for name, a, b in rows:
        lv, cen, cor = section_stats(out, a, b)
        res[name] = lv
        print(f"{name:16s} {a:6.2f} {b:6.2f} {lv:7.1f} {cen:8.0f}Hz {cor:8.2f}")
    sc_lv = [res[s["id"]] for s in plan.scapes]
    under = 10 * np.log10(np.mean([10 ** (v / 10) for v in sc_lv]))
    print(f"\nopening room tone {res['room tone only']:.1f} dB vs underwater scapes "
          f"mean {under:.1f} dB (quietest {min(sc_lv):.1f}): "
          f"{min(sc_lv) - res['room tone only']:.1f} dB below the quietest scape")
    whole_intro = section_stats(out, 0.0, plan.awakening["t0"])[0]
    print(f"whole opening incl. handling and switch (0 to awakening): {whole_intro:.1f} dB")

    names = list(stems)
    print("\nstem RMS dB per section (after normalisation gain):")
    print(f"{'section':16s} " + " ".join(f"{k:>8s}" for k in names))
    for name, a, b in rows[1:]:
        vals = [20 * np.log10(rms(stems[k][:, int(a * SR):int(b * SR)] * g)) for k in names]
        print(f"{name:16s} " + " ".join(f"{v:8.1f}" for v in vals))

    for name in ("bed", "pad", "pad_air"):
        corr = np.corrcoef(stems[name][0, int(plan.scapes[0]["t0"] * SR):int(plan.scapes[-1]["t1"] * SR)],
                           stems[name][1, int(plan.scapes[0]["t0"] * SR):int(plan.scapes[-1]["t1"] * SR)])[0, 1]
        print(f"{name} stem L/R correlation over the scapes: {corr:+.3f}")

    # Lamp click onset.
    sos = signal.butter(4, 2000, "hp", fs=SR, output="sos")
    a, b = int((LIGHT_ON_S - 0.25) * SR), int((LIGHT_ON_S + 0.25) * SR)
    hp = np.abs(signal.sosfilt(sos, out.mean(0)[a - SR:b]))[SR:]
    onset = a + int(np.argmax(hp > 0.3 * hp.max()))
    peak = a + int(np.argmax(hp))
    print(f"\nlamp click: onset {onset / SR:.4f} s, peak {peak / SR:.4f} s "
          f"(target {LIGHT_ON_S:.3f}; error {1000 * (onset / SR - LIGHT_ON_S):+.1f} ms)")

    # Whoosh peaks: mix band energy, and the whoosh stem alone.
    tt_m, e_m = band_env(out, 800, 6000, 0.5)
    tt_s, e_s = band_env(stems["whoosh"] * g, 200, 8000, 0.3)
    for tr in plan.transitions:
        sel = (tt_m > tr["t0"] - 0.5) & (tt_m < tr["t1"] + 0.5)
        pm = tt_m[sel][np.argmax(e_m[sel])]
        ps = tt_s[sel][np.argmax(e_s[sel])]
        print(f"whoosh {tr['id']}: middle {tr['mid']:.3f} s, mix peak {pm:.3f} s ({pm - tr['mid']:+.3f}), "
              f"stem peak {ps:.3f} s ({ps - tr['mid']:+.3f})")

    # Bubbles: how irregular are the underwater ones?
    t_on = np.sort([e[0] for e in events if e[0] > plan.awakening["t1"] + 0.3])
    trans_win = [(tr["t0"], tr["t1"]) for tr in plan.transitions]
    t_on = np.array([t for t in t_on if not any(a <= t <= b for a, b in trans_win)])
    ioi = np.diff(t_on)
    print(f"\nbubbles: {len(events)} total; underwater (outside transitions) {len(t_on)}, "
          f"{len(t_on) / (plan.total - plan.awakening['t1']):.2f}/s; inter-onset min {ioi.min():.3f} "
          f"median {np.median(ioi):.3f} max {ioi.max():.2f} s, CV {ioi.std() / ioi.mean():.2f} "
          f"(1 = Poisson, >1 = clustered, ~0 = metronomic)")

    # Bed periodicity: autocorrelation of its level envelope over the scapes.
    bed = stems["bed"].mean(0)[int(plan.scapes[0]["t0"] * SR):int(plan.scapes[-1]["t1"] * SR)]
    hop = int(0.05 * SR)
    ev_ = np.sqrt(np.mean(bed[: len(bed) // hop * hop].reshape(-1, hop) ** 2, axis=1))
    ev_ = np.log(ev_) - np.log(ev_).mean()
    ac = np.correlate(ev_, ev_, "full")[len(ev_) - 1:]
    ac /= ac[0]
    lags = np.arange(len(ac)) * 0.05
    # Look only past the first zero crossing: before it, ac is just the
    # slow movement decorrelating, not repetition.
    z0 = int(np.argmax(ac < 0))
    sel = (np.arange(len(ac)) > z0) & (lags <= 20.0)
    print(f"bed level-envelope autocorrelation: decorrelates by {lags[z0]:.1f} s; "
          f"largest later peak {ac[sel].max():.2f} at {lags[sel][np.argmax(ac[sel])]:.1f} s "
          f"(a loop would show a peak near 1)")

    i16 = wavfile.read(wav_path)[1]
    print(f"\nint16: max |sample| {np.abs(i16.astype(np.int32)).max()} (full scale 32767), "
          f"samples at full scale {int((np.abs(i16.astype(np.int32)) >= 32767).sum())}, "
          f"DC L {i16[:, 0].mean() / 32768:+.6f} R {i16[:, 1].mean() / 32768:+.6f}, "
          f"last sample {i16[-1].tolist()}")

    if shutil.which("ffmpeg"):
        p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(wav_path),
                            "-af", "ebur128=peak=true", "-f", "null", "-"],
                           capture_output=True, text=True)
        summ = p.stderr[p.stderr.rfind("Summary:"):]
        i_m = re.search(r"I:\s+(-?[\d.]+) LUFS", summ)
        tp_m = re.search(r"True peak:\s+Peak:\s+(-?[\d.]+) dBFS", summ)
        print(f"ffmpeg ebur128: integrated {i_m.group(1) if i_m else '?'} LUFS, "
              f"true peak {tp_m.group(1) if tp_m else '?'} dBFS")


def spectrogram_png(plan, out, path, t_a=0.0, t_b=None, title=""):
    t_b = plan.total if t_b is None else t_b
    x = out[:, int(t_a * SR):int(t_b * SR)].mean(0)
    nper = 4096 if (t_b - t_a) > 30 else 2048
    f, tt, s = signal.spectrogram(x, fs=SR, nperseg=nper, noverlap=nper * 3 // 4)
    tt = tt + t_a
    fl = np.geomspace(20, 20000, 480)
    img = np.empty((len(fl), len(tt)))
    for j in range(len(tt)):
        img[:, j] = np.interp(fl, f, s[:, j])
    img = 10 * np.log10(img + 1e-14)
    hop = int(0.05 * SR)
    xs = out[:, int(t_a * SR):int(t_b * SR)]
    m = xs.shape[1] // hop * hop
    lv = 20 * np.log10(np.sqrt(np.mean(xs[:, :m].reshape(2, -1, hop) ** 2, axis=2)) + 1e-9)
    pk = 20 * np.log10(np.abs(xs[:, :m]).reshape(2, -1, hop).max(axis=2) + 1e-9)
    tl = t_a + (np.arange(lv.shape[1]) + 0.5) * hop / SR
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        from PIL import Image
        v = np.clip((img - img.max() + 90) / 90, 0, 1)[::-1]
        Image.fromarray((v * 255).astype(np.uint8)).resize((2400, 600)).save(path)
        return
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(24, 9), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1]))
    vmax = np.percentile(img, 99.9)
    ax.imshow(img, origin="lower", aspect="auto", cmap="magma", vmin=vmax - 90, vmax=vmax,
              extent=[tt[0], tt[-1], 0, len(fl)])
    ticks = [30, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    ax.set_yticks([np.interp(np.log(v), np.log(fl), np.arange(len(fl))) for v in ticks])
    ax.set_yticklabels([f"{v:g}" for v in ticks])
    ax.set_ylabel("Hz")
    for s in plan.segments:
        if t_a <= s["t0"] <= t_b:
            for a_ in (ax, ax2):
                a_.axvline(s["t0"], color="w" if a_ is ax else "k", ls="--", lw=0.8, alpha=0.7)
            ax.text(s["t0"] + 0.1, len(fl) - 15, s["id"], color="w", fontsize=9, va="top")
        if s["kind"] == "transition" and t_a <= s["mid"] <= t_b:
            ax.axvline(s["mid"], color="c", ls=":", lw=0.8)
    for ts, kind in HANDLING:
        if t_a <= ts <= t_b:
            ax.plot([ts], [8], marker="^", color="y")
    if t_a <= LIGHT_ON_S <= t_b:
        ax.plot([LIGHT_ON_S], [8], marker="^", color="c")
    ax.set_title(title)
    ax2.plot(tl, lv[0], lw=0.6, label="RMS L (50 ms)")
    ax2.plot(tl, lv[1], lw=0.6, label="RMS R")
    ax2.plot(tl, np.maximum(pk[0], pk[1]), lw=0.4, color="0.5", label="sample peak")
    ax2.axhline(-1, color="r", lw=0.8, label="-1 dBFS")
    ax2.set_ylim(-80, 0)
    ax2.set_ylabel("dBFS")
    ax2.set_xlabel("s")
    ax2.legend(loc="lower left", fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()
    plan = Plan(paths.timeline())
    out_dir = paths.WORK / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    out, gain_db, env, stems, events = render(plan)
    wav_path = out_dir / "soundtrack.wav"
    wavfile.write(str(wav_path), SR, to_int16(out, env))
    print(f"wrote {wav_path} ({plan.n} samples, {plan.n / SR:.3f} s)")
    if not args.no_report:
        report(plan, out, gain_db, stems, events, wav_path)
        spectrogram_png(plan, out, out_dir / "spectrogram.png", title="soundtrack: whole film")
        spectrogram_png(plan, out, out_dir / "spectrogram_opening.png", 0.0, plan.scapes[0]["t0"] + 2,
                        title="opening to first scape")
        tr = plan.transitions[0]
        spectrogram_png(plan, out, out_dir / "spectrogram_transition.png", tr["t0"] - 3, tr["t1"] + 3,
                        title=f"transition {tr['id']}")
        print(f"spectrograms in {out_dir}")


if __name__ == "__main__":
    main()
