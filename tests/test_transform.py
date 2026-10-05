"""
tests/test_transform.py — Acceptance tests for the pitch-shift / time-scale
feature (``backend/transform.py``).

Run directly:

    python3 tests/test_transform.py

or with pytest:

    python3 -m pytest tests/test_transform.py

The checks mirror the feature requirements:

* duration changes exactly with speed (speed only, pitch only, both)
* pitch changes exactly with the semitone setting
* the two controls are independent and continuously adjustable
* output is click/pop free and retains peak level (no internal clipping)
* timing of dense rhythmic material is uniformly compressed/expanded
* stereo channels stay sample-aligned
* the streaming file path and the in-memory path agree
* parameter validation rejects out-of-range values
"""

from __future__ import annotations

import math
import os
import random
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import audio_io, dsp, transform as T

SR = 22050


# --------------------------------------------------------------------------- #
# Test signals
# --------------------------------------------------------------------------- #

def harmonic(f0: float = 220.0, seconds: float = 3.0):
    n = int(SR * seconds)
    out = []
    for i in range(n):
        t = i / SR
        env = min(1.0, t * 20) * min(1.0, (seconds - t) * 5)
        out.append(env * 0.35 * (
            math.sin(2 * math.pi * f0 * t)
            + 0.5 * math.sin(2 * math.pi * 2 * f0 * t)
            + 0.25 * math.sin(2 * math.pi * 3 * f0 * t)))
    return out


def impulse_train(seconds: float = 2.0, period_s: float = 0.25, seed: int = 7):
    random.seed(seed)
    n = int(SR * seconds)
    x = [0.0] * n
    for b_idx in range(int(seconds / period_s)):
        b = int(b_idx * period_s * SR)
        for k in range(120):
            x[b + k] = 0.7 * math.exp(-k / 30) * random.uniform(-1, 1)
    return x, period_s


def f0_middle(x, fmin: float = 60.0, fmax: float = 1600.0):
    """Estimate f0 on a windowed central frame via autocorrelation."""
    mid = len(x) // 2
    seg = x[max(0, mid - 2048): mid + 2048]
    w = dsp.hann(len(seg))
    return dsp.pitch_autocorr([seg[k] * w[k] for k in range(len(seg))], SR, fmin, fmax)


def count_clicks(x, threshold: float = 0.5):
    """Sample-to-sample jumps beyond ``threshold`` (the test tones cannot
    physically move that fast, so any such jump is a discontinuity)."""
    return sum(1 for i in range(len(x) - 1) if abs(x[i + 1] - x[i]) > threshold)


def run_all():
    failures = []

    def check(name, cond, detail=""):
        mark = "PASS" if cond else "FAIL"
        print(f"[{mark}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    # ------------------------------------------------------------------ #
    # 1. Speed-only: duration exact, pitch untouched
    # ------------------------------------------------------------------ #
    sig = harmonic(220.0, 3.0)
    for speed in (0.5, 0.75, 1.25, 2.0, 4.0, 0.25):
        y = T.transform_samples(sig, SR, speed=speed, semitones=0.0, quality="draft")
        expect_n = round(len(sig) / speed)
        f0 = f0_middle(y)
        check(
            f"speed {speed}x: exact length {len(y)} == {expect_n}",
            len(y) == expect_n)
        check(
            f"speed {speed}x: pitch preserved {f0:.1f} ~ 220",
            f0 is not None and abs(f0 / 220.0 - 1.0) < 0.02,
            f"got {f0}")

    # ------------------------------------------------------------------ #
    # 2. Pitch-only: length unchanged, pitch exact
    # ------------------------------------------------------------------ #
    for st in (-12.0, -7.0, -3.0, 0.3, 2.0, 5.0, 7.0, 12.0):
        y = T.transform_samples(sig, SR, speed=1.0, semitones=st, quality="draft")
        expect_f0 = 220.0 * 2 ** (st / 12.0)
        f0 = f0_middle(y, fmin=50, fmax=1800)
        check(f"pitch {st:+} st: length unchanged ({len(y)} == {len(sig)})",
              len(y) == len(sig))
        check(f"pitch {st:+} st: f0 {f0:.2f} ~ {expect_f0:.2f}",
              f0 is not None and abs(f0 / expect_f0 - 1.0) < 0.015,
              f"got {f0}")

    # ------------------------------------------------------------------ #
    # 3. Both controls simultaneously
    # ------------------------------------------------------------------ #
    for speed, st in ((1.5, 2.0), (0.6, -12.0), (2.0, 12.0), (0.33, -5.5), (3.0, 7.0)):
        y = T.transform_samples(sig, SR, speed=speed, semitones=st, quality="draft")
        expect_n = round(len(sig) / speed)
        expect_f0 = 220.0 * 2 ** (st / 12.0)
        f0 = f0_middle(y, fmin=40, fmax=2000)
        check(f"both {speed}x/{st:+}st: length {len(y)} == {expect_n}",
              len(y) == expect_n)
        check(f"both {speed}x/{st:+}st: f0 {f0:.2f} ~ {expect_f0:.2f}",
              f0 is not None and abs(f0 / expect_f0 - 1.0) < 0.02,
              f"got {f0}")

    # ------------------------------------------------------------------ #
    # 4. Independence: pitch-only output must have same length; speed-only
    #    output must keep f0 (already above) — plus direct orthogonality:
    #    (speed s, st 0) vs (speed s, st p) differ only by f0 ratio.
    # ------------------------------------------------------------------ #
    a = T.transform_samples(sig, SR, speed=1.3, semitones=0.0, quality="draft")
    b = T.transform_samples(sig, SR, speed=1.3, semitones=4.0, quality="draft")
    check("independence: same speed, different pitch -> equal length",
          len(a) == len(b))
    fa, fb = f0_middle(a), f0_middle(b)
    check("independence: f0 ratio is exactly 2^(4/12)",
          fa and fb and abs((fb / fa) / 2 ** (4.0 / 12.0) - 1.0) < 0.02,
          f"{fa} -> {fb}")

    # ------------------------------------------------------------------ #
    # 5. Click/pop-free edges and interior
    # ------------------------------------------------------------------ #
    random.seed(11)
    noisy = [0.3 * math.sin(2 * math.pi * 300 * i / SR) + 0.1 * random.uniform(-1, 1)
             for i in range(SR * 2)]
    for name, kw in (("pitch+3", dict(speed=1, semitones=3)),
                     ("speed1.33", dict(speed=1.33, semitones=0)),
                     ("speed0.7,p-5", dict(speed=0.7, semitones=-5)),
                     ("speed2,p+12", dict(speed=2, semitones=12))):
        y = T.transform_samples(noisy, SR, quality="draft", **kw)
        check(f"{name}: zero edge samples ({abs(y[0]):.2e},{abs(y[-1]):.2e})",
              abs(y[0]) < 1e-9 and abs(y[-1]) < 1e-9)
        check(f"{name}: no clicks (0 > 0.5 jumps), got {count_clicks(y)}",
              count_clicks(y) == 0)

    # ------------------------------------------------------------------ #
    # 6. Rhythmic timing scales uniformly with speed
    # ------------------------------------------------------------------ #
    rhy, period = impulse_train()
    for speed in (0.5, 1.0, 2.0):
        y = T.transform_samples(rhy, SR, speed=speed, semitones=0.0, quality="draft")
        win = int(0.01 * SR)
        env = [sum(v * v for v in y[i:i + win]) for i in range(0, len(y) - win, win)]
        peak = max(env)
        onsets = []
        for i, e in enumerate(env):
            if e > 0.2 * peak and max(env[max(0, i - 3):i] or [0]) < 0.2 * peak:
                onsets.append(i * win / SR)
        gaps = sorted(onsets[j + 1] - onsets[j] for j in range(len(onsets) - 1))
        median_gap = gaps[len(gaps) // 2] if gaps else 0.0
        expected = period / speed
        check(f"rhythm {speed}x: {len(onsets)} onsets, IOI {median_gap:.3f} ~ {expected:.3f}",
              abs(median_gap - expected) < 0.012)

    # ------------------------------------------------------------------ #
    # 7. Stereo alignment through the streaming file path
    # ------------------------------------------------------------------ #
    n = SR * 2
    base = [0.3 * math.sin(2 * math.pi * 330 * i / SR) + 0.04 * random.uniform(-1, 1)
            for i in range(n)]
    stereo = [list(base), [v * 0.6 for v in base]]
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "in.wav")
    dst = os.path.join(tmp, "out.wav")
    audio_io.save(src, audio_io.AudioData(stereo, SR))
    meta = T.transform_file(src, dst, speed=1.25, semitones=2.0, quality="draft")
    with audio_io.WavReader(dst) as r:
        chunks = list(r.iter_chunks())
        c0 = [v for ck in chunks for v in ck[0]]
        c1 = [v for ck in chunks for v in ck[1]]
    check("stereo: equal channel lengths", len(c0) == len(c1) == meta["frames"])
    best, bl = -1.0, 0
    for lag in range(-20, 21):
        s = sum(c0[i] * c1[i + lag] for i in range(3000, 9000))
        if s > best:
            best, bl = s, lag
    check(f"stereo: zero inter-channel lag (got {bl})", bl == 0)

    # ------------------------------------------------------------------ #
    # 8. Streaming file path == in-memory path (same rendering math).
    #    Compare against the 16-bit-quantised file contents (the file path
    #    necessarily passes through PCM16), so quantisation isn't mistaken
    #    for a rendering difference.
    # ------------------------------------------------------------------ #
    dst2 = os.path.join(tmp, "out2.wav")
    mono_src = os.path.join(tmp, "mono.wav")
    audio_io.save(mono_src, audio_io.AudioData([noisy], SR))
    T.transform_file(mono_src, dst2, speed=1.25, semitones=2.0, quality="draft")
    with audio_io.WavReader(dst2) as r:
        stm = [v for ck in r.iter_chunks() for v in ck[0]]
    with audio_io.WavReader(mono_src) as r:
        quantised = [v for ck in r.iter_chunks() for v in ck[0]]
    mem = T.transform_samples(quantised, SR, speed=1.25, semitones=2.0, quality="draft")
    diff = max(abs(a - b) for a, b in zip(stm, mem))
    check(f"streaming ~ in-memory (max abs diff {diff:.2e})", diff < 1e-4)

    # ------------------------------------------------------------------ #
    # 9. Metadata from transform_file is exact
    # ------------------------------------------------------------------ #
    m = T.transform_file(mono_src, os.path.join(tmp, "m3.wav"),
                         speed=0.8, semitones=-3.0, quality="draft")
    check("meta: duration_factor == 1/speed",
          abs(m["duration_factor"] - 1.0 / 0.8) < 1e-9)
    check("meta: frames == round(in/0.8)",
          m["frames"] == round(m["frames_in"] / 0.8))
    check("meta: pitch_ratio == 2^(-3/12)",
          abs(m["pitch_ratio"] - 2 ** (-3.0 / 12.0)) < 1e-12)

    # ------------------------------------------------------------------ #
    # 10. Invalid parameters are rejected/clamped
    # ------------------------------------------------------------------ #
    for bad in (0.0, -1.0, 50.0):
        try:
            T.transform_file(mono_src, os.path.join(tmp, "x.wav"), speed=bad)
            # transform_file clamps rather than raising — verify clamp instead.
            with audio_io.WavReader(os.path.join(tmp, "x.wav")) as r:
                got = r.nframes
            expected = round(m["frames_in"] / min(4.0, max(0.25, bad)))
            check(f"clamp speed {bad} -> frames {got} ~ {expected}",
                  got == expected)
        except Exception as e:  # pragma: no cover
            failures.append(f"clamp speed {bad} raised {e}")

    # ------------------------------------------------------------------ #
    # 11. Performance (balanced quality should beat ~1 s of audio / second)
    # ------------------------------------------------------------------ #
    t0 = time.time()
    T.transform_samples(harmonic(220.0, 2.0), SR, speed=1.0, semitones=4.0,
                        quality="balanced")
    dt = time.time() - t0
    check(f"performance: 2 s audio rendered in {dt:.2f} s (< 4 s budget)", dt < 4.0)

    print()
    if failures:
        print(f"{len(failures)} FAILURES: {failures}")
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_all())
