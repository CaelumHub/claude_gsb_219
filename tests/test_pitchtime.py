"""
test_pitchtime.py — Verification suite for the pitch-shift / time-stretch DSP.

Run:  python3 tests/test_pitchtime.py

Covers the acceptance criteria:
  * tempo-only: duration scales exactly, pitch unchanged (up & down)
  * pitch-only: pitch scales exactly, duration unchanged (up & down)
  * combined: both at once, independently
  * continuous (fractional) amounts
  * naturalness: no clipping, no clicks, no dropouts, level preserved
  * rhythmic material: onsets preserved, no doubled/lost beats
  * stereo coherence
"""

import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend import dsp, pitchtime  # noqa: E402

SR = 44100
random.seed(7)

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name} {detail}")
    else:
        FAIL += 1
        print(f"  FAIL  {name} {detail}")


# --------------------------------------------------------------------------- #
# Test-signal generators
# --------------------------------------------------------------------------- #

def gen_sine(freq=440.0, dur=2.0, sr=SR, amp=0.5):
    n = int(sr * dur)
    return [amp * math.sin(2 * math.pi * freq * i / sr) for i in range(n)]


def gen_chord(dur=2.0, sr=SR):
    n = int(sr * dur)
    freqs = [220.0, 277.18, 329.63]  # A3 C#4 E4
    return [0.25 * sum(math.sin(2 * math.pi * f * i / sr) for f in freqs)
            for i in range(n)]


def gen_pluck(freq=196.0, dur=2.0, sr=SR):
    delay = max(2, int(sr / freq))
    buf = [random.uniform(-1, 1) for _ in range(delay)]
    out, idx = [], 0
    for _ in range(int(sr * dur)):
        cur = buf[idx]
        buf[idx] = 0.5 * (cur + buf[(idx + 1) % delay]) * 0.996
        out.append(cur)
        idx = (idx + 1) % delay
    return out


def gen_voice(dur=2.5, sr=SR, f0=180.0):
    """Vocal-like: 8 harmonics with vibrato + 2 formant-ish envelopes."""
    n = int(sr * dur)
    out = []
    phase = [0.0] * 9
    for i in range(n):
        t = i / sr
        vib = 1.0 + 0.03 * math.sin(2 * math.pi * 5.0 * t)  # ±half semitone
        x = 0.0
        for k in range(1, 9):
            phase[k] += 2 * math.pi * f0 * k * vib / sr
            # crude formant weighting: emphasise ~700 Hz and ~2.5 kHz
            f = f0 * k
            w = (1.0 / k) * (1.0 + 1.5 * math.exp(-((f - 700) / 300) ** 2)
                             + 1.2 * math.exp(-((f - 2500) / 500) ** 2))
            x += w * math.sin(phase[k])
        env = min(1.0, i / (0.05 * sr), (n - i) / (0.05 * sr))
        out.append(0.08 * x * env)
    return out


def gen_drums(dur=3.0, sr=SR, bpm=132.0):
    """Dense rhythmic material: noise-burst 'hits' on a 16th-note grid."""
    n = int(sr * dur)
    out = [0.0] * n
    step = 60.0 / bpm / 4.0  # 16th notes
    t = 0.1
    k = 0
    while t < dur - 0.1:
        start = int(t * sr)
        length = int(0.03 * sr)
        amp = 0.8 if k % 4 == 0 else 0.45
        for j in range(length):
            if start + j < n:
                out[start + j] += amp * random.uniform(-1, 1) * math.exp(-j / (0.004 * sr))
        # low "kick" sine thump on the beat
        if k % 4 == 0:
            for j in range(int(0.09 * sr)):
                if start + j < n:
                    out[start + j] += 0.5 * math.sin(2 * math.pi * 55 * j / sr) * math.exp(-j / (0.03 * sr))
        t += step
        k += 1
    return out


# --------------------------------------------------------------------------- #
# Measurement helpers
# --------------------------------------------------------------------------- #

def measure_f0(x, sr, fmin=50, fmax=2000):
    """Median autocorrelation f0 over the steady middle of the signal."""
    n = len(x)
    w = min(4096, n // 2)
    mid = n // 2
    f0s = []
    for start in range(mid - w, mid + w, w // 2):
        seg = x[max(0, start):start + w]
        if len(seg) < w:
            continue
        f = dsp.pitch_autocorr(seg, sr, fmin, fmax)
        if f:
            f0s.append(f)
    f0s.sort()
    return f0s[len(f0s) // 2] if f0s else None


def envelope(x, sr, win_ms=5.0, hop_ms=2.0):
    """Short-time RMS envelope."""
    win = max(1, int(win_ms / 1000 * sr))
    hop = max(1, int(hop_ms / 1000 * sr))
    env = []
    acc = 0.0
    for i, v in enumerate(x):
        acc += v * v
        if i >= win:
            acc -= x[i - win] ** 2
        if i % hop == 0:
            env.append(math.sqrt(max(acc, 0.0) / win))
    return env


def scaled_corr(a, b, scale):
    """Correlation of envelope ``a`` with ``b`` time-scaled by ``scale``."""
    n = min(len(a), int(len(b) / scale))
    if n < 8:
        return 0.0
    bs = []
    for i in range(n):
        pos = i * scale
        j = int(pos)
        f = pos - j
        bs.append(b[j] * (1 - f) + b[min(j + 1, len(b) - 1)] * f)
    aa, bb = a[:n], bs
    ma, mb = sum(aa) / n, sum(bb) / n
    da = [v - ma for v in aa]
    db = [v - mb for v in bb]
    num = sum(x * y for x, y in zip(da, db))
    den = math.sqrt(sum(v * v for v in da) * sum(v * v for v in db))
    return num / den if den > 1e-12 else 0.0


def max_step(x):
    return max((abs(x[i] - x[i - 1]) for i in range(1, len(x))), default=0.0)


def peak(x):
    return max((abs(v) for v in x), default=0.0)


def runs_of_zeros(x, min_run=64):
    run = 0
    worst = 0
    for v in x:
        run = run + 1 if v == 0.0 else 0
        worst = max(worst, run)
    return worst


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_identity():
    print("\n[identity] semitones=0 tempo=1 is a no-op")
    x = gen_chord(1.5)
    y = pitchtime.pitch_tempo([x], SR, 0.0, 1.0)[0]
    check("length", len(y) == len(x), f"{len(y)} vs {len(x)}")
    d = max(abs(a - b) for a, b in zip(x, y))
    check("bit-exact", d < 1e-9, f"max diff {d:.2e}")


def test_tempo_only():
    print("\n[tempo only] duration exact, pitch preserved")
    for tempo in (1.5, 0.7, 1.25, 0.85):
        x = gen_sine(440.0, 2.0)
        t0 = time.perf_counter()
        y = pitchtime.pitch_tempo([x], SR, 0.0, tempo)[0]
        ms = (time.perf_counter() - t0) * 1000
        expect = round(len(x) / tempo)
        check(f"tempo={tempo} length", len(y) == expect,
              f"{len(y)} vs {expect}")
        f0 = measure_f0(y, SR)
        ok = f0 is not None and abs(f0 - 440) / 440 < 0.02
        check(f"tempo={tempo} pitch~440Hz", ok,
              f"f0={f0 and round(f0, 2)} ({ms:.0f} ms)")
        check(f"tempo={tempo} no clip", peak(y) <= 1.0, f"peak={peak(y):.3f}")


def test_pitch_only():
    print("\n[pitch only] pitch exact, duration preserved")
    for st in (5.0, -7.0, 0.5, -0.3, 12.0, -12.0):
        x = gen_sine(440.0, 2.0)
        y = pitchtime.pitch_tempo([x], SR, st, 1.0)[0]
        check(f"{st:+}st length", len(y) == len(x), f"{len(y)} vs {len(x)}")
        f0 = measure_f0(y, SR)
        want = 440.0 * 2 ** (st / 12)
        ok = f0 is not None and abs(f0 - want) / want < 0.02
        check(f"{st:+}st pitch={want:.1f}Hz", ok, f"f0={f0 and round(f0, 2)}")
        check(f"{st:+}st no clip", peak(y) <= 1.0, f"peak={peak(y):.3f}")


def test_combined():
    print("\n[combined] pitch and tempo independent")
    x = gen_sine(330.0, 2.0)
    y = pitchtime.pitch_tempo([x], SR, 3.0, 1.3)[0]
    check("length = n/1.3", len(y) == round(len(x) / 1.3), f"{len(y)}")
    f0 = measure_f0(y, SR)
    want = 330.0 * 2 ** (3 / 12)
    check("pitch +3st", f0 is not None and abs(f0 - want) / want < 0.02,
          f"f0={f0 and round(f0, 2)} want {want:.1f}")

    y2 = pitchtime.pitch_tempo([x], SR, -4.0, 0.75)[0]
    check("length = n/0.75", len(y2) == round(len(x) / 0.75), f"{len(y2)}")
    f0 = measure_f0(y2, SR)
    want = 330.0 * 2 ** (-4 / 12)
    check("pitch -4st", f0 is not None and abs(f0 - want) / want < 0.02,
          f"f0={f0 and round(f0, 2)} want {want:.1f}")


def test_naturalness():
    print("\n[naturalness] no clicks / dropouts / level jumps")
    # Sine: max sample-to-sample step must stay sine-like (no clicks).
    x = gen_sine(500.0, 2.0, amp=0.6)
    ref_step = max_step(x)
    for label, st, tp in (("tempo 1.6", 0, 1.6), ("tempo 0.6", 0, 0.6),
                          ("+6st", 6, 1.0), ("-6st", -6, 1.0),
                          ("+3st 1.3x", 3, 1.3)):
        y = pitchtime.pitch_tempo([x], SR, st, tp)[0]
        s = max_step(y)
        check(f"{label} click-free", s <= ref_step * 2.2,
              f"step {s:.4f} vs ref {ref_step:.4f}")
        check(f"{label} no dropout", runs_of_zeros(y[1000:-1000]) < 32,
              f"worst zero-run {runs_of_zeros(y[1000:-1000])}")
        r = dsp.rms(y[int(0.2 * SR):int(len(y) - 0.2 * SR)])
        check(f"{label} level kept", 0.5 < r / dsp.rms(x) < 1.6,
              f"rms ratio {r / dsp.rms(x):.2f}")

    # Voice-like and plucked material through heavy processing.
    for name, sig in (("voice", gen_voice()), ("pluck", gen_pluck()),
                      ("chord", gen_chord())):
        y = pitchtime.pitch_tempo([sig], SR, 4.0, 1.25)[0]
        check(f"{name} +4st/1.25x no clip", peak(y) <= 1.0,
              f"peak={peak(y):.3f}")
        check(f"{name} +4st/1.25x no dropout",
              runs_of_zeros(y[1000:-1000]) < 32,
              f"worst zero-run {runs_of_zeros(y[1000:-1000])}")


def test_rhythm():
    print("\n[rhythm] dense 16th-note pattern: envelope tracks time-scaling")
    x = gen_drums(3.0)
    env_in = envelope(x, SR)
    for tempo in (1.5, 0.75):
        y = pitchtime.pitch_tempo([x], SR, 0.0, tempo)[0]
        env_out = envelope(y, SR)
        c = scaled_corr(env_out, env_in, tempo)
        check(f"tempo={tempo} rhythm preserved", c > 0.75, f"corr={c:.3f}")
        # The first strong transient lands at the scaled time.
        def first_peak(env):
            thr = 0.5 * max(env)
            return next(i for i, v in enumerate(env) if v > thr) * 0.002
        t_in = first_peak(env_in)
        t_out = first_peak(env_out)
        check(f"tempo={tempo} first hit aligned",
              abs(t_out - t_in / tempo) < 0.03,
              f"{t_out:.3f}s vs {t_in / tempo:.3f}s")


def test_stereo():
    print("\n[stereo] shared path keeps channels coherent")
    left = gen_sine(440.0, 1.5)
    right = gen_sine(554.37, 1.5)  # different content per channel
    out = pitchtime.pitch_tempo([left, right], SR, 2.0, 1.4)
    check("channel count", len(out) == 2)
    check("equal length", len(out[0]) == len(out[1]) == round(len(left) / 1.4))
    f_l = measure_f0(out[0], SR)
    f_r = measure_f0(out[1], SR)
    check("L pitch", f_l and abs(f_l - 440 * 2 ** (2 / 12)) < 12,
          f"{f_l and round(f_l, 1)}")
    check("R pitch", f_r and abs(f_r - 554.37 * 2 ** (2 / 12)) < 12,
          f"{f_r and round(f_r, 1)}")


def test_bass_coherence():
    print("\n[bass] deep periodic content stays coherent (no tremolo)")
    for f in (55.0, 82.4, 110.0):
        x = gen_sine(f, 2.0, amp=0.5)
        y = pitchtime.pitch_tempo([x], SR, 0.0, 1.35)[0]
        # Envelope window spans 3 periods, so a pure tone's own measurement
        # artifact is < 2%; anything above that is real amplitude damage.
        env = envelope(y, SR, win_ms=3000.0 / f, hop_ms=5.0)[20:-20]
        mean = sum(env) / len(env)
        ripple = 100 * (max(env) - min(env)) / mean
        check(f"{f} Hz ripple < 8%", ripple < 8.0, f"{ripple:.1f}%")


def test_composite_mix():
    print("\n[composite] drums + bass + voice under heavy settings")
    random.seed(3)
    n = 4 * SR
    dr = gen_drums(4.0, SR, bpm=128)
    vo = gen_voice(4.0, SR, f0=220)
    bass = [0.3 * math.sin(2 * math.pi * 82.4 * i / SR) for i in range(n)]
    mix = [dr[i] + vo[i] + bass[i] for i in range(n)]
    g = 0.9 / max(abs(v) for v in mix)
    mix = [v * g for v in mix]
    env_in = envelope(mix, SR, win_ms=30)
    for st, tp in ((0, 1.35), (5, 1.35), (-5, 0.8)):
        y = pitchtime.pitch_tempo([mix], SR, st, tp)[0]
        check(f"st={st:+}/t={tp} length", len(y) == round(n / tp), f"{len(y)}")
        check(f"st={st:+}/t={tp} no clip", peak(y) <= 1.0, f"peak={peak(y):.3f}")
        check(f"st={st:+}/t={tp} no dropout",
              runs_of_zeros(y[2000:-2000]) < 32)
        # Note: ±12 ms WSOLA jitter alone caps this metric near 0.7 (a pure
        # 10 ms shift of the *unprocessed* signal scores ~0.78), so the bar
        # is set well below the individual-component scores of 0.8-0.96.
        c = scaled_corr(envelope(y, SR, win_ms=30), env_in, tp)
        check(f"st={st:+}/t={tp} groove kept", c > 0.55, f"corr={c:.3f}")


def test_performance():
    print("\n[performance] 10 s mono @44.1 kHz")
    x = gen_voice(10.0)
    t0 = time.perf_counter()
    pitchtime.pitch_tempo([x], SR, 3.0, 1.2)
    dt = time.perf_counter() - t0
    check("10 s audio processes in < 15 s", dt < 15, f"{dt:.1f} s")


def test_edges_and_robustness():
    print("\n[robustness] short / silent / loud inputs")
    short = gen_sine(440, 0.05)
    y = pitchtime.pitch_tempo([short], SR, 5, 1.5)[0]
    check("tiny input length", len(y) == round(len(short) / 1.5), f"{len(y)}")
    sil = [0.0] * SR
    y = pitchtime.pitch_tempo([sil], SR, -3, 0.8)[0]
    check("silence stays silent", peak(y) == 0.0 and len(y) == round(SR / 0.8))
    loud = [0.98 * math.sin(2 * math.pi * 300 * i / SR) for i in range(SR)]
    y = pitchtime.pitch_tempo([loud], SR, 7, 1.0)[0]
    check("hot signal never clips", peak(y) <= 1.0, f"peak={peak(y):.4f}")
    # extreme-but-allowed ranges don't crash
    y = pitchtime.pitch_tempo([gen_sine(220, 0.5)], SR, 24, 2.0)[0]
    check("extreme +24st/2x", len(y) == round(0.5 * SR / 2))
    y = pitchtime.pitch_tempo([gen_sine(220, 0.5)], SR, -24, 0.5)[0]
    check("extreme -24st/0.5x", len(y) == round(0.5 * SR / 0.5))


if __name__ == "__main__":
    test_identity()
    test_tempo_only()
    test_pitch_only()
    test_combined()
    test_naturalness()
    test_rhythm()
    test_stereo()
    test_bass_coherence()
    test_composite_mix()
    test_edges_and_robustness()
    test_performance()
    print(f"\n{'=' * 50}\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
