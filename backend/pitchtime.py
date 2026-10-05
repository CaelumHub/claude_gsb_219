"""
pitchtime.py — Independent pitch-shift (变调) and time-stretch (变速).

Two building blocks, both self-implemented in pure Python:

  1. **WSOLA time-stretch** (Waveform Similarity Overlap-Add).
     The signal is cut into overlapping frames; each output frame is placed
     at a *fixed* synthesis hop, while its *input* position is allowed to
     drift within a tolerance window around the nominal time-mapped position.
     The position that maximises the cross-correlation with the natural
     continuation of the previous frame is chosen, so the overlap-add joins
     are (nearly) phase-coherent.  Being a pure time-domain method it keeps
     transients crisp (drums, plucks, consonants) and has none of the
     "phasiness" of a phase vocoder.  The synthesis hop is exactly half the
     window and a Hann window is used, so the overlap sums to unity gain —
     no normalisation artefacts, no clicks, no level dips.

  2. **Windowed-sinc resampling** for pitch shift.
     Shifting pitch by ``s`` semitones is resampling by ``p = 2**(s/12)``
     (which also scales the duration by ``1/p``); a WSOLA stretch by ``p``
     then restores the original duration.  The resampler uses a
     Blackman-windowed sinc interpolator with 64 precomputed fractional
     phases and a ratio-dependent cutoff, so both up-sampling (pitch down)
     and down-sampling (pitch up) are alias/image free.

Combining the two: to change tempo by ``t`` (t > 1 = faster) and pitch by
``s`` semitones we resample by ``p`` and WSOLA-stretch by ``p / t`` — the
two controls are fully independent.

Stereo coherence: the WSOLA offset path is computed once on the mono mix of
the (resampled) channels and then applied identically to every channel, so
the stereo image cannot smear.

A final safety stage peak-normalises only if the result would clip, which
guarantees crackle-free (无爆音) output.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from . import audio_io, dsp

# --------------------------------------------------------------------------- #
# Tuning constants
# --------------------------------------------------------------------------- #

WIN_MS = 23.0        # WSOLA window length (ms) — good voice/music trade-off
TOL_MS = 12.0        # WSOLA search tolerance (± ms) around the nominal position
COARSE_STEP = 4      # coarse correlation search step (samples), refined to 1
SILENCE_PEAK = 1e-4  # below this template peak the search is skipped (silence)
RESAMPLE_TAPS = 16   # windowed-sinc interpolator length
RESAMPLE_PHASES = 64 # quantised fractional phases of the resampler
PEAK_LIMIT = 0.995   # output is scaled down only if it would exceed this


def _win_len(sr: float) -> int:
    """Even WSOLA window length for the sample rate (~WIN_MS milliseconds)."""
    w = int(sr * WIN_MS / 1000.0)
    return max(64, w & ~1)


def _tol_len(sr: float) -> int:
    return max(8, int(sr * TOL_MS / 1000.0))


# --------------------------------------------------------------------------- #
# 1. WSOLA time-stretch
# --------------------------------------------------------------------------- #

def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


def _best_offset(sig: Sequence[float], tmpl: Sequence[float],
                 lo: int, hi: int, corr_len: int) -> int:
    """Return the offset in [lo, hi] whose segment best matches ``tmpl``.

    Coarse-to-fine plain cross-correlation search (stride COARSE_STEP, then
    refined sample-accurately around the winner)."""
    best_off, best_val = lo, -1e300
    off = lo
    while off <= hi:
        v = _dot(sig[off:off + corr_len], tmpl)
        if v > best_val:
            best_val, best_off = v, off
        off += COARSE_STEP
    # Sample-accurate refinement around the coarse winner.
    r_lo = max(lo, best_off - COARSE_STEP + 1)
    r_hi = min(hi, best_off + COARSE_STEP - 1)
    off = r_lo
    while off <= r_hi:
        v = _dot(sig[off:off + corr_len], tmpl)
        if v > best_val:
            best_val, best_off = v, off
        off += 1
    return best_off


def wsola_path(guide: Sequence[float], sr: float, out_len: int,
               win: Optional[int] = None, tol: Optional[int] = None
               ) -> Tuple[List[Tuple[int, int]], int]:
    """Compute the WSOLA frame-placement path for a guide signal.

    Returns ``(path, win)`` where ``path`` is a list of
    ``(input_pos, output_pos)`` frame starts.  ``input_pos`` refers to the
    zero-padded signal (``win`` zeros on each side), ``output_pos`` to the
    final unpadded output timeline (may be negative for the first frame).
    """
    n = len(guide)
    W = win or _win_len(sr)
    Hs = W // 2
    tol = tol if tol is not None else _tol_len(sr)
    tol = min(tol, Hs)
    corr_len = Hs  # correlate over the overlap region

    pad = W
    inp = [0.0] * pad + list(guide) + [0.0] * pad
    limit = len(inp) - W  # largest legal frame start
    stretch = out_len / n if n else 1.0
    step_in = Hs / stretch if stretch > 0 else Hs

    # The offset search runs on a high-passed copy of the guide.  Content
    # whose period exceeds the correlation window (deep bass) cannot be
    # aligned by *any* offset within the tolerance, and its boundary-biased
    # correlation peak actively misleads the search (audible as tremolo).
    # The ideal offset is frequency-independent, so mid/high content — which
    # has many cycles per window and therefore a clean correlation peak —
    # is the reliable guide; near-stationary bass then joins coherently too.
    hp = dsp.Biquad("highpass", sr, 120.0, 0.7071)
    inp_f = hp.process_block(inp)

    # Taper for the correlation template: without it, correlating a segment
    # whose length is a non-integer number of periods produces boundary
    # wobble that biases the peak by a fraction of a period.
    corr_win = dsp.hann(corr_len)

    n_frames = int(out_len / Hs) + 2
    path: List[Tuple[int, int]] = []
    prev_in: Optional[int] = None
    for k in range(n_frames):
        natural = k * step_in + pad - Hs  # padded coords (pad == W == 2*Hs)
        lo = max(0, int(natural - tol))
        hi = min(limit, int(natural + tol))
        if prev_in is None or hi <= lo:
            pos = max(0, min(limit, int(round(natural))))
        else:
            t0 = prev_in + Hs
            tmpl = [corr_win[j] * inp_f[t0 + j] for j in range(corr_len)]
            peak = 0.0
            for v in tmpl:
                a = v if v >= 0 else -v
                if a > peak:
                    peak = a
            if peak < SILENCE_PEAK:
                pos = max(lo, min(hi, int(round(natural))))
            else:
                pos = _best_offset(inp_f, tmpl, lo, hi, corr_len)
        path.append((pos, k * Hs - Hs))
        prev_in = pos
    return path, W


def wsola_apply(samples: Sequence[float], path: Sequence[Tuple[int, int]],
                win: int, out_len: int) -> List[float]:
    """Render one channel through a precomputed WSOLA path (Hann OLA)."""
    W = win
    Hs = W // 2
    inp = [0.0] * W + list(samples) + [0.0] * W
    window = dsp.hann(W)
    out = [0.0] * out_len
    for in_pos, out_pos in path:
        j0 = out_pos if out_pos > 0 else 0
        j1 = out_pos + W
        if j1 > out_len:
            j1 = out_len
        if j1 <= j0:
            continue
        # j is the output index; window/segment index = j - out_pos.
        base = in_pos - out_pos
        o = out
        w = window
        s = inp
        for j in range(j0, j1):
            o[j] += s[base + j] * w[j - out_pos]
    return out


def time_stretch(channels: Sequence[Sequence[float]], sr: float,
                 out_len: int) -> List[List[float]]:
    """Stretch/compress every channel to exactly ``out_len`` samples.

    The offset path is derived from the mono mix and shared by all channels
    to preserve the stereo image."""
    n = len(channels[0]) if channels else 0
    if n == 0 or out_len <= 0:
        return [[] for _ in channels]
    if out_len == n:
        return [list(c) for c in channels]
    guide = audio_io.to_mono(channels)
    path, W = wsola_path(guide, sr, out_len)
    return [wsola_apply(ch, path, W, out_len) for ch in channels]


# --------------------------------------------------------------------------- #
# 2. Windowed-sinc resampler (pitch shift)
# --------------------------------------------------------------------------- #

_KERNEL_CACHE: Dict[Tuple[float, int, int], List[List[float]]] = {}


def _sinc(x: float) -> float:
    if abs(x) < 1e-9:
        return 1.0
    return math.sin(math.pi * x) / (math.pi * x)


def _resample_kernels(ratio: float, taps: int, phases: int) -> List[List[float]]:
    """Precompute normalised Blackman-windowed-sinc kernels per phase.

    The cutoff tracks the resample ratio so the same table anti-aliases
    (ratio > 1) and de-images (ratio < 1)."""
    key = (round(ratio, 6), taps, phases)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    half = taps // 2
    fc = 0.5 * min(1.0, 1.0 / ratio) * 0.92  # cycles per input sample
    kernels: List[List[float]] = []
    for q in range(phases):
        frac = q / phases
        taps_out = []
        acc = 0.0
        for m in range(taps):
            d = (m - half + 1) - frac
            # Blackman window centred on the interpolation point.
            w = 0.42 + 0.5 * math.cos(math.pi * d / half) \
                + 0.08 * math.cos(2.0 * math.pi * d / half)
            h = 2.0 * fc * _sinc(2.0 * fc * d) * w
            taps_out.append(h)
            acc += h
        inv = 1.0 / acc if abs(acc) > 1e-12 else 1.0
        kernels.append([h * inv for h in taps_out])
    _KERNEL_CACHE[key] = kernels
    return kernels


def resample_pitch(samples: Sequence[float], ratio: float,
                   taps: int = RESAMPLE_TAPS,
                   phases: int = RESAMPLE_PHASES) -> List[float]:
    """Resample by ``ratio`` (input samples advanced per output sample).

    ``ratio > 1`` shortens the signal (pitch up), ``ratio < 1`` lengthens it
    (pitch down).  Boundary samples are replicated to avoid edge clicks."""
    n = len(samples)
    if n == 0:
        return []
    if abs(ratio - 1.0) < 1e-12:
        return list(samples)
    out_len = max(1, int(round(n / ratio)))
    kernels = _resample_kernels(ratio, taps, phases)
    half = taps // 2
    src = list(samples)
    last = n - 1
    out: List[float] = []
    pos = 0.0
    for _ in range(out_len):
        c = int(pos)
        q = int((pos - c) * phases)
        if q >= phases:
            q = phases - 1
        kern = kernels[q]
        base = c - half + 1
        acc = 0.0
        for m in range(taps):
            k = base + m
            if k < 0:
                k = 0
            elif k > last:
                k = last
            acc += src[k] * kern[m]
        out.append(acc)
        pos += ratio
    return out


# --------------------------------------------------------------------------- #
# 3. Combined pitch + tempo pipeline
# --------------------------------------------------------------------------- #

def pitch_tempo(channels: Sequence[Sequence[float]], sr: float,
                semitones: float = 0.0, tempo: float = 1.0
                ) -> List[List[float]]:
    """Independently shift pitch (``semitones``) and tempo (``tempo``).

    ``tempo`` > 1 is faster (shorter), < 1 is slower (longer); ``semitones``
    may be any real number (fractional shifts allowed).  The output length is
    exactly ``round(n / tempo)`` samples regardless of the pitch shift."""
    n = len(channels[0]) if channels else 0
    if n == 0:
        return [[] for _ in channels]
    tempo = min(4.0, max(0.25, float(tempo)))
    semitones = min(24.0, max(-24.0, float(semitones)))
    p = 2.0 ** (semitones / 12.0)

    chans = [list(c) for c in channels]
    if abs(semitones) > 1e-9:
        chans = [resample_pitch(c, p) for c in chans]

    out_len = max(1, int(round(n / tempo)))
    cur_len = len(chans[0])
    if cur_len != out_len:
        chans = time_stretch(chans, sr, out_len)

    # Safety: never emit a clipped sample (prevents 爆音 on hot material).
    peak = 0.0
    for c in chans:
        for v in c:
            a = v if v >= 0 else -v
            if a > peak:
                peak = a
    if peak > PEAK_LIMIT:
        g = PEAK_LIMIT / peak
        chans = [[v * g for v in c] for c in chans]
    return chans


# --------------------------------------------------------------------------- #
# 4. File-level helpers
# --------------------------------------------------------------------------- #

def process_file(src_path: str, dst_path: str, semitones: float = 0.0,
                 tempo: float = 1.0) -> Dict:
    """Apply pitch/tempo to a whole file and write the result as 16-bit WAV."""
    audio = audio_io.load(src_path)  # raises MemoryError for very long files
    out = pitch_tempo(audio.samples, audio.sr, semitones, tempo)
    audio_io.save(dst_path, audio_io.AudioData(out, audio.sr))
    return {
        "frames": len(out[0]) if out else 0,
        "channels": len(out),
        "sr": audio.sr,
        "semitones": semitones,
        "tempo": tempo,
    }


def render_preview(src_path: str, semitones: float = 0.0, tempo: float = 1.0,
                   start_s: float = 0.0, duration_s: float = 8.0) -> bytes:
    """Process a short excerpt and return it as a 16-bit WAV byte string."""
    with audio_io.WavReader(src_path) as r:
        sr = r.sr
        start = max(0, int(start_s * sr))
        n = min(int(duration_s * sr), r.nframes - start)
        chunk = r.read_excerpt(start, max(0, n)).samples
    out = pitch_tempo(chunk, sr, semitones, tempo)
    return encode_wav_bytes(out, sr)


def encode_wav_bytes(channels: Sequence[Sequence[float]], sr: int) -> bytes:
    """Encode de-interleaved float channels as a 16-bit PCM WAV byte string."""
    import io
    import struct
    buf = io.BytesIO()
    nframes = min((len(c) for c in channels), default=0)
    nch = len(channels)
    data = bytearray()
    for i in range(nframes):
        for c in channels:
            v = int(round(max(-1.0, min(1.0, c[i])) * 32767))
            data += struct.pack("<h", v)
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(data)))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, nch, sr, sr * nch * 2, nch * 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(data)))
    buf.write(bytes(data))
    return buf.getvalue()
