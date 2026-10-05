"""
transform.py — Independent pitch shifting and time scaling.

The two operations are *independent*: either one may be changed without
affecting the other, or both may be applied at once.

  * speed change only  -> duration changes, pitch stays put
  * pitch change only  -> pitch changes, duration stays put
  * both               -> duration becomes ``1/speed`` and pitch becomes
                          ``2 ** (semitones / 12)`` regardless of the other

Like the rest of the system this is implemented from first principles in pure
Python (standard library plus :mod:`backend.dsp`).

Signal chain
------------
Both effects are produced by cascading two primitives:

  1. A **phase vocoder** stretches / compresses the time axis while leaving
     spectral content (hence pitch) untouched.  It uses 75 %-overlapped Hann
     windows, the classic phase-unwrapping recurrence of Flanagan–Golden, and
     *peak phase locking* (Laroche–Dolson): the phase of every spectral bin is
     locked to the strongest local-magnitude peak in its region, which keeps
     harmonics coherent and removes the phasy / reverberant flavour a naive
     phase vocoder gives on voices and dense mixes.
  2. A **windowed-sinc resampler** then plays the stretched stream at a
     different rate, which shifts pitch.  A Kaiser-windowed sinc kernel
     (band-limited interpolation, 16–24 zero-crossings, a 256-phase polyphase
     table) is used instead of linear interpolation, so transients stay crisp
     and there is neither aliasing whistling nor the muffled quality of cheap
     interpolation.

Pitch shift = time-stretch by ``pitch_ratio`` and then resample by
``1 / pitch_ratio``; speed change = time-stretch by ``1 / speed`` only.  With
both controls the stretch factor is ``pitch_ratio / speed``, so the final
duration is always ``original / speed`` and pitch is exactly
``2 ** (semitones / 12)``.

Everything streams channel-by-channel in bounded memory, so multi-minute
stereo files can be processed without loading them whole.
"""

from __future__ import annotations

import io
import math
import struct
from typing import Dict, List, Optional, Sequence, Tuple

from . import audio_io, dsp

# --------------------------------------------------------------------------- #
# Parameter ranges and quality presets
# --------------------------------------------------------------------------- #

MIN_SPEED = 0.25
MAX_SPEED = 4.0
MIN_SEMITONES = -12.0
MAX_SEMITONES = 12.0

# quality name -> (FFT/window size, sinc zero-crossings per side, Kaiser beta)
QUALITIES: Dict[str, Tuple[int, int, float]] = {
    "draft":    (1024,  8, 10.0),
    "balanced": (2048, 16, 14.0),
    "high":     (4096, 24, 15.0),
}

_TWO_PI = 2.0 * math.pi


def semitones_to_ratio(semitones: float) -> float:
    """Pitch ratio for a shift given in semitones (fractional allowed)."""
    return 2.0 ** (semitones / 12.0)


def _edge_fades(out: List[float], sr: int, fade_in_s: float = 0.003,
                fade_out_s: float = 0.005) -> None:
    """Apply short raised-cosine edge fades in place.

    Pure pitch/speed processing is otherwise perfectly transparent at the
    boundaries (zero-padded analysis/synthesis), but these few milliseconds
    guarantee no click regardless of the source's sample value at t = 0.
    """
    n = len(out)
    if n < 4:
        return
    fi = min(n // 2, int(fade_in_s * sr))
    fo = min(n // 2, int(fade_out_s * sr))
    for i in range(fi):
        g = 0.5 - 0.5 * math.cos(math.pi * i / fi)
        out[i] *= g
    for i in range(fo):
        g = 0.5 - 0.5 * math.cos(math.pi * i / fo)
        out[n - 1 - i] *= g


# --------------------------------------------------------------------------- #
# Kaiser-windowed sinc resampler (band-limited interpolation)
# --------------------------------------------------------------------------- #

def _bessel_i0(x: float) -> float:
    """Order-zero modified Bessel function (series), for the Kaiser window."""
    xx = x * x / 4.0
    s = 1.0
    term = 1.0
    for m in range(1, 60):
        term *= xx / (m * m)
        s += term
        if term < 1e-18 * s:
            break
    return s


class SincResampler:
    """Streaming arbitrary-ratio resampler with a Kaiser-windowed sinc kernel.

    ``ratio`` is the source-position increment per output sample, i.e.
    ``output[j] = input(j * ratio)``.  ``ratio > 1`` produces a shorter stream
    (played back at the same rate: faster and higher), ``ratio < 1`` a longer
    one (slower and lower).  A dense polyphase table of the interpolation
    kernel keeps the per-sample cost low while remaining effectively
    alias-free.
    """

    _PHASES = 256  # fractional-phase resolution of the kernel table

    def __init__(self, ratio: float, zero_crossings: int = 16, kaiser_beta: float = 14.0):
        if ratio <= 0:
            raise ValueError("resample ratio must be positive")
        self.ratio = ratio
        self.z = max(4, int(zero_crossings))
        self.beta = float(kaiser_beta)
        self.pos = 0.0          # current source position
        # Leading zero run so the interpolation kernel starts fully "before"
        # the signal: output warm-up samples are genuine zeros rather than
        # edge-clamped copies of the first sample (avoids a start click).
        self.buf: List[float] = [0.0] * self.z
        self._table = self._build_table()

    def _kernel(self, arg: float, i0_arg: float) -> float:
        """sinc(arg) * Kaiser(|arg|) for |arg| < z (0 outside)."""
        d = abs(arg)
        if d >= self.z:
            return 0.0
        if d < 1e-12:
            return 1.0
        # Kaiser window over [-z, z].
        r = d / self.z
        w = _bessel_i0(self.beta * math.sqrt(max(0.0, 1.0 - r * r))) / i0_arg
        return math.sin(math.pi * arg) / (math.pi * arg) * w

    def _build_table(self) -> List[List[float]]:
        """Tabulate kernel taps for a fine grid of sub-sample phases.

        Row ``ph`` holds the 2*z taps for fractional position ``ph/PHASES``;
        an extra row for phase 1.0 is appended so adjacent rows can be linearly
        interpolated without special-casing the wrap-around.
        """
        i0_arg = _bessel_i0(self.beta)
        z = self.z
        table: List[List[float]] = []
        for ph in range(self._PHASES + 1):
            frac = ph / self._PHASES
            row = [self._kernel(t - frac, i0_arg) for t in range(-z, z)]
            # Normalise DC gain to exactly 1 (cures window-truncation ripple).
            gain = sum(row)
            if gain != 0.0:
                row = [v / gain for v in row]
            table.append(row)
        return table

    def push(self, block: Sequence[float]) -> None:
        self.buf.extend(block)

    def _emit(self, allow_edge_clamp: bool) -> List[float]:
        out: List[float] = []
        buf = self.buf
        n = len(buf)
        z = self.z
        ratio = self.ratio
        table = self._table
        phases = self._PHASES
        nz = len(table[0])
        pos = self.pos
        if allow_edge_clamp:
            # Keep producing while the kernel window intersects the buffer;
            # out-of-range tap indices are clamped to the end sample.
            while int(pos) - z + 1 < n:
                n0 = int(pos)
                phf = (pos - n0) * phases
                ph0 = int(phf)
                if ph0 >= phases:
                    ph0 = phases - 1
                f = phf - ph0
                r0 = table[ph0]
                r1 = table[ph0 + 1]
                s = 0.0
                base = n0 - z
                for tt in range(nz):
                    idx = base + tt
                    if idx < 0:
                        idx = 0
                    elif idx >= n:
                        idx = n - 1
                    k = r0[tt] + (r1[tt] - r0[tt]) * f
                    s += buf[idx] * k
                out.append(s)
                pos += ratio
        else:
            # Interior only: every tap index must be a valid, present sample.
            while int(pos) + z <= n:
                n0 = int(pos)
                phf = (pos - n0) * phases
                ph0 = int(phf)
                if ph0 >= phases:
                    ph0 = phases - 1
                f = phf - ph0
                r0 = table[ph0]
                r1 = table[ph0 + 1]
                s = 0.0
                base = n0 - z
                for tt in range(nz):
                    k = r0[tt] + (r1[tt] - r0[tt]) * f
                    s += buf[base + tt] * k
                out.append(s)
                pos += ratio
        consumed = int(pos)
        if consumed > 0:
            del buf[:consumed]
            pos -= consumed
        self.pos = pos
        return out

    def pull(self) -> List[float]:
        """Return all interior samples (kernel fully inside the buffer)."""
        return self._emit(False)

    def flush(self) -> List[float]:
        """Produce the remaining tail samples (edge clamped)."""
        return self._emit(True)


# --------------------------------------------------------------------------- #
# Phase-vocoder time stretcher (streaming, with peak phase locking)
# --------------------------------------------------------------------------- #

class PhaseVocoder:
    """Streaming phase-vocoder time-scale modifier.

    ``ratio`` is the output/input *duration* factor: ``1.5`` is 50 % longer
    (slower), ``0.5`` is half as long (faster).  Pitch is preserved.

    Analysis positions advance by a (possibly fractional) hop so any continuous
    ratio is supported with exact duration control; synthesis uses a fixed
    integer hop with weighted overlap-add.  Fractional hops beyond the window
    (heavy compression) automatically shrink the synthesis hop so consecutive
    analysis frames always overlap by at least 50 %.
    """

    def __init__(self, ratio: float, nfft: int = 2048):
        if ratio <= 0:
            raise ValueError("stretch ratio must be positive")
        self.N = nfft
        self.ratio = ratio
        N = nfft
        # Periodic Hann window.
        self.win = [0.5 - 0.5 * math.cos(_TWO_PI * k / N) for k in range(N)]

        # Choose the synthesis hop so the analysis hop never exceeds N/2
        # (>=50 % analysis overlap even when compressing hard).
        hs = min(N // 4, int(round(0.5 * ratio * N)))
        hs = max(32, hs)
        self.hs = hs
        self.ha = hs / ratio          # fractional analysis hop
        self.ap = 0.0                 # next absolute analysis-frame position
        self.base = 0                 # absolute index of inbuf[0]
        self.inbuf: List[float] = [0.0] * (N // 2)  # leading (centred) pad

        # Overlap-add sliding buffers.
        self.ola = [0.0] * N
        self.norm = [0.0] * N

        # Per-bin phase state (positive frequencies only).
        half = N // 2
        self.prev_phase: Optional[List[float]] = None
        self.synth_phase: List[float] = [0.0] * (half + 1)
        self._first = True

        self._finished = False

    # -- input side --------------------------------------------------------- #

    def push(self, block: Sequence[float]) -> None:
        self.inbuf.extend(block)

    def _can_frame(self) -> bool:
        # Need samples at absolute indices floor(ap) .. floor(ap)+N inclusive
        # (the +1 backs the linear interpolation of the windowed segment).
        return int(self.ap) + self.N + 1 <= self.base + len(self.inbuf)

    def _trim_buffer(self) -> None:
        keep_from = int(self.ap)
        drop = keep_from - self.base
        if drop > self.N * 4 and drop > 0:
            del self.inbuf[:drop]
            self.base = keep_from

    def _analysis_frame(self) -> List[float]:
        """Windowed, linearly interpolated segment at the fractional hop."""
        N = self.N
        win = self.win
        buf = self.inbuf
        base = self.base
        ap = self.ap
        seg = [0.0] * N
        i0 = int(ap) - base
        pos = ap - int(ap)
        idx = i0
        for j in range(N):
            v0 = buf[idx]
            v1 = buf[idx + 1]
            seg[j] = (v0 + (v1 - v0) * pos) * win[j]
            pos += 1.0
            if pos >= 1.0:
                pos -= 1.0
                idx += 1
        return seg

    # -- phase processing --------------------------------------------------- #

    def _synthesize_frame(self, seg: Sequence[float]) -> List[float]:
        N = self.N
        half = N // 2
        spec = dsp.fft(seg)
        mag = [abs(spec[k]) for k in range(half + 1)]
        phase = [math.atan2(spec[k].imag, spec[k].real) for k in range(half + 1)]

        if self._first:
            # Starting from the true analysis phase avoids attacking through a
            # phase ramp-up; frame zero is reconstructed unchanged.
            self.prev_phase = list(phase)
            self.synth_phase = list(phase)
            self._first = False
            td = dsp.ifft(spec)
            return [v.real for v in td]

        prev = self.prev_phase
        sp = self.synth_phase
        ha = self.ha
        hs = self.hs
        new_phase = [0.0] * (half + 1)

        # Phase-unwrapped frequency-channel integration for every bin.
        for k in range(1, half):
            omega = _TWO_PI * k / N
            dphi = phase[k] - prev[k] - omega * ha
            dphi -= _TWO_PI * round(dphi / _TWO_PI)
            true_freq = omega + dphi / ha
            sp[k] = sp[k] + true_freq * hs
            new_phase[k] = sp[k]
        # DC / Nyquist stay real.
        new_phase[0] = 0.0
        new_phase[half] = 0.0

        # Peak phase locking (Laroche–Dolson): locate magnitude peaks and make
        # every bin rotate by the phase advance of its owning peak, preserving
        # the bin's own phase offset relative to that peak.
        is_peak = [False] * (half + 1)
        is_peak[0] = True
        is_peak[half] = True
        for k in range(1, half):
            if mag[k] >= mag[k - 1] and mag[k] >= mag[k + 1]:
                is_peak[k] = True
        # Nearest peak to the left / right via two sweeps.
        owner = [0] * (half + 1)
        last = 0
        for k in range(half + 1):
            if is_peak[k]:
                last = k
            owner[k] = last
        nxt = half
        for k in range(half, -1, -1):
            if is_peak[k]:
                nxt = k
            elif k - owner[k] > nxt - k:
                owner[k] = nxt

        locked = [0.0] * (half + 1)
        locked[0] = phase[0]
        locked[half] = phase[half]
        for k in range(1, half):
            p = owner[k]
            if p == k:
                locked[k] = sp[k]
            else:
                diff = sp[p] - phase[p]
                locked[k] = phase[k] + diff

        # Build a strictly Hermitian-symmetric spectrum -> real IFFT.
        Y = [0j] * N
        Y[0] = mag[0] + 0j
        Y[half] = mag[half] + 0j
        for k in range(1, half):
            re = mag[k] * math.cos(locked[k])
            im = mag[k] * math.sin(locked[k])
            Y[k] = complex(re, im)
            Y[N - k] = complex(re, -im)

        self.prev_phase = phase
        td = dsp.ifft(Y)
        return [v.real for v in td]

    # -- output side -------------------------------------------------------- #

    def _process_ready_frames(self) -> List[float]:
        N = self.N
        win = self.win
        out: List[float] = []
        while self._can_frame():
            seg = self._analysis_frame()
            td = self._synthesize_frame(seg)
            hs = self.hs
            ola = self.ola
            norm = self.norm
            for j in range(N):
                wv = td[j] * win[j]
                ola[j] += wv
                norm[j] += win[j] * win[j]
            # Band [0, hs) will never be touched by a later frame -> finalise.
            band = [0.0] * hs
            for j in range(hs):
                d = norm[j]
                band[j] = ola[j] / d if d > 1e-12 else 0.0
            out.extend(band)
            del ola[:hs]; ola.extend([0.0] * hs)
            del norm[:hs]; norm.extend([0.0] * hs)
            self.ap += self.ha
        self._trim_buffer()
        return out

    def pull(self) -> List[float]:
        """Return every stretched sample currently derivable."""
        return self._process_ready_frames()

    def finish(self) -> List[float]:
        """Append trailing silence, drain every frame and flush the OLA tail."""
        if self._finished:
            return []
        self._finished = True
        self.inbuf.extend([0.0] * self.N)
        out = self._process_ready_frames()
        # Flush the partial-overlap remainder of the overlap-add buffers.
        ola = self.ola
        norm = self.norm
        hs = self.hs
        i = 0
        while i < self.N:
            jmax = min(hs, self.N - i)
            for j in range(jmax):
                d = norm[i + j]
                if d > 1e-12:
                    out.append(ola[i + j] / d)
                else:
                    out.append(0.0)
            i += jmax
        return out


# --------------------------------------------------------------------------- #
# Whole-signal driver
# --------------------------------------------------------------------------- #

def _process_channels(channels: Sequence[Sequence[float]], sr: int,
                      speed: float, semitones: float,
                      quality: str = "balanced") -> List[List[float]]:
    """Transform in-memory de-interleaved channels (used by previews/tests)."""
    n_in = min((len(c) for c in channels), default=0)
    if n_in == 0:
        return [[] for _ in channels]

    nfft, zero_x, beta = QUALITIES.get(quality, QUALITIES["balanced"])
    pitch_ratio = semitones_to_ratio(semitones)
    stretch_ratio = pitch_ratio / speed          # vocoder duration factor
    resample_inc = pitch_ratio                    # playback-rate factor
    n_final = int(round(n_in / speed))
    # Leading samples to discard: the vocoder's N/2 centring delay and the
    # resampler's z-sample zero warm-up, both expressed in output samples.
    # Resampler output i reads source position i*inc, so source offset
    # (N/2 + z) starts at output index (N/2 + z) / inc.
    warmup = int(round((nfft // 2 + zero_x) / resample_inc))

    identity = abs(speed - 1.0) < 1e-9 and abs(semitones) < 1e-9

    result: List[List[float]] = []
    for ch in channels:
        data = ch[:n_in]
        if identity:
            result.append(list(data))
            continue
        pv = PhaseVocoder(stretch_ratio, nfft)
        rs = SincResampler(resample_inc, zero_x, beta)
        pv.push(data)
        stretched = pv.pull()
        rs.push(stretched)
        out = rs.pull()
        tail = pv.finish()
        if tail:
            rs.push(tail)
            out.extend(rs.pull())
        # Zero pad then edge-clamped flush pulls the final kernel window.
        rs.push([0.0] * (rs.z + 4))
        out.extend(rs.flush())
        # Remove leading centring/warm-up pads, enforce exact duration.
        out = out[warmup:]
        if len(out) >= n_final:
            out = out[:n_final]
        else:
            out.extend([0.0] * (n_final - len(out)))
        _edge_fades(out, sr)
        result.append(out)
    return result


def transform_samples(samples: Sequence[float], sr: int,
                      speed: float = 1.0, semitones: float = 0.0,
                      quality: str = "balanced") -> List[float]:
    """Transform a single mono channel held in memory."""
    return _process_channels([samples], sr, speed, semitones, quality)[0]


def transform_file(src_path: str, dst_path: str, *,
                   speed: float = 1.0, semitones: float = 0.0,
                   quality: str = "balanced") -> Dict:
    """Stream a WAV file through pitch/speed transformation.

    The transformation runs per channel in bounded memory and writes a 16-bit
    PCM WAV.  Returns metadata about the render.
    """
    speed = max(MIN_SPEED, min(MAX_SPEED, float(speed)))
    semitones = max(MIN_SEMITONES, min(MAX_SEMITONES, float(semitones)))
    if quality not in QUALITIES:
        quality = "balanced"

    nfft, zero_x, beta = QUALITIES[quality]
    pitch_ratio = semitones_to_ratio(semitones)
    stretch_ratio = pitch_ratio / speed
    resample_inc = pitch_ratio

    with audio_io.WavReader(src_path) as r:
        sr, ch, n_in = r.sr, r.channels, r.nframes
        n_final = int(round(n_in / speed))
        identity = abs(speed - 1.0) < 1e-9 and abs(semitones) < 1e-9

        if identity:
            # Nothing to do — stream straight through byte-for-byte samples.
            with audio_io.WavWriter(dst_path, sr, ch, 2) as w:
                for chunk in r.iter_chunks():
                    w.write_chunk(chunk)
            return _meta(sr, ch, n_in, n_in, speed, semitones, pitch_ratio, quality)

        vocoders = [PhaseVocoder(stretch_ratio, nfft) for _ in range(ch)]
        resamplers = [SincResampler(resample_inc, zero_x, beta) for _ in range(ch)]
        # Discard leading pad + resampler warm-up samples on every channel.
        warmup = int(round((nfft // 2 + zero_x) / resample_inc))
        discard = [warmup] * ch
        written = [0] * ch
        faded_in = [False] * ch
        fi_n = min(n_final // 2, int(0.003 * sr))
        fo_n = min(n_final // 2, int(0.005 * sr))

        def feed(block_per_ch: Sequence[Sequence[float]], finish: bool):
            """Run one streaming round; write whatever output becomes final."""
            chunks_out: List[List[float]] = []
            counts = []
            for c in range(ch):
                pv, rs = vocoders[c], resamplers[c]
                if block_per_ch is not None:
                    pv.push(block_per_ch[c])
                if finish:
                    tail = pv.finish()
                else:
                    tail = pv.pull()
                if tail:
                    rs.push(tail)
                    if finish:
                        rs.push([0.0] * (rs.z + 4))
                        out = rs.pull()
                        out.extend(rs.flush())
                    else:
                        out = rs.pull()
                else:
                    out = []
                # Leading-pad trim.
                if discard[c] > 0:
                    if len(out) <= discard[c]:
                        discard[c] -= len(out)
                        out = []
                    else:
                        out = out[discard[c]:]
                        discard[c] = 0
                # Raised-cosine fade-in on the first audible samples.
                if out and not faded_in[c] and fi_n > 1:
                    faded_in[c] = True
                    for i in range(min(fi_n, len(out))):
                        out[i] *= 0.5 - 0.5 * math.cos(math.pi * i / fi_n)
                # Truncate to the exact planned length only on the final round;
                # intermediate blocks are written whole so channels stay in
                # lock-step for their entire duration.
                budget = n_final - written[c]
                if finish and len(out) > budget:
                    out = out[:budget]
                if finish and fo_n > 1 and out:
                    # Fade out the tail of the very final block.
                    for i in range(fo_n):
                        out[len(out) - 1 - i] *= (
                            0.5 - 0.5 * math.cos(math.pi * i / fo_n))
                chunks_out.append(out)
                counts.append(len(out))
                written[c] += len(out)
            # Every channel is driven by equal-length inputs with identical
            # hop/ratio math, so their output counts match frame for frame.
            m = min(counts)
            if m:
                w.write_chunk([c_out[:m] for c_out in chunks_out])

        with audio_io.WavWriter(dst_path, sr, ch, 2) as w:
            for chunk in r.iter_chunks():
                feed(chunk, False)
            feed(None, True)
            # Exact-length guarantee (pad the odd sample if rounding was short).
            short = [max(0, n_final - written[c]) for c in range(ch)]
            if any(short):
                w.write_chunk([[0.0] * s for s in short])

    return _meta(sr, ch, n_in, n_final, speed, semitones, pitch_ratio, quality)


def _meta(sr: int, ch: int, n_in: int, n_out: int,
          speed: float, semitones: float, pitch_ratio: float, quality: str) -> Dict:
    return {
        "sr": sr,
        "channels": ch,
        "frames_in": n_in,
        "frames": n_out,
        "duration_in": n_in / sr if sr else 0.0,
        "duration": n_out / sr if sr else 0.0,
        "duration_factor": (n_out / n_in) if n_in else 1.0,
        "speed": speed,
        "semitones": semitones,
        "pitch_ratio": pitch_ratio,
        "quality": quality,
    }


# --------------------------------------------------------------------------- #
# Preview rendering (short in-memory WAV)
# --------------------------------------------------------------------------- #

def render_preview(src_path: str, speed: float = 1.0, semitones: float = 0.0,
                   quality: str = "balanced", start_s: float = 0.0,
                   duration_s: float = 6.0) -> bytes:
    """Render an excerpt through the transform and return WAV bytes."""
    duration_s = max(0.5, min(12.0, float(duration_s)))
    with audio_io.WavReader(src_path) as r:
        sr = r.sr
        start = max(0, int(start_s * sr))
        n = int(duration_s * sr)
        r._w.setpos(start)
        chunk = r.read_chunk(n)
        if chunk is None:
            chunk = [[] for _ in range(r.channels)]
        channels = chunk
        out_ch = _process_channels(channels, sr, speed, semitones, quality)

    nframes = min(len(c) for c in out_ch) if out_ch else 0
    channels_n = len(out_ch)
    data = bytearray()
    for i in range(nframes):
        for c in out_ch:
            v = int(round(max(-1.0, min(1.0, c[i])) * 32767))
            data += struct.pack("<h", v)
    buf = io.BytesIO()
    byte_rate = sr * channels_n * 2
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(data)))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, channels_n, sr, byte_rate, channels_n * 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(data)))
    buf.write(bytes(data))
    return buf.getvalue()
