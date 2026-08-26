"""Dependency-free WAV diagnostics used by the MVP manifest.

These diagnostics deliberately do not claim that a no-reference output is
"cleaner".  They catch clipping/level regressions and quantify how strongly an
MSR stage changed its MSS input; listening or clean references remain necessary.
"""

from __future__ import annotations

from array import array
import math
from pathlib import Path
import sys
import wave
from typing import Any


def _samples(path: Path) -> tuple[array, int, int, int] | None:
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as source:
            width = source.getsampwidth()
            if width not in (1, 2, 3, 4) or source.getcomptype() != "NONE":
                return None
            channels = source.getnchannels()
            rate = source.getframerate()
            data = source.readframes(source.getnframes())
    except (OSError, wave.Error):
        return None
    if width == 1:
        values = array("B", data)
        values = array("i", (value - 128 for value in values))
    elif width == 2:
        values = array("h")
        values.frombytes(data)
    elif width == 3:
        values = array("i", (
            int.from_bytes(data[index:index + 3], "little", signed=True)
            for index in range(0, len(data) - 2, 3)
        ))
    else:
        values = array("i")
        values.frombytes(data)
    if sys.byteorder != "little" and width > 1:
        values.byteswap()
    return values, channels, rate, width


def inspect_wav(path: Path) -> dict[str, Any] | None:
    loaded = _samples(path)
    if loaded is None:
        return None
    samples, channels, rate, width = loaded
    if not samples:
        return None
    scale = float(2 ** (width * 8 - 1))
    peak = max(abs(value) for value in samples) / scale
    mean = sum(samples) / len(samples) / scale
    rms = math.sqrt(sum(float(value) * value for value in samples) / len(samples)) / scale
    clipped = sum(1 for value in samples if abs(value) >= scale - 1) / len(samples)
    return {
        "sampleRate": rate,
        "channels": channels,
        "durationSeconds": round(len(samples) / channels / rate, 6),
        "peakDbfs": round(20 * math.log10(max(peak, 1e-12)), 3),
        "rmsDbfs": round(20 * math.log10(max(rms, 1e-12)), 3),
        "dcOffset": round(mean, 8),
        "clippedFraction": round(clipped, 8),
    }


def compare_wav_files(before: Path, after: Path) -> dict[str, float] | None:
    left = _samples(before)
    right = _samples(after)
    if left is None or right is None or left[1:] != right[1:]:
        return None
    a, _, _, _ = left
    b, _, _, _ = right
    length = min(len(a), len(b))
    if length == 0:
        return None
    dot = sum(float(a[i]) * b[i] for i in range(length))
    aa = sum(float(a[i]) * a[i] for i in range(length))
    bb = sum(float(b[i]) * b[i] for i in range(length))
    correlation = dot / math.sqrt(max(aa * bb, 1e-24))
    gain = dot / max(aa, 1e-24)
    error = sum((b[i] - gain * a[i]) ** 2 for i in range(length))
    change_snr = 10 * math.log10(max(gain * gain * aa, 1e-24) / max(error, 1e-24))
    rms_a = math.sqrt(aa / length)
    rms_b = math.sqrt(bb / length)
    return {
        "correlation": round(correlation, 6),
        "scaleInvariantChangeSnrDb": round(change_snr, 3),
        "rmsDeltaDb": round(20 * math.log10(max(rms_b, 1e-12) / max(rms_a, 1e-12)), 3),
    }
