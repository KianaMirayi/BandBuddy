#!/usr/bin/env python3
from __future__ import annotations

import argparse
from array import array
from pathlib import Path
import wave


def read(path: Path) -> tuple[wave._wave_params, bytes]:
    with wave.open(str(path), "rb") as source:
        return source.getparams(), source.readframes(source.getnframes())


def write(path: Path, params: wave._wave_params, frames: bytes, gain: float) -> None:
    samples = array("h")
    samples.frombytes(frames)
    scaled = array("h", (max(-32768, min(32767, round(value * gain))) for value in samples))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as destination:
        destination.setparams(params)
        destination.writeframes(scaled.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    mss = sub.add_parser("mss")
    mss.add_argument("--input", type=Path, required=True)
    mss.add_argument("--output", type=Path, required=True)
    msr = sub.add_parser("msr")
    msr.add_argument("--input", type=Path, required=True)
    msr.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    params, frames = read(args.input)
    if args.command == "mss":
        for stem, gain in (("vocals", 0.5), ("drums", 0.25), ("saxophone", 0.125)):
            write(args.output / f"{stem}.wav", params, frames, gain)
    else:
        write(args.output, params, frames, 0.8)


if __name__ == "__main__":
    main()

