#!/usr/bin/env python3
"""Dynamic-stem Demucs adapter for the MSS stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def safe_name(value: str) -> str:
    name = re.sub(r"[^a-z0-9._-]+", "_", value.strip().lower()).strip("._-")
    if not name:
        raise RuntimeError(f"INVALID_STEM_NAME:{value}")
    return name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="htdemucs_6s")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--segment", type=int, default=7)
    args = parser.parse_args()

    import torch
    from demucs.api import Separator, save_audio

    if not args.input.is_file():
        raise RuntimeError("SOURCE_FILE_MISSING")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_NOT_AVAILABLE")
    args.output.mkdir(parents=True, exist_ok=True)
    separator = Separator(
        model=args.model,
        repo=args.repo.resolve() if args.repo else None,
        device=args.device,
        shifts=1,
        overlap=0.25,
        split=True,
        segment=args.segment,
        jobs=0,
        progress=True,
    )
    _, separated = separator.separate_audio_file(args.input.resolve())
    if not separated:
        raise RuntimeError("NO_STEMS_PRODUCED")
    files: dict[str, str] = {}
    for label, audio in separated.items():
        stem = safe_name(label)
        destination = args.output / f"{stem}.wav"
        save_audio(audio.cpu(), str(destination), samplerate=separator.samplerate, bits_per_sample=24, clip="rescale")
        files[stem] = str(destination.resolve())
    print(json.dumps({"type": "result", "model": args.model, "stems": sorted(files), "files": files}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

