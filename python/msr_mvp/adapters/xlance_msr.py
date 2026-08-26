#!/usr/bin/env python3
"""Apply one or more official X-Lance checkpoints to a single MSS stem.

The adapter intentionally treats this as an experimental post-MSS use.  The
official X-Lance recipe runs denoise before separation and only applies the
released dereverb checkpoint to vocals.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Iterator


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--chunk-seconds", type=float, default=2.0)
    parser.add_argument("--overlap-seconds", type=float, default=0.25)
    args = parser.parse_args()

    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio.functional as audio_functional

    # The released model only needs librosa's mel filter builder during model
    # construction. Recreate that tiny surface with Torchaudio so the MVP does
    # not pull a second NumPy/SciPy/Numba stack into BandBuddy's runtime.
    librosa_shim = ModuleType("librosa")

    def mel_filter(*, sr: int, n_fft: int, n_mels: int = 128, fmin: float = 0.0, fmax: float | None = None, **_kwargs):
        return audio_functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=fmin,
            f_max=float(fmax if fmax is not None else sr / 2),
            n_mels=n_mels,
            sample_rate=sr,
            norm="slaney",
            mel_scale="slaney",
        ).T.numpy()

    librosa_shim.filters = SimpleNamespace(mel=mel_filter)  # type: ignore[attr-defined]
    sys.modules.setdefault("librosa", librosa_shim)

    repo = args.repo.resolve()
    if not (repo / "inference.py").is_file():
        raise RuntimeError(f"XLANCE_REPO_INVALID:{repo}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_NOT_AVAILABLE")
    if args.overlap_seconds < 0 or args.chunk_seconds <= args.overlap_seconds:
        raise RuntimeError("INVALID_CHUNK_SETTINGS")
    sys.path.insert(0, str(repo))
    with working_directory(repo):
        from inference import load_config_and_state_dict, load_generator  # type: ignore

        audio, source_rate = sf.read(args.input.resolve(), always_2d=True, dtype="float32")
        current = torch.from_numpy(audio.T.copy())
        for checkpoint_arg in args.checkpoint:
            checkpoint = checkpoint_arg.resolve()
            config, state = load_config_and_state_dict(str(checkpoint), args.device)
            model = load_generator(config, state, device=args.device)
            model_rate = int(config["data"]["sample_rate"])
            if source_rate != model_rate:
                current = audio_functional.resample(current, source_rate, model_rate)
            chunk = max(1, round(args.chunk_seconds * model_rate))
            overlap = max(0, round(args.overlap_seconds * model_rate))
            hop = chunk - overlap
            original_length = current.shape[-1]
            padded_length = max(chunk, ((max(0, original_length - overlap) + hop - 1) // hop) * hop + overlap)
            padded = torch.nn.functional.pad(current, (0, padded_length - original_length))
            accumulation = torch.zeros_like(padded)
            weights = torch.zeros_like(padded)
            fade = torch.ones(chunk)
            if overlap:
                ramp = torch.linspace(0, 1, overlap + 2)[1:-1]
                fade[:overlap] = ramp
                fade[-overlap:] = torch.flip(ramp, dims=(0,))
            for offset in range(0, padded_length - chunk + 1, hop):
                block = padded[:, offset:offset + chunk].unsqueeze(0).to(args.device)
                with torch.inference_mode():
                    if args.device == "cuda":
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            estimate = model(block)
                    else:
                        estimate = model(block)
                estimate = estimate[0].detach().float().cpu()[..., :chunk]
                accumulation[:, offset:offset + chunk] += estimate * fade
                weights[:, offset:offset + chunk] += fade
            current = accumulation[:, :original_length] / weights[:, :original_length].clamp_min(1e-6)
            if source_rate != model_rate:
                current = audio_functional.resample(current, model_rate, source_rate)
            del model, state
            if args.device == "cuda":
                torch.cuda.empty_cache()

    peak = float(current.abs().max())
    gain = min(1.0, 0.999 / peak) if peak > 0 else 1.0
    current = current * gain
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, current.T.numpy(), source_rate, subtype="PCM_24")
    print(json.dumps({
        "type": "result",
        "model": "xlance-msr",
        "checkpoints": [path.name for path in args.checkpoint],
        "officialOrder": False,
        "antiClippingGainDb": round(20 * math.log10(max(gain, 1e-12)), 4),
        "output": str(args.output.resolve()),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
