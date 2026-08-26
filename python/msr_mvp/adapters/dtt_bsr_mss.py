#!/usr/bin/env python3
"""Eight-target DTT-BSR (stage 1) adapter.

This is not DTT-BSR+: the published plus system needs an additional, per-stem
Demucs-L checkpoint that is not present in the authors' public repositories.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


CHECKPOINTS = {
    "vocals": "Vocals.pth",
    "guitar": "Guitars.pth",
    "keyboards": "Keyboards.pth",
    "synthesizers": "Synthesizers.pth",
    "bass": "Bass.pth",
    "orchestral": "Orchestra.pth",
    "drums": "Drums.pth",
    "percussion": "Percussions.pth",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--chunk-seconds", type=float, default=3.0)
    parser.add_argument("--overlap-seconds", type=float, default=0.25)
    parser.add_argument("--stems", nargs="*", choices=tuple(CHECKPOINTS), default=list(CHECKPOINTS))
    args = parser.parse_args()

    import soundfile as sf
    import torch
    import torchaudio.functional as audio_functional
    import yaml

    librosa_shim = ModuleType("librosa")

    def mel_filter(*, sr: int, n_fft: int, n_mels: int = 128, fmin: float = 0.0, fmax: float | None = None, norm=None, **_kwargs):
        return audio_functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=fmin,
            f_max=float(fmax if fmax is not None else sr / 2),
            n_mels=n_mels,
            sample_rate=sr,
            norm="slaney" if norm == "slaney" else None,
            mel_scale="slaney",
        ).T.numpy()

    librosa_shim.filters = SimpleNamespace(mel=mel_filter)  # type: ignore[attr-defined]
    sys.modules.setdefault("librosa", librosa_shim)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_NOT_AVAILABLE")
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    from models.DTTNet.dp_tdf.dp_tdf_net import DPTDFNet  # type: ignore

    config = yaml.safe_load(args.config.read_text("utf-8"))
    model_rate = int(config["model"]["params"]["sample_rate"])
    samples, source_rate = sf.read(args.input.resolve(), always_2d=True, dtype="float32")
    mixture = torch.from_numpy(samples.T.copy())
    if source_rate != model_rate:
        mixture = audio_functional.resample(mixture, source_rate, model_rate)
    chunk = max(1, round(args.chunk_seconds * model_rate))
    overlap = max(0, round(args.overlap_seconds * model_rate))
    if chunk <= overlap:
        raise RuntimeError("INVALID_CHUNK_SETTINGS")
    hop = chunk - overlap
    original_length = mixture.shape[-1]
    padded_length = max(chunk, ((max(0, original_length - overlap) + hop - 1) // hop) * hop + overlap)
    padded = torch.nn.functional.pad(mixture, (0, padded_length - original_length))
    fade = torch.ones(chunk)
    if overlap:
        ramp = torch.linspace(0, 1, overlap + 2)[1:-1]
        fade[:overlap] = ramp
        fade[-overlap:] = torch.flip(ramp, dims=(0,))

    args.output.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    output_gain_db: dict[str, float] = {}
    for stem in args.stems:
        checkpoint = args.checkpoint_dir / CHECKPOINTS[stem]
        if not checkpoint.is_file():
            raise RuntimeError(f"CHECKPOINT_MISSING:{checkpoint}")
        model = DPTDFNet(**config["model"]["params"])
        state = torch.load(checkpoint, map_location=args.device, weights_only=True)
        model.load_state_dict(state)
        model.to(args.device).eval()
        accumulation = torch.zeros_like(padded)
        weights = torch.zeros_like(padded)
        for offset in range(0, padded_length - chunk + 1, hop):
            # The released audio_ch=1 model treats stereo channels as a batch,
            # matching the authors' inference.py behavior.
            block = padded[:, offset:offset + chunk].to(args.device)
            with torch.inference_mode():
                if args.device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        estimate = model(block)
                else:
                    estimate = model(block)
            estimate = estimate.detach().float().cpu()[..., :chunk]
            accumulation[:, offset:offset + chunk] += estimate * fade
            weights[:, offset:offset + chunk] += fade
        output = accumulation[:, :original_length] / weights[:, :original_length].clamp_min(1e-6)
        if source_rate != model_rate:
            output = audio_functional.resample(output, model_rate, source_rate)
        peak = float(output.abs().max())
        gain = min(1.0, 0.999 / peak) if peak > 0 else 1.0
        output = output * gain
        destination = args.output / f"{stem}.wav"
        sf.write(destination, output.T.numpy(), source_rate, subtype="PCM_24")
        files[stem] = str(destination.resolve())
        output_gain_db[stem] = round(20 * math.log10(max(gain, 1e-12)), 4)
        del model, state, accumulation, weights, output
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    print(json.dumps({
        "type": "result",
        "model": "dtt-bsr-stage1",
        "stems": list(files),
        "files": files,
        "antiClippingGainDb": output_gain_db,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
