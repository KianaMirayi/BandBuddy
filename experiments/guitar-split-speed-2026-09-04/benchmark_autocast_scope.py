from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from guitar_separator_hq.inference import _build_model, _checkpoint_state, load_config
from guitar_separator_hq.separator import load_audio
from guitar_separator_hq.specs import MODEL_BY_KEY


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODEL_BY_KEY), required=True)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / ".codex-tmp" / "bd-analysis" / "source-60s-15s.wav",
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=REPO_ROOT / ".codex-tmp" / "guitar-hq-models",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    spec = MODEL_BY_KEY[args.model]
    config = load_config(spec)
    chunk_size = int(config["inference"].get("chunk_size", config["audio"]["chunk_size"]))
    use_amp = bool(config["training"].get("use_amp", True))

    mix = load_audio(args.input)
    part = torch.from_numpy(np.ascontiguousarray(mix[:, :chunk_size], dtype=np.float32))
    if part.shape[-1] < chunk_size:
        missing = chunk_size - part.shape[-1]
        mode = "reflect" if part.shape[-1] > chunk_size // 2 else "constant"
        part = F.pad(part, (0, missing), mode=mode)
    gpu_part = part.unsqueeze(0).to(device)

    model = _build_model(spec, config)
    model.load_state_dict(_checkpoint_state(args.models / spec.filename), strict=True)
    model.eval().to(device)

    def run(scope: str) -> tuple[np.ndarray, float, int]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        outputs = []
        started = time.perf_counter()
        with torch.inference_mode():
            if scope == "outer":
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp, cache_enabled=True):
                    for _ in range(args.iterations):
                        outputs.append(model(gpu_part).float().cpu())
            elif scope == "per_chunk":
                for _ in range(args.iterations):
                    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp, cache_enabled=True):
                        outputs.append(model(gpu_part).float().cpu())
            elif scope == "outer_no_cache":
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp, cache_enabled=False):
                    for _ in range(args.iterations):
                        outputs.append(model(gpu_part).float().cpu())
            else:
                raise ValueError(scope)
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        peak = int(torch.cuda.max_memory_allocated(device))
        result = outputs[-1].numpy()
        del outputs
        return result, seconds, peak

    # One unmeasured call initializes FFT plans and CUDA kernels.
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        model(gpu_part).float().cpu()
    torch.cuda.synchronize(device)

    results = []
    reference = None
    # Run the current behavior twice around the candidate to reduce ordering bias.
    for scope in ("per_chunk", "outer", "outer_no_cache", "per_chunk"):
        output, seconds, peak = run(scope)
        if reference is None:
            reference = output
        delta = output.astype(np.float64) - reference.astype(np.float64)
        delta_rms = float(np.sqrt(np.mean(np.square(delta))))
        row = {
            "scope": scope,
            "iterations": args.iterations,
            "seconds": seconds,
            "seconds_per_forward": seconds / args.iterations,
            "peak_cuda_bytes": peak,
            "difference_rms_vs_first": delta_rms,
            "difference_peak_vs_first": float(np.max(np.abs(delta))),
        }
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    report = {
        "model": args.model,
        "chunk_size": chunk_size,
        "amp": use_amp,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "trials": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
