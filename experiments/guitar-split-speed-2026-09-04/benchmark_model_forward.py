from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from guitar_separator_hq._vendor.msst.attend import Attend, FlashAttentionConfig
from guitar_separator_hq.inference import _build_model, _checkpoint_state, load_config
from guitar_separator_hq.separator import load_audio
from guitar_separator_hq.specs import MODEL_BY_KEY


BACKENDS: dict[str, FlashAttentionConfig] = {
    "legacy_windows": FlashAttentionConfig(False, True, True),
    "auto": FlashAttentionConfig(True, True, True),
    "flash": FlashAttentionConfig(True, False, False),
    "efficient": FlashAttentionConfig(False, False, True),
    "math": FlashAttentionConfig(False, True, False),
}


def set_attention_backend(model: torch.nn.Module, backend: str) -> int:
    config = BACKENDS[backend]
    count = 0
    for module in model.modules():
        if isinstance(module, Attend):
            module.cuda_config = config
            count += 1
    return count


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref = reference.astype(np.float64, copy=False)
    test = candidate.astype(np.float64, copy=False)
    delta = test - ref
    ref_rms = float(np.sqrt(np.mean(np.square(ref))))
    delta_rms = float(np.sqrt(np.mean(np.square(delta))))
    denom = float(np.linalg.norm(ref.ravel()) * np.linalg.norm(test.ravel()))
    correlation = float(np.dot(ref.ravel(), test.ravel()) / denom) if denom else 1.0
    return {
        "reference_rms": ref_rms,
        "difference_rms": delta_rms,
        "difference_peak": float(np.max(np.abs(delta))),
        "difference_snr_db": (
            float("inf") if delta_rms == 0.0 else 20.0 * math.log10(ref_rms / delta_rms)
        ),
        "correlation": correlation,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODEL_BY_KEY), required=True)
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
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=tuple(BACKENDS),
        default=["legacy_windows", "auto", "flash", "efficient", "math"],
    )
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--batch-backend", choices=tuple(BACKENDS), default="auto")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    device = torch.device("cuda:0")
    spec = MODEL_BY_KEY[args.model]
    config = load_config(spec)
    inference = config["inference"]
    audio_config = config["audio"]
    chunk_size = int(inference.get("chunk_size", audio_config["chunk_size"]))
    checkpoint = args.models / spec.filename

    mix = load_audio(args.input)
    part = torch.from_numpy(np.ascontiguousarray(mix[:, :chunk_size], dtype=np.float32))
    if part.shape[-1] < chunk_size:
        missing = chunk_size - part.shape[-1]
        mode = "reflect" if part.shape[-1] > chunk_size // 2 else "constant"
        part = F.pad(part, (0, missing), mode=mode)

    model = _build_model(spec, config)
    model.load_state_dict(_checkpoint_state(checkpoint), strict=True)
    model.eval().to(device)
    amp_enabled = bool(config["training"].get("use_amp", True))

    report: dict[str, object] = {
        "model": args.model,
        "architecture": spec.architecture,
        "chunk_size": chunk_size,
        "chunk_seconds": chunk_size / 44_100,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "attention_modules": sum(isinstance(module, Attend) for module in model.modules()),
        "amp": amp_enabled,
        "backend_trials": [],
        "batch_trials": [],
    }

    def run(batch_size: int) -> tuple[np.ndarray, float, int]:
        host_batch = part.unsqueeze(0).repeat(batch_size, 1, 1)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        gpu_batch = host_batch.to(device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp_enabled
        ):
            prediction = model(gpu_batch)
        output = prediction.float().cpu().numpy()
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        peak = int(torch.cuda.max_memory_allocated(device))
        del host_batch, gpu_batch, prediction
        return output, seconds, peak

    reference: np.ndarray | None = None
    for backend in args.backends:
        trial: dict[str, object] = {"backend": backend}
        try:
            set_attention_backend(model, backend)
            for _ in range(args.warmups):
                warm, _, _ = run(1)
                del warm
            timings: list[float] = []
            peaks: list[int] = []
            output = None
            for _ in range(args.repeats):
                output, seconds, peak = run(1)
                timings.append(seconds)
                peaks.append(peak)
            assert output is not None
            if backend == "legacy_windows" or reference is None:
                reference = output[0].copy()
            trial.update(
                {
                    "status": "ok",
                    "seconds": timings,
                    "median_seconds": float(np.median(timings)),
                    "peak_cuda_bytes": max(peaks),
                    "vs_reference": compare(reference, output[0]),
                }
            )
            del output
        except Exception as error:
            trial.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
            gc.collect()
            torch.cuda.empty_cache()
        report["backend_trials"].append(trial)
        print(json.dumps(trial, ensure_ascii=False), flush=True)

    set_attention_backend(model, args.batch_backend)
    for batch_size in args.batches:
        trial = {"backend": args.batch_backend, "batch_size": batch_size}
        try:
            for _ in range(args.warmups):
                warm, _, _ = run(batch_size)
                del warm
            timings = []
            peaks = []
            output = None
            for _ in range(args.repeats):
                output, seconds, peak = run(batch_size)
                timings.append(seconds)
                peaks.append(peak)
            assert output is not None
            batch_consistency = compare(output[0], output[-1])
            trial.update(
                {
                    "status": "ok",
                    "seconds": timings,
                    "median_seconds": float(np.median(timings)),
                    "median_seconds_per_item": float(np.median(timings)) / batch_size,
                    "peak_cuda_bytes": max(peaks),
                    "within_batch_consistency": batch_consistency,
                }
            )
            del output
        except Exception as error:
            trial.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
            gc.collect()
            torch.cuda.empty_cache()
        report["batch_trials"].append(trial)
        print(json.dumps(trial, ensure_ascii=False), flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
