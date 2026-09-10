from __future__ import annotations

import argparse
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

from guitar_separator_hq._vendor.msst import BSRoformer
from guitar_separator_hq.inference import _build_model, _checkpoint_state, load_config
from guitar_separator_hq.separator import load_audio
from guitar_separator_hq.specs import MODEL_BY_KEY


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref = reference.astype(np.float64, copy=False)
    test = candidate.astype(np.float64, copy=False)
    delta = test - ref
    ref_rms = float(np.sqrt(np.mean(np.square(ref))))
    error_rms = float(np.sqrt(np.mean(np.square(delta))))
    return {
        "difference_rms": error_rms,
        "difference_peak": float(np.max(np.abs(delta))),
        "difference_snr_db": (
            float("inf") if error_rms == 0 else 20 * math.log10(ref_rms / error_rms)
        ),
    }


def combined_state(
    acoustic: dict[str, torch.Tensor], electric: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    if acoustic.keys() != electric.keys():
        raise ValueError("checkpoint keys differ")
    result = {}
    for key, value in acoustic.items():
        if key.startswith("mask_estimators.0."):
            suffix = key.removeprefix("mask_estimators.0.")
            result[f"mask_estimators.0.{suffix}"] = value
            result[f"mask_estimators.1.{suffix}"] = electric[key]
        else:
            if not torch.equal(value, electric[key]):
                raise ValueError(f"shared trunk mismatch: {key}")
            result[key] = value
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3)
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    acoustic_spec = MODEL_BY_KEY["acoustic"]
    electric_spec = MODEL_BY_KEY["electric"]
    acoustic_config = load_config(acoustic_spec)
    electric_config = load_config(electric_spec)
    if acoustic_config["model"] != electric_config["model"]:
        raise ValueError("model configurations differ")

    acoustic_state = _checkpoint_state(args.models / acoustic_spec.filename)
    electric_state = _checkpoint_state(args.models / electric_spec.filename)
    merged_state = combined_state(acoustic_state, electric_state)

    acoustic = _build_model(acoustic_spec, acoustic_config)
    acoustic.load_state_dict(acoustic_state, strict=True)
    electric = _build_model(electric_spec, electric_config)
    electric.load_state_dict(electric_state, strict=True)
    kwargs = dict(acoustic_config["model"])
    kwargs["num_stems"] = 2
    combined = BSRoformer(**kwargs)
    combined.load_state_dict(merged_state, strict=True)
    del acoustic_state, electric_state, merged_state
    acoustic.eval().requires_grad_(False).to(device)
    electric.eval().requires_grad_(False).to(device)
    combined.eval().requires_grad_(False).to(device)

    chunk_size = int(acoustic_config["inference"]["chunk_size"])
    mix = load_audio(args.input)
    part = torch.from_numpy(np.ascontiguousarray(mix[:, :chunk_size], dtype=np.float32))
    if part.shape[-1] < chunk_size:
        missing = chunk_size - part.shape[-1]
        part = F.pad(
            part,
            (0, missing),
            mode="reflect" if part.shape[-1] > chunk_size // 2 else "constant",
        )
    data = part.unsqueeze(0).to(device)

    def run_separate() -> tuple[np.ndarray, float, int]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        output = None
        for _ in range(args.iterations):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                left = acoustic(data)
                right = electric(data)
            output = torch.cat((left, right), dim=1).float().cpu().numpy()
        torch.cuda.synchronize(device)
        return output, time.perf_counter() - started, int(torch.cuda.max_memory_allocated(device))

    def run_combined() -> tuple[np.ndarray, float, int]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        output = None
        for _ in range(args.iterations):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                output = combined(data).float().cpu().numpy()
        torch.cuda.synchronize(device)
        return output, time.perf_counter() - started, int(torch.cuda.max_memory_allocated(device))

    # Initialize kernels and FFT plans for all three module instances.
    run_separate()
    run_combined()
    separate_output, separate_seconds, separate_peak = run_separate()
    combined_output, combined_seconds, combined_peak = run_combined()

    report = {
        "iterations": args.iterations,
        "chunk_size": chunk_size,
        "separate_seconds": separate_seconds,
        "combined_seconds": combined_seconds,
        "speedup": separate_seconds / combined_seconds,
        "separate_peak_cuda_bytes": separate_peak,
        "combined_peak_cuda_bytes": combined_peak,
        "combined_parameters": sum(parameter.numel() for parameter in combined.parameters()),
        "separate_parameters": sum(parameter.numel() for parameter in acoustic.parameters())
        + sum(parameter.numel() for parameter in electric.parameters()),
        "outputs": {
            "acoustic": metrics(separate_output[:, 0], combined_output[:, 0]),
            "electric": metrics(separate_output[:, 1], combined_output[:, 1]),
        },
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
