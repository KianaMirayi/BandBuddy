from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
MSST_ROOT = REPO_ROOT / ".codex-tmp" / "msst"
MSST_DEPS = REPO_ROOT / ".codex-tmp" / "msst-deps"
SHARED_EXPERIMENT_DEPS = EXPERIMENT_DIR.parent / "pydeps"
SAMPLE_RATE = 44_100

sys.path.insert(0, str(MSST_DEPS))
sys.path.insert(0, str(SHARED_EXPERIMENT_DEPS))
sys.path.insert(0, str(MSST_ROOT))

import torch

from utils.model_utils import bigshifts_wrapper
from utils.settings import get_model_from_config


def read_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"unexpected sample rate for {path}: {sample_rate}")
    return np.ascontiguousarray(audio.T)


def write_audio(path: Path, audio: np.ndarray) -> None:
    sf.write(path, audio.T, SAMPLE_RATE, format="WAV", subtype="FLOAT")


def audio_stats(audio: np.ndarray) -> dict[str, float]:
    values = audio.astype(np.float64, copy=False)
    return {
        "peak": float(np.max(np.abs(values))),
        "rms": float(np.sqrt(np.mean(values * values))),
        "mean": float(np.mean(values)),
    }


def compare_streaming(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {candidate.shape}")
    ref_square = 0.0
    candidate_square = 0.0
    difference_square = 0.0
    cross = 0.0
    difference_peak = 0.0
    for start in range(0, reference.shape[-1], SAMPLE_RATE):
        ref = reference[:, start : start + SAMPLE_RATE].astype(np.float64)
        test = candidate[:, start : start + SAMPLE_RATE].astype(np.float64)
        delta = test - ref
        ref_square += float(np.sum(ref * ref))
        candidate_square += float(np.sum(test * test))
        difference_square += float(np.sum(delta * delta))
        cross += float(np.sum(ref * test))
        difference_peak = max(difference_peak, float(np.max(np.abs(delta))))
    values = reference.size
    denominator = math.sqrt(ref_square * candidate_square)
    return {
        "reference_rms": math.sqrt(ref_square / values),
        "candidate_rms": math.sqrt(candidate_square / values),
        "difference_rms": math.sqrt(difference_square / values),
        "difference_peak": difference_peak,
        "difference_snr_db": (
            float("inf")
            if difference_square == 0.0
            else 10.0 * math.log10(ref_square / difference_square)
        ),
        "correlation": cross / denominator if denominator else 1.0,
    }


def unwrap_checkpoint(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"unexpected checkpoint type: {type(checkpoint)}")
    for key in ("state", "state_dict", "model_state_dict"):
        nested = checkpoint.get(key)
        if isinstance(nested, dict):
            return nested
    return checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=REPO_ROOT / ".codex-tmp" / "worker-nine-full-v2-20260903",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            REPO_ROOT
            / ".codex-tmp"
            / "bd-analysis"
            / "models"
            / "demucs4_lead_rhythm_guitar_drypaint_config.yaml"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            REPO_ROOT
            / ".codex-tmp"
            / "bd-analysis"
            / "models"
            / "demucs4_lead_rhythm_guitar_drypaint.ckpt"
        ),
    )
    parser.add_argument(
        "--electric-input",
        type=Path,
        help="Optional electric-guitar WAV; defaults to the current HQ6 lead+rhythm sum.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--bigshifts", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    args.output.mkdir(parents=True, exist_ok=True)

    reference_lead = read_audio(args.reference / "lead_guitar.wav")
    reference_rhythm = read_audio(args.reference / "rhythm_guitar.wav")
    if args.electric_input is None:
        electric = np.add(reference_lead, reference_rhythm, dtype=np.float32)
        input_contract = "current HQ6 electric = current HQ6 lead + rhythm"
        electric_input = None
    else:
        electric = read_audio(args.electric_input)
        if electric.shape != reference_lead.shape:
            raise ValueError(
                f"electric input shape mismatch: {electric.shape} vs {reference_lead.shape}"
            )
        input_contract = "external electric-guitar candidate"
        electric_input = str(args.electric_input.resolve())

    load_started = time.perf_counter()
    model, config = get_model_from_config("htdemucs", str(args.config))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(unwrap_checkpoint(checkpoint), strict=True)
    del checkpoint
    model.eval().to(device)
    config.inference.num_overlap = args.overlap
    config.inference.batch_size = args.batch_size
    parameters = sum(parameter.numel() for parameter in model.parameters())
    load_seconds = time.perf_counter() - load_started

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    inference_started = time.perf_counter()
    sources = bigshifts_wrapper(
        config,
        model,
        electric,
        device,
        model_type="htdemucs",
        pbar=False,
        bigshifts=args.bigshifts,
    )
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    peak_cuda_bytes = int(torch.cuda.max_memory_allocated(device))

    if not isinstance(sources, dict) or "lead" not in sources or "rhythm" not in sources:
        raise RuntimeError(f"unexpected model outputs: {type(sources)} {getattr(sources, 'keys', lambda: [])()}")
    raw_lead = np.ascontiguousarray(sources["lead"][..., : electric.shape[-1]], dtype=np.float32)
    raw_rhythm = np.ascontiguousarray(sources["rhythm"][..., : electric.shape[-1]], dtype=np.float32)
    # Keep BandBuddy's exact mixture-consistency contract.
    residual_rhythm = np.subtract(electric, raw_lead, dtype=np.float32)
    raw_reconstruction = np.add(raw_lead, raw_rhythm, dtype=np.float32)

    lead_path = args.output / "lead_guitar_htdemucs.wav"
    rhythm_path = args.output / "rhythm_guitar_residual.wav"
    raw_rhythm_path = args.output / "rhythm_guitar_htdemucs_raw.wav"
    write_audio(lead_path, raw_lead)
    write_audio(rhythm_path, residual_rhythm)
    write_audio(raw_rhythm_path, raw_rhythm)

    duration_seconds = electric.shape[-1] / SAMPLE_RATE
    report = {
        "input_contract": input_contract,
        "electric_input": electric_input,
        "reference": str(args.reference.resolve()),
        "duration_seconds": duration_seconds,
        "frames": int(electric.shape[-1]),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "model": {
            "architecture": "HTDemucs",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_size_bytes": args.checkpoint.stat().st_size,
            "parameters": parameters,
            "segment_seconds": int(config.training.segment),
            "overlap": args.overlap,
            "batch_size": args.batch_size,
            "bigshifts": args.bigshifts,
            "load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "realtime_factor": inference_seconds / duration_seconds,
            "times_realtime": duration_seconds / inference_seconds,
            "peak_cuda_bytes": peak_cuda_bytes,
            "speedup_vs_current_mbr_hq6": 201.16 / inference_seconds,
            "speedup_vs_mbr_one_pass": 34.58235939999577 / inference_seconds,
        },
        "lead": {
            "output": str(lead_path.resolve()),
            "stats": audio_stats(raw_lead),
            "comparison_to_current_mbr_hq6": compare_streaming(reference_lead, raw_lead),
        },
        "rhythm_residual": {
            "output": str(rhythm_path.resolve()),
            "stats": audio_stats(residual_rhythm),
            "comparison_to_current_mbr_hq6": compare_streaming(
                reference_rhythm, residual_rhythm
            ),
        },
        "raw_model_rhythm": {
            "output": str(raw_rhythm_path.resolve()),
            "stats": audio_stats(raw_rhythm),
            "reconstruction_error_vs_electric": compare_streaming(
                electric, raw_reconstruction
            ),
        },
    }
    report_path = args.output / "htdemucs-lead-benchmark.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    model.to("cpu")
    del model, sources
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
