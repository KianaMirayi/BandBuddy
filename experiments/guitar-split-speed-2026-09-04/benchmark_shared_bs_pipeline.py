from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from guitar_separator_hq._vendor.msst import BSRoformer
from guitar_separator_hq.inference import _checkpoint_state, _linear_window, load_config
from guitar_separator_hq.separator import load_audio
from guitar_separator_hq.specs import MODEL_BY_KEY

shared_spec = importlib.util.spec_from_file_location(
    "shared_forward_benchmark", EXPERIMENT_DIR / "benchmark_shared_bs.py"
)
if shared_spec is None or shared_spec.loader is None:
    raise RuntimeError("cannot load benchmark_shared_bs.py")
shared = importlib.util.module_from_spec(shared_spec)
shared_spec.loader.exec_module(shared)


ALL_PASSES = (
    ("identity", 0),
    ("identity", 1),
    ("channel_swap", 0),
    ("channel_swap", 1),
    ("polarity", 0),
    ("polarity", 1),
)


def augment(mix: np.ndarray, variant: str) -> np.ndarray:
    if variant == "identity":
        return mix
    if variant == "channel_swap":
        return mix[::-1].copy()
    if variant == "polarity":
        return -mix
    raise ValueError(variant)


def undo_augment(estimate: np.ndarray, variant: str) -> np.ndarray:
    if variant == "identity":
        return estimate
    if variant == "channel_swap":
        return estimate[:, ::-1].copy()
    if variant == "polarity":
        return -estimate
    raise ValueError(variant)


def demix_once(
    model: torch.nn.Module,
    mix: np.ndarray,
    *,
    chunk_size: int,
    overlap: int,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    step = chunk_size // overlap
    border = chunk_size - step
    original_length = mix.shape[-1]
    padded = torch.from_numpy(np.ascontiguousarray(mix, dtype=np.float32))
    used_border = original_length > 2 * border and border > 0
    if used_border:
        padded = F.pad(padded, (border, border), mode="reflect")
    starts = list(range(0, padded.shape[-1], step))
    window_template = _linear_window(chunk_size)
    result = torch.zeros((2, 2, padded.shape[-1]), dtype=torch.float32)
    counter = torch.zeros(padded.shape[-1], dtype=torch.float32)

    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.float16, enabled=use_amp and device.type == "cuda"
    ):
        for index, start in enumerate(starts):
            part = padded[:, start : start + chunk_size]
            segment_length = part.shape[-1]
            missing = chunk_size - segment_length
            if missing:
                mode = "reflect" if segment_length > chunk_size // 2 else "constant"
                part = F.pad(part, (0, missing), mode=mode)
            prediction = model(part.unsqueeze(0).to(device))
            if prediction.shape[:3] != (1, 2, 2):
                raise RuntimeError(f"invalid output shape: {tuple(prediction.shape)}")
            prediction = prediction[0, :, :, :segment_length].float().cpu()
            window = window_template.clone()
            if index == 0:
                window[: chunk_size // 10] = 1.0
            if index == len(starts) - 1:
                window[-chunk_size // 10 :] = 1.0
            weights = window[:segment_length]
            result[..., start : start + segment_length] += prediction * weights
            counter[start : start + segment_length] += weights
            del prediction

    estimate = (result / counter.clamp_min_(1e-10)[None, None]).numpy()
    if used_border:
        estimate = estimate[..., border:-border]
    return np.ascontiguousarray(estimate[..., :original_length], dtype=np.float32)


def compare_reference(
    candidate: np.ndarray, first_path: Path, second_path: Path | None = None
) -> dict[str, float]:
    sum_ref_sq = 0.0
    sum_test_sq = 0.0
    sum_error_sq = 0.0
    dot = 0.0
    peak = 0.0
    count = 0
    block_frames = 262_144
    with sf.SoundFile(first_path) as first:
        second_context = sf.SoundFile(second_path) if second_path else None
        try:
            position = 0
            while position < candidate.shape[-1]:
                frames = min(block_frames, candidate.shape[-1] - position)
                reference = first.read(frames, dtype="float32", always_2d=True).T
                if reference.shape[-1] != frames:
                    raise ValueError("short reference")
                if second_context is not None:
                    other = second_context.read(frames, dtype="float32", always_2d=True).T
                    reference = np.add(reference, other, dtype=np.float32)
                test = candidate[:, position : position + frames]
                ref64 = reference.astype(np.float64)
                test64 = test.astype(np.float64)
                delta = test64 - ref64
                sum_ref_sq += float(np.sum(ref64 * ref64))
                sum_test_sq += float(np.sum(test64 * test64))
                sum_error_sq += float(np.sum(delta * delta))
                dot += float(np.sum(ref64 * test64))
                peak = max(peak, float(np.max(np.abs(delta))))
                count += delta.size
                position += frames
        finally:
            if second_context is not None:
                second_context.close()
    ref_rms = math.sqrt(sum_ref_sq / count)
    error_rms = math.sqrt(sum_error_sq / count)
    return {
        "reference_rms": ref_rms,
        "candidate_rms": math.sqrt(sum_test_sq / count),
        "difference_rms": error_rms,
        "difference_peak": peak,
        "difference_snr_db": (
            float("inf") if error_rms == 0 else 20 * math.log10(ref_rms / error_rms)
        ),
        "correlation": dot / math.sqrt(sum_ref_sq * sum_test_sq),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / ".codex-tmp" / "bd-analysis" / "source-60s-15s.wav",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=REPO_ROOT / ".codex-tmp" / "bd-analysis" / "hq6-smoke",
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=REPO_ROOT / ".codex-tmp" / "guitar-hq-models",
    )
    parser.add_argument("--baseline-acoustic-seconds", type=float, default=15.799791799974628)
    parser.add_argument("--baseline-electric-seconds", type=float, default=15.615065400023013)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    acoustic_spec = MODEL_BY_KEY["acoustic"]
    electric_spec = MODEL_BY_KEY["electric"]
    config = load_config(acoustic_spec)
    other_config = load_config(electric_spec)
    if config["model"] != other_config["model"]:
        raise ValueError("model configurations differ")

    load_started = time.perf_counter()
    acoustic_state = _checkpoint_state(args.models / acoustic_spec.filename)
    electric_state = _checkpoint_state(args.models / electric_spec.filename)
    state = shared.combined_state(acoustic_state, electric_state)
    kwargs = dict(config["model"])
    kwargs["num_stems"] = 2
    model = BSRoformer(**kwargs)
    model.load_state_dict(state, strict=True)
    del acoustic_state, electric_state, state
    model.eval().requires_grad_(False).to(device)
    load_seconds = time.perf_counter() - load_started
    chunk_size = int(config["inference"]["chunk_size"])
    overlap = int(config["inference"]["num_overlap"])
    use_amp = bool(config["training"].get("use_amp", True))
    mix = load_audio(args.input)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    accumulation = np.zeros((2, 2, mix.shape[-1]), dtype=np.float32)
    pass_seconds = {}
    shift_size = mix.shape[-1] // 2
    for variant, shift_index in ALL_PASSES:
        source = augment(mix, variant)
        shift = shift_index * shift_size
        shifted = np.roll(source, shift, axis=-1) if shift else source
        started = time.perf_counter()
        estimate = demix_once(
            model,
            shifted,
            chunk_size=chunk_size,
            overlap=overlap,
            device=device,
            use_amp=use_amp,
        )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if shift:
            estimate = np.roll(estimate, -shift, axis=-1)
        estimate = undo_augment(estimate, variant)
        accumulation += estimate
        name = f"{variant}:{shift_index}"
        pass_seconds[name] = elapsed
        print(json.dumps({"pass": name, "seconds": elapsed}, ensure_ascii=False), flush=True)
        del estimate, shifted, source
        gc.collect()
    accumulation /= float(len(ALL_PASSES))

    reference_acoustic = args.reference / "acoustic_guitar.wav"
    reference_lead = args.reference / "lead_guitar.wav"
    reference_rhythm = args.reference / "rhythm_guitar.wav"
    comparisons = {
        "acoustic": compare_reference(accumulation[0], reference_acoustic),
        "electric": compare_reference(accumulation[1], reference_lead, reference_rhythm),
    }
    baseline = args.baseline_acoustic_seconds + args.baseline_electric_seconds
    inference_seconds = sum(pass_seconds.values())
    report = {
        "input": str(args.input.resolve()),
        "reference": str(args.reference.resolve()),
        "duration_seconds": mix.shape[-1] / 44_100,
        "frames": int(mix.shape[-1]),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "passes": len(ALL_PASSES),
        "load_seconds": load_seconds,
        "pass_seconds": pass_seconds,
        "inference_seconds": inference_seconds,
        "baseline_separate_seconds": baseline,
        "speedup": baseline / inference_seconds,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "comparisons": comparisons,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
