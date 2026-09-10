from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from guitar_separator_hq.inference import (
    _build_model,
    _checkpoint_state,
    _linear_window,
    load_config,
)
from guitar_separator_hq.separator import load_audio
from guitar_separator_hq.specs import MODEL_BY_KEY


PassKey = tuple[str, int]
ALL_PASSES: tuple[PassKey, ...] = (
    ("identity", 0),
    ("identity", 1),
    ("channel_swap", 0),
    ("channel_swap", 1),
    ("polarity", 0),
    ("polarity", 1),
)
POLICIES: dict[str, tuple[PassKey, ...]] = {
    "hq6": ALL_PASSES,
    "hq4_no_polarity": ALL_PASSES[:4],
    "tta3_no_bigshift": (ALL_PASSES[0], ALL_PASSES[2], ALL_PASSES[4]),
    "bigshift2_no_tta": ALL_PASSES[:2],
    "single": (ALL_PASSES[0],),
}


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
        return estimate[::-1].copy()
    if variant == "polarity":
        return -estimate
    raise ValueError(variant)


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {candidate.shape}")
    ref = reference.astype(np.float64, copy=False)
    test = candidate.astype(np.float64, copy=False)
    delta = test - ref
    ref_rms = float(np.sqrt(np.mean(np.square(ref))))
    test_rms = float(np.sqrt(np.mean(np.square(test))))
    error_rms = float(np.sqrt(np.mean(np.square(delta))))
    denom = float(np.linalg.norm(ref.ravel()) * np.linalg.norm(test.ravel()))
    correlation = float(np.dot(ref.ravel(), test.ravel()) / denom) if denom else 1.0

    window = 44_100
    window_snrs = []
    for start in range(0, ref.shape[-1], window):
        ref_part = ref[..., start : start + window]
        err_part = delta[..., start : start + window]
        part_ref_rms = float(np.sqrt(np.mean(np.square(ref_part))))
        part_err_rms = float(np.sqrt(np.mean(np.square(err_part))))
        if part_ref_rms > 0 and part_err_rms > 0:
            window_snrs.append(20.0 * math.log10(part_ref_rms / part_err_rms))

    return {
        "reference_rms": ref_rms,
        "candidate_rms": test_rms,
        "difference_rms": error_rms,
        "difference_peak": float(np.max(np.abs(delta))),
        "difference_snr_db": (
            float("inf") if error_rms == 0 else 20.0 * math.log10(ref_rms / error_rms)
        ),
        "correlation": correlation,
        "one_second_snr_min_db": min(window_snrs) if window_snrs else float("inf"),
        "one_second_snr_median_db": (
            float(np.median(window_snrs)) if window_snrs else float("inf")
        ),
    }


def average_passes(outputs: dict[str, np.ndarray], passes: Iterable[PassKey]) -> np.ndarray:
    names = [f"{variant}:{shift}" for variant, shift in passes]
    return np.mean([outputs[name] for name in names], axis=0, dtype=np.float32)


def demix_once_batched(
    model: torch.nn.Module,
    mix: np.ndarray,
    *,
    chunk_size: int,
    overlap: int,
    batch_size: int,
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
    result = torch.zeros((1, 2, padded.shape[-1]), dtype=torch.float32)
    counter = torch.zeros_like(result)

    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.float16, enabled=use_amp and device.type == "cuda"
    ):
        for batch_start in range(0, len(starts), batch_size):
            selected_starts = starts[batch_start : batch_start + batch_size]
            parts = []
            lengths = []
            for start in selected_starts:
                part = padded[:, start : start + chunk_size]
                segment_length = part.shape[-1]
                missing = chunk_size - segment_length
                if missing:
                    pad_mode = "reflect" if segment_length > chunk_size // 2 else "constant"
                    part = F.pad(part, (0, missing), mode=pad_mode)
                parts.append(part)
                lengths.append(segment_length)

            batch = torch.stack(parts, dim=0).to(device)
            predicted = model(batch)
            if predicted.ndim == 3:
                predicted = predicted.unsqueeze(1)
            if predicted.ndim != 4 or predicted.shape[1] != 1 or predicted.shape[2] != 2:
                raise RuntimeError(f"invalid model output: {tuple(predicted.shape)}")
            predicted = predicted.float().cpu()

            for local_index, (start, segment_length) in enumerate(zip(selected_starts, lengths)):
                global_index = batch_start + local_index
                window = window_template.clone()
                if global_index == 0:
                    window[: chunk_size // 10] = 1.0
                if global_index == len(starts) - 1:
                    window[-chunk_size // 10 :] = 1.0
                weights = window[:segment_length]
                result[..., start : start + segment_length] += (
                    predicted[local_index, :, :, :segment_length] * weights
                )
                counter[..., start : start + segment_length] += weights
            del batch, predicted

    estimate = (result / counter.clamp_min_(1e-10)).numpy()[0]
    if used_border:
        estimate = estimate[:, border:-border]
    return np.ascontiguousarray(estimate[:, :original_length], dtype=np.float32)


def load_model(key: str, model_root: Path, device: torch.device):
    spec = MODEL_BY_KEY[key]
    config = load_config(spec)
    model = _build_model(spec, config)
    model.load_state_dict(_checkpoint_state(model_root / spec.filename), strict=True)
    model.eval().to(device)
    chunk_size = int(config["inference"].get("chunk_size", config["audio"]["chunk_size"]))
    overlap = int(config["inference"]["num_overlap"])
    use_amp = bool(config["training"].get("use_amp", True))
    return model, chunk_size, overlap, use_amp


def run_passes(
    model: torch.nn.Module,
    mix: np.ndarray,
    passes: Iterable[PassKey],
    *,
    chunk_size: int,
    overlap: int,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    outputs = {}
    timings = {}
    shift_size = mix.shape[-1] // 2
    for variant, shift_index in passes:
        source = augment(mix, variant)
        shift = shift_index * shift_size
        shifted = np.roll(source, shift, axis=-1) if shift else source
        started = time.perf_counter()
        estimate = demix_once_batched(
            model,
            shifted,
            chunk_size=chunk_size,
            overlap=overlap,
            batch_size=batch_size,
            device=device,
            use_amp=use_amp,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if shift:
            estimate = np.roll(estimate, -shift, axis=-1)
        estimate = undo_augment(estimate, variant)
        name = f"{variant}:{shift_index}"
        outputs[name] = estimate
        timings[name] = elapsed
        print(json.dumps({"pass": name, "seconds": elapsed}, ensure_ascii=False), flush=True)
    return outputs, timings


def read_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != 44_100:
        raise ValueError(path)
    return np.ascontiguousarray(audio.T)


def main() -> int:
    parser = argparse.ArgumentParser()
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
        "--reference",
        type=Path,
        default=REPO_ROOT / ".codex-tmp" / "bd-analysis" / "hq6-smoke",
    )
    parser.add_argument("--lead-batch", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    mix = load_audio(args.input)
    report: dict[str, object] = {
        "input": str(args.input.resolve()),
        "frames": int(mix.shape[-1]),
        "duration_seconds": mix.shape[-1] / 44_100,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "lead_batch": args.lead_batch,
        "models": {},
        "full_policies": {},
    }

    stage_outputs: dict[str, dict[str, np.ndarray]] = {}
    stage_timings: dict[str, dict[str, float]] = {}
    for key in ("acoustic", "electric"):
        model, chunk_size, overlap, use_amp = load_model(key, args.models, device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        outputs, timings = run_passes(
            model,
            mix,
            ALL_PASSES,
            chunk_size=chunk_size,
            overlap=overlap,
            batch_size=1,
            device=device,
            use_amp=use_amp,
        )
        stage_outputs[key] = outputs
        stage_timings[key] = timings
        hq6 = average_passes(outputs, ALL_PASSES)
        report["models"][key] = {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "batch_size": 1,
            "pass_seconds": timings,
            "total_hq6_seconds": sum(timings.values()),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
            "policy_vs_hq6": {
                name: compare(hq6, average_passes(outputs, passes))
                for name, passes in POLICIES.items()
            },
        }
        model.to("cpu")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    policy_electric = {
        name: average_passes(stage_outputs["electric"], passes)
        for name, passes in POLICIES.items()
    }
    lead_model, lead_chunk, lead_overlap, lead_amp = load_model("lead", args.models, device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    lead_policy_outputs = {}
    lead_policy_timings = {}
    lead_stage_ablation = None
    for policy_name, passes in POLICIES.items():
        outputs, timings = run_passes(
            lead_model,
            policy_electric[policy_name],
            passes,
            chunk_size=lead_chunk,
            overlap=lead_overlap,
            batch_size=args.lead_batch,
            device=device,
            use_amp=lead_amp,
        )
        lead = average_passes(outputs, passes)
        lead_policy_outputs[policy_name] = lead
        lead_policy_timings[policy_name] = timings
        if policy_name == "hq6":
            lead_stage_ablation = {
                name: compare(lead, average_passes(outputs, subset))
                for name, subset in POLICIES.items()
            }

    report["models"]["lead"] = {
        "chunk_size": lead_chunk,
        "overlap": lead_overlap,
        "batch_size": args.lead_batch,
        "policy_seconds": {
            name: sum(timings.values()) for name, timings in lead_policy_timings.items()
        },
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "policy_vs_hq6_on_same_hq6_electric": lead_stage_ablation,
    }

    reference_acoustic = read_audio(args.reference / "acoustic_guitar.wav")
    reference_lead = read_audio(args.reference / "lead_guitar.wav")
    reference_rhythm = read_audio(args.reference / "rhythm_guitar.wav")
    reference_electric = np.add(reference_lead, reference_rhythm, dtype=np.float32)
    policy_acoustic = {
        name: average_passes(stage_outputs["acoustic"], passes)
        for name, passes in POLICIES.items()
    }

    original_manifest = json.loads((args.reference / "separation_manifest.json").read_text("utf-8"))
    original_seconds = {
        key: float(original_manifest["models"][key]["inference"]["seconds"])
        for key in ("acoustic", "electric", "lead")
    }
    original_total = sum(original_seconds.values())

    for name, passes in POLICIES.items():
        acoustic = policy_acoustic[name]
        electric = policy_electric[name]
        lead = lead_policy_outputs[name]
        rhythm = np.subtract(electric, lead, dtype=np.float32)
        acoustic_seconds = sum(stage_timings["acoustic"][f"{v}:{s}"] for v, s in passes)
        electric_seconds = sum(stage_timings["electric"][f"{v}:{s}"] for v, s in passes)
        lead_seconds = sum(lead_policy_timings[name].values())
        total_seconds = acoustic_seconds + electric_seconds + lead_seconds
        report["full_policies"][name] = {
            "passes_per_model": len(passes),
            "seconds": {
                "acoustic": acoustic_seconds,
                "electric": electric_seconds,
                "lead": lead_seconds,
                "total": total_seconds,
            },
            "speedup_vs_original_hq6_manifest": original_total / total_seconds,
            "outputs_vs_original_hq6": {
                "acoustic": compare(reference_acoustic, acoustic),
                "electric": compare(reference_electric, electric),
                "lead": compare(reference_lead, lead),
                "rhythm": compare(reference_rhythm, rhythm),
            },
        }

    report["original_hq6_manifest_seconds"] = {
        **original_seconds,
        "total": original_total,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": report["full_policies"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
