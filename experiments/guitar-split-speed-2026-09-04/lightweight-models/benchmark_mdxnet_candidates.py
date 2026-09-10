from __future__ import annotations

import argparse
import gc
import json
import logging
import math
from pathlib import Path
import sys
import time
import types

import numpy as np
import soundfile as sf
import torch


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PYDEPS = EXPERIMENT_DIR / "pydeps"
BD_RUNTIME = REPO_ROOT / ".codex-tmp" / "bd-runtime"

# Keep the official GPU ORT wheel ahead of the extracted audio-separator runtime,
# which contains a CPU-only ORT build. Nothing is installed into BandBuddy itself.
sys.path.insert(0, str(BD_RUNTIME))
sys.path.insert(0, str(PYDEPS))
sys.path.insert(0, str(REPO_ROOT / "python"))

import onnxruntime as ort

ort.preload_dlls()

# MDXSeparator imports onnx2torch eagerly, although the fixed segment size below
# always selects ONNX Runtime. Avoid loading an incompatible, unused torchvision
# conversion stack from the extracted reference runtime.
onnx2torch_stub = types.ModuleType("onnx2torch")
onnx2torch_stub.convert = lambda *args, **kwargs: (_ for _ in ()).throw(
    RuntimeError("onnx2torch is disabled in this ONNX Runtime-only benchmark")
)
sys.modules["onnx2torch"] = onnx2torch_stub

from audio_separator.separator.architectures import mdx_separator as mdx_module
from audio_separator.separator.architectures.mdx_separator import MDXSeparator
from guitar_separator_hq.separator import audio_stats, load_audio, write_float_wav


SAMPLE_RATE = 44_100
CURRENT_HQ6_SECONDS = {
    "acoustic_guitar": 320.65,
    "electric_guitar": 336.37,
}
MODEL_SPECS = {
    "acoustic_guitar": {
        "filename": "mdx_6s_acoustic_guitar_anvuew.onnx",
        "sha256": "2bd8f2af629b279cc1a568f895ee9636f7ce2d76c69aa601e6744eaab8b4916a",
    },
    "electric_guitar": {
        "filename": "mdx_6s_electric_guitar_anvuew.onnx",
        "sha256": "bd6fcf40659771568ee180ea69bd4576a9c3d2423ae0f5f6f5afcc8b6a6fd938",
    },
}


def read_float_wav(path: Path, frames: int | None = None) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"unexpected sample rate for {path}: {sample_rate}")
    result = np.ascontiguousarray(audio.T)
    return result if frames is None else result[:, :frames]


def compare_streaming(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {candidate.shape}")

    ref_square = 0.0
    candidate_square = 0.0
    difference_square = 0.0
    cross = 0.0
    difference_peak = 0.0
    one_second_snrs: list[float] = []
    value_count = reference.size

    for start in range(0, reference.shape[-1], SAMPLE_RATE):
        ref = reference[:, start : start + SAMPLE_RATE].astype(np.float64)
        test = candidate[:, start : start + SAMPLE_RATE].astype(np.float64)
        delta = test - ref
        local_ref_square = float(np.sum(ref * ref))
        local_difference_square = float(np.sum(delta * delta))
        ref_square += local_ref_square
        candidate_square += float(np.sum(test * test))
        difference_square += local_difference_square
        cross += float(np.sum(ref * test))
        difference_peak = max(difference_peak, float(np.max(np.abs(delta))))
        if local_ref_square > 0.0 and local_difference_square > 0.0:
            one_second_snrs.append(
                10.0 * math.log10(local_ref_square / local_difference_square)
            )

    reference_rms = math.sqrt(ref_square / value_count)
    candidate_rms = math.sqrt(candidate_square / value_count)
    difference_rms = math.sqrt(difference_square / value_count)
    correlation_denominator = math.sqrt(ref_square * candidate_square)
    return {
        "reference_rms": reference_rms,
        "candidate_rms": candidate_rms,
        "difference_rms": difference_rms,
        "difference_peak": difference_peak,
        "difference_snr_db": (
            float("inf")
            if difference_square == 0.0
            else 10.0 * math.log10(ref_square / difference_square)
        ),
        "correlation": (
            cross / correlation_denominator if correlation_denominator else 1.0
        ),
        "one_second_snr_min_db": (
            min(one_second_snrs) if one_second_snrs else float("inf")
        ),
        "one_second_snr_median_db": (
            float(np.median(one_second_snrs))
            if one_second_snrs
            else float("inf")
        ),
    }


def build_separator(target: str, model_path: Path, output_dir: Path) -> MDXSeparator:
    logger = logging.getLogger(f"mdxnet.{target}")
    common_config = {
        "logger": logger,
        "log_level": logging.INFO,
        "torch_device": torch.device("cuda:0"),
        "torch_device_cpu": torch.device("cpu"),
        "torch_device_mps": None,
        "onnx_execution_provider": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "model_name": model_path.name,
        "model_path": str(model_path),
        "model_data": {
            "compensate": 1.03,
            "mdx_dim_f_set": 2048,
            "mdx_dim_t_set": 8,
            "mdx_n_fft_scale_set": 4096,
            "primary_stem": target,
            "training": {
                "target_instrument": target,
                "instruments": [target, "other"],
            },
        },
        "output_dir": str(output_dir),
        "output_format": "WAV",
        "output_bitrate": None,
        "normalization_threshold": 0.9,
        "amplification_threshold": 0.0,
        "enable_denoise": False,
        "output_single_stem": target,
        "invert_using_spec": False,
        "sample_rate": SAMPLE_RATE,
        "use_soundfile": True,
        "use_autocast": False,
        "use_native_fp16": False,
        "use_torch_compile": False,
    }
    arch_config = {
        "segment_size": 256,
        "overlap": 0.5,
        "batch_size": 1,
        "hop_length": 1024,
        "enable_denoise": False,
    }
    return MDXSeparator(common_config, arch_config)


def session_providers(separator: MDXSeparator) -> list[str]:
    closure = getattr(separator.model_run, "__closure__", None) or ()
    for cell in closure:
        value = cell.cell_contents
        if isinstance(value, ort.InferenceSession):
            return value.get_providers()
    return []


def load_reference(reference_dir: Path, target: str, frames: int) -> np.ndarray:
    if target == "acoustic_guitar":
        return read_float_wav(reference_dir / "acoustic_guitar.wav", frames)
    lead = read_float_wav(reference_dir / "lead_guitar.wav", frames)
    rhythm = read_float_wav(reference_dir / "rhythm_guitar.wav", frames)
    return np.add(lead, rhythm, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(r"C:\CloudMusic\新裤子 - 没有理想的人不伤心.mp3"),
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=EXPERIMENT_DIR / "models" / "mdxnet",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=REPO_ROOT / ".codex-tmp" / "worker-nine-full-v2-20260903",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(f"CUDA ORT provider unavailable: {ort.get_available_providers()}")

    # Keep the benchmark log concise; this changes only progress rendering.
    mdx_module.tqdm = lambda iterable, *unused_args, **unused_kwargs: iterable
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)

    decode_started = time.perf_counter()
    mix = load_audio(args.input)
    if args.max_seconds is not None:
        mix = np.ascontiguousarray(mix[:, : int(args.max_seconds * SAMPLE_RATE)])
    decode_seconds = time.perf_counter() - decode_started

    report: dict[str, object] = {
        "input": str(args.input.resolve()),
        "reference": str(args.reference.resolve()),
        "frames": int(mix.shape[-1]),
        "duration_seconds": mix.shape[-1] / SAMPLE_RATE,
        "decode_seconds": decode_seconds,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "onnxruntime": ort.__version__,
        "models": {},
    }

    combined_seconds = 0.0
    for target, spec in MODEL_SPECS.items():
        model_path = args.models / spec["filename"]
        load_started = time.perf_counter()
        separator = build_separator(target, model_path, args.output)
        load_seconds = time.perf_counter() - load_started
        providers = session_providers(separator)
        if not providers or providers[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"{target} did not activate CUDA: {providers}")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        inference_started = time.perf_counter()
        candidate = separator.demix(mix)
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started
        combined_seconds += inference_seconds
        candidate = np.ascontiguousarray(candidate[:, : mix.shape[-1]], dtype=np.float32)
        if candidate.shape != mix.shape or not np.isfinite(candidate).all():
            raise RuntimeError(f"invalid {target} output: {candidate.shape}")

        output_path = args.output / f"{target}_mdxnet.wav"
        write_float_wav(output_path, candidate)
        reference = load_reference(args.reference, target, mix.shape[-1])
        comparison = compare_streaming(reference, candidate)
        report["models"][target] = {
            "model_path": str(model_path.resolve()),
            "model_size_bytes": model_path.stat().st_size,
            "expected_sha256": spec["sha256"],
            "providers": providers,
            "segment_size": 256,
            "overlap": 0.5,
            "batch_size": 1,
            "load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "realtime_factor": inference_seconds / report["duration_seconds"],
            "times_realtime": report["duration_seconds"] / inference_seconds,
            "current_hq6_seconds": CURRENT_HQ6_SECONDS[target],
            "speedup_vs_current_hq6": CURRENT_HQ6_SECONDS[target] / inference_seconds,
            "torch_peak_cuda_bytes_excludes_ort": int(torch.cuda.max_memory_allocated()),
            "output": str(output_path.resolve()),
            "output_stats": audio_stats(candidate),
            "comparison_to_current_hq6": comparison,
        }
        print(json.dumps({target: report["models"][target]}, ensure_ascii=False), flush=True)

        del reference, candidate, separator
        gc.collect()
        torch.cuda.empty_cache()

    report["combined_inference_seconds"] = combined_seconds
    report["combined_current_hq6_seconds"] = sum(CURRENT_HQ6_SECONDS.values())
    report["combined_speedup_vs_current_hq6"] = (
        report["combined_current_hq6_seconds"] / combined_seconds
    )
    report_path = args.output / "mdxnet-benchmark.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "combined_seconds": combined_seconds}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
