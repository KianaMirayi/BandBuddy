from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time
from typing import Callable, Mapping

import librosa
import numpy as np
import soundfile as sf
import torch

from .inference import predict_model, resolve_device
from .model_store import ensure_models, sha256_file
from .specs import MODEL_BY_KEY, MODEL_SET_REVISION, MODEL_SPECS, MSST_SOURCE_COMMIT


SAMPLE_RATE = 44_100
OUTPUT_NAMES = {
    "acoustic_guitar": "acoustic_guitar.wav",
    "lead_guitar": "lead_guitar.wav",
    "rhythm_guitar": "rhythm_guitar.wav",
}
ProgressCallback = Callable[[str, float, str], None]


@dataclass(frozen=True)
class GuitarArrayResult:
    acoustic_guitar: np.ndarray
    lead_guitar: np.ndarray
    rhythm_guitar: np.ndarray
    electric_guitar: np.ndarray
    reports: dict[str, dict[str, object]]
    reconstruction: dict[str, float]

    @property
    def stems(self) -> dict[str, np.ndarray]:
        return {
            "acoustic_guitar": self.acoustic_guitar,
            "lead_guitar": self.lead_guitar,
            "rhythm_guitar": self.rhythm_guitar,
        }


@dataclass(frozen=True)
class SeparationResult:
    acoustic_guitar: Path
    lead_guitar: Path
    rhythm_guitar: Path
    manifest: Path


def audio_stats(audio: np.ndarray) -> dict[str, float]:
    values = audio.astype(np.float64, copy=False)
    return {
        "peak": float(np.max(np.abs(values))),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "mean": float(np.mean(values)),
    }


def validate_audio_array(audio: np.ndarray, *, name: str, frames: int | None = None) -> np.ndarray:
    if audio.ndim != 2 or audio.shape[0] != 2 or audio.shape[1] == 0:
        raise RuntimeError(f"AUDIO_SHAPE_INVALID:{name}:{audio.shape}")
    if frames is not None and audio.shape[1] != frames:
        raise RuntimeError(
            f"AUDIO_LENGTH_MISMATCH:{name}:expected={frames}:actual={audio.shape[1]}"
        )
    if not np.isfinite(audio).all():
        raise RuntimeError(f"AUDIO_NON_FINITE:{name}")
    return np.ascontiguousarray(audio, dtype=np.float32)


def load_audio(path: Path) -> np.ndarray:
    audio, sample_rate = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=False,
        dtype=np.float32,
        res_type="soxr_hq",
    )
    if sample_rate != SAMPLE_RATE:
        raise RuntimeError(f"SAMPLE_RATE_CONVERSION_FAILED:{sample_rate}")
    if audio.ndim == 1:
        audio = np.stack((audio, audio))
    return validate_audio_array(audio, name="input")


def write_float_wav(path: Path, audio: np.ndarray) -> None:
    audio = validate_audio_array(audio, name=path.stem)
    partial = path.with_name(path.name + ".part")
    try:
        sf.write(partial, audio.T, SAMPLE_RATE, format="WAV", subtype="FLOAT")
        info = sf.info(partial)
        if info.frames != audio.shape[-1] or info.channels != 2 or info.samplerate != SAMPLE_RATE:
            raise RuntimeError(f"OUTPUT_AUDIO_VERIFY_FAILED:{path.name}")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _output_metadata(path: Path, stats: dict[str, float]) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "format": "WAV IEEE float32",
        "sample_rate": SAMPLE_RATE,
        "channels": 2,
        "frames": sf.info(path).frames,
        "stats": stats,
    }


def _emit(callback: ProgressCallback | None, stage: str, fraction: float, message: str) -> None:
    if callback:
        callback(stage, min(1.0, max(0.0, fraction)), message)


def separate_guitar_arrays(
    mix: np.ndarray,
    model_root: Path,
    *,
    device_name: str = "auto",
    download_missing: bool = True,
    weights: Mapping[str, Path] | None = None,
    progress: ProgressCallback | None = None,
) -> GuitarArrayResult:
    """Run the fixed HQ6 chain without decoding or quantizing the input."""

    mix = validate_audio_array(mix, name="input")
    model_root = model_root.expanduser().resolve()
    if weights is None:
        _emit(progress, "models", 0.0, "正在校验分轨资源")

        def model_progress(key: str, fraction: float, message: str) -> None:
            model_index = next(index for index, spec in enumerate(MODEL_SPECS) if spec.key == key)
            _emit(
                progress,
                "models",
                (model_index + fraction) / len(MODEL_SPECS),
                "正在准备分轨资源",
            )

        resolved_weights = ensure_models(
            model_root,
            download_missing=download_missing,
            callback=model_progress,
        )
    else:
        resolved_weights = {key: Path(path).resolve() for key, path in weights.items()}
        missing = sorted(set(MODEL_BY_KEY) - set(resolved_weights))
        if missing:
            raise RuntimeError(f"MODEL_PATHS_MISSING:{','.join(missing)}")

    device = resolve_device(device_name)
    reports: dict[str, dict[str, object]] = {}

    def infer(key: str, source: np.ndarray) -> np.ndarray:
        spec = MODEL_BY_KEY[key]

        def model_inference_progress(fraction: float, message: str) -> None:
            _emit(progress, key, fraction, "正在分轨")

        estimate, report = predict_model(
            source,
            spec,
            resolved_weights[key],
            device,
            model_inference_progress,
        )
        reports[key] = report.to_dict()
        return validate_audio_array(estimate, name=key, frames=mix.shape[-1])

    acoustic = infer("acoustic", mix)
    electric = infer("electric", mix)
    lead = infer("lead", electric)
    rhythm = np.subtract(electric, lead, dtype=np.float32)
    rhythm = validate_audio_array(rhythm, name="rhythm", frames=mix.shape[-1])
    reconstruction_error = (
        lead.astype(np.float64) + rhythm.astype(np.float64) - electric.astype(np.float64)
    )
    reconstruction = {
        "rms": float(np.sqrt(np.mean(np.square(reconstruction_error)))),
        "peak": float(np.max(np.abs(reconstruction_error))),
    }
    return GuitarArrayResult(
        acoustic_guitar=acoustic,
        lead_guitar=lead,
        rhythm_guitar=rhythm,
        electric_guitar=electric,
        reports=reports,
        reconstruction=reconstruction,
    )


def separate_guitars(
    input_path: Path,
    output_dir: Path,
    model_root: Path,
    *,
    device_name: str = "auto",
    download_missing: bool = True,
    overwrite: bool = False,
    progress: ProgressCallback | None = None,
) -> SeparationResult:
    """Standalone, auditable three-stem entry point used by QA tooling."""

    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = {key: output_dir / name for key, name in OUTPUT_NAMES.items()}
    manifest_path = output_dir / "separation_manifest.json"
    existing = [path for path in (*destinations.values(), manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("OUTPUT_EXISTS:" + ",".join(str(path) for path in existing))

    started = time.perf_counter()
    _emit(progress, "decode", 0.0, "正在解码")
    mix = load_audio(input_path)
    _emit(progress, "decode", 1.0, "解码完成")
    arrays = separate_guitar_arrays(
        mix,
        model_root,
        device_name=device_name,
        download_missing=download_missing,
        progress=progress,
    )
    _emit(progress, "write", 0.0, "正在写入浮点中间轨")
    for index, (key, audio) in enumerate(arrays.stems.items()):
        write_float_wav(destinations[key], audio)
        _emit(progress, "write", (index + 1) / len(destinations), "正在写入分轨")

    output_metadata = {
        key: _output_metadata(destinations[key], audio_stats(audio))
        for key, audio in arrays.stems.items()
    }
    manifest = {
        "schema": 2,
        "pipeline": MODEL_SET_REVISION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "decoded_sample_rate": SAMPLE_RATE,
            "decoded_channels": 2,
            "decoded_frames": mix.shape[-1],
        },
        "runtime": {
            "device": str(resolve_device(device_name)),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "quality_policy": {
            "only_mode": "HQ6",
            "uncompressed_float_audio_path": True,
            "host_audio_and_overlap_add": "float32",
            "intermediate_peak_normalization": False,
            "lead_rhythm_mixture_consistency": "rhythm = electric - lead",
        },
        "models": {
            spec.key: {
                "name": spec.display_name,
                "architecture": spec.architecture,
                "target": spec.target,
                "file": spec.filename,
                "sha256": spec.sha256,
                "size": spec.size,
                "source": spec.source_url,
                "repository_revision": spec.repository_revision,
                "config_sha256": spec.config_sha256,
                "inference": arrays.reports[spec.key],
            }
            for spec in MODEL_SPECS
        },
        "implementation": {
            "msst_source_commit": MSST_SOURCE_COMMIT,
            "method": [
                "acoustic guitar from the original decoded float32 mix",
                "electric guitar from the same original decoded float32 mix",
                "lead guitar from the electric-guitar intermediate",
                "rhythm is the float32 complementary electric-guitar residual",
            ],
        },
        "intermediates": {"electric_guitar_stats": audio_stats(arrays.electric_guitar)},
        "consistency": {"lead_plus_rhythm_minus_electric": arrays.reconstruction},
        "outputs": output_metadata,
    }
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".part")
    try:
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return SeparationResult(
        acoustic_guitar=destinations["acoustic_guitar"],
        lead_guitar=destinations["lead_guitar"],
        rhythm_guitar=destinations["rhythm_guitar"],
        manifest=manifest_path,
    )
