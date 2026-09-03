from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import hashlib
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as torch_functional
import yaml

from ._vendor.msst import BSRoformer, MelBandRoformer
from .specs import ModelSpec, config_path


HQ_BIG_SHIFTS = 2
HQ_TTA_VARIANTS = ("identity", "channel_swap", "polarity")
InferenceProgress = Callable[[float, str], None]


@dataclass(frozen=True)
class InferenceReport:
    architecture: str
    target: str
    chunk_size: int
    overlap: int
    big_shifts: int
    tta_variants: tuple[str, ...]
    passes: int
    amp_enabled: bool
    parameters: int
    seconds: float
    peak_cuda_bytes: int

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["tta_variants"] = list(self.tta_variants)
        return value


def resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_NOT_AVAILABLE")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS_NOT_AVAILABLE")
    return device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(spec: ModelSpec) -> dict[str, object]:
    path = config_path(spec)
    if _sha256(path) != spec.config_sha256:
        raise RuntimeError(f"CONFIG_HASH_MISMATCH:{spec.key}:{path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
    if not isinstance(config, dict):
        raise RuntimeError(f"CONFIG_INVALID:{spec.key}")
    training = config.get("training")
    if not isinstance(training, dict) or training.get("target_instrument") != spec.target:
        raise RuntimeError(f"CONFIG_TARGET_MISMATCH:{spec.key}")
    return config


def _build_model(spec: ModelSpec, config: dict[str, object]) -> torch.nn.Module:
    kwargs = config.get("model")
    if not isinstance(kwargs, dict):
        raise RuntimeError(f"CONFIG_MODEL_MISSING:{spec.key}")
    if spec.architecture == "bs_roformer":
        return BSRoformer(**kwargs)
    if spec.architecture == "mel_band_roformer":
        return MelBandRoformer(**kwargs)
    raise RuntimeError(f"ARCHITECTURE_UNSUPPORTED:{spec.architecture}")


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"CHECKPOINT_INVALID:{path}")
    for key in ("state", "state_dict", "model_state_dict"):
        nested = checkpoint.get(key)
        if isinstance(nested, dict):
            checkpoint = nested
            break
    if not checkpoint or not all(isinstance(key, str) for key in checkpoint):
        raise RuntimeError(f"CHECKPOINT_STATE_INVALID:{path}")
    return checkpoint


def _linear_window(chunk_size: int) -> torch.Tensor:
    fade_size = chunk_size // 10
    window = torch.ones(chunk_size, dtype=torch.float32)
    window[:fade_size] = torch.linspace(0.0, 1.0, fade_size)
    window[-fade_size:] = torch.linspace(1.0, 0.0, fade_size)
    return window


def _demix_once(
    model: torch.nn.Module,
    mix: np.ndarray,
    *,
    chunk_size: int,
    overlap: int,
    use_amp: bool,
    device: torch.device,
    progress: InferenceProgress | None,
) -> np.ndarray:
    if mix.ndim != 2 or mix.shape[0] != 2 or mix.shape[1] == 0:
        raise ValueError(f"EXPECTED_STEREO_CHANNEL_FIRST:{mix.shape}")
    if chunk_size <= 0 or overlap <= 0 or chunk_size // overlap <= 0:
        raise ValueError("INVALID_CHUNK_SETTINGS")

    step = chunk_size // overlap
    border = chunk_size - step
    original_length = mix.shape[-1]
    padded = torch.from_numpy(np.ascontiguousarray(mix, dtype=np.float32))
    used_border = original_length > 2 * border and border > 0
    if used_border:
        padded = torch_functional.pad(padded, (border, border), mode="reflect")

    window_template = _linear_window(chunk_size)
    result = torch.zeros((1, 2, padded.shape[-1]), dtype=torch.float32)
    counter = torch.zeros_like(result)
    starts = list(range(0, padded.shape[-1], step))
    amp_enabled = use_amp and device.type == "cuda"

    with torch.inference_mode():
        for index, start in enumerate(starts):
            part = padded[:, start : start + chunk_size]
            segment_length = part.shape[-1]
            missing = chunk_size - segment_length
            if missing:
                pad_mode = "reflect" if segment_length > chunk_size // 2 else "constant"
                part = torch_functional.pad(part, (0, missing), mode=pad_mode)
            part = part.unsqueeze(0).to(device)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                predicted = model(part)
            if predicted.ndim == 3:
                predicted = predicted.unsqueeze(1)
            if predicted.ndim != 4 or predicted.shape[1] != 1 or predicted.shape[2] != 2:
                raise RuntimeError(f"MODEL_OUTPUT_SHAPE_INVALID:{tuple(predicted.shape)}")
            predicted = predicted[0, :, :, :segment_length].float().cpu()

            window = window_template.clone()
            if index == 0:
                window[: chunk_size // 10] = 1.0
            if index == len(starts) - 1:
                window[-chunk_size // 10 :] = 1.0
            weights = window[:segment_length]
            result[..., start : start + segment_length] += predicted * weights
            counter[..., start : start + segment_length] += weights
            if progress:
                progress((index + 1) / len(starts), "分块推理")

    estimate = (result / counter.clamp_min_(1e-10)).numpy()[0]
    if used_border:
        estimate = estimate[:, border:-border]
    estimate = estimate[:, :original_length]
    if not np.isfinite(estimate).all():
        raise RuntimeError("MODEL_OUTPUT_NON_FINITE")
    return np.ascontiguousarray(estimate, dtype=np.float32)


def _augment(mix: np.ndarray, variant: str) -> np.ndarray:
    if variant == "identity":
        return mix
    if variant == "channel_swap":
        return mix[::-1].copy()
    if variant == "polarity":
        return -mix
    raise ValueError(f"UNKNOWN_TTA_VARIANT:{variant}")


def _undo_augment(estimate: np.ndarray, variant: str) -> np.ndarray:
    if variant == "identity":
        return estimate
    if variant == "channel_swap":
        return estimate[::-1].copy()
    if variant == "polarity":
        return -estimate
    raise ValueError(f"UNKNOWN_TTA_VARIANT:{variant}")


def _high_quality_predict(
    model: torch.nn.Module,
    mix: np.ndarray,
    *,
    chunk_size: int,
    overlap: int,
    use_amp: bool,
    device: torch.device,
    progress: InferenceProgress | None,
) -> np.ndarray:
    pass_count = len(HQ_TTA_VARIANTS) * HQ_BIG_SHIFTS
    accumulation = np.zeros_like(mix, dtype=np.float32)
    pass_index = 0
    shift_size = mix.shape[-1] // HQ_BIG_SHIFTS

    for variant in HQ_TTA_VARIANTS:
        augmented = _augment(mix, variant)
        for shift_index in range(HQ_BIG_SHIFTS):
            shift = shift_index * shift_size
            shifted = np.roll(augmented, shift, axis=-1) if shift else augmented

            def report_pass(fraction: float, message: str) -> None:
                if progress:
                    progress(
                        (pass_index + fraction) / pass_count,
                        f"HQ6 {variant} {shift_index + 1}/{HQ_BIG_SHIFTS} · {message}",
                    )

            estimate = _demix_once(
                model,
                shifted,
                chunk_size=chunk_size,
                overlap=overlap,
                use_amp=use_amp,
                device=device,
                progress=report_pass,
            )
            if shift:
                estimate = np.roll(estimate, -shift, axis=-1)
            accumulation += _undo_augment(estimate, variant)
            pass_index += 1
    accumulation /= float(pass_count)
    return accumulation


def predict_model(
    mix: np.ndarray,
    spec: ModelSpec,
    checkpoint_path: Path,
    device: torch.device,
    progress: InferenceProgress | None = None,
) -> tuple[np.ndarray, InferenceReport]:
    config = load_config(spec)
    inference = config.get("inference")
    audio = config.get("audio")
    training = config.get("training")
    if not isinstance(inference, dict) or not isinstance(audio, dict) or not isinstance(training, dict):
        raise RuntimeError(f"CONFIG_SECTIONS_MISSING:{spec.key}")
    chunk_size = int(inference.get("chunk_size", audio["chunk_size"]))
    overlap = int(inference["num_overlap"])
    use_amp = bool(training.get("use_amp", True))

    model = _build_model(spec, config)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != spec.trainable_parameters:
        raise RuntimeError(
            f"MODEL_PARAMETER_COUNT_MISMATCH:{spec.key}:expected={spec.trainable_parameters}:actual={parameters}"
        )
    state = _checkpoint_state(checkpoint_path)
    model.load_state_dict(state, strict=True)
    del state
    model.eval().to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    try:
        estimate = _high_quality_predict(
            model,
            mix,
            chunk_size=chunk_size,
            overlap=overlap,
            use_amp=use_amp,
            device=device,
            progress=progress,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        peak_cuda_bytes = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )
    finally:
        model.to("cpu")
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = InferenceReport(
        architecture=spec.architecture,
        target=spec.target,
        chunk_size=chunk_size,
        overlap=overlap,
        big_shifts=HQ_BIG_SHIFTS,
        tta_variants=HQ_TTA_VARIANTS,
        passes=len(HQ_TTA_VARIANTS) * HQ_BIG_SHIFTS,
        amp_enabled=use_amp and device.type == "cuda",
        parameters=parameters,
        seconds=seconds,
        peak_cuda_bytes=peak_cuda_bytes,
    )
    return estimate, report
