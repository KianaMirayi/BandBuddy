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

from guitar_separator_hq.inference import _build_model, _checkpoint_state, load_config
from guitar_separator_hq.separator import load_audio
from guitar_separator_hq.specs import MODEL_BY_KEY


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref = reference.astype(np.float64, copy=False)
    test = candidate.astype(np.float64, copy=False)
    delta = test - ref
    ref_rms = float(np.sqrt(np.mean(np.square(ref))))
    error_rms = float(np.sqrt(np.mean(np.square(delta))))
    return {
        "reference_rms": ref_rms,
        "difference_rms": error_rms,
        "difference_peak": float(np.max(np.abs(delta))),
        "difference_snr_db": (
            float("inf") if error_rms == 0 else 20.0 * math.log10(ref_rms / error_rms)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODEL_BY_KEY), required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=30)
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
    host_input = part.unsqueeze(0).repeat(args.batch, 1, 1)
    gpu_input = host_input.to(device)

    model = _build_model(spec, config)
    model.load_state_dict(_checkpoint_state(args.models / spec.filename), strict=True)
    model.eval().requires_grad_(False).to(device)

    def eager_once() -> torch.Tensor:
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.float16, enabled=use_amp
        ):
            return model(gpu_input)

    # Initialize FFT plans and attention kernels before timing or capture.
    eager_once().float().cpu()
    torch.cuda.synchronize(device)
    eager_reference = eager_once().float().cpu().numpy()
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    eager_output = None
    for _ in range(args.iterations):
        eager_output = eager_once().float().cpu()
    torch.cuda.synchronize(device)
    eager_seconds = time.perf_counter() - started
    eager_peak = int(torch.cuda.max_memory_allocated(device))
    assert eager_output is not None

    static_input = torch.empty_like(gpu_input)
    static_input.copy_(gpu_input)
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.float16, enabled=use_amp
            ):
                model(static_input)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize(device)

    torch.cuda.empty_cache()
    capture_started = time.perf_counter()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.float16, enabled=use_amp
        ):
            static_output = model(static_input)
    torch.cuda.synchronize(device)
    capture_seconds = time.perf_counter() - capture_started

    static_input.copy_(gpu_input)
    graph.replay()
    graph_reference = static_output.float().cpu().numpy()
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    graph_output = None
    for _ in range(args.iterations):
        static_input.copy_(gpu_input)
        graph.replay()
        graph_output = static_output.float().cpu()
    torch.cuda.synchronize(device)
    graph_seconds = time.perf_counter() - started
    graph_peak = int(torch.cuda.max_memory_allocated(device))
    assert graph_output is not None

    report = {
        "model": args.model,
        "batch": args.batch,
        "iterations": args.iterations,
        "chunk_size": chunk_size,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "capture_seconds": capture_seconds,
        "eager": {
            "seconds": eager_seconds,
            "seconds_per_forward": eager_seconds / args.iterations,
            "peak_cuda_bytes": eager_peak,
        },
        "cuda_graph": {
            "seconds": graph_seconds,
            "seconds_per_forward": graph_seconds / args.iterations,
            "peak_cuda_bytes": graph_peak,
        },
        "speedup": eager_seconds / graph_seconds,
        "graph_vs_eager": compare(eager_reference, graph_reference),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
