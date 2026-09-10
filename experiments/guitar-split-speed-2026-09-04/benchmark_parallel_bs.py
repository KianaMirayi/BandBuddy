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


def load(key: str, root: Path, device: torch.device):
    spec = MODEL_BY_KEY[key]
    config = load_config(spec)
    model = _build_model(spec, config)
    model.load_state_dict(_checkpoint_state(root / spec.filename), strict=True)
    return model.eval().requires_grad_(False).to(device), config


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
    acoustic, config = load("acoustic", args.models, device)
    electric, _ = load("electric", args.models, device)
    chunk_size = int(config["inference"]["chunk_size"])
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

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        acoustic(data).float().cpu()
        electric(data).float().cpu()
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    sequential = None
    for _ in range(args.iterations):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            left = acoustic(data)
            right = electric(data)
        sequential = (left.float().cpu().numpy(), right.float().cpu().numpy())
    torch.cuda.synchronize(device)
    sequential_seconds = time.perf_counter() - started
    sequential_peak = int(torch.cuda.max_memory_allocated(device))

    stream_a = torch.cuda.Stream()
    stream_e = torch.cuda.Stream()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    concurrent = None
    for _ in range(args.iterations):
        stream_a.wait_stream(torch.cuda.current_stream())
        stream_e.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream_a), torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.float16
        ):
            left = acoustic(data)
        with torch.cuda.stream(stream_e), torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.float16
        ):
            right = electric(data)
        torch.cuda.current_stream().wait_stream(stream_a)
        torch.cuda.current_stream().wait_stream(stream_e)
        concurrent = (left.float().cpu().numpy(), right.float().cpu().numpy())
    torch.cuda.synchronize(device)
    concurrent_seconds = time.perf_counter() - started
    concurrent_peak = int(torch.cuda.max_memory_allocated(device))
    assert sequential is not None and concurrent is not None

    deltas = []
    for ref, test in zip(sequential, concurrent):
        delta = test.astype(np.float64) - ref.astype(np.float64)
        deltas.append(
            {
                "rms": float(np.sqrt(np.mean(np.square(delta)))),
                "peak": float(np.max(np.abs(delta))),
            }
        )
    report = {
        "iterations": args.iterations,
        "chunk_size": chunk_size,
        "sequential_seconds": sequential_seconds,
        "concurrent_seconds": concurrent_seconds,
        "speedup": sequential_seconds / concurrent_seconds,
        "sequential_peak_cuda_bytes": sequential_peak,
        "concurrent_peak_cuda_bytes": concurrent_peak,
        "differences": {"acoustic": deltas[0], "electric": deltas[1]},
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
