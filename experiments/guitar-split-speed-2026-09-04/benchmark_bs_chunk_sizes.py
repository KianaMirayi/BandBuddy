from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

module_spec = importlib.util.spec_from_file_location(
    "hq_policy_benchmark", EXPERIMENT_DIR / "benchmark_hq_policies.py"
)
if module_spec is None or module_spec.loader is None:
    raise RuntimeError("cannot load benchmark_hq_policies.py")
hq = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(hq)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("acoustic", "electric"), default="acoustic")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "测试用例" / "新裤子 - 没有理想的人不伤心.mp3",
    )
    parser.add_argument("--start", type=float, default=60.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument(
        "--chunk-seconds", nargs="+", type=float, default=[10.0, 12.771, 15.0, 20.0]
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
    full_mix = hq.load_audio(args.input)
    start = int(round(args.start * 44_100))
    frames = int(round(args.duration * 44_100))
    mix = np.ascontiguousarray(full_mix[:, start : start + frames])
    if mix.shape[-1] != frames:
        raise ValueError("requested segment exceeds input")
    del full_mix

    model, configured_chunk, overlap, use_amp = hq.load_model(args.model, args.models, device)
    trials = []
    outputs = {}
    # Run the configured reference last so its cold-start does not penalize candidates.
    ordered_seconds = [value for value in args.chunk_seconds if value != 20.0]
    if 20.0 in args.chunk_seconds:
        ordered_seconds.append(20.0)

    for chunk_seconds in ordered_seconds:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        chunk_size = int(round(chunk_seconds * 44_100))
        output, timings = hq.run_passes(
            model,
            mix,
            (("identity", 0),),
            chunk_size=chunk_size,
            overlap=overlap,
            batch_size=1,
            device=device,
            use_amp=use_amp,
        )
        estimate = output["identity:0"]
        outputs[chunk_seconds] = estimate
        row = {
            "chunk_seconds": chunk_seconds,
            "chunk_size": chunk_size,
            "seconds": timings["identity:0"],
            "realtime_factor": timings["identity:0"] / args.duration,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
        trials.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    reference_key = 20.0 if 20.0 in outputs else ordered_seconds[-1]
    reference = outputs[reference_key]
    for row in trials:
        row["vs_reference_chunk"] = hq.compare(reference, outputs[row["chunk_seconds"]])

    report = {
        "model": args.model,
        "input": str(args.input.resolve()),
        "segment_start_seconds": args.start,
        "segment_duration_seconds": args.duration,
        "configured_chunk_samples": configured_chunk,
        "training_chunk_samples": 441_000,
        "overlap": overlap,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "reference_chunk_seconds": reference_key,
        "trials": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
