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
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8])
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    reference_lead = hq.read_audio(args.reference / "lead_guitar.wav")
    reference_rhythm = hq.read_audio(args.reference / "rhythm_guitar.wav")
    electric = np.add(reference_lead, reference_rhythm, dtype=np.float32)
    model, chunk_size, overlap, use_amp = hq.load_model("lead", args.models, device)

    report = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "trials": [],
    }
    for batch_size in args.batches:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        outputs, timings = hq.run_passes(
            model,
            electric,
            hq.ALL_PASSES,
            chunk_size=chunk_size,
            overlap=overlap,
            batch_size=batch_size,
            device=device,
            use_amp=use_amp,
        )
        candidate = hq.average_passes(outputs, hq.ALL_PASSES)
        row = {
            "batch_size": batch_size,
            "seconds": sum(timings.values()),
            "pass_seconds": timings,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
            "vs_original_hq6": hq.compare(reference_lead, candidate),
        }
        report["trials"].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
