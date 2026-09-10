from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]

from guitar_separator_hq.separator import load_audio, separate_guitar_arrays
from guitar_separator_hq.specs import model_specs_for_quality


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(r"C:\CloudMusic\新裤子 - 没有理想的人不伤心.mp3"),
    )
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quality", choices=("fast", "balanced", "high"), default="fast")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mix = load_audio(args.input)
    if args.seconds:
        mix = np.ascontiguousarray(mix[:, : int(args.seconds * 44_100)])
    bundle = EXPERIMENT_DIR / "model-bundle-v2.0.1"
    weights = {
        spec.key: bundle / spec.filename
        for spec in model_specs_for_quality(args.quality)
    }
    started = time.perf_counter()
    result = separate_guitar_arrays(
        mix,
        bundle,
        device_name=args.device,
        download_missing=False,
        weights=weights,
        quality=args.quality,
        progress=lambda stage, fraction, message: print(
            json.dumps({"stage": stage, "fraction": fraction, "message": message}),
            flush=True,
        ),
    )
    elapsed = time.perf_counter() - started
    args.output.mkdir(parents=True, exist_ok=True)
    for name, audio in result.stems.items():
        sf.write(args.output / f"{name}.wav", audio.T, 44_100, subtype="FLOAT")
    report = {
        "input": str(args.input.resolve()),
        "frames": int(mix.shape[-1]),
        "duration_seconds": mix.shape[-1] / 44_100,
        "quality": args.quality,
        "elapsed_seconds": elapsed,
        "reports": result.reports,
        "reconstruction": result.reconstruction,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
