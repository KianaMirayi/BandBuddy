from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import warnings

from .separator import separate_guitars
from .specs import MODEL_SET_REVISION


def _default_model_root() -> Path:
    override = os.environ.get("BAND_BUDDY_GUITAR_MODELS")
    if override:
        return Path(override)
    local_data = os.environ.get("LOCALAPPDATA")
    if local_data:
        return Path(local_data) / "BandBuddy" / "guitar-hq-models"
    return Path.home() / ".cache" / "bandbuddy" / "guitar-hq-models"


def _default_output(input_path: Path) -> Path:
    return input_path.parent / f"{input_path.stem}-guitar-hq6"


class ConsoleProgress:
    def __init__(self) -> None:
        self._last_stage = ""
        self._last_bucket = -1
        self._last_time = 0.0

    def __call__(self, stage: str, fraction: float, message: str) -> None:
        now = time.monotonic()
        bucket = int(fraction * 20)
        changed = stage != self._last_stage
        if not (changed or fraction >= 1.0 or bucket != self._last_bucket or now - self._last_time >= 10):
            return
        print(f"[{stage:8}] {fraction * 100:6.1f}%  {message}", flush=True)
        self._last_stage = stage
        self._last_bucket = bucket
        self._last_time = now


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="guitar-separator-hq",
        description="运行 BandBuddy 固定高质量链路并输出木吉他、Lead 和 Rhythm 三轨。",
    )
    parser.add_argument("input", type=Path, help="输入音频")
    parser.add_argument("--output", type=Path, help="输出目录")
    parser.add_argument("--model-dir", type=Path, default=_default_model_root())
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda、cuda:0 或 mps")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings(
        "ignore",
        message=r"`torch\.backends\.cuda\.sdp_kernel\(\)` is deprecated.*",
        category=FutureWarning,
    )
    args = build_parser().parse_args(argv)
    input_path = args.input.expanduser().resolve()
    output_dir = (args.output or _default_output(input_path)).expanduser().resolve()
    try:
        result = separate_guitars(
            input_path,
            output_dir,
            args.model_dir,
            device_name=args.device,
            download_missing=not args.offline,
            overwrite=args.overwrite,
            progress=ConsoleProgress(),
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "pipeline": MODEL_SET_REVISION,
                "acoustic_guitar": str(result.acoustic_guitar),
                "lead_guitar": str(result.lead_guitar),
                "rhythm_guitar": str(result.rhythm_guitar),
                "manifest": str(result.manifest),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0
