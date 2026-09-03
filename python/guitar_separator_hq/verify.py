from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from .model_store import sha256_file
from .separator import OUTPUT_NAMES, SAMPLE_RATE


READ_FRAMES = SAMPLE_RATE * 10


def _stream_stats(path: Path) -> dict[str, object]:
    total = 0
    sum_value = 0.0
    sum_square = 0.0
    peak = 0.0
    non_finite = 0
    with sf.SoundFile(path) as source:
        for block in source.blocks(READ_FRAMES, dtype="float32", always_2d=True):
            values = block.astype(np.float64)
            finite = np.isfinite(values)
            non_finite += int(values.size - np.count_nonzero(finite))
            safe = np.where(finite, values, 0.0)
            peak = max(peak, float(np.max(np.abs(safe))))
            total += safe.size
            sum_value += float(np.sum(safe))
            sum_square += float(np.sum(np.square(safe)))
    if total == 0:
        raise RuntimeError(f"OUTPUT_EMPTY:{path.name}")
    return {
        "peak": peak,
        "rms": math.sqrt(sum_square / total),
        "mean": sum_value / total,
        "non_finite_samples": non_finite,
    }


def _lead_rhythm_blocks(lead_path: Path, rhythm_path: Path) -> dict[str, object]:
    reconstructed_hash = hashlib.sha256()
    rows: list[dict[str, float | int]] = []
    with sf.SoundFile(lead_path) as lead_source, sf.SoundFile(rhythm_path) as rhythm_source:
        index = 0
        while True:
            lead = lead_source.read(READ_FRAMES, dtype="float32", always_2d=True)
            rhythm = rhythm_source.read(READ_FRAMES, dtype="float32", always_2d=True)
            if lead.shape != rhythm.shape:
                raise RuntimeError("LEAD_RHYTHM_LENGTH_MISMATCH")
            if lead.size == 0:
                break
            electric = np.add(lead, rhythm, dtype=np.float32)
            reconstructed_hash.update(np.ascontiguousarray(electric.astype("<f4")).tobytes())
            rows.append(
                {
                    "start_seconds": index * 10,
                    "duration_seconds": lead.shape[0] / SAMPLE_RATE,
                    "lead_rms": float(np.sqrt(np.mean(lead.astype(np.float64) ** 2))),
                    "rhythm_rms": float(np.sqrt(np.mean(rhythm.astype(np.float64) ** 2))),
                    "electric_rms": float(np.sqrt(np.mean(electric.astype(np.float64) ** 2))),
                }
            )
            index += 1
    return {
        "reconstructed_electric_raw_float32_sha256": reconstructed_hash.hexdigest(),
        "blocks": rows,
    }


def verify_output_directory(output_dir: Path) -> dict[str, object]:
    output_dir = output_dir.expanduser().resolve()
    manifest_path = output_dir / "separation_manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    files: dict[str, dict[str, object]] = {}
    frame_counts: set[int] = set()
    for key, filename in OUTPUT_NAMES.items():
        path = output_dir / filename
        info = sf.info(path)
        expected = manifest["outputs"][key]
        actual_hash = sha256_file(path)
        if actual_hash != expected["sha256"]:
            raise RuntimeError(f"OUTPUT_HASH_MISMATCH:{key}")
        if info.samplerate != SAMPLE_RATE or info.channels != 2 or info.subtype != "FLOAT":
            raise RuntimeError(f"OUTPUT_FORMAT_MISMATCH:{key}:{info}")
        if info.frames != expected["frames"]:
            raise RuntimeError(f"OUTPUT_FRAMES_MISMATCH:{key}")
        frame_counts.add(info.frames)
        stats = _stream_stats(path)
        if stats["non_finite_samples"]:
            raise RuntimeError(f"OUTPUT_NON_FINITE:{key}")
        files[key] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual_hash,
            "frames": info.frames,
            "seconds": info.frames / SAMPLE_RATE,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "subtype": info.subtype,
            "stats": stats,
        }
    if len(frame_counts) != 1:
        raise RuntimeError(f"OUTPUT_LENGTHS_DIFFER:{sorted(frame_counts)}")
    input_path = Path(manifest["input"]["path"])
    if sha256_file(input_path) != manifest["input"]["sha256"]:
        raise RuntimeError("QA_INPUT_HASH_MISMATCH")
    return {
        "schema": 2,
        "status": "passed",
        "manifest": str(manifest_path),
        "pipeline": manifest["pipeline"],
        "files": files,
        "lead_rhythm_analysis": _lead_rhythm_blocks(
            output_dir / OUTPUT_NAMES["lead_guitar"],
            output_dir / OUTPUT_NAMES["rhythm_guitar"],
        ),
        "inference_reconstruction": manifest["consistency"]["lead_plus_rhythm_minus_electric"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="独立重读并校验 HQ6 三轨")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = verify_output_directory(args.output_dir)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        destination = args.report.expanduser().resolve()
        temporary = destination.with_name(destination.name + ".part")
        try:
            temporary.write_text(rendered, "utf-8")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
