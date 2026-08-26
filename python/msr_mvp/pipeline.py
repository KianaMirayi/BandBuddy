#!/usr/bin/env python3
"""Run a model-agnostic MSS -> MSR experiment.

Stages are ordinary argv arrays.  They never run through a shell, and exchange
audio through a tiny filesystem contract:

* MSS writes one audio file per stem directly under ``{output}``.
* MSR is invoked once per selected stem with ``{input}``, ``{output}`` and
  ``{stem}`` available as template variables.

This keeps the MVP independent of Demucs, a fixed stem taxonomy, and any one
research repository while still making every run reproducible in manifest.json.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable

from metrics import compare_wav_files, inspect_wav


SCHEMA_VERSION = 1
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".aac"}
STEM_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


class PipelineError(RuntimeError):
    """A bounded, user-actionable pipeline failure."""


@dataclass(frozen=True)
class CommandStage:
    id: str
    argv: tuple[str, ...]
    timeout_seconds: float
    environment: dict[str, str]
    select_stems: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineConfig:
    id: str
    mss: CommandStage
    msr: CommandStage | None


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PipelineError(f"CONFIG_INVALID:{label} must be an object")
    return value


def _stage(value: Any, label: str, *, restorer: bool) -> CommandStage:
    root = _object(value, label)
    stage_id = root.get("id")
    argv = root.get("argv")
    if not isinstance(stage_id, str) or not stage_id.strip():
        raise PipelineError(f"CONFIG_INVALID:{label}.id")
    if not isinstance(argv, list) or not argv or any(not isinstance(item, str) or not item for item in argv):
        raise PipelineError(f"CONFIG_INVALID:{label}.argv")
    timeout = root.get("timeoutSeconds", 0)
    if not isinstance(timeout, (int, float)) or timeout < 0:
        raise PipelineError(f"CONFIG_INVALID:{label}.timeoutSeconds")
    environment = root.get("environment", {})
    if not isinstance(environment, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in environment.items()):
        raise PipelineError(f"CONFIG_INVALID:{label}.environment")
    select = root.get("selectStems", []) if restorer else []
    if not isinstance(select, list) or any(not isinstance(item, str) or not STEM_NAME.fullmatch(item) for item in select):
        raise PipelineError(f"CONFIG_INVALID:{label}.selectStems")
    if len(set(select)) != len(select):
        raise PipelineError(f"CONFIG_INVALID:{label}.selectStems contains duplicates")
    return CommandStage(stage_id, tuple(argv), float(timeout), dict(environment), tuple(select))


def load_config(path: Path) -> PipelineConfig:
    try:
        raw = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PipelineError(f"CONFIG_READ_FAILED:{error}") from error
    root = _object(raw, "root")
    if root.get("schemaVersion") != SCHEMA_VERSION:
        raise PipelineError(f"CONFIG_SCHEMA_UNSUPPORTED:{root.get('schemaVersion')}")
    pipeline_id = root.get("id")
    if not isinstance(pipeline_id, str) or not pipeline_id.strip():
        raise PipelineError("CONFIG_INVALID:id")
    mss = _stage(root.get("mss"), "mss", restorer=False)
    msr_value = root.get("msr")
    msr = None if msr_value is None else _stage(msr_value, "msr", restorer=True)
    return PipelineConfig(pipeline_id, mss, msr)


def parse_variables(items: Iterable[str]) -> dict[str, str]:
    variables: dict[str, str] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", key):
            raise PipelineError(f"VARIABLE_INVALID:{item}")
        variables[key] = value
    return variables


class StrictVariables(dict[str, str]):
    def __missing__(self, key: str) -> str:
        raise PipelineError(f"VARIABLE_MISSING:{key}")


def expand(value: str, variables: dict[str, str]) -> str:
    try:
        return value.format_map(StrictVariables(variables))
    except ValueError as error:
        raise PipelineError(f"TEMPLATE_INVALID:{value}:{error}") from error


def emit(kind: str, **payload: Any) -> None:
    print(json.dumps({"type": kind, **payload}, ensure_ascii=False), flush=True)


def run_stage(stage: CommandStage, variables: dict[str, str], *, phase: str) -> dict[str, Any]:
    argv = [expand(item, variables) for item in stage.argv]
    environment = os.environ.copy()
    environment.update({key: expand(value, variables) for key, value in stage.environment.items()})
    emit("progress", phase=phase, stage=stage.id, message=f"运行 {stage.id}")
    started = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            check=False,
            shell=False,
            env=environment,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=stage.timeout_seconds or None,
        )
    except FileNotFoundError as error:
        raise PipelineError(f"STAGE_EXECUTABLE_MISSING:{stage.id}:{argv[0]}") from error
    except subprocess.TimeoutExpired as error:
        raise PipelineError(f"STAGE_TIMEOUT:{stage.id}:{stage.timeout_seconds}") from error
    elapsed = time.monotonic() - started
    if completed.stdout:
        for line in completed.stdout.splitlines():
            emit("stageOutput", stage=stage.id, stream="stdout", message=line[-2000:])
    if completed.stderr:
        for line in completed.stderr.splitlines():
            emit("stageOutput", stage=stage.id, stream="stderr", message=line[-2000:])
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-2000:]
        raise PipelineError(f"STAGE_FAILED:{stage.id}:{completed.returncode}:{detail}")
    return {"id": stage.id, "elapsedSeconds": round(elapsed, 3), "argv": argv}


def normalize_stem_name(filename: str) -> str:
    stem = Path(filename).stem.strip().lower().replace(" ", "_")
    if not STEM_NAME.fullmatch(stem):
        raise PipelineError(f"INVALID_STEM_FILENAME:{filename}")
    return stem


def discover_stems(directory: Path) -> dict[str, Path]:
    stems: dict[str, Path] = {}
    for candidate in sorted(directory.iterdir() if directory.is_dir() else []):
        if not candidate.is_file() or candidate.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        stem = normalize_stem_name(candidate.name)
        if stem in stems:
            raise PipelineError(f"DUPLICATE_STEM:{stem}")
        if candidate.stat().st_size <= 44:
            raise PipelineError(f"EMPTY_STEM:{stem}")
        stems[stem] = candidate.resolve()
    if not stems:
        raise PipelineError("NO_STEMS_PRODUCED")
    return stems


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_destination(root: Path, stem: str, suffix: str) -> Path:
    destination = root / f"{stem}{suffix.lower()}"
    if destination.exists():
        raise PipelineError(f"OUTPUT_COLLISION:{destination}")
    return destination


def run_pipeline(config: PipelineConfig, source: Path, destination: Path, supplied: dict[str, str]) -> dict[str, Any]:
    if not source.is_file():
        raise PipelineError(f"SOURCE_MISSING:{source}")
    if destination.exists() and any(destination.iterdir()):
        raise PipelineError(f"OUTPUT_NOT_EMPTY:{destination}")
    destination.mkdir(parents=True, exist_ok=True)
    work = destination / ".work"
    mss_root = work / "mss"
    msr_root = work / "msr"
    final_root = work / "final"
    for directory in (mss_root, msr_root, final_root):
        directory.mkdir(parents=True, exist_ok=False)

    variables = {
        "workspace": str(Path.cwd().resolve()),
        "python": sys.executable,
        "source": str(source.resolve()),
        "input": str(source.resolve()),
        "output": str(mss_root.resolve()),
        "work": str(work.resolve()),
        **supplied,
    }
    stage_runs: list[dict[str, Any]] = []
    stage_runs.append(run_stage(config.mss, variables, phase="mss"))
    stems = discover_stems(mss_root)
    emit("progress", phase="mss", stage=config.mss.id, stems=sorted(stems), message=f"MSS 生成 {len(stems)} 轨")

    restored: set[str] = set()
    selected = set(config.msr.select_stems if config.msr else ())
    if config.msr and not selected:
        selected = set(stems)
    missing = selected - set(stems)
    if missing:
        raise PipelineError(f"MSR_SELECTED_STEM_MISSING:{','.join(sorted(missing))}")

    for index, (stem, input_path) in enumerate(stems.items(), start=1):
        output_path = unique_destination(msr_root, stem, input_path.suffix) if stem in selected else None
        if config.msr and output_path:
            per_stem = {
                **variables,
                "input": str(input_path),
                "output": str(output_path),
                "stem": stem,
            }
            run_info = run_stage(config.msr, per_stem, phase="msr")
            run_info["stem"] = stem
            stage_runs.append(run_info)
            if not output_path.is_file() or output_path.stat().st_size <= 44:
                raise PipelineError(f"MSR_OUTPUT_MISSING:{stem}")
            restored.add(stem)
            final_source = output_path
        else:
            final_source = input_path
        final_path = unique_destination(final_root, stem, final_source.suffix)
        shutil.copy2(final_source, final_path)
        emit("progress", phase="msr", progress=index / len(stems), stem=stem, restored=stem in restored)

    files: list[dict[str, Any]] = []
    final_stems = discover_stems(final_root)
    for stem, final_path in final_stems.items():
        published = destination / final_path.name
        os.replace(final_path, published)
        item: dict[str, Any] = {
            "stem": stem,
            "file": published.name,
            "restored": stem in restored,
            "bytes": published.stat().st_size,
            "sha256": sha256(published),
        }
        inspection = inspect_wav(published)
        if inspection is not None:
            item["audio"] = inspection
        before = stems[stem]
        comparison = compare_wav_files(before, published) if stem in restored else None
        if comparison is not None:
            item["msrChange"] = comparison
        files.append(item)

    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "pipeline": config.id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "source": {"name": source.name, "sha256": sha256(source)},
        "stems": [item["stem"] for item in files],
        "restoredStems": sorted(restored),
        "files": files,
        "stageRuns": stage_runs,
        "qualityNote": "No clean reference was supplied; msrChange measures signal change, not restoration quality.",
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    shutil.rmtree(work)
    emit("result", output=str(destination.resolve()), manifest=str(manifest_path.resolve()), stems=manifest["stems"])
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BandBuddy model-agnostic MSS -> MSR MVP")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run one configured pipeline")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--var", action="append", default=[], metavar="NAME=VALUE")
    validate = subparsers.add_parser("validate", help="validate and summarize a config")
    validate.add_argument("--config", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_config(args.config.resolve())
        if args.command == "validate":
            emit("result", pipeline=config.id, mss=config.mss.id, msr=config.msr.id if config.msr else None)
        else:
            run_pipeline(config, args.input.resolve(), args.output.resolve(), parse_variables(args.var))
        return 0
    except KeyboardInterrupt:
        emit("error", code="CANCELLED", message="任务已取消")
        return 130
    except PipelineError as error:
        emit("error", code=str(error).split(":", 1)[0], message=str(error))
        return 1
    except Exception as error:
        emit("error", code="MVP_FAILED", message=f"{type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
