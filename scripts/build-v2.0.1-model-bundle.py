#!/usr/bin/env python3
"""Build and audit the immutable BandBuddy 2.0.1 model bundle.

The acoustic/electric BS-RoFormer checkpoints have an identical trunk.  This
script verifies that invariant and emits one two-head checkpoint containing the
trunk once.  All other model files are copied byte-for-byte after validating
their pinned upstream hashes.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))

from guitar_separator_hq._vendor.msst import BSRoformer  # noqa: E402


SOURCE_FILES = {
    "six_stem": (
        "5c90dfd2-34c22ccb.th",
        54_996_327,
        "34c22ccb381c6f9fdbf324f04e1e2fe21aaaf293f5ded163a162697ff9a02ddd",
    ),
    "acoustic_original": (
        "bs_mega_53stem_acoustic-guitar_mvsep.ckpt",
        77_624_038,
        "fa386b2e7b1ea4f12b9b5c557444c0dc78648ef4ee299de2759d86457e182b3e",
    ),
    "electric_original": (
        "bs_mega_53stem_electric-guitar_mvsep.ckpt",
        77_624_038,
        "cd506bfce9474f91a31001967f2c4935ce4e67f643da3df20d04058da927c553",
    ),
    "lead_hq": (
        "mbr_lead_rhythm_guitar_listra92.ckpt",
        337_073_664,
        "b3c47bca33609ca1ba0bb2d2076410bfd1eb941b051b72afc1f3e24d12b17eef",
    ),
    "acoustic_fast": (
        "mdx_6s_acoustic_guitar_anvuew.onnx",
        27_147_460,
        "2bd8f2af629b279cc1a568f895ee9636f7ce2d76c69aa601e6744eaab8b4916a",
    ),
    "electric_fast": (
        "mdx_6s_electric_guitar_anvuew.onnx",
        27_147_623,
        "bd6fcf40659771568ee180ea69bd4576a9c3d2423ae0f5f6f5afcc8b6a6fd938",
    ),
    "lead_fast": (
        "demucs4_lead_rhythm_guitar_drypaint.ckpt",
        109_822_623,
        "946ffd50d7f2fd87e447d880525283e88bd9061e1b428d4f1d380e764d54d618",
    ),
}
SHARED_FILENAME = "bs_mega_53stem_acoustic-electric_shared_mvsep.ckpt"
CONFIG_NAMES = (
    "shared_acoustic_electric.yaml",
    "lead_rhythm.yaml",
    "fast_acoustic_mdx.yaml",
    "fast_electric_mdx.yaml",
    "fast_lead_rhythm_htdemucs.yaml",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(path: Path, *, size: int, sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != size:
        raise RuntimeError(f"SOURCE_SIZE_MISMATCH:{path.name}")
    actual = sha256_file(path)
    if actual != sha256:
        raise RuntimeError(f"SOURCE_HASH_MISMATCH:{path.name}:{actual}")


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    value: Any = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"CHECKPOINT_INVALID:{path}")
    for key in ("state", "state_dict", "model_state_dict"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            value = nested
            break
    if not value or not all(isinstance(key, str) and isinstance(tensor, torch.Tensor) for key, tensor in value.items()):
        raise RuntimeError(f"CHECKPOINT_STATE_INVALID:{path}")
    return dict(value)


def combine_bs_state(
    acoustic: Mapping[str, torch.Tensor],
    electric: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], int, int]:
    if acoustic.keys() != electric.keys():
        raise RuntimeError("BS_CHECKPOINT_KEYS_DIFFER")
    combined: dict[str, torch.Tensor] = {}
    shared_values = 0
    head_values = 0
    for key, acoustic_tensor in acoustic.items():
        electric_tensor = electric[key]
        if key.startswith("mask_estimators.0."):
            suffix = key.removeprefix("mask_estimators.0.")
            combined[f"mask_estimators.0.{suffix}"] = acoustic_tensor
            combined[f"mask_estimators.1.{suffix}"] = electric_tensor
            head_values += acoustic_tensor.numel() + electric_tensor.numel()
            continue
        if not torch.equal(acoustic_tensor, electric_tensor):
            raise RuntimeError(f"BS_SHARED_TRUNK_MISMATCH:{key}")
        combined[key] = acoustic_tensor
        shared_values += acoustic_tensor.numel()
    return combined, shared_values, head_values


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".part")
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(state: Mapping[str, torch.Tensor], destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".part")
    try:
        with temporary.open("wb") as handle:
            torch.save(dict(state), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def file_record(key: str, path: Path, config: str | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "key": key,
        "path": path.name,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if config:
        config_path = path.parent / config
        record.update({"config": config, "configSha256": sha256_file(config_path)})
    return record


def build(args: argparse.Namespace) -> Path:
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    locations = {
        "six_stem": args.six_stem,
        "acoustic_original": args.acoustic,
        "electric_original": args.electric,
        "lead_hq": args.lead_hq,
        "acoustic_fast": args.acoustic_fast,
        "electric_fast": args.electric_fast,
        "lead_fast": args.lead_fast,
    }
    for key, source in locations.items():
        _, size, expected_hash = SOURCE_FILES[key]
        verify_source(source, size=size, sha256=expected_hash)

    acoustic_state = checkpoint_state(args.acoustic)
    electric_state = checkpoint_state(args.electric)
    combined_state, shared_values, head_values = combine_bs_state(
        acoustic_state, electric_state
    )
    shared_path = output / SHARED_FILENAME
    atomic_torch_save(combined_state, shared_path)

    import yaml

    config_root = REPOSITORY_ROOT / "python" / "guitar_separator_hq" / "configs"
    for config_name in CONFIG_NAMES:
        atomic_copy(config_root / config_name, output / config_name)
    shared_config = yaml.load(
        (config_root / "shared_acoustic_electric.yaml").read_text("utf-8"),
        Loader=yaml.FullLoader,
    )
    model = BSRoformer(**shared_config["model"])
    model.load_state_dict(combined_state, strict=True)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != 51_057_684:
        raise RuntimeError(f"SHARED_PARAMETER_COUNT_MISMATCH:{parameters}")

    for key in ("six_stem", "lead_hq", "acoustic_fast", "electric_fast", "lead_fast"):
        filename, _, _ = SOURCE_FILES[key]
        atomic_copy(locations[key], output / filename)

    files = [
        file_record("six_stem", output / SOURCE_FILES["six_stem"][0]),
        file_record("shared_acoustic_electric", shared_path, "shared_acoustic_electric.yaml"),
        file_record("lead_rhythm_hq", output / SOURCE_FILES["lead_hq"][0], "lead_rhythm.yaml"),
        file_record("acoustic_guitar_fast", output / SOURCE_FILES["acoustic_fast"][0], "fast_acoustic_mdx.yaml"),
        file_record("electric_guitar_fast", output / SOURCE_FILES["electric_fast"][0], "fast_electric_mdx.yaml"),
        file_record("lead_rhythm_fast", output / SOURCE_FILES["lead_fast"][0], "fast_lead_rhythm_htdemucs.yaml"),
    ]
    manifest = {
        "schemaVersion": 2,
        "bundleVersion": "v2.0.1",
        "minimumAppVersion": "2.0.1",
        "profiles": {
            "high": ["six_stem", "shared_acoustic_electric", "lead_rhythm_hq"],
            "balanced": ["six_stem", "shared_acoustic_electric", "lead_rhythm_hq"],
            "fast": ["six_stem", "acoustic_guitar_fast", "electric_guitar_fast", "lead_rhythm_fast"],
        },
        "files": files,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", "utf-8")
    audit = {
        "sourceHashesVerified": True,
        "sharedTrunkTensorValues": shared_values,
        "twoMaskHeadTensorValues": head_values,
        "combinedStateTensorValues": sum(tensor.numel() for tensor in combined_state.values()),
        "trainableParameters": parameters,
        "sharedCheckpoint": file_record("shared_acoustic_electric", shared_path),
    }
    (output / "build-audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    print(json.dumps({"output": str(output), "manifest": manifest, "audit": audit}, ensure_ascii=False, indent=2))
    return output


def parser() -> argparse.ArgumentParser:
    product = REPOSITORY_ROOT / ".codex-tmp" / "product-model-root-v2" / "bandbuddy-stems-v2.0.0"
    hq = REPOSITORY_ROOT / ".codex-tmp" / "guitar-hq-models"
    lightweight = REPOSITORY_ROOT / "experiments" / "guitar-split-speed-2026-09-04" / "lightweight-models" / "models" / "mdxnet"
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "experiments" / "guitar-split-speed-2026-09-04" / "model-bundle-v2.0.1")
    root.add_argument("--six-stem", type=Path, default=product / SOURCE_FILES["six_stem"][0])
    root.add_argument("--acoustic", type=Path, default=hq / SOURCE_FILES["acoustic_original"][0])
    root.add_argument("--electric", type=Path, default=hq / SOURCE_FILES["electric_original"][0])
    root.add_argument("--lead-hq", type=Path, default=hq / SOURCE_FILES["lead_hq"][0])
    root.add_argument("--acoustic-fast", type=Path, default=lightweight / SOURCE_FILES["acoustic_fast"][0])
    root.add_argument("--electric-fast", type=Path, default=lightweight / SOURCE_FILES["electric_fast"][0])
    root.add_argument("--lead-fast", type=Path, default=REPOSITORY_ROOT / ".codex-tmp" / "bd-analysis" / "models" / SOURCE_FILES["lead_fast"][0])
    return root


if __name__ == "__main__":
    build(parser().parse_args())
