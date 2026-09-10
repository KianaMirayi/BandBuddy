from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def unwrap(path: Path) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state", "state_dict", "model_state_dict"):
        if isinstance(value, dict) and isinstance(value.get(key), dict):
            value = value[key]
            break
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("acoustic", type=Path)
    parser.add_argument("electric", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    left = unwrap(args.acoustic)
    right = unwrap(args.electric)
    if left.keys() != right.keys():
        raise ValueError("checkpoint keys differ")
    groups: dict[str, dict[str, float | int]] = {}
    total_numel = 0
    identical_numel = 0
    identical_tensors = 0
    for key in left:
        a = left[key]
        b = right[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(key)
        group = key.split(".", 1)[0]
        row = groups.setdefault(
            group,
            {
                "tensors": 0,
                "numel": 0,
                "identical_tensors": 0,
                "identical_numel": 0,
                "sum_squared_reference": 0.0,
                "sum_squared_difference": 0.0,
            },
        )
        numel = a.numel()
        same = torch.equal(a, b)
        row["tensors"] += 1
        row["numel"] += numel
        total_numel += numel
        if same:
            row["identical_tensors"] += 1
            row["identical_numel"] += numel
            identical_tensors += 1
            identical_numel += numel
        if a.is_floating_point():
            af = a.float()
            delta = b.float() - af
            row["sum_squared_reference"] += float(torch.sum(af * af))
            row["sum_squared_difference"] += float(torch.sum(delta * delta))

    for row in groups.values():
        ref = float(row.pop("sum_squared_reference"))
        delta = float(row.pop("sum_squared_difference"))
        row["relative_l2_difference"] = (delta / ref) ** 0.5 if ref else 0.0
        row["identical_fraction"] = int(row["identical_numel"]) / int(row["numel"])

    report = {
        "tensors": len(left),
        "total_numel": total_numel,
        "identical_tensors": identical_tensors,
        "identical_numel": identical_numel,
        "identical_fraction": identical_numel / total_numel,
        "groups": groups,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
