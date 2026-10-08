"""Render/check only the generated plan block from authoritative JSON."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from config import PROJECT_ROOT
from .configuration import CONFIG_PATH, INTEGRITY_FIELDS, load_config, seal_config, validate_config
BEGIN = b"<!-- BEGIN GENERATED MODELING CONFIG -->"
END = b"<!-- END GENERATED MODELING CONFIG -->"


def render_block(config: dict) -> bytes:
    sections = [
        ("14.3 Global experiment policy", ["schema_version", "plan_revision", "environment", "data", "experiment"]),
        ("14.4 Frozen model scopes and search spaces", ["models", "neural", "stacking", "adaboost"]),
        ("14.5 Preprocessing matrix", ["preprocessing"]),
        ("14.6 Selection, threshold and reporting policies", ["selection", "losses"]),
        ("14.7 Test access gate", ["test_gate"]),
        ("14.8 Save/load, artifacts and resume contract", ["persistence"]),
        ("14.9 Artifact requirements by task/lifecycle", ["artifacts"]),
        ("14.10 Scratch/reference comparison contract", ["references"]),
    ]
    fence = chr(96) * 3
    parts = ["\nSource: configs/modeling.json (authoritative). Model entries are configuration, not implementations.\n"]
    for title, keys in sections:
        summary = {key: config[key] for key in keys if key not in INTEGRITY_FIELDS}
        text = json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        parts.append("\n### " + title + "\n\n" + fence + "json\n" + text + "\n" + fence + "\n")
    return "".join(parts).encode("utf-8")


def synchronize(config_path: Path, plan_path: Path, *, write: bool = False) -> None:
    config = load_config(config_path, verify_integrity=not write)
    sealed = seal_config(config)
    validate_config(sealed)
    # Compare LF semantics even when the source was checked out with CRLF.
    raw = plan_path.read_bytes().decode("utf-8-sig").replace("\r\n", "\n").encode("utf-8")
    if raw.count(BEGIN) != 1 or raw.count(END) != 1 or raw.index(BEGIN) > raw.index(END):
        raise ValueError("Plan must contain one ordered BEGIN/END marker pair")
    start, stop = raw.index(BEGIN) + len(BEGIN), raw.index(END)
    block = render_block(sealed)
    if write:
        # Preserve prose outside the block; canonical UTF-8 LF, no BOM.
        plan_path.write_bytes((raw[:start] + block + raw[stop:]).rstrip(b"\n") + b"\n")
        config_path.write_bytes((json.dumps(sealed, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
    elif raw[start:stop] != block:
        raise ValueError("Generated modeling config drift; run --write-plan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write-plan", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--plan", type=Path, default=PROJECT_ROOT / "MODEL_TRAINING_PLAN.md")
    args = parser.parse_args()
    synchronize(args.config, args.plan, write=args.write_plan)
    print("Modeling config and generated plan block match")


if __name__ == "__main__":
    main()
