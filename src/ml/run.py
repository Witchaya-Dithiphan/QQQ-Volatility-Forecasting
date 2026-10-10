"""CLI for the modeling layer: train / report / finalize.

report and finalize return 1 and say why rather than pretending to work: the
previous runner's habit of silently producing empty artifacts is what made the old
documentation untrustworthy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core.contracts import VARIANTS
from .core.trainer import train_model
from .registry import MODELS, model_names


def _load_grid(path: str | None, name: str) -> dict:
    if not path:
        return {}
    grids = json.loads(Path(path).read_text(encoding="utf-8"))
    return grids.get(name, {})


def _train(args) -> int:
    names = model_names() if args.all else [args.model]
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        print(f"Unknown model {unknown[0]!r}; available: {model_names()}", file=sys.stderr)
        return 1

    failures = 0
    for name in names:
        result = train_model(
            MODELS[name],
            grid=_load_grid(args.grid, name),
            variant=args.variant,
            output_root=Path(args.output_root) if args.output_root else None,
        )
        passed = result["load_verification"]["passed"]
        failures += 0 if passed else 1
        status = "ok" if passed else "RELOAD FAILED"
        print(
            f"{name:>20} | {args.variant:<10} | candidates={len(result['search_results']):>2} "
            f"| selected={result['selected']} | reload={status}"
        )
    return 1 if failures else 0


def _report(args) -> int:
    print(
        "report is not implemented yet; it needs every model trained first "
        "(see docs/tasks/claude-linear-family.md, Phase C1)",
        file=sys.stderr,
    )
    return 1


def _finalize(args) -> int:
    if not args.allow_test:
        print("finalize reads the Test split and requires --allow-test", file=sys.stderr)
        return 1
    print(
        "finalize is not implemented yet; it must run only after every model is selected "
        "(see docs/tasks/claude-linear-family.md, Phase C4)",
        file=sys.stderr,
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="src.ml.run", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train one model or all of them")
    group = train.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="model name from the registry")
    group.add_argument("--all", action="store_true", help="train every registered model")
    train.add_argument("--variant", choices=VARIANTS, required=True)
    train.add_argument("--grid", help="path to a JSON file of {model: grid}")
    train.add_argument("--output-root")
    train.set_defaults(handler=_train)

    report = sub.add_parser("report", help="build figures and the leaderboard")
    report.add_argument("--variant", choices=VARIANTS, required=True)
    report.add_argument("--output-root")
    report.set_defaults(handler=_report)

    finalize = sub.add_parser("finalize", help="evaluate selected models on the Test split")
    finalize.add_argument("--allow-test", action="store_true")
    finalize.add_argument("--output-root")
    finalize.set_defaults(handler=_finalize)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
