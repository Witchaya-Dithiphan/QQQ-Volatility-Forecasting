"""CLI for the modeling layer: train / report / compare / analyze / finalize.

A command that cannot do its job returns 1 and says why rather than pretending to
work: the previous runner's habit of silently producing empty artifacts is what made
the old documentation untrustworthy. finalize opens the Test split exactly once, and
only after `finalize --dry-run` has locked every run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import PROJECT_ROOT

from .core.contracts import VARIANTS
from .core.trainer import train_model
from .registry import MODELS, model_names


def _load_grid(path: str | None, name: str) -> dict:
    """Grids come from configs/modeling.json unless an explicit file overrides it.

    That config is the accepted inventory, so hyperparameter ranges stay in one place
    rather than being retyped per model.
    """
    if path:
        grids = json.loads(Path(path).read_text(encoding="utf-8"))
        return grids.get(name, {})
    config = json.loads((PROJECT_ROOT / "configs" / "modeling.json").read_text(encoding="utf-8"))
    model = config.get("models", {}).get(name, {})
    grid = model.get("grid", {})
    if "tuples" in grid:
        # The config keeps the field names beside the grid rather than inside it.
        return {**grid, "tuple_fields": model["parameters"]["tuple_fields"]}
    return grid


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
    from .core.report import build_report

    try:
        result = build_report(
            args.variant,
            artifact_root=Path(args.output_root) if args.output_root else None,
            report_root=Path(args.report_root) if args.report_root else None,
            figure_root=Path(args.figure_root) if args.figure_root else None,
            replicates=args.replicates,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for path in result["written"]:
        print(path)
    return 0


def _compare(args) -> int:
    from .core.comparison import paired_comparison, write_comparison
    from .core.report import FIGURE_ROOT
    from .core.trainer import OUTPUT_ROOT

    artifact_root = Path(args.output_root) if args.output_root else OUTPUT_ROOT
    try:
        rows = paired_comparison(artifact_root=artifact_root, replicates=args.replicates)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for path in write_comparison(
        rows,
        report_root=Path(args.report_root) if args.report_root else artifact_root / "report",
        figure_root=Path(args.figure_root) if args.figure_root else FIGURE_ROOT,
    ):
        print(path)
    return 0


def _analyze(args) -> int:
    import numpy as np

    from .core import figures
    from .core.contracts import FEATURE_COLUMNS
    from .core.importance import feature_importance, linear_coefficients
    from .core.report import FIGURE_ROOT
    from .core.trainer import OUTPUT_ROOT
    from .core.walk_forward import walk_forward_table

    artifact_root = Path(args.output_root) if args.output_root else OUTPUT_ROOT
    report = (Path(args.report_root) if args.report_root else artifact_root / "report") / args.variant
    figure = (Path(args.figure_root) if args.figure_root else FIGURE_ROOT) / args.variant

    importance = feature_importance(args.variant, artifact_root=artifact_root, repeats=args.repeats)
    figures.write_leaderboard(importance, report / "feature_importance.csv", sort_by="model")
    names = sorted({row["model"] for row in importance if row["task"] == "classification"})
    matrix = np.array([[next(row["importance_mean"] for row in importance
                             if row["model"] == name and row["feature"] == feature)
                        for feature in FEATURE_COLUMNS] for name in names])
    figures.plot_heatmap(matrix, names, list(FEATURE_COLUMNS), figure / "feature_importance.png",
                         title=f"Block-permutation importance on Validation — {args.variant}",
                         colorbar_label="drop in average precision")
    figures.write_leaderboard(linear_coefficients(args.variant, artifact_root=artifact_root),
                              report / "linear_coefficients.csv", sort_by="model")
    written = [report / "feature_importance.csv", figure / "feature_importance.png",
               report / "linear_coefficients.csv"]
    if not args.skip_walk_forward:
        rows = walk_forward_table(args.variant, artifact_root=artifact_root,
                                  models=args.models.split(",") if args.models else None,
                                  replicates=args.replicates)
        figures.write_leaderboard(rows, report / "walk_forward.csv", sort_by="model")
        written.append(report / "walk_forward.csv")
    for path in written:
        print(path)
    return 0


def _finalize(args) -> int:
    from . import finalize as finalize_module
    from .core import data as data_module
    from .core.trainer import OUTPUT_ROOT

    root = Path(args.output_root) if args.output_root else OUTPUT_ROOT
    if args.dry_run:
        try:
            path = finalize_module.write_lock(root)
        except (FileNotFoundError, FileExistsError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        lock = json.loads(path.read_text(encoding="utf-8"))
        print(f"locked {lock['run_count']} runs in {path}; headline: {lock['headline']}")
        return 0
    if not args.allow_test:
        print("finalize reads the Test split and requires --allow-test", file=sys.stderr)
        return 1
    try:
        # Both refusals happen before the held-out files are opened.
        finalize_module.verify_lock(root)
        finalize_module.check_receipt(root, args.rerun_reason)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    holdout = data_module.load_test(allow_test=True)
    segment_dates, segments = data_module.load_segments("test", allow_test=True)
    result = finalize_module.evaluate_holdout(
        holdout, segment_dates, segments, artifact_root=root, replicates=args.replicates,
        command=" ".join(sys.argv), rerun_reason=args.rerun_reason,
    )
    for path in result["written"][-6:]:
        print(path)
    print(f"wrote {len(result['written'])} files; this was the one permitted read of the Test split")
    return 0


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

    report = sub.add_parser("report", help="build figures and the validation leaderboards")
    report.add_argument("--variant", choices=VARIANTS, required=True)
    report.add_argument("--output-root", help="where the trained runs live")
    report.add_argument("--report-root", help="where tables are written (default <output-root>/report)")
    report.add_argument("--figure-root", help="where figures are written (default outputs/figures/modeling)")
    report.add_argument("--replicates", type=int, default=2000, help="bootstrap replicates")
    report.set_defaults(handler=_report)

    compare = sub.add_parser("compare", help="paired with_spike vs non_spike table on Validation")
    compare.add_argument("--output-root")
    compare.add_argument("--report-root")
    compare.add_argument("--figure-root")
    compare.add_argument("--replicates", type=int, default=2000)
    compare.set_defaults(handler=_compare)

    analyze = sub.add_parser("analyze", help="feature importance and walk-forward robustness")
    analyze.add_argument("--variant", choices=VARIANTS, required=True)
    analyze.add_argument("--output-root")
    analyze.add_argument("--report-root")
    analyze.add_argument("--figure-root")
    analyze.add_argument("--repeats", type=int, default=30, help="reorderings per feature")
    analyze.add_argument("--replicates", type=int, default=2000, help="bootstrap replicates")
    analyze.add_argument("--models", help="comma-separated classifiers for the walk-forward table")
    analyze.add_argument("--skip-walk-forward", action="store_true")
    analyze.set_defaults(handler=_analyze)

    finalize = sub.add_parser("finalize", help="evaluate the locked models on the Test split, once")
    finalize.add_argument("--allow-test", action="store_true")
    finalize.add_argument("--dry-run", action="store_true",
                          help="write selection_lock.json from the artifacts; never opens Test")
    finalize.add_argument("--rerun-reason",
                          help="required to evaluate again after a receipt exists; recorded in the receipt")
    finalize.add_argument("--output-root")
    finalize.add_argument("--replicates", type=int, default=2000)
    finalize.set_defaults(handler=_finalize)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
