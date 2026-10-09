"""M2 CLI checks and a test-only no-op artifact proof; no training commands."""
from __future__ import annotations
from .test_access import install_test_access_guard
install_test_access_guard()

import argparse
import json
from pathlib import Path
import traceback
from config import PROJECT_ROOT
from .artifacts import code_snapshot_hash, file_sha256, runtime_fingerprint, write_json
from .configuration import load_config
from .datasets import load_train_validation
from .persistence import create_run_directory, make_manifest, save_npz, load_npz, verify_reload, transition, capture_artifacts, _git_state
from .preprocessing import Standardizer, PCA


def current_hashes(variant: str) -> dict:
    config = load_config()
    inputs = load_train_validation(variant)
    return {"config": config["config_sha256"], "inputs": {"train": inputs.train.sha256, "validation": inputs.validation.sha256},
            "code": code_snapshot_hash(), "dependency_lock": file_sha256(PROJECT_ROOT / "requirements-lock.txt"), "runtime": runtime_fingerprint()["sha256"]}


def run_noop(variant: str, output_root: Path, run_id: str) -> Path:
    """Used only by tests. Persist/reload preprocessing; fit no predictive model."""
    config = load_config()
    inputs = load_train_validation(variant)
    fingerprint = runtime_fingerprint()
    hashes = current_hashes(variant)
    path = create_run_directory(output_root, variant, "infrastructure", "noop", "dummy", run_id)
    manifest = make_manifest(run_id=run_id, variant=variant, task="infrastructure", model="noop", implementation="dummy",
                             lifecycle="infrastructure_noop", hashes=hashes, role="infrastructure_test", required=False)
    manifest["config_id"] = config["config_id"]
    manifest["git_dirty"], manifest["git_revision"] = _git_state()
    manifest["input_identity"] = {
        name: {"path": Path(split.source).relative_to(PROJECT_ROOT).as_posix(), "sha256": split.sha256,
               "rows": len(split.X), "date_range": {"start": str(split.dates[0])[:10], "end": str(split.dates[-1])[:10]}}
        for name, split in (("train", inputs.train), ("validation", inputs.validation))
    }
    transition(manifest, "running", stage="preprocessing_proof")
    try:
        write_json(path / "config.json", config)
        write_json(path / "runtime.json", fingerprint)
        scaler = Standardizer(config["preprocessing"]["standardizer"]["variance_floor"]).fit(inputs.train.X)
        train_scaled = scaler.transform(inputs.train.X)
        pca_config = config["preprocessing"]["pca"]
        pca = PCA(pca_config["components"][0], rank_floor=pca_config["rank_eigenvalue_floor"], degenerate_rtol=pca_config["degenerate_relative_tolerance"], negative_tolerance=pca_config["negative_eigenvalue_tolerance"]).fit(train_scaled)
        before = pca.transform(scaler.transform(inputs.validation.X))
        arrays = {"mean": scaler.mean_, "variance": scaler.variance_, "scale": scaler.scale_, "constant": scaler.constant_, "components": pca.components_}
        save_npz(path / "state.npz", arrays, {"kind": "infrastructure_noop", "trained_model": False, "feature_columns": config["data"]["feature_columns"], "warnings": scaler.warnings_})
        reloaded, metadata = load_npz(path / "state.npz")
        restored = Standardizer()
        restored.mean_, restored.variance_, restored.scale_, restored.constant_ = [reloaded[key] for key in ("mean", "variance", "scale", "constant")]
        restored_pca = PCA(len(reloaded["components"]))
        restored_pca.components_, restored_pca.n_features_ = reloaded["components"], len(restored.mean_)
        verification = verify_reload(before, restored_pca.transform(restored.transform(inputs.validation.X)), rtol=config["persistence"]["reload_rtol"], atol=config["persistence"]["reload_atol"])
        manifest["load_verification"] = verification
        write_json(path / "load_verification.json", verification)
        capture_artifacts(manifest, path)
        transition(manifest, "completed", run_dir=path, stage="infrastructure_proof_complete")
    except Exception as error:
        (path / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        transition(manifest, "failed", stage=manifest["stage"], error={"exception_type": type(error).__name__, "message": str(error), "traceback_path": "traceback.txt"})
        write_json(path / "manifest.json", manifest)
        raise
    write_json(path / "manifest.json", manifest)
    return path


def main() -> None:
    import sys
    if "--help" in sys.argv or "-h" in sys.argv or "--milestone" in sys.argv:
        from .orchestration import main as orchestrate
        raise SystemExit(orchestrate())

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check-config", "check-data", "train", "tune", "evaluate", "finalize-test"])
    parser.add_argument("--variant", choices=["with_spike", "non_spike"], default="with_spike")
    args = parser.parse_args()
    if args.command in ("train", "tune", "evaluate", "finalize-test"):
        parser.error("M2 provides shared infrastructure only; model execution is not implemented")
    config = load_config()
    if args.command == "check-data":
        pair = load_train_validation(args.variant)
        print(json.dumps({"variant": args.variant, "train_rows": len(pair.train.X), "validation_rows": len(pair.validation.X), "test_accessed": False}, allow_nan=False))
    else:
        print(config["config_id"])


if __name__ == "__main__":
    main()
