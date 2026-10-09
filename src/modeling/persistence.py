"""Immutable artifacts, lifecycle manifests, reload equality and resume gates."""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any
import numpy as np
from .artifacts import canonical_bytes, file_sha256, write_json
from .configuration import load_config


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_run_directory(root: Path, variant: str, task: str, model: str, implementation: str, run_id: str) -> Path:
    parts = (variant, task, model, implementation, run_id)
    if variant not in ("with_spike", "non_spike") or task not in ("regression", "classification", "clustering", "infrastructure") or implementation not in ("scratch", "reference", "dummy"):
        raise ValueError("Invalid run identity")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", part) or part.upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]} for part in parts):
        raise ValueError("Unsafe run path component")
    path = root.joinpath(*parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Run path escapes output root")
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_npz(path: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    metadata_bytes = canonical_bytes(metadata) + b"\n"
    safe = {}
    for name, value in arrays.items():
        data = np.asarray(value)
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or data.dtype.kind not in "biufcUS" or (data.dtype.kind in "fcu" and not np.isfinite(data).all()):
            raise ValueError("Unsafe array name/dtype or nonfinite array")
        safe[name] = data
    if not safe: raise ValueError("No scratch arrays")
    sidecar = path.with_suffix(".json")
    if path.exists() or sidecar.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("xb") as stream:
        stream.write(metadata_bytes)
    try:
        with path.open("xb") as stream:
            np.savez(stream, **safe)
    except Exception:
        sidecar.unlink()
        raise


def load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    canonical_bytes(metadata)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    for array in arrays.values():
        if array.dtype.kind not in "biufcUS" or (array.dtype.kind in "fcu" and not np.isfinite(array).all()):
            raise ValueError("Unsafe loaded array")
    return arrays, metadata


def save_reference(path: Path, value: Any) -> None:
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        joblib.dump(value, stream)


def load_reference(path: Path, *, expected_sha256: str):
    import joblib
    if file_sha256(path) != expected_sha256: raise ValueError("Reference artifact checksum mismatch")
    return joblib.load(path)


def verify_reload(before, after, *, labels: bool = False, rtol: float = 1e-10, atol: float = 1e-12) -> dict:
    before, after = np.asarray(before), np.asarray(after)
    matching_shape = before.shape == after.shape
    finite = np.isfinite(before).all() and np.isfinite(after).all()
    passed = bool(matching_shape and finite and (np.array_equal(before, after) if labels else np.allclose(before, after, rtol=rtol, atol=atol)))
    return {"passed": passed, "labels_exact": labels, "rtol": 0.0 if labels else rtol, "atol": 0.0 if labels else atol,
            "reason": None if passed else "shape, nonfinite value or reload output mismatch"}


def make_manifest(*, run_id: str, variant: str, task: str, model: str, implementation: str, lifecycle: str, hashes: dict, role: str, required: bool = True) -> dict:
    if set(hashes) != {"config", "inputs", "code", "dependency_lock", "runtime"}:
        raise ValueError("Missing or unknown provenance hashes")
    if lifecycle not in ("infrastructure_noop", "supervised_tuned", "supervised_finalized", "clustering_train", "clustering_extension"):
        raise ValueError("Unknown run lifecycle")
    return {"manifest_version": 1, "run_id": run_id, "variant": variant, "task": task, "model": model, "implementation": implementation,
            "lifecycle": lifecycle, "role": role, "pilot_only": role == "pilot_only", "required": required, "status": "pending", "stage": "initialized",
            "hashes": copy.deepcopy(hashes), "artifact_checksums": {}, "load_verification": None, "test_accessed": False, "test_access": None,
            "selected_hyperparameters": None, "output_policy": None, "reason": None, "error": None, "mismatches": {}, "git_revision": None, "git_dirty": None, "config_id": None, "input_identity": {},
            "timing": {"started_utc": None, "finished_utc": None, "elapsed_seconds": None}}


def required_artifacts(manifest: dict) -> list[str]:
    lifecycle = manifest["lifecycle"]
    if lifecycle == "infrastructure_noop": return ["config.json", "state.npz", "state.json", "load_verification.json"]
    config = load_config()
    mapping = config["artifacts"]
    base = mapping["supervised_tuned"] if lifecycle.startswith("supervised") else mapping["clustering_train"]
    names = [name for name in base if name != "manifest.json"]
    if lifecycle == "supervised_finalized": names += mapping["supervised_finalized"]
    if lifecycle == "clustering_extension": names += mapping["clustering_extension"]
    if manifest["implementation"] == "reference": names = ["model.joblib" if name == "model.npz" else name for name in names]
    else: names += ["model.json"]
    names += ["preprocessor.json"]
    return names


def validate_lifecycle(manifest: dict, fields: dict | None = None) -> None:
    fields = fields or {}
    if manifest["lifecycle"].startswith("clustering") and any(key in fields for key in ("threshold", "confusion_matrix", "f1", "accuracy")):
        raise ValueError("Clustering structure forbids supervised fields")
    if manifest["lifecycle"] in ("supervised_tuned", "clustering_train", "infrastructure_noop") and manifest["test_accessed"]:
        raise ValueError("Test access prohibited at this lifecycle")
    if manifest["lifecycle"] == "supervised_finalized" and not manifest["test_accessed"]:
        raise ValueError("Finalized run requires Test access identity")


def _artifact_files(run_dir: Path) -> list[Path]:
    files = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(run_dir.resolve()):
            raise ValueError("Artifact symlinks/path escapes are forbidden")
        if path.is_file() and path.name != "manifest.json": files.append(path)
    return files


def capture_artifacts(manifest: dict, run_dir: Path) -> None:
    validate_lifecycle(manifest)
    files = _artifact_files(run_dir)
    if manifest["lifecycle"] in ("supervised_tuned", "clustering_train", "infrastructure_noop") and any(path.name.startswith("test_") for path in files):
        raise ValueError("Test artifacts prohibited before finalization")
    manifest["artifact_checksums"] = {path.relative_to(run_dir).as_posix(): file_sha256(path) for path in files}


def _check_artifacts(manifest: dict, run_dir: Path) -> dict:
    issues = {}
    for name in required_artifacts(manifest):
        path = run_dir / name
        if not path.exists() or (path.is_dir() and not any(p.is_file() for p in path.rglob("*"))):
            issues[name] = {"expected": "required artifact", "actual": "missing"}
        elif path.is_file() and name not in manifest["artifact_checksums"]:
            issues[name] = {"expected": "recorded checksum", "actual": "unrecorded"}
    for name, expected in manifest["artifact_checksums"].items():
        path = run_dir / name
        if not path.resolve().is_relative_to(run_dir.resolve()): raise ValueError("Artifact path escape")
        actual = file_sha256(path) if path.is_file() else None
        if actual != expected: issues[name] = {"expected": expected, "actual": actual}
        if path.is_file() and path.suffix.lower() == ".npz":
            try:
                with np.load(path, allow_pickle=False) as archive:
                    for member in archive.files:
                        archive[member]  # Materialize every member; opening alone permits object arrays.
            except Exception:
                issues.setdefault("unsafe_object_npz", []).append(name)
    actual_names = {path.relative_to(run_dir).as_posix() for path in _artifact_files(run_dir)}
    if actual_names != set(manifest["artifact_checksums"]):
        issues["file_inventory"] = {"expected": sorted(manifest["artifact_checksums"]), "actual": sorted(actual_names)}
    receipt_path = run_dir / "load_verification.json"
    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            canonical_bytes(receipt)
            if receipt.get("passed") is not True or receipt != manifest["load_verification"]:
                issues["load_verification_receipt"] = {"expected": manifest["load_verification"], "actual": receipt}
        except (ValueError, AttributeError):
            issues["load_verification_receipt"] = {"expected": "valid verified receipt", "actual": "invalid JSON/receipt"}
    return issues


def transition(manifest: dict, status: str, *, run_dir: Path | None = None, stage: str | None = None, reason: str | None = None, error: dict | None = None, mismatches: dict | None = None) -> None:
    allowed = {"pending": {"running", "skipped", "incompatible"}, "running": {"completed", "failed", "skipped", "incompatible"}}
    if status not in allowed.get(manifest["status"], set()): raise ValueError("Invalid lifecycle status transition")
    if status == "completed":
        validate_lifecycle(manifest)
        if run_dir is None or not manifest["load_verification"] or manifest["load_verification"].get("passed") is not True or _check_artifacts(manifest, run_dir):
            raise ValueError("Cannot complete run without required artifacts and verified reload")
    if status == "failed" and (not stage or not error or not {"exception_type", "message", "traceback_path"} <= set(error)):
        raise ValueError("Failed run requires stage and exception evidence")
    if status == "skipped" and not reason: raise ValueError("Skipped run requires explicit reason")
    if status == "incompatible" and not mismatches: raise ValueError("Incompatible run requires mismatches")
    manifest.update(status=status, stage=stage or manifest["stage"], reason=reason, error=error, mismatches=mismatches or {})
    if status == "running": manifest["timing"]["started_utc"] = utc_now()
    else:
        finished = utc_now()
        manifest["timing"]["finished_utc"] = finished
        started = manifest["timing"]["started_utc"]
        manifest["timing"]["elapsed_seconds"] = (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds() if started else 0.0


def resume_check(manifest: dict, current_hashes: dict, run_dir: Path) -> dict:
    mismatches = {key: {"expected": manifest["hashes"].get(key), "actual": current_hashes.get(key)} for key in set(manifest["hashes"]) | set(current_hashes) if manifest["hashes"].get(key) != current_hashes.get(key)}
    artifact_issues = _check_artifacts(manifest, run_dir)
    if artifact_issues: mismatches["artifacts"] = artifact_issues
    if manifest["status"] != "completed": mismatches["status"] = {"expected": "completed", "actual": manifest["status"]}
    if not manifest["load_verification"] or manifest["load_verification"].get("passed") is not True:
        mismatches["load_verification"] = {"expected": True, "actual": manifest["load_verification"]}
    validate_lifecycle(manifest)
    return {"can_skip": not mismatches, "status": "completed" if not mismatches else "incompatible", "mismatches": mismatches}


def _git_state() -> tuple[bool, str]:
    import subprocess
    from config import PROJECT_ROOT
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", ".", ":(top,exclude)outputs/modeling/**"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    return bool(status.stdout.strip()), revision.stdout.strip()


def read_manifest(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    canonical_bytes(manifest)
    identity_keys = ("run_id", "variant", "task", "model", "implementation", "lifecycle", "hashes", "role", "required")
    if any(key not in manifest for key in identity_keys): raise ValueError("Missing manifest identity")
    expected = make_manifest(**{key: manifest[key] for key in identity_keys})
    if set(expected) != set(manifest): raise ValueError("Missing or unknown manifest keys")
    if manifest["manifest_version"] != 1 or manifest["status"] not in ("pending", "running", "completed", "failed", "skipped", "incompatible"):
        raise ValueError("Invalid manifest version/status")
    return manifest


def _finalization_prerequisites(run_dir: Path, caller_command: str) -> tuple[dict, str]:
    from .runner import current_hashes
    if not caller_command.split() or caller_command.split()[0] != "finalize-test":
        raise PermissionError("Only explicit finalize-test callers may authorize Test")
    manifest = read_manifest(run_dir)
    if manifest["lifecycle"] != "supervised_tuned" or manifest["task"] not in ("regression", "classification") or manifest["implementation"] not in ("scratch", "reference") or manifest["pilot_only"] or manifest["role"] == "infrastructure_test" or manifest["test_accessed"]:
        raise PermissionError("Test requires completed supervised Validation, excluding pilot/no-op runs")
    config = load_config()
    if manifest["hashes"]["config"] != config["config_sha256"] or not isinstance(manifest["selected_hyperparameters"], dict):
        raise PermissionError("Frozen config and selected hyperparameters required")
    policy = manifest["output_policy"]
    if not isinstance(policy, dict): raise PermissionError("Frozen threshold/clipping policy required")
    if manifest["task"] == "regression" and policy.get("clipping_min") != config["selection"]["regression_clip_min"]:
        raise PermissionError("Invalid regression clipping policy")
    if manifest["task"] == "classification":
        threshold = policy.get("threshold")
        kind = policy.get("score_kind")
        if type(threshold) not in (float, int) or not np.isfinite(threshold) or kind not in ("probability", "decision") or (kind == "probability" and not 0 <= threshold <= 1):
            raise PermissionError("Invalid classification threshold policy")
    if not config["test_gate"]["freeze_non_spike_before_test"] or config["data"]["variants"] != ["with_spike", "non_spike"]:
        raise PermissionError("Non-Spike protocol must be frozen before Test")
    if not resume_check(manifest, current_hashes(manifest["variant"]), run_dir)["can_skip"]:
        raise PermissionError("Run is incomplete or incompatible with current hashes/runtime")
    verified = json.loads((run_dir / "load_verification.json").read_text(encoding="utf-8"))
    if verified.get("passed") is not True: raise PermissionError("Persisted load verification failed")
    dirty, revision = _git_state()
    if dirty: raise PermissionError("finalize-test requires a clean Git working tree")
    if (run_dir / "test_access.json").exists(): raise PermissionError("Test already accessed; create a new authorized run")
    return manifest, revision


from dataclasses import dataclass


@dataclass(frozen=True)
class FinalizeTestAuthorization:
    run_dir: Path
    caller_command: str
    manifest_sha256: str

    def validate(self) -> None:
        if file_sha256(self.run_dir / "manifest.json") != self.manifest_sha256:
            raise PermissionError("Manifest changed after Test authorization")
        _finalization_prerequisites(self.run_dir, self.caller_command)

    def record_access(self, test_sha256: str) -> None:
        self.validate()
        manifest, revision = _finalization_prerequisites(self.run_dir, self.caller_command)
        record = {"test_accessed": True, "utc_timestamp": utc_now(),
                   "config_sha256": manifest["hashes"]["config"], "test_input_sha256": test_sha256, "git_revision": revision,
                   "code_snapshot_sha256": manifest["hashes"]["code"], "runtime_sha256": manifest["hashes"]["runtime"],
                   "dependency_lock_sha256": manifest["hashes"]["dependency_lock"], "caller_command": self.caller_command}
        write_json(self.run_dir / "test_access.json", record)
        # Append an immutable lifecycle snapshot, preserving the tuned manifest.
        snapshot = copy.deepcopy(manifest)
        snapshot.update(test_accessed=True, test_access=record, lifecycle="supervised_finalized", status="running", stage="test_accessed")
        snapshot["timing"] = {"started_utc": record["utc_timestamp"], "finished_utc": None, "elapsed_seconds": None}
        write_json(self.run_dir / "test_access_manifest.json", snapshot)


def _test_ledger_path() -> Path:
    """Atomic Test-access ledger at output root level."""
    from config import PROJECT_ROOT
    ledger = PROJECT_ROOT / "outputs" / ".test_ledger.json"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    if not ledger.exists():
        ledger.write_text("{}\n", encoding="utf-8")
    return ledger


def _record_test_access_intent(manifest: dict) -> None:
    """Write intent entry to ledger BEFORE reading Test data."""
    ledger_path = _test_ledger_path()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    canonical_bytes(ledger)
    
    key = f"{manifest['hashes']['config']}/{manifest['variant']}/{manifest['task']}/{manifest['model']}/{manifest['implementation']}/{manifest['run_id']}"
    
    if key in ledger and ledger[key].get("status") == "completed":
        raise PermissionError(f"Test already accessed for this run identity: {key}")
    
    intent_entry = {
        "config_sha256": manifest["hashes"]["config"],
        "variant": manifest["variant"],
        "task": manifest["task"],
        "model": manifest["model"],
        "implementation": manifest["implementation"],
        "run_id": manifest["run_id"],
        "attempted_utc": utc_now(),
        "status": "intent"
    }
    
    ledger[key] = intent_entry
    # Use text write (overwrite mode) for ledger updates
    ledger_path.write_text(json.dumps(ledger, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _finalize_test_access_ledger(manifest: dict, test_sha256: str) -> None:
    """Finalize ledger entry AFTER successful Test read."""
    ledger_path = _test_ledger_path()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    canonical_bytes(ledger)
    
    key = f"{manifest['hashes']['config']}/{manifest['variant']}/{manifest['task']}/{manifest['model']}/{manifest['implementation']}/{manifest['run_id']}"
    
    if key not in ledger or ledger[key].get("status") != "intent":
        raise PermissionError(f"No intent found in ledger for {key}")
    
    ledger[key].update({
        "test_sha256": test_sha256,
        "finalized_utc": utc_now(),
        "status": "completed"
    })
    # Use text write (overwrite mode) for ledger updates
    ledger_path.write_text(json.dumps(ledger, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def classify_run_directory(run_dir: Path) -> dict:
    """Return {"status": "legacy_incompatible", "reasons": [...]} for any run that lacks a valid modern manifest.

    Returns completed only for a valid manifest, safe complete artifacts and passed reload.
    Never deletes or modifies files.
    """
    reasons: list[str] = []
    if not (run_dir / "manifest.json").exists():
        reasons.append("missing_manifest")
        if (run_dir / "metadata.json").exists():
            reasons.append("has_legacy_metadata")
        for npz_path in run_dir.glob("*.npz"):
            try:
                with np.load(npz_path, allow_pickle=False) as archive:
                    for key in archive.files:
                        _ = archive[key]  # triggers ValueError on object-dtype arrays
            except Exception:
                reasons.append("unsafe_object_npz")
                break
        return {"status": "legacy_incompatible", "reasons": reasons}
    try:
        manifest = read_manifest(run_dir)
    except Exception:
        return {"status": "legacy_incompatible", "reasons": ["invalid_manifest"]}
    if manifest["status"] != "completed":
        reasons.append("not_completed")
        return {"status": "legacy_incompatible", "reasons": reasons}
    try:
        artifact_issues = _check_artifacts(manifest, run_dir)
    except Exception:
        return {"status": "legacy_incompatible", "reasons": ["incomplete_artifacts"]}
    if artifact_issues:
        reasons.append("incomplete_artifacts")
        if "unsafe_object_npz" in artifact_issues:
            reasons.append("unsafe_object_npz")
    lv = manifest.get("load_verification")
    if not isinstance(lv, dict) or lv.get("passed") is not True or not (run_dir / "load_verification.json").exists():
        reasons.append("missing_or_failed_load_verification")
    if reasons:
        return {"status": "legacy_incompatible", "reasons": reasons}
    return {"status": manifest["status"], "reasons": []}


def authorize_finalize_test(run_dir: Path, *, caller_command: str) -> FinalizeTestAuthorization:
    manifest, _ = _finalization_prerequisites(run_dir, caller_command)
    _record_test_access_intent(manifest)  # Write intent BEFORE authorization
    return FinalizeTestAuthorization(run_dir, caller_command, file_sha256(run_dir / "manifest.json"))
