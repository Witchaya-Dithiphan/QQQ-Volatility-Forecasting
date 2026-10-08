"""Offline import and installed-version verification against the environment lock."""
from __future__ import annotations
import importlib
import importlib.metadata
import json
import platform
from config import PROJECT_ROOT
from .configuration import load_config


def verify_dependencies() -> dict:
    python = platform.python_version()
    if python != load_config()["environment"]["python"]:
        raise RuntimeError("Modeling Python version does not match the frozen environment")
    locked = {}
    for line in (PROJECT_ROOT / "requirements-lock.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"): continue
        name, version = line.split("==", 1)
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f"Installed dependency differs from lock: {name}")
        locked[name.lower().replace("_", "-")] = version
    direct = (PROJECT_ROOT / "requirements.in").read_text(encoding="utf-8").splitlines()
    extras = ["scipy", "cloudpickle", "narwhals", "threadpoolctl"]
    imported = {}
    for name in dict.fromkeys([*direct, *extras]):
        if not name or name.startswith("#"): continue
        module = "sklearn" if name == "scikit-learn" else name
        importlib.import_module(module)
        version = importlib.metadata.version(name)
        if locked.get(name.lower().replace("_", "-")) != version:
            raise RuntimeError(f"Imported dependency missing from lock: {name}")
        imported[name] = version
    return {"python": python, "imported_versions": imported, "locked_distributions": len(locked)}


if __name__ == "__main__":
    print(json.dumps(verify_dependencies(), sort_keys=True, allow_nan=False))
