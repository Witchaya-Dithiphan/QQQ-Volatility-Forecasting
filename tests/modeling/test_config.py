"""Executable config identity and exact generated-plan synchronization."""
import copy
import hashlib
import json
import pytest
from config import PROJECT_ROOT
from src.modeling.configuration import load_config, seal_config, validate_config
from src.modeling.config_sync import render_block, synchronize, BEGIN, END
from src.modeling.artifacts import canonical_bytes


def test_hash_and_identity():
    config = load_config()
    body = {k: v for k, v in config.items() if k not in ("config_id", "config_sha256")}
    digest = hashlib.sha256(canonical_bytes(body)).hexdigest()
    assert config["config_sha256"] == digest
    assert config["config_id"] == "cfg-" + digest[:12]
    assert config["plan_section_sha256"] == hashlib.sha256(render_block(config)).hexdigest()
    assert len(config["models"]) == 19
    assert config["models"]["polynomial"]["grid"]["degree"] == [2, 3]


@pytest.mark.parametrize("mutation", ["unknown", "missing", "nested_unknown", "nested_missing", "pca", "bool_component", "duplicate_component", "budget", "test_default", "leakage", "probability", "nonfinite"])
def test_invalid_configs(mutation):
    config = load_config()
    if mutation == "unknown": config["surprise"] = 1
    if mutation == "missing": del config["data"]
    if mutation == "nested_unknown": config["models"]["svm"]["parameters"]["kernel_extra"] = "rbf"
    if mutation == "nested_missing": del config["models"]["mlp"]["grid"]["l2"]
    if mutation == "pca": config["preprocessing"]["pca"]["components"] = [9]
    if mutation == "bool_component": config["preprocessing"]["pca"]["components"] = [True]
    if mutation == "duplicate_component": config["preprocessing"]["pca"]["components"] = [2, 2]
    if mutation == "budget": config["models"]["knn"]["grid"]["k"] = list(range(1, 20))
    if mutation == "test_default": config["test_gate"]["default_allow_test"] = True
    if mutation == "leakage": config["data"]["feature_columns"][0] = "target_high_volatility"
    if mutation == "probability": config["selection"]["probability_epsilon"] = 0.6
    if mutation == "nonfinite": config["experiment"]["seed"] = float("nan")
    with pytest.raises(ValueError): validate_config(seal_config(config), verify_integrity=False)


def test_integrity_tamper_rejected():
    config = load_config()
    config["experiment"]["seed"] = 7
    with pytest.raises(ValueError, match="hash"):
        validate_config(config)


def test_exact_sync_and_outside_preservation(tmp_path):
    config_path = tmp_path / "modeling.json"
    config_path.write_bytes((PROJECT_ROOT / "configs/modeling.json").read_bytes())
    plan = tmp_path / "plan.md"
    before = b"user text\r\n" + BEGIN + b"\r\nold\r\n" + END + b"\r\nuser tail"
    plan.write_bytes(before)
    with pytest.raises(ValueError, match="drift"):
        synchronize(config_path, plan, write=False)
    synchronize(config_path, plan, write=True)
    assert plan.read_bytes().startswith(b"user text\n" + BEGIN)
    assert plan.read_bytes().endswith(END + b"\nuser tail\n")
    synchronize(config_path, plan, write=False)
    block = render_block(load_config(config_path))
    assert b"\r" not in block and not block.startswith(b"\xef\xbb\xbf")
    assert block.endswith(b"\n") and not block.endswith(b"\n\n")
    assert b'"config_id"' not in block
    saved = plan.read_bytes()
    synchronize(config_path, plan, write=True)
    assert saved == plan.read_bytes()
    plan.write_bytes(saved.replace(b'"seed": 42', b'"seed": 41'))
    with pytest.raises(ValueError, match="drift"):
        synchronize(config_path, plan, write=False)


def test_repo_sync():
    synchronize(PROJECT_ROOT / "configs/modeling.json", PROJECT_ROOT / "MODEL_TRAINING_PLAN.md", write=False)


def test_generated_section_anchors_and_unsupported_modes():
    config = load_config()
    block = render_block(config)
    for section in range(3, 11): assert f"### 14.{section} ".encode() in block
    config["models"]["svm"]["parameters"]["kernel"] = "rbf"
    with pytest.raises(ValueError): validate_config(seal_config(config), verify_integrity=False)


@pytest.mark.parametrize("path,value", [("preprocessing.standardizer.ddof", 1), ("preprocessing.standardizer.variance_floor", -1), ("preprocessing.pca.covariance_ddof", 0), ("preprocessing.pca.rank_eigenvalue_floor", -1), ("neural.internal_tail_fraction", 2), ("persistence.reload_rtol", -1), ("experiment.max_candidates", 100), ("models.random_forest.parameters.final_candidates", 6)])
def test_invalid_numeric_policies(path, value):
    config = load_config()
    target = config
    keys = path.split(".")
    for key in keys[:-1]: target = target[key]
    target[keys[-1]] = value
    with pytest.raises(ValueError): validate_config(seal_config(config), verify_integrity=False)



def test_crlf_plan_check_and_canonical_write_are_stable(tmp_path):
    config_path = tmp_path / "modeling.json"
    config_path.write_bytes((PROJECT_ROOT / "configs/modeling.json").read_bytes())
    config = load_config(config_path)
    plan = tmp_path / "plan.md"
    lf = b"user prose\n" + BEGIN + render_block(config) + END + b"\nuser tail\n"
    plan.write_bytes(lf.replace(b"\n", b"\r\n"))
    original = plan.read_bytes()
    synchronize(config_path, plan)
    assert plan.read_bytes() == original  # --check is read-only.
    synchronize(config_path, plan, write=True)
    assert plan.read_bytes() == lf
    assert not plan.read_bytes().startswith(b"\xef\xbb\xbf")
    first_config = config_path.read_bytes()
    synchronize(config_path, plan)
    synchronize(config_path, plan, write=True)
    assert plan.read_bytes() == lf and config_path.read_bytes() == first_config


def test_protocol_text_git_attributes_enforce_lf():
    import subprocess
    names = ["src/modeling/preprocessing.py", "configs/modeling.json", "MODEL_TRAINING_PLAN.md", "requirements.txt", "requirements-lock.txt", "requirements.in"]
    result = subprocess.run(["git", "check-attr", "eol", "--", *names], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == [name + ": eol: lf" for name in names]
