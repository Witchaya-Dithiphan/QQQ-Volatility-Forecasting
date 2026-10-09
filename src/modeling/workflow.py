"""Shared identity, provenance and immutable run preflight for milestone workflows."""
from pathlib import Path
import re
import numpy as np
from config import PROJECT_ROOT
from .artifacts import code_snapshot_hash, file_sha256, runtime_fingerprint, canonical_hash
from .configuration import load_config
from .persistence import classify_run_directory, read_manifest, resume_check, _git_state, make_manifest


def run_path(root, variant, task, name, implementation, run_id):
    if variant not in ('with_spike','non_spike') or implementation not in ('scratch','reference'):
        raise ValueError('Invalid variant/implementation')
    if task not in ('regression','classification','clustering'):
        raise ValueError('Invalid task')
    reserved = {'CON','PRN','AUX','NUL', *[f'COM{i}' for i in range(1,10)], *[f'LPT{i}' for i in range(1,10)]}
    parts = (variant,task,name,implementation,run_id)
    if any(not isinstance(p,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',p) or p.upper() in reserved for p in parts):
        raise ValueError('Unsafe run path component')
    root = Path(root)
    path = root.joinpath(*parts)
    if not path.resolve().is_relative_to(root.resolve()) or any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
        raise ValueError('Unsafe run path escape/symlink')
    return path


def preflight(root, variant, task, name, implementations, run_id):
    paths = [run_path(root, variant, task, name, role, run_id) for role in implementations]
    for path in paths:
        if path.exists():
            classification = classify_run_directory(path)
            raise FileExistsError(f'Run already exists ({classification["status"]}); use a fresh run ID: {path}')
    return paths


def identity(split, config):
    source = Path(split.source)
    if source.is_absolute() and source.is_relative_to(PROJECT_ROOT): source = source.relative_to(PROJECT_ROOT)
    return {'path': source.as_posix(), 'sha256': split.sha256, 'rows': len(split.X),
            'date_range': {'start': str(split.dates[0])[:10], 'end': str(split.dates[-1])[:10]},
            'dates_sha256': canonical_hash(np.asarray(split.dates).astype('datetime64[ns]').astype(str).tolist()),
            'feature_order': list(config['data']['feature_columns'])}


def provenance(config, inputs):
    runtime = runtime_fingerprint()
    return {'config': config['config_sha256'], 'inputs': {'train': inputs.train.sha256, 'validation': inputs.validation.sha256},
            'code': code_snapshot_hash(), 'dependency_lock': file_sha256(PROJECT_ROOT / config['environment']['dependency_lock']),
            'runtime': runtime['sha256']}, runtime


def manifest_for(path, config, inputs, hashes, task, name, implementation, *, pilot_only=False, lifecycle='supervised_tuned'):
    manifest = make_manifest(run_id=path.name, variant=inputs.variant, task=task, model=name, implementation=implementation,
        lifecycle=lifecycle, hashes=hashes, role='pilot_only' if pilot_only else 'finalized_'+implementation, required=not pilot_only)
    manifest['config_id'] = config['config_id']
    manifest['git_dirty'], manifest['git_revision'] = _git_state()
    manifest['input_identity'] = {key: identity(split,config) for key,split in [('train',inputs.train),('validation',inputs.validation)]}
    return manifest


def status(root, variant, task, name, implementation, run_id, *, config=None, inputs=None, hashes=None, loader=None):
    path = run_path(root,variant,task,name,implementation,run_id)
    if not path.exists(): return {'can_skip':False,'status':'missing','mismatches':{}}
    classification = classify_run_directory(path)
    if classification['status'] != 'completed':
        return {'can_skip':False,'status':'legacy_incompatible','mismatches':{'classification':classification}}
    manifest = read_manifest(path)
    expected = dict(variant=variant,task=task,model=name,implementation=implementation,run_id=run_id)
    if any(manifest[k] != v for k,v in expected.items()):
        return {'can_skip':False,'status':'incompatible','mismatches':{'identity':expected}}
    if hashes is None:
        from .datasets import load_train_validation
        config = load_config() if config is None else config
        inputs = (loader or load_train_validation)(variant) if inputs is None else inputs
        hashes, _ = provenance(config,inputs)
    return resume_check(manifest,hashes,path)


def validate_inputs(inputs, variant, config):
    from .m6_training import _validate_inputs
    _validate_inputs(inputs,variant,config)
    for split in [inputs.train,inputs.validation]:
        y = np.asarray(split.y_regression)
        if y.shape != (len(split.X),) or not np.isfinite(y).all() or (y<0).any():
            raise ValueError('Invalid aligned regression target')
