"""Shared CLI uses only synthetic inputs; no production pilot is executed."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from src.modeling import orchestration as cli
from src.modeling.configuration import load_config, seal_config
from src.modeling.datasets import DatasetSplit, TrainValidation
from src.modeling.persistence import read_manifest, required_artifacts


def synthetic(variant='with_spike'):
    rng = np.random.default_rng(6)
    def split(start, n):
        dates = np.arange(np.datetime64(start), np.datetime64(start) + np.timedelta64(n, "D"))
        X = rng.normal(size=(n, 8))
        y = np.maximum(.5 + .15 * X[:, 0] + .1 * X[:, 1], .01)
        return DatasetSplit(dates, X, y, (y > .6).astype(int), str(n) * 32, 'synthetic')
    return TrainValidation(variant, split('2020-01-01', 100), split('2021-01-01', 35))


@pytest.fixture
def env(monkeypatch):
    config = copy.deepcopy(load_config())
    for name, model in config['models'].items():
        if name in cli.FAMILIES['M5']:
            model['grid'] = {k: v[:1] for k, v in model['grid'].items()}
            for key in ('max_iter', 'max_epochs'):
                if key in model['parameters']: model['parameters'][key] = 10
    config = seal_config(config)
    monkeypatch.setattr(cli, 'load_config', lambda: copy.deepcopy(config))
    monkeypatch.setattr(cli, 'load_train_validation', synthetic)
    from src.modeling import datasets
    monkeypatch.setattr(datasets, 'load_test', lambda **k: pytest.fail('Test accessed'))
    return config


def args(tmp_path, milestone='M3', model='multiple_linear', action='train', implementation='scratch', run_id='synthetic'):
    return [action, '--milestone', milestone, '--model', model, '--variant', 'with_spike',
            '--implementation', implementation, '--output-root', str(tmp_path), '--run-id', run_id]


def test_help_and_process_exit_codes(tmp_path):
    command = [sys.executable, '-m', 'src.modeling.runner']
    help_result = subprocess.run([*command, '--help'], capture_output=True, text=True)
    assert help_result.returncode == 0 and '--milestone' in help_result.stdout
    bad = subprocess.run([*command, *args(tmp_path, model='unknown')], capture_output=True, text=True)
    assert bad.returncode != 0


@pytest.mark.parametrize('milestone,model', [('M3','multiple_linear'), ('M4','simple_linear'), ('M4','polynomial'),
    ('M4','elastic_net'), ('M5','gaussian_nb'), ('M5','logistic'), ('M5','knn'), ('M5','perceptron'),
    ('M5','slp'), ('M5','decision_tree'), ('M7','kmeans'), ('M7','agglomerative')])
def test_synthetic_success_and_lifecycle(env, tmp_path, milestone, model):
    assert cli.main(args(tmp_path, milestone, model)) == 0
    task = load_config()['models'][model]['task']
    path = tmp_path / 'with_spike' / task / model / 'scratch/synthetic'
    manifest = read_manifest(path)
    assert manifest['status'] == 'completed' and not manifest['test_accessed']
    assert manifest['pilot_only'] == (milestone == 'M3')
    assert manifest['lifecycle'] == ('clustering_train' if milestone == 'M7' else 'supervised_tuned')
    assert all((path / name).exists() for name in required_artifacts(manifest))
    assert manifest['input_identity']['train']['rows'] == 100
    assert manifest['input_identity']['train']['feature_order'] == load_config()['data']['feature_columns']
    assert manifest['load_verification']['passed']
    assert not list(path.glob('test_*'))


@pytest.mark.parametrize('milestone,model', [('M3','multiple_linear'), ('M4','elastic_net'), ('M5','gaussian_nb'), ('M7','agglomerative')])
@pytest.mark.parametrize('implementation', ['scratch', 'reference', 'both'])
def test_implementation_selection(env, tmp_path, milestone, model, implementation):
    assert cli.main(args(tmp_path, milestone, model, implementation=implementation)) == 0
    task = load_config()['models'][model]['task']
    base = tmp_path / 'with_spike' / task / model
    for role in ['scratch', 'reference']:
        path = base / role / 'synthetic'
        assert path.exists() == (implementation in [role, 'both'])
        if path.exists():
            manifest = read_manifest(path)
            assert manifest['implementation'] == role and manifest['pilot_only'] == (milestone == 'M3')
            assert ('model.joblib' in required_artifacts(manifest)) == (role == 'reference')


@pytest.mark.parametrize('field,value', [('--model','unknown'), ('--variant','invalid'), ('--implementation','invalid'),
    ('--milestone','M8'), ('--run-id','../escape'), ('--run-id','CON'), ('--seed','-1')])
def test_invalid_arguments_fail_before_loading(env, monkeypatch, tmp_path, field, value):
    monkeypatch.setattr(cli, 'load_train_validation', lambda *a: pytest.fail('invalid arguments read input'))
    command = args(tmp_path)
    if field in command: command[command.index(field)+1] = value
    else: command += [field, value]
    try: code = cli.main(command)
    except SystemExit as error: code = error.code
    assert code != 0 and not list(tmp_path.iterdir())


@pytest.mark.parametrize('kind', ['legacy', 'corrupt', 'incomplete', 'object_npz'])
def test_existing_bad_run_fails_closed_without_data(env, monkeypatch, tmp_path, kind):
    path = tmp_path / 'with_spike/regression/multiple_linear/scratch/synthetic'
    path.mkdir(parents=True)
    if kind == 'corrupt': (path / 'manifest.json').write_text('{')
    if kind == 'incomplete': (path / 'manifest.json').write_text('{}')
    if kind == 'object_npz': np.savez(path / 'model.npz', state=np.array([{}], dtype=object))
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    monkeypatch.setattr(cli, 'load_train_validation', lambda *a: pytest.fail('bad run read inputs'))
    assert cli.main(args(tmp_path, action='status')) != 0
    assert cli.main(args(tmp_path)) != 0
    assert before == {p.name: p.read_bytes() for p in path.iterdir()}


def test_missing_status_dry_run_and_m6_routing(env, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, 'load_train_validation', lambda *a: pytest.fail('status/dry-run read data'))
    assert cli.main(args(tmp_path, action='status')) == 0
    assert 'missing' in capsys.readouterr().out
    for model in cli.FAMILIES['M6']:
        assert cli.main(args(tmp_path, 'M6', model, action='dry-run')) == 0
    assert not list(tmp_path.iterdir())


def test_no_overwrite_completed_or_partial_both(env, monkeypatch, tmp_path):
    assert cli.main(args(tmp_path)) == 0
    monkeypatch.setattr(cli, 'load_train_validation', lambda *a: pytest.fail('overwrite reads inputs'))
    assert cli.main(args(tmp_path)) != 0
    assert cli.main(args(tmp_path, implementation='both')) != 0
    assert not (tmp_path / 'with_spike/regression/multiple_linear/reference').exists()


def test_seed_identity_is_deterministic_and_rejects_mismatch(env, tmp_path):
    first = args(tmp_path, run_id='a') + ['--seed','7']
    second = args(tmp_path, run_id='b') + ['--seed','7']
    assert cli.main(first) == cli.main(second) == 0
    base = tmp_path / 'with_spike/regression/multiple_linear/scratch'
    a, b = read_manifest(base/'a'), read_manifest(base/'b')
    assert a['hashes'] == b['hashes'] and a['config_id'] == b['config_id']
    assert a['output_policy']['seed'] == 7
    assert cli.main(args(tmp_path, action='status', run_id='a') + ['--seed','7']) == 0
    assert cli.main(args(tmp_path, action='status', run_id='a') + ['--seed','42']) != 0


def test_symlink_escape_is_rejected(env, tmp_path):
    # resolve validation also applies when status checks a missing final directory.
    outside = tmp_path.parent / (tmp_path.name + '-outside')
    outside.mkdir()
    import os
    os.symlink(outside, tmp_path/'with_spike', target_is_directory=True)
    assert cli.main(args(tmp_path, action='status')) != 0


@pytest.mark.parametrize('model',['kmeans','agglomerative'])
@pytest.mark.parametrize('implementation',['scratch','reference'])
def test_clustering_extension_is_explicit_and_reload_verified(env,tmp_path,model,implementation):
    assert cli.main(args(tmp_path,'M7',model,implementation=implementation)+['--extension'])==0
    path=tmp_path/'with_spike/clustering'/model/implementation/'synthetic'
    manifest=read_manifest(path)
    assert manifest['lifecycle']=='clustering_extension' and not manifest['test_accessed']
    assert all((path/name).exists() for name in required_artifacts(manifest))
    assert manifest['load_verification']['extension']['passed']
    import pandas as pd
    frame=pd.read_csv(path/'validation_assignments.csv')
    assert set(frame['assignment_method'])=={'frozen_nearest_centroid_extension'}


def test_cli_synthetic_process_success(tmp_path):
    script="from src.modeling import orchestration as c; from tests.modeling.test_orchestration import synthetic; c.load_train_validation=synthetic; raise SystemExit(c.main("+repr(args(tmp_path))+"))"
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('action',['unknown','finalize-test','tune'])
def test_invalid_action_nonzero(env,tmp_path,action):
    with pytest.raises(SystemExit) as error: cli.main(args(tmp_path,action=action))
    assert error.value.code!=0


@pytest.mark.parametrize('milestone,model',[('M4','polynomial'),('M5','gaussian_nb'),('M7','kmeans')])
def test_modern_unsafe_npz_fails_even_with_matching_checksum(env,tmp_path,milestone,model):
    assert cli.main(args(tmp_path,milestone,model))==0
    task=load_config()['models'][model]['task']
    path=tmp_path/'with_spike'/task/model/'scratch/synthetic'
    manifest=read_manifest(path)
    np.savez(path/'model.npz',state=np.array([{'hidden':1}],dtype=object))
    from src.modeling.artifacts import file_sha256
    manifest['artifact_checksums']['model.npz']=file_sha256(path/'model.npz')
    (path/'manifest.json').write_text(json.dumps(manifest))
    assert cli.main(args(tmp_path,milestone,model,action='status'))!=0


@pytest.mark.parametrize('milestone,model',[('M5','gaussian_nb'),('M6','xgboost')])
def test_library_both_preflights_reference_before_any_loading(monkeypatch,tmp_path,milestone,model):
    from unittest.mock import Mock
    loader = Mock(side_effect=AssertionError('collision read data'))
    path=tmp_path/'with_spike/classification'/model/'reference/existing'
    path.mkdir(parents=True)
    if milestone=='M5':
        from src.modeling import classification_training as training
        monkeypatch.setattr(training,'load_train_validation',loader)
        invoke=lambda: training.train_classifier(model,'with_spike',tmp_path,'existing',reference=True)
    else:
        from src.modeling import m6_training as training
        monkeypatch.setattr(training,'load_train_validation',loader)
        invoke=lambda: training.train_m6_model(model,'with_spike',tmp_path,'existing',reference=True)
    with pytest.raises(FileExistsError): invoke()
    loader.assert_not_called()
    assert not (path.parent.parent/'scratch').exists()
