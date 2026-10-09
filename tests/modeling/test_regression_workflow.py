import json
import numpy as np
import pytest
from src.modeling.regression_training import train_regressor, load_regressor
from src.modeling.metrics import regression_outputs, select_candidate
from src.modeling.configuration import load_config
from tests.modeling.test_orchestration import synthetic


@pytest.mark.parametrize('name', ['simple_linear','multiple_linear','polynomial','elastic_net'])
def test_grid_selection_metrics_preprocessing_and_reload(tmp_path, name):
    inputs = synthetic()
    path = train_regressor(name, 'with_spike', tmp_path, 'grid', inputs=inputs, implementation='scratch')
    search = json.loads((path/'search_results.json').read_text())
    selected = select_candidate([r for r in search['candidates'] if r['error'] is None], task='regression')
    assert search['selected']['id'] == selected['id']
    model = load_regressor(path)
    raw = model.predict(inputs.validation.X)
    stored = json.loads((path/'validation_metrics.json').read_text())
    expected = regression_outputs(inputs.validation.y_regression, raw)
    assert stored['raw_metrics'] == expected['raw_metrics'] and stored['clipped_metrics'] == expected['clipped_metrics']
    assert stored['negative_count'] == expected['negative_count']
    assert stored['baselines']['train_mean']['raw_metrics'] == regression_outputs(inputs.validation.y_regression,
        np.full(len(raw), inputs.train.y_regression.mean()))['raw_metrics']
    if name in ['multiple_linear','elastic_net']:
        np.testing.assert_allclose(model.scaler.mean_, inputs.train.X.mean(axis=0))
    with np.load(path/'model.npz', allow_pickle=False) as archive:
        assert all(archive[k].dtype.kind in 'biuf' for k in archive.files)


def test_reference_matches_linear_scratch_and_pilot_is_not_final(tmp_path):
    inputs = synthetic()
    paths = [train_regressor('multiple_linear', 'with_spike', tmp_path, 'pilot', inputs=inputs,
        implementation=role, pilot_only=True) for role in ['scratch','reference']]
    np.testing.assert_allclose(load_regressor(paths[0]).predict(inputs.validation.X), load_regressor(paths[1]).predict(inputs.validation.X), atol=1e-12)
    from src.modeling.persistence import authorize_finalize_test
    with pytest.raises(PermissionError, match='pilot'):
        authorize_finalize_test(paths[0], caller_command='finalize-test pilot')


def test_polynomial_exact_ols_and_constant_target():
    from src.modeling.regression.polynomial import PolynomialRegression
    x = np.linspace(-1,1,40)
    y = .2 + 1e-9 * x**2
    model = PolynomialRegression(2).fit(x,y)
    np.testing.assert_allclose(model.predict(x),y,rtol=0,atol=1e-15)
    constant = PolynomialRegression(2).fit(x,np.full(40,.2))
    np.testing.assert_allclose(constant.predict(x),.2,rtol=0,atol=1e-15)


def test_multiple_linear_uses_frozen_pseudoinverse(monkeypatch):
    from src.modeling.regression.multiple_linear import MultiLinearRegression
    original = np.linalg.pinv
    calls = []
    monkeypatch.setattr(np.linalg, "pinv", lambda X: calls.append(X.copy()) or original(X))
    X = np.random.default_rng(0).normal(size=(20,3))
    y = .4 + X @ np.array([.1,.2,.3])
    model = MultiLinearRegression().fit(X,y)
    assert len(calls)==1 and calls[0].shape==(20,4)
    np.testing.assert_allclose(model.predict(X),y,atol=1e-12)
