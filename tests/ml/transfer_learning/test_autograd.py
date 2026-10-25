"""Central finite differences against every autograd op and both full backbones (float64)."""
import numpy as np
import pytest

from src.ml.transfer_learning import autograd as ag
from src.ml.transfer_learning.nn import AdamW, Model


def numeric_grad(f, array, eps=1e-6):
    grad = np.zeros_like(array)
    for index in np.ndindex(array.shape):
        old = array[index]
        array[index] = old + eps
        up = f()
        array[index] = old - eps
        down = f()
        array[index] = old
        grad[index] = (up - down) / (2 * eps)
    return grad


def check(build, *shapes, seed=0, tol=1e-6):
    """build(*tensors) -> scalar Tensor; compare analytic and numeric grads of every input."""
    rng = np.random.RandomState(seed)
    params = [ag.parameter(rng.normal(size=s)) for s in shapes]
    out = build(*params)
    out.backward()
    for p in params:
        numeric = numeric_grad(lambda: float(build(*[ag.Tensor(q.data) for q in params]).data), p.data)
        err = np.abs(p.grad - numeric).max() / max(1.0, np.abs(numeric).max())
        assert err < tol, err


def weighted(t, seed=1):
    """Scalar that touches every output element with a different weight."""
    w = np.random.RandomState(seed).normal(size=t.shape)
    return ag.tensor_sum(t * w)


@pytest.mark.parametrize("name,build,shapes", [
    ("add_broadcast", lambda a, b: weighted(a + b), [(3, 4), (4,)]),
    ("sub_rsub", lambda a, b: weighted(1.0 - a - b), [(2, 3), (1, 3)]),
    ("mul_broadcast", lambda a, b: weighted(a * b), [(2, 3, 4), (3, 1)]),
    ("matmul_batched_weight", lambda a, b: weighted(a @ b), [(2, 3, 4), (4, 5)]),
    ("matmul_batched_both", lambda a, b: weighted(ag.matmul(a, b)), [(2, 2, 3, 4), (2, 2, 4, 3)]),
    ("sum_axis", lambda a: weighted(ag.tensor_sum(a, axis=1)), [(3, 4, 2)]),
    ("mean_axis", lambda a: weighted(ag.mean(a, axis=1)), [(3, 4, 2)]),
    ("reshape_transpose", lambda a: weighted(ag.transpose(ag.reshape(a, (2, 3, 4)), (2, 0, 1))), [(6, 4)]),
    ("getitem_slice", lambda a: weighted(a[:, 1:3, :] + a[:, -1, :][:, None, :]), [(2, 4, 3)]),
    ("stack", lambda a, b: weighted(ag.stack([a, b, a], axis=1)), [(2, 3), (2, 3)]),
    ("tanh", lambda a: weighted(ag.tanh(a)), [(3, 4)]),
    ("sigmoid", lambda a: weighted(ag.sigmoid(a * 3.0)), [(3, 4)]),
    ("gelu", lambda a: weighted(ag.gelu(a)), [(3, 4)]),
    ("softmax", lambda a: weighted(ag.softmax(a, axis=-1)), [(2, 3, 5)]),
    ("layer_norm", lambda x, g, b: weighted(ag.layer_norm(x, g, b)), [(2, 3, 6), (6,), (6,)]),
])
def test_op_gradients(name, build, shapes):
    check(build, *shapes)


def test_losses_gradients():
    y = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    check(lambda z: ag.bce_with_logits(z * 4.0, y, pos_weight=2.5), (5,))
    mask = np.array([[1, 0, 1], [0, 0, 1]], dtype=bool)
    target = np.random.RandomState(3).normal(size=(2, 3, 4))
    check(lambda p: ag.masked_mse(p, target, mask), (2, 3, 4))


def test_bce_matches_closed_form_and_survives_large_logits():
    z = np.array([-50.0, -1.0, 0.0, 2.0, 60.0])
    y = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    # -log(sigmoid(z)) = log1p(exp(-z)), exact in float64 for |z| <= 60.
    expected = np.mean(3 * y * np.log1p(np.exp(-z)) + (1 - y) * np.log1p(np.exp(z)))
    got = float(ag.bce_with_logits(ag.Tensor(z), y, pos_weight=3.0).data)
    assert np.isfinite(got)
    assert abs(got - expected) / expected < 1e-6


def test_softmax_rows_sum_to_one_and_mask_mse_ignores_unmasked():
    out = ag.softmax(ag.Tensor(np.array([[1000.0, 1001.0, 999.0]])), axis=-1).data
    assert np.allclose(out.sum(), 1.0) and np.isfinite(out).all()
    pred = ag.Tensor(np.zeros((1, 2, 1)))
    assert float(ag.masked_mse(pred, np.array([[[5.0], [1.0]]]), np.array([[False, True]])).data) == 1.0


@pytest.mark.parametrize("backbone", ["transformer", "gru"])
def test_full_model_gradients(backbone):
    model = Model(backbone, n_features=3, lookback=5, d_model=4, n_heads=2, n_layers=2, d_ff=6,
                  dropout=0.0, seed=1, dtype=np.float64)
    rng = np.random.RandomState(2)
    x = rng.normal(size=(3, 5, 3))
    y = np.array([1.0, 0.0, 1.0])
    mask = np.array([[0, 1, 1, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 1, 1]], dtype=bool)
    # Nonzero mask token so its gradient path is exercised too.
    model.params["enc.mask_token"].data[:] = rng.normal(size=3)

    def loss():
        return ag.bce_with_logits(model.logits(x), y, 1.7) + ag.masked_mse(model.reconstruct(x, mask), x, mask)

    loss().backward()
    analytic = {n: t.grad.copy() for n, t in model.params.items()}
    for name, tensor in model.params.items():
        numeric = numeric_grad(lambda: float(loss().data), tensor.data)
        err = np.abs(analytic[name] - numeric).max() / max(1.0, np.abs(numeric).max())
        assert err < 1e-6, (name, err)


def test_adamw_overfits_a_tiny_batch_and_respects_frozen_names():
    model = Model("transformer", n_features=3, lookback=6, d_model=8, n_heads=2, n_layers=1, d_ff=8,
                  dropout=0.0, seed=0, dtype=np.float64)
    rng = np.random.RandomState(0)
    x = rng.normal(size=(16, 6, 3))
    y = (x[:, -1, 0] > 0).astype(float)
    frozen = model.params["enc.in.w"].data.copy()
    trainable = {n: 1e-2 for n in model.params if n != "enc.in.w"}
    opt = AdamW(model.params, trainable, weight_decay=0.0)
    first = None
    for _ in range(150):
        opt.zero_grad()
        loss = ag.bce_with_logits(model.logits(x), y)
        loss.backward()
        opt.step()
        first = float(loss.data) if first is None else first
    assert float(loss.data) < 0.1 * first
    assert np.array_equal(model.params["enc.in.w"].data, frozen)


def test_same_seed_same_scores_and_state_round_trip():
    x = np.random.RandomState(5).normal(size=(4, 60, 8)).astype(np.float32)
    a, b = Model(seed=7), Model(seed=7)
    assert np.array_equal(a.predict_proba(x), b.predict_proba(x))
    c = Model(seed=8).load_state(a.state_dict())
    assert np.array_equal(a.predict_proba(x), c.predict_proba(x))
    assert a.predict_proba(x).shape == (4,)
