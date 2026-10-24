"""Minimal reverse-mode autodiff on NumPy arrays, written for this experiment.

Only the operations a small Transformer/GRU needs are here. Softmax, layer norm and the
two losses are fused (one node with a hand-derived backward) because composing them
from primitives is slower and numerically worse. Every op is checked against central
finite differences in tests/ml/transfer_learning/test_autograd.py.
"""
from __future__ import annotations

import numpy as np


class Tensor:
    __array_priority__ = 100  # make ndarray (op) Tensor defer to Tensor's reflected op

    def __init__(self, data, parents=(), backward=None, requires_grad=False):
        self.data = np.asarray(data)
        self.grad = None
        self.parents = parents
        self.backward_fn = backward
        self.requires_grad = requires_grad or any(p.requires_grad for p in parents)

    shape = property(lambda self: self.data.shape)
    ndim = property(lambda self: self.data.ndim)

    def backward(self, grad=None) -> None:
        order, seen = [], set()

        def visit(node):
            # Iterative DFS: a 60-step GRU graph is deeper than Python's recursion limit.
            stack = [(node, False)]
            while stack:
                current, done = stack.pop()
                if done:
                    order.append(current)
                    continue
                if id(current) in seen or not current.requires_grad:
                    continue
                seen.add(id(current))
                stack.append((current, True))
                stack.extend((parent, False) for parent in current.parents)

        visit(self)
        self.grad = np.ones_like(self.data) if grad is None else grad
        for node in reversed(order):
            if node.backward_fn is not None and node.grad is not None:
                for parent, g in zip(node.parents, node.backward_fn(node.grad)):
                    if parent.requires_grad and g is not None:
                        parent.grad = g if parent.grad is None else parent.grad + g
                if node.parents:
                    node.grad = None  # intermediate: every consumer has already been processed

    def __add__(self, other): return add(self, other)
    __radd__ = __add__
    def __sub__(self, other): return add(self, mul(as_tensor(other), -1.0))
    def __rsub__(self, other): return add(as_tensor(other), mul(self, -1.0))
    def __mul__(self, other): return mul(self, other)
    __rmul__ = __mul__
    def __neg__(self): return mul(self, -1.0)
    def __matmul__(self, other): return matmul(self, other)
    def __getitem__(self, index): return getitem(self, index)


def as_tensor(value) -> Tensor:
    return value if isinstance(value, Tensor) else Tensor(value)


def parameter(data) -> Tensor:
    return Tensor(np.array(data), requires_grad=True)


def unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum a broadcast gradient back down to the operand's shape."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def add(a, b) -> Tensor:
    a, b = as_tensor(a), as_tensor(b)
    return Tensor(a.data + b.data, (a, b),
                  lambda g: (unbroadcast(g, a.shape), unbroadcast(g, b.shape)))


def mul(a, b) -> Tensor:
    a, b = as_tensor(a), as_tensor(b)
    return Tensor(a.data * b.data, (a, b),
                  lambda g: (unbroadcast(g * b.data, a.shape), unbroadcast(g * a.data, b.shape)))


def matmul(a, b) -> Tensor:
    a, b = as_tensor(a), as_tensor(b)

    def backward(g):
        ga = g @ np.swapaxes(b.data, -1, -2)
        gb = np.swapaxes(a.data, -1, -2) @ g
        return unbroadcast(ga, a.shape), unbroadcast(gb, b.shape)
    return Tensor(a.data @ b.data, (a, b), backward)


def tensor_sum(a: Tensor, axis=None, keepdims=False) -> Tensor:
    def backward(g):
        if axis is not None and not keepdims:
            g = np.expand_dims(g, axis)
        return (np.broadcast_to(g, a.shape).copy(),)
    return Tensor(a.data.sum(axis=axis, keepdims=keepdims), (a,), backward)


def mean(a: Tensor, axis=None, keepdims=False) -> Tensor:
    count = a.data.size if axis is None else np.prod([a.shape[i] for i in np.atleast_1d(axis)])
    return mul(tensor_sum(a, axis, keepdims), 1.0 / count)


def reshape(a: Tensor, shape) -> Tensor:
    return Tensor(a.data.reshape(shape), (a,), lambda g: (g.reshape(a.shape),))


def transpose(a: Tensor, axes) -> Tensor:
    inverse = np.argsort(axes)
    return Tensor(a.data.transpose(axes), (a,), lambda g: (g.transpose(inverse),))


def getitem(a: Tensor, index) -> Tensor:
    def backward(g):
        full = np.zeros_like(a.data)
        np.add.at(full, index, g)
        return (full,)
    return Tensor(a.data[index], (a,), backward)


def stack(tensors: list[Tensor], axis: int) -> Tensor:
    def backward(g):
        return tuple(np.take(g, i, axis=axis) for i in range(len(tensors)))
    return Tensor(np.stack([t.data for t in tensors], axis=axis), tuple(tensors), backward)


def tanh(a: Tensor) -> Tensor:
    out = np.tanh(a.data)
    return Tensor(out, (a,), lambda g: (g * (1 - out * out),))


def sigmoid(a: Tensor) -> Tensor:
    out = stable_sigmoid(a.data)
    return Tensor(out, (a,), lambda g: (g * out * (1 - out),))


def gelu(a: Tensor) -> Tensor:
    """tanh approximation (Hendrycks & Gimpel), the variant GPT-2/BERT ship."""
    x = a.data
    c = np.sqrt(2 / np.pi)
    inner = c * (x + 0.044715 * x ** 3)
    t = np.tanh(inner)
    out = 0.5 * x * (1 + t)

    def backward(g):
        d_inner = c * (1 + 3 * 0.044715 * x ** 2)
        return (g * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * d_inner),)
    return Tensor(out, (a,), backward)


def softmax(a: Tensor, axis: int = -1) -> Tensor:
    shifted = a.data - a.data.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    out = e / e.sum(axis=axis, keepdims=True)
    return Tensor(out, (a,), lambda g: (out * (g - (g * out).sum(axis=axis, keepdims=True)),))


def layer_norm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    mu = x.data.mean(axis=-1, keepdims=True)
    centered = x.data - mu
    var = (centered ** 2).mean(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = centered * inv
    out = xhat * gamma.data + beta.data

    def backward(g):
        gxhat = g * gamma.data
        n = x.shape[-1]
        gx = inv / n * (n * gxhat - gxhat.sum(axis=-1, keepdims=True)
                        - xhat * (gxhat * xhat).sum(axis=-1, keepdims=True))
        return gx, unbroadcast(g * xhat, gamma.shape), unbroadcast(g, beta.shape)
    return Tensor(out, (x, gamma, beta), backward)


def dropout(a: Tensor, rate: float, rng: np.random.RandomState | None) -> Tensor:
    """Inverted dropout; a no-op when rng is None (evaluation) or rate is 0."""
    if rng is None or rate == 0:
        return a
    keep = (rng.uniform(size=a.shape) >= rate).astype(a.data.dtype) / (1 - rate)
    return mul(a, keep)


def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1 / (1 + np.exp(-x[positive]))
    e = np.exp(x[~positive])
    out[~positive] = e / (1 + e)
    return out


def bce_with_logits(logits: Tensor, y: np.ndarray, pos_weight: float = 1.0) -> Tensor:
    """Mean of -[w*y*log(s) + (1-y)*log(1-s)], computed from logits without overflow."""
    z = logits.data
    log_sig = -np.logaddexp(0, -z)
    log_one_minus = -np.logaddexp(0, z)
    loss = -(pos_weight * y * log_sig + (1 - y) * log_one_minus).mean()
    s = stable_sigmoid(z)

    def backward(g):
        return (g * (pos_weight * y * (s - 1) + (1 - y) * s) / z.size,)
    return Tensor(np.asarray(loss, dtype=z.dtype), (logits,), backward)


def masked_mse(prediction: Tensor, target: np.ndarray, mask: np.ndarray) -> Tensor:
    """Mean squared error over the masked time steps only; mask is (B, L), target (B, L, F)."""
    weight = np.broadcast_to(mask[..., None], prediction.shape).astype(prediction.data.dtype)
    count = max(weight.sum(), 1.0)
    diff = (prediction.data - target) * weight
    loss = (diff ** 2).sum() / count
    return Tensor(np.asarray(loss, dtype=prediction.data.dtype), (prediction,),
                  lambda g: (g * 2 * diff / count,))
