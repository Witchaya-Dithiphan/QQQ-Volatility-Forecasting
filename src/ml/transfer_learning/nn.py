"""Layers, the two sequence backbones and AdamW, all on the NumPy autograd in autograd.py.

A model keeps its parameters in one flat {name: Tensor} dict, so saving, loading,
freezing the encoder and giving the head its own learning rate are dict operations.
"""
from __future__ import annotations

import numpy as np

from . import autograd as ag
from .autograd import Tensor, parameter


def uniform(rng, fan_in: int, shape, dtype) -> Tensor:
    bound = 1.0 / np.sqrt(fan_in)  # PyTorch's default nn.Linear/GRU range
    return parameter(rng.uniform(-bound, bound, size=shape).astype(dtype))


class Model:
    """Sequence encoder + reconstruction head (pretraining) + probability head (fine-tune)."""

    def __init__(self, backbone: str = "transformer", n_features: int = 8, lookback: int = 60,
                 d_model: int = 32, n_heads: int = 4, n_layers: int = 2, d_ff: int = 64,
                 dropout: float = 0.1, seed: int = 0, dtype=np.float32):
        if backbone not in ("transformer", "gru") or d_model % n_heads:
            raise ValueError("Unknown backbone or d_model not divisible by n_heads")
        self.config = dict(backbone=backbone, n_features=n_features, lookback=lookback, d_model=d_model,
                           n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, dropout=dropout, seed=seed)
        rng = np.random.RandomState(seed)
        d, f, p = d_model, n_features, {}
        p["enc.mask_token"] = parameter(np.zeros(f, dtype=dtype))
        if backbone == "transformer":
            p["enc.in.w"], p["enc.in.b"] = uniform(rng, f, (f, d), dtype), uniform(rng, f, (d,), dtype)
            p["enc.pos"] = parameter((rng.normal(size=(lookback, d)) * 0.02).astype(dtype))
            for i in range(n_layers):
                k = f"enc.block{i}."
                for ln in ("ln1", "ln2"):
                    p[k + ln + ".g"], p[k + ln + ".b"] = parameter(np.ones(d, dtype)), parameter(np.zeros(d, dtype))
                p[k + "qkv.w"], p[k + "qkv.b"] = uniform(rng, d, (d, 3 * d), dtype), uniform(rng, d, (3 * d,), dtype)
                p[k + "out.w"], p[k + "out.b"] = uniform(rng, d, (d, d), dtype), uniform(rng, d, (d,), dtype)
                p[k + "ff1.w"], p[k + "ff1.b"] = uniform(rng, d, (d, d_ff), dtype), uniform(rng, d, (d_ff,), dtype)
                p[k + "ff2.w"], p[k + "ff2.b"] = uniform(rng, d_ff, (d_ff, d), dtype), uniform(rng, d_ff, (d,), dtype)
            p["enc.lnf.g"], p["enc.lnf.b"] = parameter(np.ones(d, dtype)), parameter(np.zeros(d, dtype))
        else:
            # Gates ordered (reset, update, candidate) as in PyTorch's nn.GRU.
            p["enc.gru.wx"], p["enc.gru.bx"] = uniform(rng, d, (f, 3 * d), dtype), uniform(rng, d, (3 * d,), dtype)
            p["enc.gru.wh"], p["enc.gru.bh"] = uniform(rng, d, (d, 3 * d), dtype), uniform(rng, d, (3 * d,), dtype)
        p["recon.w"], p["recon.b"] = uniform(rng, d, (d, f), dtype), uniform(rng, d, (f,), dtype)
        p["head.w"], p["head.b"] = uniform(rng, d, (d, 1), dtype), parameter(np.zeros(1, dtype))
        self.params = p

    # -- forward ---------------------------------------------------------------
    def encode(self, x: np.ndarray, mask: np.ndarray | None = None, rng=None) -> Tensor:
        """(B, L, F) windows -> (B, L, d) states. mask (B, L) marks days replaced by the mask token."""
        p, c = self.params, self.config
        inp = Tensor(x)
        if mask is not None:
            m = mask[..., None].astype(x.dtype)
            inp = inp * (1 - m) + p["enc.mask_token"] * m
        if c["backbone"] == "gru":
            return self._gru(inp)
        h = inp @ p["enc.in.w"] + p["enc.in.b"] + p["enc.pos"][-x.shape[1]:]
        for i in range(c["n_layers"]):
            k = f"enc.block{i}."
            h = h + ag.dropout(self._attention(ag.layer_norm(h, p[k + "ln1.g"], p[k + "ln1.b"]), k), c["dropout"], rng)
            z = ag.layer_norm(h, p[k + "ln2.g"], p[k + "ln2.b"])
            z = ag.gelu(z @ p[k + "ff1.w"] + p[k + "ff1.b"]) @ p[k + "ff2.w"] + p[k + "ff2.b"]
            h = h + ag.dropout(z, c["dropout"], rng)
        return ag.layer_norm(h, p["enc.lnf.g"], p["enc.lnf.b"])

    def _attention(self, x: Tensor, k: str) -> Tensor:
        p, heads = self.params, self.config["n_heads"]
        b, length, d = x.shape
        dh = d // heads
        qkv = ag.reshape(x @ p[k + "qkv.w"] + p[k + "qkv.b"], (b, length, 3, heads, dh))
        qkv = ag.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, H, L, dh)
        q, key, v = qkv[0], qkv[1], qkv[2]
        scores = ag.matmul(q, ag.transpose(key, (0, 1, 3, 2))) * (1.0 / np.sqrt(dh))
        out = ag.matmul(ag.softmax(scores, axis=-1), v)  # (B, H, L, dh)
        out = ag.reshape(ag.transpose(out, (0, 2, 1, 3)), (b, length, d))
        return out @ p[k + "out.w"] + p[k + "out.b"]

    def _gru(self, inp: Tensor) -> Tensor:
        p, d = self.params, self.config["d_model"]
        b, length, _ = inp.shape
        gx = inp @ p["enc.gru.wx"] + p["enc.gru.bx"]  # all input projections at once
        h = Tensor(np.zeros((b, d), dtype=inp.data.dtype))
        states = []
        for t in range(length):
            xt = gx[:, t, :]
            gh = h @ p["enc.gru.wh"] + p["enc.gru.bh"]
            r = ag.sigmoid(xt[:, :d] + gh[:, :d])
            z = ag.sigmoid(xt[:, d:2 * d] + gh[:, d:2 * d])
            n = ag.tanh(xt[:, 2 * d:] + r * gh[:, 2 * d:])
            h = (1 - z) * n + z * h
            states.append(h)
        return ag.stack(states, axis=1)

    def pooled(self, states: Tensor) -> Tensor:
        if self.config["backbone"] == "gru":
            return states[:, -1, :]
        return ag.mean(states, axis=1)

    def reconstruct(self, x, mask, rng=None) -> Tensor:
        return self.encode(x, mask, rng) @ self.params["recon.w"] + self.params["recon.b"]

    def logits(self, x, rng=None) -> Tensor:
        z = self.pooled(self.encode(x, None, rng)) @ self.params["head.w"] + self.params["head.b"]
        return ag.reshape(z, (x.shape[0],))

    def predict_proba(self, X: np.ndarray, batch: int = 512) -> np.ndarray:
        dtype = self.params["head.w"].data.dtype
        out = [ag.stable_sigmoid(self.logits(X[i:i + batch].astype(dtype)).data.astype(np.float64))
               for i in range(0, len(X), batch)]
        return np.concatenate(out)

    # -- parameters ------------------------------------------------------------
    def names(self, group: str) -> list[str]:
        return [n for n in self.params if n.startswith(group + ".")]

    def state_dict(self) -> dict[str, np.ndarray]:
        return {n: t.data.copy() for n, t in self.params.items()}

    def load_state(self, arrays) -> "Model":
        missing = set(self.params) - set(arrays)
        if missing:
            raise ValueError(f"State is missing {sorted(missing)}")
        for n, t in self.params.items():
            if arrays[n].shape != t.shape:
                raise ValueError(f"Shape mismatch for {n}")
            t.data = np.array(arrays[n], dtype=t.data.dtype)
        return self

    def reset_head(self, seed: int) -> None:
        """Fresh probability head, drawn the same way for every arm with the same seed."""
        rng = np.random.RandomState(10_000 + seed)
        d, dtype = self.config["d_model"], self.params["head.w"].data.dtype
        self.params["head.w"] = uniform(rng, d, (d, 1), dtype)
        self.params["head.b"] = parameter(np.zeros(1, dtype))


class AdamW:
    """Adam with decoupled weight decay and global grad-norm clipping (Loshchilov & Hutter)."""

    def __init__(self, params: dict[str, Tensor], lrs: dict[str, float], weight_decay: float = 1e-4,
                 betas=(0.9, 0.999), eps: float = 1e-8, clip: float = 1.0):
        self.params, self.lrs = params, lrs  # lrs: name -> learning rate; absent names are frozen
        self.wd, self.b1, self.b2, self.eps, self.clip = weight_decay, betas[0], betas[1], eps, clip
        self.m = {n: np.zeros_like(params[n].data) for n in lrs}
        self.v = {n: np.zeros_like(params[n].data) for n in lrs}
        self.t = 0

    def zero_grad(self) -> None:
        for t in self.params.values():
            t.grad = None

    def step(self) -> float:
        grads = {n: self.params[n].grad for n in self.lrs if self.params[n].grad is not None}
        norm = float(np.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads.values())))
        scale = min(1.0, self.clip / (norm + 1e-12)) if self.clip else 1.0
        self.t += 1
        for n, g in grads.items():
            g = g * scale
            self.m[n] = self.b1 * self.m[n] + (1 - self.b1) * g
            self.v[n] = self.b2 * self.v[n] + (1 - self.b2) * g * g
            m_hat = self.m[n] / (1 - self.b1 ** self.t)
            v_hat = self.v[n] / (1 - self.b2 ** self.t)
            data = self.params[n].data
            decay = self.wd if data.ndim > 1 else 0.0  # no decay on biases / norms / mask token
            self.params[n].data = (data - self.lrs[n] * (m_hat / (np.sqrt(v_hat) + self.eps) + decay * data)).astype(data.dtype)
        return norm
