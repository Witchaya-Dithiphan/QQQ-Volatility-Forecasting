"""Masked-reconstruction pretraining and early-stopped fine-tuning for nn.Model."""
from __future__ import annotations

import numpy as np

from ..core.metrics import classification_metrics
from . import autograd as ag
from .nn import AdamW, Model

SPAN = 5          # masked spans are one trading week long
SPANS = 2         # 2 x 5 of 60 days = ~17% of tokens hidden
TAIL_FRACTION = 0.15
PURGE = 5         # the label looks 5 trading days ahead, as in mlp.py


def span_mask(rng: np.random.RandomState, batch: int, length: int) -> np.ndarray:
    mask = np.zeros((batch, length), dtype=bool)
    starts = rng.randint(0, length - SPAN + 1, size=(batch, SPANS))
    for row in range(batch):
        for start in starts[row]:
            mask[row, start:start + SPAN] = True
    return mask


def order(rng: np.random.RandomState, n: int) -> np.ndarray:
    # Random visiting order without replacement (argsort of uniforms keeps RandomState's stream).
    return np.argsort(rng.uniform(size=n), kind="stable")


def pretrain(model: Model, windows: np.ndarray, *, epochs: int, lr: float, batch: int, seed: int,
             monitor: np.ndarray | None = None, weight_decay: float = 1e-4) -> dict:
    """Reconstruct masked days of every window; monitor = fixed windows whose masked loss is logged."""
    rng = np.random.RandomState(seed)
    names = [n for n in model.params if n.startswith(("enc.", "recon."))]
    opt = AdamW(model.params, {n: lr for n in names}, weight_decay=weight_decay)
    dtype = model.params["head.w"].data.dtype
    windows = windows.astype(dtype)
    monitor_mask = None if monitor is None else span_mask(np.random.RandomState(999), len(monitor), monitor.shape[1])
    history = {"iteration": [], "loss": [], "monitor_loss": []}
    for epoch in range(1, epochs + 1):
        losses = []
        idx = order(rng, len(windows))
        for start in range(0, len(idx), batch):
            x = windows[idx[start:start + batch]]
            mask = span_mask(rng, len(x), x.shape[1])
            opt.zero_grad()
            loss = ag.masked_mse(model.reconstruct(x, mask, rng), x, mask)
            loss.backward()
            opt.step()
            losses.append(float(loss.data) * len(x))
        history["iteration"].append(epoch)
        history["loss"].append(sum(losses) / len(windows))
        if monitor is not None:
            m = monitor.astype(dtype)
            history["monitor_loss"].append(float(ag.masked_mse(model.reconstruct(m, monitor_mask), m, monitor_mask).data))
    return history


def tail_split(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(subtrain rows, tail rows): last 15% watches early stopping; 5 days before it are purged."""
    cut = int(len(positions) * (1 - TAIL_FRACTION))
    sub = np.arange(cut)
    sub = sub[positions[:cut] < positions[cut] - PURGE]
    return sub, np.arange(cut, len(positions))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    return float(classification_metrics(y, score)["average_precision"]["value"])


def finetune(model: Model, X: np.ndarray, y: np.ndarray, positions: np.ndarray, *, seed: int,
             encoder_lr: float | None, head_lr: float, max_epochs: int, patience: int, batch: int,
             weight_decay: float = 1e-4) -> dict:
    """BCE fine-tune with early stopping on tail AP; the best epoch's weights are kept.

    encoder_lr=None freezes the encoder (linear probe). The mask token and the
    reconstruction head never train here.
    """
    sub, tail = tail_split(positions)
    rng = np.random.RandomState(20_000 + seed)
    lrs = {n: head_lr for n in model.names("head")}
    if encoder_lr is not None:
        lrs.update({n: encoder_lr for n in model.names("enc") if n != "enc.mask_token"})
    opt = AdamW(model.params, lrs, weight_decay=weight_decay)
    dtype = model.params["head.w"].data.dtype
    Xs, ys = X[sub].astype(dtype), y[sub].astype(dtype)
    pos_weight = float((ys == 0).sum() / max((ys == 1).sum(), 1))
    best, best_epoch, since, best_state = -np.inf, 0, 0, model.state_dict()
    history = {"iteration": [], "loss": [], "tail_ap": []}
    for epoch in range(1, max_epochs + 1):
        idx = order(rng, len(Xs))
        total = 0.0
        for start in range(0, len(idx), batch):
            b = idx[start:start + batch]
            opt.zero_grad()
            loss = ag.bce_with_logits(model.logits(Xs[b], rng), ys[b], pos_weight)
            loss.backward()
            opt.step()
            total += float(loss.data) * len(b)
        ap = average_precision(y[tail], model.predict_proba(X[tail]))
        history["iteration"].append(epoch)
        history["loss"].append(total / len(Xs))
        history["tail_ap"].append(ap)
        if ap > best + 1e-6:  # ties keep the earliest epoch
            best, best_epoch, since, best_state = ap, epoch, 0, model.state_dict()
        else:
            since += 1
            if since >= patience:
                break
    model.load_state(best_state)
    history.update(best_epoch=best_epoch, best_tail_ap=best, pos_weight=pos_weight,
                   n_subtrain=int(len(sub)), n_tail=int(len(tail)), tail_positives=int(y[tail].sum()))
    return history
