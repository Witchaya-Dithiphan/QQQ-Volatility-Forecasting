"""M9 stability checks: seed variance, blocked 5-fold CV and row-shuffle stability. Train/Validation only; Test is never read."""
from __future__ import annotations
import json
import warnings
import numpy as np
from config import PROJECT_ROOT
from .datasets import load_train_validation
from .classification.decision_tree import DecisionTreeClassifier
from .classification.logistic import LogisticRegression
from .classification.naive_bayes import GaussianNaiveBayes
from .classification.perceptron import Perceptron
from .classification.slp import SLP
from .ensemble.adaboost import AdaBoost
from .ensemble.gradient_boosting import GradientBoosting
from .ensemble.random_forest import RandomForest
from .ensemble.stacking import StackingClassifier
from .ensemble.xgboost import XGBoostClassifier
from .regression.elastic_net import ElasticNet
from .regression.multiple_linear import MultiLinearRegression
from .regression.polynomial import PolynomialRegression
from .regression.simple_linear import SimpleLinearRegression

SEEDS = (42, 123, 999)
SHUFFLE_SEEDS = (1, 2, 3)
HORIZON = 5  # target_volatility_5d overlaps 5 rows, so CV purges 5 rows next to the held-out block
FEATURES = ["return_1d", "return_5d", "historical_volatility_5d", "historical_volatility_20d",
            "intraday_range", "sma_ratio_5_20", "rsi_14", "volume_zscore_20"]
SEED_TOL, CV_REL_STD_TOL, SHUFFLE_TOL = 0.02, 0.05, 0.02

# name -> (task, feature used or None for all, factory(seed) or None when the model takes no seed). Hyperparameters
# are the ones recorded in outputs/modeling/**/metadata.json.
SPECS = {
    "MLR": ("regression", None, None, lambda s: MultiLinearRegression()),
    "Simple Linear": ("regression", "return_1d", None, lambda s: SimpleLinearRegression()),
    "Polynomial": ("regression", "historical_volatility_20d", None, lambda s: PolynomialRegression(3)),
    "Elastic Net": ("regression", None, None, lambda s: ElasticNet(alpha=0.01, l1_ratio=0.5)),
    "Logistic": ("classification", None, "random_state", lambda s: LogisticRegression(random_state=s)),
    "Stacking": ("classification", None, "random_state", lambda s: StackingClassifier(random_state=s)),
    "Decision Tree": ("classification", None, "random_state", lambda s: DecisionTreeClassifier(random_state=s)),
    "Naive Bayes": ("classification", None, None, lambda s: GaussianNaiveBayes()),
    "XGBoost": ("classification", None, None, lambda s: XGBoostClassifier(n_estimators=10)),
    "Random Forest": ("classification", None, "random_state", lambda s: RandomForest(n_estimators=10, random_state=s)),
    "Gradient Boosting": ("classification", None, "random_state", lambda s: GradientBoosting(n_estimators=10, random_state=s)),
    "Perceptron": ("classification", None, None, lambda s: Perceptron(max_iter=20)),
    "SLP": ("classification", None, "random_state", lambda s: SLP(random_state=s)),
    "AdaBoost": ("classification", None, "random_state", lambda s: AdaBoost(n_estimators=10, random_state=s)),
}


def blocked_folds(n: int, k: int = 5, purge: int = HORIZON):
    """Contiguous held-out blocks; `purge` rows on each side of the block are dropped from the training indices."""
    edges = np.linspace(0, n, k + 1).astype(int)
    for lo, hi in zip(edges[:-1], edges[1:]):
        yield np.r_[0:max(lo - purge, 0), min(hi + purge, n):n], np.arange(lo, hi)


def _metric(task: str, y, pred) -> float:
    return float(np.mean((y - pred) ** 2)) if task == "regression" else float(np.mean(y == pred))


def _fit_predict(spec, seed, Xa, ya, Xb, *, dates=None, original_dates=None):
    cols = [FEATURES.index(spec[1])] if spec[1] else slice(None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = spec[3](seed)
        if isinstance(model, StackingClassifier):
            model.fit(Xa[:, cols], ya, dates=dates, original_dates=original_dates)
        else:
            model.fit(Xa[:, cols], ya)
        return np.asarray(model.predict(Xb[:, cols]), dtype=float)


def _disagreement(task, ref, other) -> float:
    """Percent of predictions that differ (classification) or mean relative absolute change (regression)."""
    return 100 * float(np.mean(ref != other) if task == "classification" else np.mean(np.abs(ref - other)) / np.mean(np.abs(ref)))


def check_model(name, tr, va, y_tr, y_va, *, original_dates=None) -> dict:
    spec = SPECS[name]
    task = spec[0]
    original_dates = tr.dates if original_dates is None else original_dates
    preds = [_fit_predict(spec, s, tr.X, y_tr, va.X, dates=tr.dates, original_dates=original_dates) for s in SEEDS]
    seed_var = max(_disagreement(task, preds[0], p) for p in preds[1:])
    base = _metric(task, y_va, preds[0])
    if name == "Stacking":
        # Outer evaluation is expanding too; inner fits get only the historical timeline.
        position = np.searchsorted(original_dates, tr.dates)
        folds = []
        start = int(.4 * len(original_dates))
        for te in np.array_split(np.flatnonzero(position >= start), 5):
            if not len(te): raise ValueError("Empty chronological stability block")
            fit = np.flatnonzero(position < position[te[0]] - HORIZON)
            if not len(fit): raise ValueError("Empty purged stability Train")
            timeline = original_dates[:position[fit[-1]] + 1]
            pred = _fit_predict(spec, 42, tr.X[fit], y_tr[fit], tr.X[te],
                                dates=tr.dates[fit], original_dates=timeline)
            folds.append(_metric(task, y_tr[te], pred))
    else:
        folds = [_metric(task, y_tr[te], _fit_predict(spec, 42, tr.X[fit], y_tr[fit], tr.X[te])) for fit, te in blocked_folds(len(tr.X))]
    shuffled = []
    for s in ([] if name == "Stacking" else SHUFFLE_SEEDS):
        order = np.random.default_rng(s).permutation(len(tr.X))
        shuffled.append(_metric(task, y_va, _fit_predict(spec, 42, tr.X[order], y_tr[order], va.X)))
    # regression: relative MSE change; classification: absolute accuracy change
    shuffle_var = max(abs(m - base) / base if task == "regression" else abs(m - base) for m in shuffled) * 100 if shuffled else None
    cv_mean, cv_std = float(np.mean(folds)), float(np.std(folds, ddof=1))
    result = {"task": task, "seeded": spec[2] is not None, "seed_var_pct": seed_var, "val_metric": base,
              "cv_folds": folds, "cv_mean": cv_mean, "cv_std": cv_std, "cv_rel_std": cv_std / cv_mean,
              "shuffle_var_pct": shuffle_var}
    result["checks"] = {"seed": seed_var < 100 * SEED_TOL, "cv": result["cv_rel_std"] < CV_REL_STD_TOL, "shuffle": shuffle_var <= 100 * SHUFFLE_TOL if shuffle_var is not None else None}
    if name == "Stacking":
        result["shuffle_reason"] = "Row shuffle is incompatible with mandatory chronological Stacking"
        result["cv_protocol"] = "expanding_original_timeline_five_day_purge"
    return result


def run(output=PROJECT_ROOT / "outputs" / "reports" / "stability_results.json") -> dict:
    data = load_train_validation("with_spike")
    tr, va = data.train, data.validation
    targets = {"contract_q75": (tr.y_classification, va.y_classification)}
    results = {}
    for name, spec in SPECS.items():
        if spec[0] == "regression":
            results[name] = {"regression": check_model(name, tr, va, tr.y_regression, va.y_regression)}
        else:
            results[name] = {t: check_model(name, tr, va, *ys) for t, ys in targets.items()}
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    run()
