"""Tree-based baseline models and multi-seed cross-validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
from sklearn.model_selection import StratifiedKFold

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None


@dataclass
class TreeBaselineConfig:
    n_estimators: int = 200
    max_depth: int | None = 10


class RandomForestSpreadModel:
    def __init__(self, random_state: int, config: TreeBaselineConfig | None = None) -> None:
        cfg = config or TreeBaselineConfig()
        self.model = RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, y)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)[:, 1]


class XGBoostSpreadModel:
    def __init__(self, random_state: int) -> None:
        if XGBClassifier is None:
            raise ImportError("xgboost is required for XGBoostSpreadModel.")

        self.model = XGBClassifier(
            objective="binary:logistic",
            n_estimators=250,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, y)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)[:, 1]


def _compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    preds = (scores >= 0.5).astype(int)
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, scores)
    return {
        "auc": float(sklearn.metrics.auc(fpr, tpr)),
        "precision": float(sklearn.metrics.precision_score(y_true, preds, zero_division=0)),
        "recall": float(sklearn.metrics.recall_score(y_true, preds, zero_division=0)),
        "f1": float(sklearn.metrics.f1_score(y_true, preds, zero_division=0)),
        "loss": float(sklearn.metrics.log_loss(y_true, np.clip(scores, 1e-6, 1 - 1e-6))),
    }


def evaluate_tree_baselines_multiseed_cv(
    x: np.ndarray,
    y: np.ndarray,
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    n_splits: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Run 5-seed cross-validation and report mean/std metrics for RF + XGBoost."""
    metric_values: Dict[str, Dict[str, list[float]]] = {
        "random_forest": {"auc": [], "precision": [], "recall": [], "f1": [], "loss": []},
    }
    if XGBClassifier is not None:
        metric_values["xgboost"] = {"auc": [], "precision": [], "recall": [], "f1": [], "loss": []}

    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, test_idx in cv.split(x, y):
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            rf_model = RandomForestSpreadModel(random_state=seed)
            rf_model.fit(x_train, y_train)
            rf_metrics = _compute_metrics(y_test, rf_model.predict_proba(x_test))
            for key, value in rf_metrics.items():
                metric_values["random_forest"][key].append(value)

            if XGBClassifier is not None:
                xgb_model = XGBoostSpreadModel(random_state=seed)
                xgb_model.fit(x_train, y_train)
                xgb_metrics = _compute_metrics(y_test, xgb_model.predict_proba(x_test))
                for key, value in xgb_metrics.items():
                    metric_values["xgboost"][key].append(value)

    summary: Dict[str, Dict[str, float]] = {}
    for model_name, values in metric_values.items():
        summary[model_name] = {}
        for metric_name, metric_series in values.items():
            arr = np.asarray(metric_series, dtype=float)
            summary[model_name][f"{metric_name}_mean"] = float(arr.mean())
            summary[model_name][f"{metric_name}_std"] = float(arr.std(ddof=1))

    return summary
