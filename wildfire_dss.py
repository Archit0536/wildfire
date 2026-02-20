"""Wildfire spatial decision-support system.

Implements:
1) Multi-day wildfire spread prediction from FIRMS-like fire detections.
2) Baseline vs quantum-inspired model comparison with ROC/AUC metrics.
3) Wildlife evacuation simulation on a spatial graph.
4) Ecological risk reduction quantification.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_curve
from sklearn.model_selection import train_test_split


@dataclass
class GridSpec:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    rows: int = 30
    cols: int = 30


def load_firms_csv(csv_path: Path) -> pd.DataFrame:
    """Load FIRMS-style CSV.

    Expected columns include latitude, longitude, acq_date.
    """
    df = pd.read_csv(csv_path)
    required = {"latitude", "longitude", "acq_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in FIRMS CSV: {missing}")

    df["acq_date"] = pd.to_datetime(df["acq_date"])
    return df


def generate_synthetic_firms_data(n_days: int = 20, points_per_day: int = 1000) -> pd.DataFrame:
    """Generate synthetic California-like fire detections when no local FIRMS file exists."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-07-01", periods=n_days, freq="D")

    all_rows = []
    center_lat, center_lon = 37.2, -120.2
    for i, day in enumerate(dates):
        drift_lat = center_lat + 0.05 * i + rng.normal(0, 0.02)
        drift_lon = center_lon + 0.08 * i + rng.normal(0, 0.02)

        lats = rng.normal(drift_lat, 1.5, size=points_per_day)
        lons = rng.normal(drift_lon, 1.5, size=points_per_day)
        conf = rng.uniform(30, 100, size=points_per_day)
        frp = np.clip(rng.normal(35 + i, 15, size=points_per_day), 0, None)

        day_df = pd.DataFrame(
            {
                "latitude": np.clip(lats, 32.5, 42.2),
                "longitude": np.clip(lons, -124.5, -114.0),
                "acq_date": day,
                "confidence": conf,
                "frp": frp,
            }
        )
        all_rows.append(day_df)

    return pd.concat(all_rows, ignore_index=True)


def detections_to_grid(df: pd.DataFrame, grid: GridSpec) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Convert detections into [day, row, col] count tensor."""
    days = pd.DatetimeIndex(sorted(df["acq_date"].dt.normalize().unique()))
    day_to_idx = {d: i for i, d in enumerate(days)}

    lat_bins = np.linspace(grid.lat_min, grid.lat_max, grid.rows + 1)
    lon_bins = np.linspace(grid.lon_min, grid.lon_max, grid.cols + 1)

    day_ix = df["acq_date"].dt.normalize().map(day_to_idx).to_numpy()
    row_ix = np.digitize(df["latitude"], lat_bins) - 1
    col_ix = np.digitize(df["longitude"], lon_bins) - 1

    valid = (row_ix >= 0) & (row_ix < grid.rows) & (col_ix >= 0) & (col_ix < grid.cols)
    tensor = np.zeros((len(days), grid.rows, grid.cols), dtype=np.float32)

    for d, r, c in zip(day_ix[valid], row_ix[valid], col_ix[valid]):
        tensor[d, r, c] += 1

    return tensor, days


def neighbor_sum(arr_2d: np.ndarray) -> np.ndarray:
    padded = np.pad(arr_2d, 1, mode="constant")
    out = np.zeros_like(arr_2d)
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue
            out += padded[i : i + arr_2d.shape[0], j : j + arr_2d.shape[1]]
    return out


def build_features_targets(tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build X, y where X at day t predicts active-fire presence at t+1."""
    features = []
    targets = []

    for t in range(tensor.shape[0] - 1):
        cur = tensor[t]
        nxt = tensor[t + 1]

        cur_norm = cur / (cur.max() + 1e-6)
        neigh = neighbor_sum(cur)
        neigh_norm = neigh / (neigh.max() + 1e-6)

        x_t = np.stack([cur_norm, neigh_norm], axis=-1).reshape(-1, 2)
        y_t = (nxt.reshape(-1) > 0).astype(int)

        features.append(x_t)
        targets.append(y_t)

    return np.vstack(features), np.concatenate(targets)


def quantum_feature_map(x: np.ndarray) -> np.ndarray:
    """Quantum-inspired nonlinear encoding using phase-like sinusoidal projections."""
    x1 = x[:, 0]
    x2 = x[:, 1]
    return np.column_stack(
        [
            x1,
            x2,
            np.sin(math.pi * x1),
            np.cos(math.pi * x2),
            np.sin(2 * math.pi * x1 * x2),
            np.cos(2 * math.pi * (x1 - x2)),
        ]
    )


class BaselineSpreadModel:
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        return None

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        # Baseline persistence: use current cell intensity only.
        return np.clip(x[:, 0], 0, 1)


class QuantumInspiredSpreadModel:
    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=300)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        z = quantum_feature_map(x)
        self.model.fit(z, y)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = quantum_feature_map(x)
        return self.model.predict_proba(z)[:, 1]


def evaluate_models(x: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, np.ndarray | float]]:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )

    baseline = BaselineSpreadModel()
    qmodel = QuantumInspiredSpreadModel()

    baseline.fit(x_train, y_train)
    qmodel.fit(x_train, y_train)

    out = {}
    for name, model in {"baseline": baseline, "quantum_inspired": qmodel}.items():
        scores = model.predict_proba(x_test)
        preds = (scores >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y_test, scores)
        out[name] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": float(auc(fpr, tpr)),
            "scores": scores,
            "y_test": y_test,
            "confusion_matrix": confusion_matrix(y_test, preds, labels=[0, 1]),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "loss": float(log_loss(y_test, np.clip(scores, 1e-6, 1 - 1e-6))),
        }
    return out


def predict_risk_map_next_day(model: QuantumInspiredSpreadModel, day_grid: np.ndarray) -> np.ndarray:
    cur = day_grid
    cur_norm = cur / (cur.max() + 1e-6)
    neigh_norm = neighbor_sum(cur) / (neighbor_sum(cur).max() + 1e-6)
    x = np.stack([cur_norm, neigh_norm], axis=-1).reshape(-1, 2)
    return model.predict_proba(x).reshape(day_grid.shape)


def multi_day_forecast(model: QuantumInspiredSpreadModel, seed_grid: np.ndarray, horizon: int) -> List[np.ndarray]:
    maps = []
    current = seed_grid.copy()
    for _ in range(horizon):
        risk = predict_risk_map_next_day(model, current)
        maps.append(risk)
        current = (risk > 0.5).astype(float)
    return maps


def build_spatial_graph(rows: int, cols: int) -> nx.Graph:
    g = nx.grid_2d_graph(rows, cols)
    for u, v in g.edges:
        g.edges[u, v]["distance"] = 1.0
    return g


def route_cost(path: Sequence[Tuple[int, int]], risk_map: np.ndarray) -> float:
    return float(sum(risk_map[r, c] for r, c in path))


def baseline_route(g: nx.Graph, start: Tuple[int, int], safe_nodes: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    best_path = None
    best_len = float("inf")
    for sn in safe_nodes:
        try:
            p = nx.shortest_path(g, start, sn, weight="distance")
            if len(p) < best_len:
                best_path = p
                best_len = len(p)
        except nx.NetworkXNoPath:
            continue
    return best_path if best_path else [start]


def quantum_inspired_route(
    g: nx.Graph,
    start: Tuple[int, int],
    safe_nodes: Sequence[Tuple[int, int]],
    risk_map: np.ndarray,
) -> List[Tuple[int, int]]:
    # "Quantum-inspired" phase perturbation in edge weights + risk-avoiding potential.
    g2 = g.copy()
    for u, v in g2.edges:
        phase = math.sin((u[0] + v[1]) * 0.7) * 0.2
        risk_term = 0.7 * (risk_map[u] + risk_map[v])
        g2.edges[u, v]["qweight"] = 1.0 + risk_term + phase

    best_path = None
    best_score = float("inf")
    for sn in safe_nodes:
        try:
            p = nx.shortest_path(g2, start, sn, weight="qweight")
            s = route_cost(p, risk_map)
            if s < best_score:
                best_score = s
                best_path = p
        except nx.NetworkXNoPath:
            continue

    return best_path if best_path else [start]


def ecological_risk(population_nodes: Dict[Tuple[int, int], int], risk_map: np.ndarray) -> float:
    return float(sum(pop * risk_map[node] for node, pop in population_nodes.items()))


def simulate_evacuation(
    risk_map: np.ndarray,
    populations: Dict[Tuple[int, int], int],
    safe_nodes: Sequence[Tuple[int, int]],
) -> Dict[str, float]:
    rows, cols = risk_map.shape
    g = build_spatial_graph(rows, cols)

    pre_risk = ecological_risk(populations, risk_map)

    moved_baseline = {}
    moved_quantum = {}

    for node, pop in populations.items():
        bpath = baseline_route(g, node, safe_nodes)
        qpath = quantum_inspired_route(g, node, safe_nodes, risk_map)
        moved_baseline[bpath[-1]] = moved_baseline.get(bpath[-1], 0) + pop
        moved_quantum[qpath[-1]] = moved_quantum.get(qpath[-1], 0) + pop

    baseline_post = ecological_risk(moved_baseline, risk_map)
    quantum_post = ecological_risk(moved_quantum, risk_map)

    return {
        "pre_risk": pre_risk,
        "baseline_post": baseline_post,
        "quantum_post": quantum_post,
        "baseline_reduction_pct": 100 * (pre_risk - baseline_post) / (pre_risk + 1e-6),
        "quantum_reduction_pct": 100 * (pre_risk - quantum_post) / (pre_risk + 1e-6),
    }


def plot_roc_curves(metrics: Dict[str, Dict[str, np.ndarray | float]], out_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    for name, m in metrics.items():
        plt.plot(m["fpr"], m["tpr"], label=f"{name} AUC={m['auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Wildfire Spread Prediction ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_fire_spread_maps(
    actual: np.ndarray,
    predicted_maps: Sequence[np.ndarray],
    out_path: Path,
) -> None:
    k = len(predicted_maps)
    cols = min(3, k)
    rows = int(np.ceil((k + 1) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(-1)

    axes[0].imshow(actual, cmap="hot")
    axes[0].set_title("Actual (reference day)")
    axes[0].axis("off")

    for i, pred in enumerate(predicted_maps, start=1):
        axes[i].imshow(pred, cmap="hot", vmin=0, vmax=1)
        axes[i].set_title(f"Predicted Day +{i}")
        axes[i].axis("off")

    for j in range(k + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--firms_csv", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--horizon_days", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.firms_csv and args.firms_csv.exists():
        df = load_firms_csv(args.firms_csv)
    else:
        df = generate_synthetic_firms_data(n_days=25, points_per_day=900)

    grid = GridSpec(lat_min=32.5, lat_max=42.2, lon_min=-124.5, lon_max=-114.0, rows=28, cols=28)
    tensor, days = detections_to_grid(df, grid)

    x, y = build_features_targets(tensor)
    metrics = evaluate_models(x, y)

    qmodel = QuantumInspiredSpreadModel()
    qmodel.fit(x, y)

    seed_day = tensor[-2]
    preds = multi_day_forecast(qmodel, seed_day, horizon=args.horizon_days)

    roc_path = args.output_dir / "roc_curve.png"
    map_path = args.output_dir / "fire_spread_maps.png"
    plot_roc_curves(metrics, roc_path)
    plot_fire_spread_maps(actual=tensor[-1], predicted_maps=preds, out_path=map_path)

    # Example wildlife populations on high-risk cells and edge safe nodes
    last_risk = preds[0]
    flat_idx = np.argsort(last_risk.reshape(-1))[-20:]
    pop_nodes = {(int(i // grid.cols), int(i % grid.cols)): 30 for i in flat_idx}
    safe_nodes = [(0, c) for c in range(0, grid.cols, 5)] + [(grid.rows - 1, c) for c in range(0, grid.cols, 5)]

    evac_report = simulate_evacuation(last_risk, pop_nodes, safe_nodes)

    summary = {
        "n_days": len(days),
        "n_detections": len(df),
        "baseline_auc": metrics["baseline"]["auc"],
        "baseline_precision": metrics["baseline"]["precision"],
        "baseline_recall": metrics["baseline"]["recall"],
        "baseline_f1": metrics["baseline"]["f1"],
        "baseline_loss": metrics["baseline"]["loss"],
        "baseline_confusion_matrix": metrics["baseline"]["confusion_matrix"].tolist(),
        "quantum_auc": metrics["quantum_inspired"]["auc"],
        "quantum_precision": metrics["quantum_inspired"]["precision"],
        "quantum_recall": metrics["quantum_inspired"]["recall"],
        "quantum_f1": metrics["quantum_inspired"]["f1"],
        "quantum_loss": metrics["quantum_inspired"]["loss"],
        "quantum_confusion_matrix": metrics["quantum_inspired"]["confusion_matrix"].tolist(),
        **evac_report,
        "roc_curve": str(roc_path),
        "fire_spread_map": str(map_path),
    }

    summary_path = args.output_dir / "summary_metrics.json"
    pd.Series(summary).to_json(summary_path, indent=2)

    print("Run complete. Outputs:")
    print(f"- {roc_path}")
    print(f"- {map_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
