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
from ca_algorithms import DeterministicCAModel, PersistenceCAModel
from tree_baselines import (
    XGBClassifier,
    RandomForestSpreadModel,
    XGBoostSpreadModel,
    evaluate_tree_baselines_holdout,
    evaluate_tree_baselines_multiseed_cv,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_curve
from sklearn.model_selection import train_test_split

from ml_evacuationn import build_tree_model_risk_maps, evacuation_routes_for_maps, plot_tree_evacuation_routes

DEFAULT_FIRMS_CSV = Path(__file__).with_name("fire_archive_SV-C2_716767.csv")

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
    persistence_ca = PersistenceCAModel()
    deterministic_ca = DeterministicCAModel()
    qmodel = QuantumInspiredSpreadModel()

    baseline.fit(x_train, y_train)
    persistence_ca.fit(x_train, y_train)
    deterministic_ca.fit(x_train, y_train)
    qmodel.fit(x_train, y_train)

    out = {}
    for name, model in {
        "baseline": baseline,
        "persistence_ca": persistence_ca,
        "deterministic_ca": deterministic_ca,
        "quantum_inspired": qmodel,
    }.items():
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


def select_dynamic_safe_nodes(
    risk_map: np.ndarray,
    n_safe_nodes: int = 12,
    boundary_bias: float = 0.2,
) -> List[Tuple[int, int]]:
    """Select low-risk refuge cells dynamically from the current risk map.

    Prioritizes low-risk boundary cells while still allowing interior refuges if needed.
    """
    rows, cols = risk_map.shape
    candidates = []
    for r in range(rows):
        for c in range(cols):
            is_boundary = int(r in (0, rows - 1) or c in (0, cols - 1))
            score = float(risk_map[r, c] - boundary_bias * is_boundary)
            candidates.append(((r, c), score))

    candidates.sort(key=lambda item: item[1])

    selected: List[Tuple[int, int]] = []
    min_gap = max(1, min(rows, cols) // 8)
    for node, _ in candidates:
        if all(abs(node[0] - r2) + abs(node[1] - c2) >= min_gap for r2, c2 in selected):
            selected.append(node)
        if len(selected) >= n_safe_nodes:
            break

    return selected if selected else [(0, 0)]


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


def evacuation_routes(
    risk_map: np.ndarray,
    populations: Dict[Tuple[int, int], int],
    safe_nodes: Sequence[Tuple[int, int]],
) -> Tuple[Dict[Tuple[int, int], List[Tuple[int, int]]], Dict[Tuple[int, int], List[Tuple[int, int]]]]:
    rows, cols = risk_map.shape
    g = build_spatial_graph(rows, cols)

    baseline_paths = {}
    quantum_paths = {}
    for node in populations:
        baseline_paths[node] = baseline_route(g, node, safe_nodes)
        quantum_paths[node] = quantum_inspired_route(g, node, safe_nodes, risk_map)

    return baseline_paths, quantum_paths


def simulate_dynamic_evacuation(
    risk_forecast: Sequence[np.ndarray],
    populations: Dict[Tuple[int, int], int],
    safe_nodes: Sequence[Tuple[int, int]],
    traversal_output_dir: Path | None = None,
    max_steps: int | None = None,
) -> Dict[str, float]:
    """Day-by-day adaptive evacuation that re-routes as risk evolves.

    Continues beyond forecast horizon (using the final risk map) until all clusters
    reach safe nodes or max_steps is reached.
    """
    g = build_spatial_graph(*risk_forecast[0].shape)
    baseline_positions = populations.copy()
    quantum_positions = populations.copy()

    pre_risk = ecological_risk(populations, risk_forecast[0])
    cumulative_baseline_risk = 0.0
    cumulative_quantum_risk = 0.0

    traversal_frames = 0
    step_idx = 0
    effective_max_steps = max_steps if max_steps is not None else max(len(risk_forecast) * 6, 30)

    while step_idx < effective_max_steps:
        all_baseline_safe = all(node in safe_nodes for node in baseline_positions)
        all_quantum_safe = all(node in safe_nodes for node in quantum_positions)
        if all_baseline_safe and all_quantum_safe:
            break

        step_idx += 1
        day_risk = risk_forecast[min(step_idx - 1, len(risk_forecast) - 1)]
        cumulative_baseline_risk += ecological_risk(baseline_positions, day_risk)
        cumulative_quantum_risk += ecological_risk(quantum_positions, day_risk)

        if traversal_output_dir is not None:
            traversal_output_dir.mkdir(parents=True, exist_ok=True)
            frame_path = traversal_output_dir / f"traversal_step_{step_idx:02d}.png"
            plot_dynamic_traversal_step(
                risk_map=day_risk,
                baseline_positions=baseline_positions,
                quantum_positions=quantum_positions,
                safe_nodes=safe_nodes,
                step_idx=step_idx,
                out_path=frame_path,
            )
            traversal_frames += 1

        next_baseline: Dict[Tuple[int, int], int] = {}
        next_quantum: Dict[Tuple[int, int], int] = {}

        for node, pop in baseline_positions.items():
            if node in safe_nodes:
                next_baseline[node] = next_baseline.get(node, 0) + pop
                continue
            path = baseline_route(g, node, safe_nodes)
            moved = path[1] if len(path) > 1 else path[0]
            next_baseline[moved] = next_baseline.get(moved, 0) + pop

        for node, pop in quantum_positions.items():
            if node in safe_nodes:
                next_quantum[node] = next_quantum.get(node, 0) + pop
                continue
            path = quantum_inspired_route(g, node, safe_nodes, day_risk)
            moved = path[1] if len(path) > 1 else path[0]
            next_quantum[moved] = next_quantum.get(moved, 0) + pop

        baseline_positions = next_baseline
        quantum_positions = next_quantum

    horizon = max(step_idx, 1)
    final_risk = risk_forecast[-1]
    baseline_post = ecological_risk(baseline_positions, final_risk)
    quantum_post = ecological_risk(quantum_positions, final_risk)

    return {
        "pre_risk": pre_risk,
        "baseline_post": baseline_post,
        "quantum_post": quantum_post,
        "baseline_reduction_pct": 100 * (pre_risk - baseline_post) / (pre_risk + 1e-6),
        "quantum_reduction_pct": 100 * (pre_risk - quantum_post) / (pre_risk + 1e-6),
        "baseline_cumulative_risk": cumulative_baseline_risk,
        "quantum_cumulative_risk": cumulative_quantum_risk,
        "baseline_avg_daily_risk": cumulative_baseline_risk / horizon,
        "quantum_avg_daily_risk": cumulative_quantum_risk / horizon,
        "traversal_frames": traversal_frames,
        "evacuation_steps_executed": step_idx,
        "evacuation_max_steps": effective_max_steps,
        "baseline_reached_safe_state": all(node in safe_nodes for node in baseline_positions),
        "quantum_reached_safe_state": all(node in safe_nodes for node in quantum_positions),
    }


def generate_steps_to_safe_state(
    risk_forecast: Sequence[np.ndarray],
    populations: Dict[Tuple[int, int], int],
    safe_nodes: Sequence[Tuple[int, int]],
    max_steps: int | None = None,
) -> Dict[str, object]:
    """Generate all movement steps until populations reach a safe state.

    Safe state means all population mass is located on safe nodes.
    """
    g = build_spatial_graph(*risk_forecast[0].shape)
    baseline_positions = populations.copy()
    quantum_positions = populations.copy()

    steps: List[Dict[str, object]] = []

    def serialize_positions(position_map: Dict[Tuple[int, int], int]) -> Dict[str, int]:
        return {f"{r},{c}": int(pop) for (r, c), pop in sorted(position_map.items())}

    effective_max_steps = max_steps if max_steps is not None else max(len(risk_forecast) * 6, 30)
    for step_idx in range(1, effective_max_steps + 1):
        day_risk = risk_forecast[min(step_idx - 1, len(risk_forecast) - 1)]

        all_baseline_safe = all(node in safe_nodes for node in baseline_positions)
        all_quantum_safe = all(node in safe_nodes for node in quantum_positions)
        if all_baseline_safe and all_quantum_safe:
            break

        baseline_moves = []
        quantum_moves = []
        next_baseline: Dict[Tuple[int, int], int] = {}
        next_quantum: Dict[Tuple[int, int], int] = {}

        for node, pop in baseline_positions.items():
            if node in safe_nodes:
                moved = node
            else:
                path = baseline_route(g, node, safe_nodes)
                moved = path[1] if len(path) > 1 else path[0]
            next_baseline[moved] = next_baseline.get(moved, 0) + pop
            baseline_moves.append({"from": [node[0], node[1]], "to": [moved[0], moved[1]], "population": int(pop)})

        for node, pop in quantum_positions.items():
            if node in safe_nodes:
                moved = node
            else:
                path = quantum_inspired_route(g, node, safe_nodes, day_risk)
                moved = path[1] if len(path) > 1 else path[0]
            next_quantum[moved] = next_quantum.get(moved, 0) + pop
            quantum_moves.append({"from": [node[0], node[1]], "to": [moved[0], moved[1]], "population": int(pop)})

        baseline_positions = next_baseline
        quantum_positions = next_quantum

        steps.append(
            {
                "step": step_idx,
                "baseline_moves": baseline_moves,
                "quantum_moves": quantum_moves,
                "baseline_positions": serialize_positions(baseline_positions),
                "quantum_positions": serialize_positions(quantum_positions),
                "baseline_all_safe": all(node in safe_nodes for node in baseline_positions),
                "quantum_all_safe": all(node in safe_nodes for node in quantum_positions),
            }
        )

    return {
        "safe_nodes": [[r, c] for (r, c) in safe_nodes],
        "steps": steps,
        "baseline_reached_safe_state": all(node in safe_nodes for node in baseline_positions),
        "quantum_reached_safe_state": all(node in safe_nodes for node in quantum_positions),
        "total_steps_generated": len(steps),
        "max_steps": effective_max_steps,
    }


def simulate_evacuation(
    risk_map: np.ndarray,
    populations: Dict[Tuple[int, int], int],
    safe_nodes: Sequence[Tuple[int, int]],
) -> Dict[str, float]:
    pre_risk = ecological_risk(populations, risk_map)
    baseline_paths, quantum_paths = evacuation_routes(risk_map, populations, safe_nodes)

    moved_baseline = {}
    moved_quantum = {}

    for node, pop in populations.items():
        bpath = baseline_paths[node]
        qpath = quantum_paths[node]
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


def plot_quantum_vs_ml_roc(
    quantum_metrics: Dict[str, np.ndarray | float],
    ml_metrics: Dict[str, Dict[str, np.ndarray | float]],
    out_path: Path,
) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(
        quantum_metrics["fpr"],
        quantum_metrics["tpr"],
        label=f"quantum_inspired AUC={quantum_metrics['auc']:.3f}",
        linewidth=2.2,
    )

    for model_name, metrics in ml_metrics.items():
        plt.plot(metrics["fpr"], metrics["tpr"], label=f"{model_name} AUC={metrics['auc']:.3f}")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Quantum ROC vs ML Model ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_evacuation_routes(
    risk_map: np.ndarray,
    populations: Dict[Tuple[int, int], int],
    safe_nodes: Sequence[Tuple[int, int]],
    baseline_paths: Dict[Tuple[int, int], List[Tuple[int, int]]],
    quantum_paths: Dict[Tuple[int, int], List[Tuple[int, int]]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    for ax, title, paths in [
        (axes[0], "Baseline Evacuation Routes", baseline_paths),
        (axes[1], "Quantum-Inspired Evacuation Routes", quantum_paths),
    ]:
        ax.imshow(risk_map, cmap="hot", vmin=0, vmax=1, alpha=0.8)
        for start, path in paths.items():
            rows = [pt[0] for pt in path]
            cols = [pt[1] for pt in path]
            ax.plot(cols, rows, color="cyan", linewidth=1.6, alpha=0.9)
            ax.scatter(start[1], start[0], color="white", s=20, edgecolor="black", linewidth=0.5)

        safe_cols = [n[1] for n in safe_nodes]
        safe_rows = [n[0] for n in safe_nodes]
        ax.scatter(safe_cols, safe_rows, color="lime", marker="s", s=36, label="Safe nodes")

        ax.set_title(title)
        ax.set_xlabel("Grid column")
        ax.set_ylabel("Grid row")

    handles = [
        plt.Line2D([0], [0], color="cyan", lw=2, label="Route"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="black", lw=0, label="Population start"),
        plt.Line2D([0], [0], marker="s", color="lime", lw=0, label="Safe node"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(out_path)
    plt.close(fig)



def plot_dynamic_traversal_step(
    risk_map: np.ndarray,
    baseline_positions: Dict[Tuple[int, int], int],
    quantum_positions: Dict[Tuple[int, int], int],
    safe_nodes: Sequence[Tuple[int, int]],
    step_idx: int,
    out_path: Path,
) -> None:
    """Render a step-wise evacuation snapshot under the current fire risk map."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    for ax, title, positions in [
        (axes[0], "Baseline dynamic traversal", baseline_positions),
        (axes[1], "Quantum-inspired dynamic traversal", quantum_positions),
    ]:
        ax.imshow(risk_map, cmap="hot", vmin=0, vmax=1, alpha=0.85)

        if positions:
            total_pop = max(sum(positions.values()), 1)
            rows = [node[0] for node in positions]
            cols = [node[1] for node in positions]
            sizes = [28 + 180 * (pop / total_pop) for pop in positions.values()]
            ax.scatter(cols, rows, color="cyan", s=sizes, edgecolor="black", linewidth=0.5, alpha=0.9)

        safe_cols = [n[1] for n in safe_nodes]
        safe_rows = [n[0] for n in safe_nodes]
        ax.scatter(safe_cols, safe_rows, color="lime", marker="s", s=44, label="Safe nodes")

        ax.set_title(f"{title} - step {step_idx}")
        ax.set_xlabel("Grid column")
        ax.set_ylabel("Grid row")

    handles = [
        plt.Line2D([0], [0], marker="o", color="cyan", markeredgecolor="black", lw=0, label="Population position"),
        plt.Line2D([0], [0], marker="s", color="lime", lw=0, label="Safe node"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(out_path)
    plt.close(fig)

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
    parser.add_argument("--firms_csv", type=Path, default=DEFAULT_FIRMS_CSV)
    parser.add_argument("--output_dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--horizon_days", type=int, default=5)
    parser.add_argument("--max_evacuation_steps", type=int, default=60)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.firms_csv.exists():
        raise FileNotFoundError(
            f"FIRMS CSV not found at {args.firms_csv}. Provide --firms_csv with a valid path."
        )
    df = load_firms_csv(args.firms_csv)

    grid = GridSpec(lat_min=32.5, lat_max=42.2, lon_min=-124.5, lon_max=-114.0, rows=28, cols=28)
    tensor, days = detections_to_grid(df, grid)

    x, y = build_features_targets(tensor)
    metrics = evaluate_models(x, y)
    tree_cv_metrics = evaluate_tree_baselines_multiseed_cv(x, y, seeds=(0, 1, 2, 3, 4), n_splits=5)


    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    ml_holdout_metrics = evaluate_tree_baselines_holdout(
        x_train,
        y_train,
        x_test,
        y_test,
        seed=42
    )

    qmodel = QuantumInspiredSpreadModel()
    qmodel.fit(x, y)

    seed_day = tensor[-2]
    preds = multi_day_forecast(qmodel, seed_day, horizon=args.horizon_days)

    roc_path = args.output_dir / "roc_curve.png"
    quantum_vs_ml_roc_path = args.output_dir / "quantum_vs_ml_roc_curve.png"
    confusion_dir = args.output_dir / "confusion_matrices"
    confusion_dir.mkdir(parents=True, exist_ok=True)
    map_path = args.output_dir / "fire_spread_maps.png"
    plot_roc_curves(metrics, roc_path)
    plot_quantum_vs_ml_roc(metrics["quantum_inspired"], ml_holdout_metrics, quantum_vs_ml_roc_path)
    confusion_paths = {}
    for model_name, m in metrics.items():
        model_confusion_path = confusion_dir / f"{model_name}_confusion_matrix.png"
        plot_confusion_matrix(np.array(m["confusion_matrix"]), f"{model_name} Confusion Matrix", model_confusion_path)
        confusion_paths[model_name] = str(model_confusion_path)
    for model_name, m in ml_holdout_metrics.items():
        model_confusion_path = confusion_dir / f"{model_name}_confusion_matrix.png"
        plot_confusion_matrix(np.array(m["confusion_matrix"]), f"{model_name} Confusion Matrix", model_confusion_path)
        confusion_paths[model_name] = str(model_confusion_path)

    plot_fire_spread_maps(actual=tensor[-1], predicted_maps=preds, out_path=map_path)

    # Dynamic evacuation setup from forecasted risk and high-risk source cells.
    last_risk = preds[0]
    flat_idx = np.argsort(last_risk.reshape(-1))[-20:]
    pop_nodes = {(int(i // grid.cols), int(i % grid.cols)): 30 for i in flat_idx}
    safe_nodes = select_dynamic_safe_nodes(last_risk, n_safe_nodes=12)

    rf_spread_model = RandomForestSpreadModel(random_state=42)
    rf_spread_model.fit(x, y)
    xgb_spread_model = None
    if XGBClassifier is not None:
        xgb_spread_model = XGBoostSpreadModel(random_state=42)
        xgb_spread_model.fit(x, y)

    tree_risk_maps = build_tree_model_risk_maps(
        day_grid=seed_day,
        random_forest_model=rf_spread_model,
        xgboost_model=xgb_spread_model,
    )
    tree_routes = evacuation_routes_for_maps(tree_risk_maps, pop_nodes, safe_nodes)
    tree_evacuation_map_path = args.output_dir / "tree_evacuation_routes.png"
    plot_tree_evacuation_routes(
        risk_maps=tree_risk_maps,
        populations=pop_nodes,
        safe_nodes=safe_nodes,
        routes_by_model=tree_routes,
        out_path=str(tree_evacuation_map_path),
    )

    baseline_paths, quantum_paths = evacuation_routes(last_risk, pop_nodes, safe_nodes)
    traversal_dir = args.output_dir / "dynamic_traversal"
    evac_report = simulate_dynamic_evacuation(
        preds,
        pop_nodes,
        safe_nodes,
        traversal_output_dir=traversal_dir,
        max_steps=args.max_evacuation_steps,
    )
    safe_state_steps = generate_steps_to_safe_state(
        preds,
        pop_nodes,
        safe_nodes,
        max_steps=args.max_evacuation_steps,
    )
    safe_state_steps_path = args.output_dir / "safe_state_steps.json"
    pd.Series(safe_state_steps).to_json(safe_state_steps_path, indent=2)
    evacuation_map_path = args.output_dir / "evacuation_routes.png"
    plot_evacuation_routes(
        risk_map=last_risk,
        populations=pop_nodes,
        safe_nodes=safe_nodes,
        baseline_paths=baseline_paths,
        quantum_paths=quantum_paths,
        out_path=evacuation_map_path,
    )

    summary = {
        "n_days": len(days),
        "n_detections": len(df),
        "baseline_auc": metrics["baseline"]["auc"],
        "baseline_precision": metrics["baseline"]["precision"],
        "baseline_recall": metrics["baseline"]["recall"],
        "baseline_f1": metrics["baseline"]["f1"],
        "baseline_loss": metrics["baseline"]["loss"],
        "baseline_confusion_matrix": metrics["baseline"]["confusion_matrix"].tolist(),
        "persistence_ca_auc": metrics["persistence_ca"]["auc"],
        "persistence_ca_precision": metrics["persistence_ca"]["precision"],
        "persistence_ca_recall": metrics["persistence_ca"]["recall"],
        "persistence_ca_f1": metrics["persistence_ca"]["f1"],
        "persistence_ca_loss": metrics["persistence_ca"]["loss"],
        "persistence_ca_confusion_matrix": metrics["persistence_ca"]["confusion_matrix"].tolist(),
        "deterministic_ca_auc": metrics["deterministic_ca"]["auc"],
        "deterministic_ca_precision": metrics["deterministic_ca"]["precision"],
        "deterministic_ca_recall": metrics["deterministic_ca"]["recall"],
        "deterministic_ca_f1": metrics["deterministic_ca"]["f1"],
        "deterministic_ca_loss": metrics["deterministic_ca"]["loss"],
        "deterministic_ca_confusion_matrix": metrics["deterministic_ca"]["confusion_matrix"].tolist(),
        "quantum_auc": metrics["quantum_inspired"]["auc"],
        "quantum_precision": metrics["quantum_inspired"]["precision"],
        "quantum_recall": metrics["quantum_inspired"]["recall"],
        "quantum_f1": metrics["quantum_inspired"]["f1"],
        "quantum_loss": metrics["quantum_inspired"]["loss"],
        "quantum_confusion_matrix": metrics["quantum_inspired"]["confusion_matrix"].tolist(),
        "random_forest_cv": tree_cv_metrics["random_forest"],
        "random_forest_holdout_auc": ml_holdout_metrics["random_forest"]["auc"],
        "random_forest_holdout_confusion_matrix": ml_holdout_metrics["random_forest"]["confusion_matrix"].tolist(),
        "random_forest_auc_mean_pm_std": f"{tree_cv_metrics['random_forest']['auc_mean']:.4f} ± {tree_cv_metrics['random_forest']['auc_std']:.4f}",
        **({
            "xgboost_cv": tree_cv_metrics["xgboost"],
            "xgboost_holdout_auc": ml_holdout_metrics["xgboost"]["auc"],
            "xgboost_holdout_confusion_matrix": ml_holdout_metrics["xgboost"]["confusion_matrix"].tolist(),
            "xgboost_auc_mean_pm_std": f"{tree_cv_metrics['xgboost']['auc_mean']:.4f} ± {tree_cv_metrics['xgboost']['auc_std']:.4f}",
        } if "xgboost" in tree_cv_metrics else {"xgboost_cv": "xgboost_not_installed"}),
        **evac_report,
        "roc_curve": str(roc_path),
        "quantum_vs_ml_roc_curve": str(quantum_vs_ml_roc_path),
        "confusion_matrices": confusion_paths,
        "fire_spread_map": str(map_path),
        "evacuation_route_map": str(evacuation_map_path),
        "tree_evacuation_route_map": str(tree_evacuation_map_path),
        "tree_risk_maps": {name: risk_map.tolist() for name, risk_map in tree_risk_maps.items()},
        "dynamic_traversal_dir": str(traversal_dir),
        "safe_state_steps": str(safe_state_steps_path),
        "safe_state_total_steps": safe_state_steps["total_steps_generated"],
        "safe_state_baseline_reached": safe_state_steps["baseline_reached_safe_state"],
        "safe_state_quantum_reached": safe_state_steps["quantum_reached_safe_state"],
    }


    summary_path = args.output_dir / "summary_metrics.json"
    pd.Series(summary).to_json(summary_path, indent=2)

    print("Run complete. Outputs:")
    print(f"- {roc_path}")
    print(f"- {quantum_vs_ml_roc_path}")
    print(f"- {map_path}")
    print("- confusion matrices:")
    for name, path in confusion_paths.items():
        print(f"  - {name}: {path}")
    print(f"- {evacuation_map_path}")
    print(f"- {tree_evacuation_map_path}")
    print(f"- {traversal_dir} ({evac_report['traversal_frames']} frames)")
    print(f"- {safe_state_steps_path} ({safe_state_steps['total_steps_generated']} steps)")
    print(f"- RandomForest 5-seed CV AUC: {tree_cv_metrics['random_forest']['auc_mean']:.4f} ± {tree_cv_metrics['random_forest']['auc_std']:.4f}")
    if "xgboost" in tree_cv_metrics:
        print(f"- XGBoost 5-seed CV AUC: {tree_cv_metrics['xgboost']['auc_mean']:.4f} ± {tree_cv_metrics['xgboost']['auc_std']:.4f}")
    else:
        print("- XGBoost 5-seed CV AUC: unavailable (xgboost not installed)")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
