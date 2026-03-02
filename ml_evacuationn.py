"""Evacuation mapping utilities backed by tree-based wildfire spread models."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def neighbor_sum(arr_2d: np.ndarray) -> np.ndarray:
    padded = np.pad(arr_2d, 1, mode="constant")
    out = np.zeros_like(arr_2d)
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue
            out += padded[i : i + arr_2d.shape[0], j : j + arr_2d.shape[1]]
    return out


def _build_day_features(day_grid: np.ndarray) -> np.ndarray:
    cur_norm = day_grid / (day_grid.max() + 1e-6)
    neigh = neighbor_sum(day_grid)
    neigh_norm = neigh / (neigh.max() + 1e-6)
    return np.stack([cur_norm, neigh_norm], axis=-1).reshape(-1, 2)


def predict_next_day_risk(model: object, day_grid: np.ndarray) -> np.ndarray:
    x = _build_day_features(day_grid)
    return model.predict_proba(x).reshape(day_grid.shape)


def build_tree_model_risk_maps(
    day_grid: np.ndarray,
    random_forest_model: object,
    xgboost_model: object | None = None,
) -> Dict[str, np.ndarray]:
    """Create next-day risk maps for tree-based models used for evacuation planning."""
    maps = {"random_forest": predict_next_day_risk(random_forest_model, day_grid)}
    if xgboost_model is not None:
        maps["xgboost"] = predict_next_day_risk(xgboost_model, day_grid)
    return maps


def build_spatial_graph(rows: int, cols: int) -> nx.Graph:
    g = nx.grid_2d_graph(rows, cols)
    for u, v in g.edges:
        g.edges[u, v]["distance"] = 1.0
    return g


def risk_aware_route(
    g: nx.Graph,
    start: Tuple[int, int],
    safe_nodes: Sequence[Tuple[int, int]],
    risk_map: np.ndarray,
) -> List[Tuple[int, int]]:
    g2 = g.copy()
    for u, v in g2.edges:
        risk_term = 0.8 * (risk_map[u] + risk_map[v])
        g2.edges[u, v]["risk_weight"] = 1.0 + risk_term

    best_path = None
    best_score = float("inf")
    for safe in safe_nodes:
        try:
            path = nx.shortest_path(g2, start, safe, weight="risk_weight")
            score = float(sum(risk_map[r, c] for r, c in path))
            if score < best_score:
                best_score = score
                best_path = path
        except nx.NetworkXNoPath:
            continue

    return best_path if best_path else [start]


def evacuation_routes_for_maps(
    risk_maps: Dict[str, np.ndarray],
    populations: Dict[Tuple[int, int], int],
    safe_nodes: Sequence[Tuple[int, int]],
) -> Dict[str, Dict[Tuple[int, int], List[Tuple[int, int]]]]:
    """Compute evacuation routes for each supplied model risk map."""
    rows, cols = next(iter(risk_maps.values())).shape
    g = build_spatial_graph(rows, cols)

    all_routes: Dict[str, Dict[Tuple[int, int], List[Tuple[int, int]]]] = {}
    for model_name, risk_map in risk_maps.items():
        model_routes: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for node in populations:
            model_routes[node] = risk_aware_route(g, node, safe_nodes, risk_map)
        all_routes[model_name] = model_routes

    return all_routes


def plot_tree_evacuation_routes(
    risk_maps: Dict[str, np.ndarray],
    populations: Dict[Tuple[int, int], int],
    safe_nodes: Sequence[Tuple[int, int]],
    routes_by_model: Dict[str, Dict[Tuple[int, int], List[Tuple[int, int]]]],
    out_path: str,
) -> None:
    """Render evacuation-route overlays for Random Forest and optional XGBoost maps."""
    model_names = list(risk_maps)
    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5), sharex=True, sharey=True)
    if len(model_names) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        risk_map = risk_maps[model_name]
        ax.imshow(risk_map, cmap="hot", vmin=0, vmax=1, alpha=0.82)

        for start in populations:
            path = routes_by_model[model_name][start]
            rows = [pt[0] for pt in path]
            cols = [pt[1] for pt in path]
            ax.plot(cols, rows, color="cyan", linewidth=1.5, alpha=0.9)
            ax.scatter(start[1], start[0], color="white", s=18, edgecolor="black", linewidth=0.5)

        safe_cols = [n[1] for n in safe_nodes]
        safe_rows = [n[0] for n in safe_nodes]
        ax.scatter(safe_cols, safe_rows, color="lime", marker="s", s=36)

        ax.set_title(f"{model_name.replace('_', ' ').title()} evacuation")
        ax.set_xlabel("Grid column")
        ax.set_ylabel("Grid row")

    fig.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
