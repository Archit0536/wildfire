import numpy as np

from ml_evacuationn import build_tree_model_risk_maps, evacuation_routes_for_maps
from tree_baselines import RandomForestSpreadModel, XGBClassifier, XGBoostSpreadModel, evaluate_tree_baselines_multiseed_cv
from wildfire_dss import (
    GridSpec,
    build_features_targets,
    detections_to_grid,
    evaluate_models,
    generate_synthetic_firms_data,
)


def test_pipeline_shapes_and_metrics():
    df = generate_synthetic_firms_data(n_days=8, points_per_day=100)
    grid = GridSpec(lat_min=32.5, lat_max=42.2, lon_min=-124.5, lon_max=-114.0, rows=10, cols=10)
    tensor, days = detections_to_grid(df, grid)

    assert tensor.shape == (len(days), 10, 10)

    x, y = build_features_targets(tensor)
    assert x.shape[1] == 2
    assert set(np.unique(y)).issubset({0, 1})

    metrics = evaluate_models(x, y)
    assert "baseline" in metrics
    assert "persistence_ca" in metrics
    assert "deterministic_ca" in metrics
    assert "quantum_inspired" in metrics
    assert 0 <= metrics["baseline"]["auc"] <= 1
    assert 0 <= metrics["persistence_ca"]["auc"] <= 1
    assert 0 <= metrics["deterministic_ca"]["auc"] <= 1
    assert 0 <= metrics["quantum_inspired"]["auc"] <= 1


def test_tree_baselines_multiseed_cv_reports_mean_std():
    df = generate_synthetic_firms_data(n_days=10, points_per_day=80)
    grid = GridSpec(lat_min=32.5, lat_max=42.2, lon_min=-124.5, lon_max=-114.0, rows=8, cols=8)
    tensor, _ = detections_to_grid(df, grid)
    x, y = build_features_targets(tensor)

    cv_metrics = evaluate_tree_baselines_multiseed_cv(x, y, seeds=(0, 1, 2, 3, 4), n_splits=5)

    assert "random_forest" in cv_metrics
    assert 0 <= cv_metrics["random_forest"]["auc_mean"] <= 1
    assert cv_metrics["random_forest"]["auc_std"] >= 0

    if "xgboost" in cv_metrics:
        assert 0 <= cv_metrics["xgboost"]["auc_mean"] <= 1
        assert cv_metrics["xgboost"]["auc_std"] >= 0


def test_tree_evacuation_mapping_module():
    df = generate_synthetic_firms_data(n_days=8, points_per_day=100)
    grid = GridSpec(lat_min=32.5, lat_max=42.2, lon_min=-124.5, lon_max=-114.0, rows=8, cols=8)
    tensor, _ = detections_to_grid(df, grid)
    x, y = build_features_targets(tensor)

    rf_model = RandomForestSpreadModel(random_state=42)
    rf_model.fit(x, y)

    xgb_model = None
    if XGBClassifier is not None:
        xgb_model = XGBoostSpreadModel(random_state=42)
        xgb_model.fit(x, y)

    risk_maps = build_tree_model_risk_maps(tensor[-2], rf_model, xgb_model)
    assert "random_forest" in risk_maps
    assert risk_maps["random_forest"].shape == (8, 8)

    pop_nodes = {(2, 2): 20, (5, 5): 30}
    safe_nodes = [(0, 0), (7, 7)]
    routes = evacuation_routes_for_maps(risk_maps, pop_nodes, safe_nodes)

    assert "random_forest" in routes
    for model_routes in routes.values():
        for start, path in model_routes.items():
            assert path[0] == start
            assert path[-1] in safe_nodes
