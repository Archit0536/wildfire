import numpy as np

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
    assert "quantum_inspired" in metrics
    assert 0 <= metrics["baseline"]["auc"] <= 1
    assert 0 <= metrics["quantum_inspired"]["auc"] <= 1
