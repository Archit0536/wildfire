"""Animation utilities for fire spread and evacuation overlap."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


GridPoint = Tuple[int, int]


def _deserialize_positions(positions: Dict[str, int]) -> Dict[GridPoint, int]:
    out: Dict[GridPoint, int] = {}
    for key, pop in positions.items():
        r_str, c_str = key.split(",")
        out[(int(r_str), int(c_str))] = int(pop)
    return out


def _to_sorted_lists(points: Set[GridPoint]) -> List[List[int]]:
    return [[r, c] for (r, c) in sorted(points)]


def build_timestamp_accessible_points(
    risk_forecast: Sequence[np.ndarray],
    safe_state_steps: Dict[str, object],
) -> List[Dict[str, object]]:
    """Build per-step evacuation-accessible points with overlap markers."""
    steps = safe_state_steps.get("steps", [])
    safe_nodes = {tuple(node) for node in safe_state_steps.get("safe_nodes", [])}

    frames: List[Dict[str, object]] = []
    for idx, step in enumerate(steps, start=1):
        baseline_positions = _deserialize_positions(step["baseline_positions"])
        quantum_positions = _deserialize_positions(step["quantum_positions"])

        baseline_points = set(baseline_positions.keys())
        quantum_points = set(quantum_positions.keys())
        overlap_points = baseline_points & quantum_points

        frame_risk = risk_forecast[min(idx - 1, len(risk_forecast) - 1)]

        frames.append(
            {
                "timestamp_step": idx,
                "baseline_accessible_points": _to_sorted_lists(baseline_points),
                "quantum_accessible_points": _to_sorted_lists(quantum_points),
                "overlap_points": _to_sorted_lists(overlap_points),
                "safe_nodes": _to_sorted_lists(safe_nodes),
                "baseline_population_total": int(sum(baseline_positions.values())),
                "quantum_population_total": int(sum(quantum_positions.values())),
                "mean_risk": float(np.mean(frame_risk)),
                "max_risk": float(np.max(frame_risk)),
            }
        )

    return frames


def render_fire_evacuation_gif(
    risk_forecast: Sequence[np.ndarray],
    safe_state_steps: Dict[str, object],
    out_path: Path,
    duration_ms: int = 500,
) -> None:
    """Render GIF showing fire spread with baseline/quantum evacuation overlap."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    steps = safe_state_steps.get("steps", [])
    safe_nodes = [tuple(node) for node in safe_state_steps.get("safe_nodes", [])]

    frame_paths: List[Path] = []
    tmp_dir = out_path.parent / "_gif_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for idx, step in enumerate(steps, start=1):
        risk_map = risk_forecast[min(idx - 1, len(risk_forecast) - 1)]
        baseline_positions = _deserialize_positions(step["baseline_positions"])
        quantum_positions = _deserialize_positions(step["quantum_positions"])

        baseline_points = set(baseline_positions.keys())
        quantum_points = set(quantum_positions.keys())
        overlap_points = baseline_points & quantum_points

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(risk_map, cmap="hot", vmin=0, vmax=1, alpha=0.85)

        if baseline_points:
            br = [pt[0] for pt in baseline_points]
            bc = [pt[1] for pt in baseline_points]
            ax.scatter(bc, br, c="cyan", s=26, label="A* evacuation", edgecolor="black", linewidth=0.4)

        if quantum_points:
            qr = [pt[0] for pt in quantum_points]
            qc = [pt[1] for pt in quantum_points]
            ax.scatter(qc, qr, c="dodgerblue", s=24, label="Quantum evacuation", alpha=0.85)

        if overlap_points:
            orows = [pt[0] for pt in overlap_points]
            ocols = [pt[1] for pt in overlap_points]
            ax.scatter(ocols, orows, c="magenta", s=38, marker="X", label="Overlap")

        if safe_nodes:
            sr = [pt[0] for pt in safe_nodes]
            sc = [pt[1] for pt in safe_nodes]
            ax.scatter(sc, sr, c="lime", marker="s", s=34, label="Safe nodes")

        ax.set_title(f"Fire spread + evacuation overlap (step {idx})")
        ax.set_xlabel("Grid column")
        ax.set_ylabel("Grid row")
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()

        frame_path = tmp_dir / f"frame_{idx:03d}.png"
        plt.savefig(frame_path)
        plt.close(fig)
        frame_paths.append(frame_path)

    if not frame_paths:
        return

    first = Image.open(frame_paths[0])
    appended = [Image.open(path) for path in frame_paths[1:]]
    first.save(
        out_path,
        save_all=True,
        append_images=appended,
        duration=duration_ms,
        loop=0,
    )

    first.close()
    for image in appended:
        image.close()

    for path in frame_paths:
        path.unlink(missing_ok=True)
    tmp_dir.rmdir()
