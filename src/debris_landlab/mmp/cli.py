from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from debris_landlab.mmp.config import load_mmp_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the modular multi-model landslide probability pipeline."
    )
    parser.add_argument(
        "--config",
        default="config/mmp_landslide.yaml",
        help="Path to the base MMP YAML config.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Optional YAML override. Pass multiple times to layer scenarios.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path for a compact run summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_mmp_config(args.config, overrides=args.override)

    print(f"Project: {config.project_name}")
    print(f"Scenario directory: {config.scenario_dir}")
    print(f"Forcing dates: {config.forcing.start_date} to {config.forcing.end_date}")
    print(f"Forcing CSV: {config.forcing.forcing_csv}")
    print(f"Landslide iterations: {config.landslide.number_of_iterations}")

    from debris_landlab.mmp.pipeline import run_pipeline

    result = run_pipeline(config)
    grid = result.grid
    ls_prob = grid.at_node["landslide__probability_of_failure"]
    summary = {
        "project": config.project_name,
        "forcing_days": int(len(result.forcing.forcing_df)),
        "number_of_nodes": int(grid.number_of_nodes),
        "number_of_core_nodes": int(grid.number_of_core_nodes),
        "number_of_landslide_iterations": int(config.landslide.number_of_iterations),
        "forcing_csv": str(result.forcing.forcing_csv),
        "forcing_manifest": str(result.forcing.manifest_path) if result.forcing.manifest_path else None,
        "ls_probability_min": float(np.nanmin(ls_prob[grid.core_nodes])),
        "ls_probability_mean": float(np.nanmean(ls_prob[grid.core_nodes])),
        "ls_probability_max": float(np.nanmax(ls_prob[grid.core_nodes])),
        "exported_paths": [str(path) for path in result.exported_paths],
    }

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved summary: {summary_path}")

    print(
        "Pipeline complete for "
        f"{summary['forcing_days']} forcing days, "
        f"{summary['number_of_core_nodes']} core nodes, and "
        f"{summary['number_of_landslide_iterations']} landslide iterations."
    )
    print(
        "Landslide probability core-node range: "
        f"{summary['ls_probability_min']:.4f} to {summary['ls_probability_max']:.4f} "
        f"(mean {summary['ls_probability_mean']:.4f})."
    )


if __name__ == "__main__":
    main()
