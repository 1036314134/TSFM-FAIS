"""Evaluate TimesFM's documented preprocessing of incomplete inputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tsfm_fais.contracts import ForecastSpec  # noqa: E402
from tsfm_fais.forecasting.adapters.timesfm import TimesFM2p5Adapter  # noqa: E402
from tsfm_fais.forecasting.metrics import macro_mase  # noqa: E402
from tsfm_fais.forecasting.registry import default_forecast_registry  # noqa: E402
from tsfm_fais.forecasting.runner import ForecastRunner  # noqa: E402
from tsfm_fais.routing.utility import family_macro  # noqa: E402
from tsfm_fais.utility_experiment import file_sha256, load_utility_config  # noqa: E402


class TimesFMVendorMissingAdapter(TimesFM2p5Adapter):
    """The vendor removes leading NaNs and interpolates remaining gaps."""

    capabilities = replace(TimesFM2p5Adapter.capabilities, supports_missing_context=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_utility_config(args.config)
    import torch
    from timesfm.timesfm_2p5 import timesfm_2p5_base

    torch.set_num_threads(4)
    root = config.output_root
    manifest = json.loads((root / "episodes_manifest.json").read_text(encoding="utf-8"))
    output = root / "timesfm-vendor-missing-v001"
    output.mkdir(parents=True, exist_ok=True)
    prediction_dir = output / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    identity = {
        "episode_manifest_sha256": file_sha256(root / "episodes_manifest.json"),
        "script_sha256": file_sha256(Path(__file__)),
        "vendor_preprocessing_sha256": file_sha256(Path(timesfm_2p5_base.__file__)),
        "checkpoint": str(config.forecaster_artifacts["timesfm2p5"]),
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("vendor baseline identity changed; preserve the earlier run")
    identity_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    adapter = TimesFMVendorMissingAdapter(
        model_name=str(config.forecaster_artifacts["timesfm2p5"]),
        device=config.device,
        batch_size=config.forecast_batch_size,
    )
    runner = ForecastRunner(default_forecast_registry(), {"timesfm2p5": adapter})
    spec = ForecastSpec(
        "timesfm2p5",
        "independent_univariate",
        config.horizon,
        context_length=config.context_length,
        target_indices=config.target_indices,
    )
    groups = defaultdict(list)
    for record in manifest["episodes"]:
        groups[record["dataset_id"]].append(record)
    rows, prediction_records = [], []
    started = perf_counter()
    for dataset_id, records in groups.items():
        for start in range(0, len(records), config.forecast_batch_size):
            batch = records[start : start + config.forecast_batch_size]
            pending, contexts = [], []
            for record in batch:
                path = root / record["path"]
                if file_sha256(path) != record["sha256"]:
                    raise ValueError("candidate episode changed")
                if not (prediction_dir / path.name).exists():
                    with np.load(path, allow_pickle=False) as episode:
                        contexts.append(episode["context"])
                    pending.append(record)
            if pending:
                result = runner.predict_missing(np.stack(contexts), spec)
                for index, record in enumerate(pending):
                    destination = prediction_dir / Path(record["path"]).name
                    np.savez_compressed(
                        destination,
                        point=result.point[index],
                        quantiles=result.quantiles[index],
                        candidate_sha256=np.asarray(record["sha256"]),
                    )
            for record in batch:
                episode_path = root / record["path"]
                forecast_path = prediction_dir / episode_path.name
                with (
                    np.load(episode_path, allow_pickle=False) as episode,
                    np.load(forecast_path, allow_pickle=False) as forecast,
                ):
                    if str(forecast["candidate_sha256"]) != record["sha256"]:
                        raise ValueError("vendor forecast is bound to another episode")
                    targets = list(config.target_indices)
                    truth = episode["future"][:, targets]
                    loss = macro_mase(
                        forecast["point"][None],
                        truth[None],
                        episode["mase_scales"][targets],
                    )[0]
                rows.append(
                    {
                        key: record[key]
                        for key in ("episode_id", "origin_id", "dataset_id", "family_id", "split")
                    }
                    | {"method": "timesfm_vendor_missing", "loss": float(loss)}
                )
                prediction_records.append(
                    {"episode_id": record["episode_id"], "sha256": file_sha256(forecast_path)}
                )
        print(
            json.dumps(
                {
                    "stage": "timesfm_vendor_missing",
                    "dataset": dataset_id,
                    "completed": len(rows),
                    "total": len(manifest["episodes"]),
                }
            ),
            flush=True,
        )
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "rows.parquet", index=False)
    score = family_macro(frame[frame.split == "validation"])
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "identity": identity,
                "evidence_role": "development",
                "validation_family_macro_mase": score,
                "elapsed_seconds": perf_counter() - started,
                "resources_this_execution": runner.resource_metrics(),
                "episodes": prediction_records,
                "semantics": "vendor interface removes leading missing values, interpolates remaining NaNs, and pads with a padding mask; this does not establish a learned missing-value mechanism",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": "completed", "validation_family_macro_mase": score}), flush=True)


if __name__ == "__main__":
    main()
