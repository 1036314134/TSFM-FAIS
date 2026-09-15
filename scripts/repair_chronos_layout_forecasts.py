"""Recompute only Chronos-2 tasks affected by ambiguous legacy axis inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import monotonic

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsfm_fais.contracts import ForecastSpec  # noqa: E402
from tsfm_fais.forecasting import ForecastRunner, default_forecast_registry  # noqa: E402
from tsfm_fais.forecasting.adapters import Chronos2Adapter  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def request_from_json(payload):
    values = dict(payload)
    for key in ("target_indices", "quantile_levels"):
        if values.get(key) is not None:
            values[key] = tuple(values[key])
    return ForecastSpec(**values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    root, output = args.source_root.resolve(), args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads((root / "episodes_manifest.json").read_text(encoding="utf-8"))
    prior = json.loads((root / "chronos2/forecast_identity.json").read_text(encoding="utf-8"))
    spec = request_from_json(prior["forecast_spec"])
    identity = {
        "source_episode_manifest_sha256": file_sha256(root / "episodes_manifest.json"),
        "adapter_sha256": file_sha256(ROOT / "src/tsfm_fais/forecasting/adapters/chronos.py"),
        "script_sha256": file_sha256(Path(__file__)),
        "checkpoint": prior["checkpoint"],
        "device": "cuda",
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("repair identity changed; preserve earlier output")
    _write_json(identity_path, identity)
    dimensions = {}
    selected = []
    for record in source["episodes"]:
        key = (record["dataset_id"], record["item_id"])
        if key not in dimensions:
            with np.load(root / record["path"], allow_pickle=False) as e:
                dimensions[key] = len(e["mase_scales"])
        if dimensions[key] in (len(spec.quantile_levels), spec.horizon):
            selected.append(record)
    adapter = Chronos2Adapter(model_name=prior["checkpoint"], device="cuda", batch_size=32)
    runner = ForecastRunner(default_forecast_registry(), {"chronos2": adapter})
    records = []
    started = monotonic()
    for index, record in enumerate(selected):
        path = root / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("candidate cache changed")
        destination = output / "predictions" / path.name
        with np.load(path, allow_pickle=False) as episode:
            if not destination.exists():
                complete = runner.predict(
                    np.concatenate([episode["candidate_values"], episode["clean_context"][None]]),
                    spec,
                )
                direct = runner.predict_missing(episode["context"][None], spec)
                _save_npz(
                    destination,
                    point=np.concatenate([complete.point[:-1], direct.point]),
                    quantiles=np.concatenate([complete.quantiles[:-1], direct.quantiles]),
                    clean_point=complete.point[-1],
                    candidate_sha256=np.asarray(record["sha256"]),
                )
            with np.load(destination, allow_pickle=False) as prediction:
                if str(prediction["candidate_sha256"]) != record["sha256"]:
                    raise ValueError("corrected forecast belongs to another episode")
        records.append(
            {
                "episode_id": record["episode_id"],
                "path": str(destination.relative_to(output)),
                "sha256": file_sha256(destination),
            }
        )
        if (index + 1) % 50 == 0 or index + 1 == len(selected):
            progress = {
                "completed": index + 1,
                "total": len(selected),
                "elapsed_seconds": monotonic() - started,
            }
            _write_json(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "evidence_role": "development",
            "identity": identity,
            "episodes": records,
            "affected_datasets": sorted({record["dataset_id"] for record in selected}),
            "resources_this_execution": runner.resource_metrics(),
            "elapsed_seconds": monotonic() - started,
        },
    )


if __name__ == "__main__":
    main()
