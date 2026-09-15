"""Check both horizons against public vendor calls without scoring new futures."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probe_differentiable_imputation import parameter_digest  # noqa: E402
from r6_runtime import forecast_spec, make_forecaster  # noqa: E402

from tsfm_fais.routing.preforecast_replay import assemble_selected_context  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "prepared-root",
        "input-audit-root",
        "legacy-bundle",
        "previous-forecasts",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve the completed horizon interface check")
    prep_path = args.prepared_root / "manifest.json"
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    audit = json.loads((args.input_audit_root / "manifest.json").read_text(encoding="utf-8"))
    if (
        audit["status"] != "completed"
        or audit["prepared_sha256"] != file_sha256(prep_path)
        or audit["failed_imputer_fits"]
    ):
        raise ValueError("complete the input audit and resolve any failed fits first")
    selected = []
    for dataset in sorted({row["dataset_id"] for row in prep["episodes"]}):
        eligible = [
            row
            for row in prep["episodes"]
            if row["dataset_id"] == dataset and row["realized_context_missing_fraction"] > 0
        ]
        if not eligible:
            raise ValueError("each source requires a missing-input interface probe")
        selected.append(eligible[0])
    torch.set_num_threads(1)
    records, models = [], []
    for model_id in ("chronos2", "timesfm2p5"):
        runner, adapter, backbone, digest, joint = make_forecaster(
            model_id, args.legacy_bundle, args.previous_forecasts
        )
        backend = adapter._ensure_backend()
        for row in selected:
            path = args.prepared_root / row["path"]
            if file_sha256(path) != row["sha256"]:
                raise ValueError("a probe context changed")
            with np.load(path, allow_pickle=False) as saved:
                context, candidates, actions = (
                    saved["context"],
                    saved["candidate_values"],
                    saved["candidate_ids"].tolist(),
                )
            guarded = assemble_selected_context(
                context,
                candidates,
                actions,
                ["guarded_direct"] if joint else ["guarded_direct"] * 2,
                [0, 1],
                joint=joint,
            )
            for horizon in (96, 192):
                spec = forecast_spec(model_id, horizon, joint)
                for kind, values in (
                    ("finite_locf", candidates[actions.index("locf")]),
                    ("native_guarded", guarded),
                ):
                    predicted = runner.predict_missing(values[None], spec).point[0]
                    if joint:
                        raw = backend.predict_quantiles(
                            inputs=[{"target": values.astype(np.float32).T}],
                            prediction_length=horizon,
                            batch_size=8,
                            quantile_levels=list(spec.quantile_levels),
                            predict_batches_jointly=False,
                        )
                        quantiles = raw[0][0]
                        quantiles = (
                            quantiles.detach().cpu().numpy()
                            if hasattr(quantiles, "detach")
                            else np.asarray(quantiles)
                        )
                        if quantiles.shape != (values.shape[1], horizon, 3):
                            raise ValueError("vendor Chronos output axes changed")
                        direct = quantiles[:2, :, 1].T
                    else:
                        raw = backend.forecast(
                            horizon=horizon,
                            inputs=[values[:, target].astype(np.float32) for target in (0, 1)],
                        )
                        point = raw[0]
                        point = (
                            point.detach().cpu().numpy()
                            if hasattr(point, "detach")
                            else np.asarray(point)
                        )
                        if point.shape != (2, horizon):
                            raise ValueError("vendor TimesFM output axes changed")
                        direct = point.T
                    if (
                        predicted.shape != (horizon, 2)
                        or not np.isfinite(predicted).all()
                        or not np.isfinite(direct).all()
                    ):
                        raise ValueError(
                            "the requested horizon was truncated or produced invalid values"
                        )
                    np.testing.assert_allclose(predicted, direct, rtol=1e-6, atol=1e-6)
                    records.append(
                        {
                            "model_id": model_id,
                            "dataset_id": row["dataset_id"],
                            "episode_id": row["episode_id"],
                            "horizon": horizon,
                            "input_kind": kind,
                            "maximum_vendor_difference": float(abs(predicted - direct).max()),
                        }
                    )
        if parameter_digest(backbone) != digest:
            raise ValueError("the frozen predictor changed during interface checks")
        models.append({"model_id": model_id, "parameter_sha256": digest})
        del runner, adapter, backbone, backend
        torch.cuda.empty_cache()
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "prepared_sha256": file_sha256(prep_path),
            "runtime_sha256": file_sha256(ROOT / "scripts/r6_runtime.py"),
            "checks": records,
            "models": models,
            "future_arrays_read": False,
            "limits": "vendor output and horizon interface check only; no confirmation prediction errors were examined",
        },
    )


if __name__ == "__main__":
    main()
