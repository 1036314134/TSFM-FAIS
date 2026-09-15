"""Verify frozen decisions and incomplete-future scores on the registered cohort."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from replay_preforecast_student import extend_forecast_features, rank_context_actions  # noqa: E402
from run_native_confirmation import chosen_prediction, hierarchical_metrics  # noqa: E402

from tsfm_fais.forecasting.observed_accuracy import observed_future_errors  # noqa: E402
from tsfm_fais.routing.forecast_response import forecast_response_inputs  # noqa: E402
from tsfm_fais.routing.preforecast_replay import candidate_feature_frame  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-root", "source-bundle", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root, output = args.input_root.resolve(), args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed native confirmation audits")
    output.mkdir(parents=True, exist_ok=True)
    prepared = json.loads((root / "prepared/manifest.json").read_text(encoding="utf-8"))
    bundle = json.loads((args.source_bundle / "manifest.json").read_text(encoding="utf-8"))
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads((root / "prepared/standardizers.json").read_text(encoding="utf-8"))
    }
    if file_sha256(root / "prepared/standardizers.json") != prepared["standardizers_sha256"]:
        raise ValueError("confirmation standardizers changed")
    models = {}
    for item in bundle["models"]:
        record_path = args.source_bundle / item["path"]
        if file_sha256(record_path) != item["sha256"]:
            raise ValueError("a frozen selector record changed")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if file_sha256(args.source_bundle / record["model_path"]) != record["model_sha256"]:
            raise ValueError("a frozen selector changed")
        models[(record["model_id"], record["objective"])] = joblib.load(
            args.source_bundle / record["model_path"]
        )
    sources, verified, max_metric, all_recomputed = {}, 0, 0.0, []
    for model in ("chronos2", "timesfm2p5"):
        directory = root / model
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest["status"] != "completed"
            or not manifest["parameters_unchanged"]
            or len(manifest["predictions"]) != 373
        ):
            raise ValueError("complete the full frozen confirmation first")
        if manifest["identity"]["source_bundle_sha256"] != file_sha256(
            args.source_bundle / "manifest.json"
        ) or manifest["identity"]["prepared_manifest_sha256"] != file_sha256(
            root / "prepared/manifest.json"
        ):
            raise ValueError("a confirmation result has different provenance")
        sources[model] = file_sha256(directory / "manifest.json")
        files = {record["episode_id"]: record for record in manifest["predictions"]}
        reported = pd.read_parquet(directory / "episode_results.parquet")
        recomputed = []
        for index, episode in enumerate(prepared["episodes"]):
            source = root / "prepared" / episode["path"]
            record = files[episode["episode_id"]]
            path = directory / record["path"]
            if file_sha256(source) != episode["sha256"] or file_sha256(path) != record["sha256"]:
                raise ValueError("a confirmation input or prediction changed")
            with np.load(source, allow_pickle=False) as saved:
                context, candidates, future, observed = (
                    saved["context"],
                    saved["candidate_values"],
                    saved["future"],
                    saved["future_observed"],
                )
                actions, coverage = saved["candidate_ids"].tolist(), saved["native_coverage"]
            with np.load(path, allow_pickle=False) as saved:
                points, methods = saved["point_z"], saved["methods"].tolist()
                diagnostics = json.loads(str(saved["metadata"]))
                if (
                    str(saved["identity_sha256"]) != manifest["identity_sha256"]
                    or str(saved["parameter_sha256"]) != manifest["parameter_sha256"]
                ):
                    raise ValueError("a prediction cache changed identity")
            scaler = scalers[(episode["dataset_id"], episode["item_id"])]
            mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
            np.testing.assert_array_equal(
                candidates[:, np.isfinite(context)],
                np.broadcast_to(
                    context[np.isfinite(context)], (len(actions), int(np.isfinite(context).sum()))
                ),
            )
            all_actions = [*actions, "guarded_direct"]
            bank = points[[methods.index(name) for name in all_actions]]
            generated = candidate_feature_frame(
                context,
                candidates,
                actions,
                coverage,
                mean,
                scale,
                [0, 1],
                joint=model == "chronos2",
                period=episode["period"],
                metadata={
                    **episode,
                    "model_id": model,
                    "episode_index": index,
                    "split": "confirmation",
                },
            )
            last_z = (candidates[actions.index("locf"), -1, [0, 1]] - mean[:2]) / scale[:2]
            generated = forecast_response_inputs(
                extend_forecast_features(generated, bank, actions, last_z)
            )
            for objective in ("clean_forecast_mse", "future_mse"):
                choices = rank_context_actions(models[(model, objective)], generated, 3)
                if (
                    choices != diagnostics["choices"][objective]
                    or diagnostics["decisions_use_current_outcomes"]
                ):
                    raise ValueError("a native confirmation choice failed independent replay")
                selected = np.stack(
                    [
                        chosen_prediction(bank, all_actions, choice, joint=model == "chronos2")
                        for choice in choices
                    ]
                )
                name = "teacher" if objective == "clean_forecast_mse" else "future_supervised"
                np.testing.assert_allclose(
                    points[methods.index(name + "_rank1")], selected[0], rtol=0, atol=1e-12
                )
                np.testing.assert_allclose(
                    points[methods.index(name + "_rank3")],
                    np.median(selected, axis=0),
                    rtol=0,
                    atol=1e-12,
                )
            for method, point in zip(methods, points, strict=True):
                expected_row = reported[
                    (reported.episode_id == episode["episode_id"]) & (reported.method == method)
                ]
                if len(expected_row) != 1:
                    raise ValueError("duplicate or missing confirmation score rows")
                if method in diagnostics["errors"]:
                    if (
                        not expected_row.failed.item()
                        or expected_row[["mae", "mse", "raw_mae", "raw_mse"]].notna().any().any()
                    ):
                        raise ValueError("a failed native diagnostic was silently scored")
                    values = {key: np.nan for key in ("mae", "mse", "raw_mae", "raw_mse")}
                else:
                    errors, counts = observed_future_errors(
                        (point * scale[:2] + mean[:2])[None], future, observed, scale[:2]
                    )
                    if counts.tolist() != episode["window"]["future_observed_by_target"]:
                        raise ValueError("the original future label mask changed")
                    values = {key: float(value.mean()) for key, value in errors.items()}
                    actual = expected_row[list(values)].to_numpy()[0]
                    expected = np.asarray(list(values.values()))
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)
                    max_metric = max(max_metric, float(np.abs(actual - expected).max()))
                recomputed.append(
                    {
                        "model_id": model,
                        "method": method,
                        "episode_id": episode["episode_id"],
                        "family_id": episode["family_id"],
                        "dataset_id": episode["dataset_id"],
                        "item_id": episode["item_id"],
                        "native_missing_context": episode["window"]["context_has_missing"],
                        **values,
                    }
                )
            verified += 1
        recomputed = pd.DataFrame(recomputed)
        for _, group in recomputed.groupby("method"):
            if len(group) != 373 or group.native_missing_context.sum() != 164:
                raise ValueError("a method changed the registered evaluation population")
        summaries = []
        for panel, rows in (
            ("all_registered", recomputed),
            ("naturally_missing", recomputed[recomputed.native_missing_context]),
        ):
            _, _, summary = hierarchical_metrics(rows)
            summaries.append(summary.assign(panel=panel))
        regenerated = pd.concat(summaries).set_index(["panel", "model_id", "method"]).sort_index()
        original = (
            pd.read_csv(directory / "summary.csv", float_precision="round_trip")
            .set_index(["panel", "model_id", "method"])
            .sort_index()
        )
        np.testing.assert_allclose(
            regenerated[["mae", "mse", "raw_mae", "raw_mse"]],
            original[["mae", "mse", "raw_mae", "raw_mse"]],
            rtol=0,
            atol=1e-9,
            equal_nan=True,
        )
        all_recomputed.append(regenerated.reset_index())
    pd.concat(all_recomputed).to_csv(output / "recomputed_summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "sources": sources,
            "source_bundle_sha256": file_sha256(args.source_bundle / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "verified_model_episodes": verified,
            "maximum_metric_difference": max_metric,
            "decision_replay": "all frozen teacher and future-supervised rankings reproduced",
            "limits": "observability, model-selection and scoring audit; statistical interpretation remains separate",
        },
    )
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps({"verified_model_episodes": verified, "maximum_metric_difference": max_metric}),
        flush=True,
    )


if __name__ == "__main__":
    main()
