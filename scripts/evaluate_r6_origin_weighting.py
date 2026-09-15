"""Audit the source weighting intervention and replay its two policies on used R6 data."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aligned_portfolio_io import decision_truth, decision_vectors, load_prepared_model  # noqa: E402
from apply_followup_policies import result_panels  # noqa: E402
from audit_shared_forecast_gate import replay_network  # noqa: E402
from evaluate_r6_policies import restore  # noqa: E402
from r6_policy_inputs import pack_gate_features  # noqa: E402
from run_native_confirmation import hierarchical_metrics  # noqa: E402
from train_shared_forecast_gate import predict_weights  # noqa: E402

from tsfm_fais.forecasting.observed_accuracy import observed_future_errors  # noqa: E402
from tsfm_fais.routing.forecast_gate import (  # noqa: E402
    SharedForecastGate,
    compose_forecasts,
    teacher_quadratics,
)
from tsfm_fais.routing.forecast_projection import simplex_quadratic_weights  # noqa: E402
from tsfm_fais.routing.forecast_response import FORECAST_FEATURES  # noqa: E402
from tsfm_fais.routing.utility import _family_weights  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_audit(args, source, controls):
    prep, accuracy = (
        read_json(args.aligned_root / "manifest.json"),
        read_json(args.accuracy_root / "manifest.json"),
    )
    if source["identity"]["aligned_sha256"] != file_sha256(
        args.aligned_root / "manifest.json"
    ) or source["identity"]["accuracy_sha256"] != file_sha256(args.accuracy_root / "manifest.json"):
        raise ValueError("source provenance changed")
    if len(source["checkpoints"]) != 96 or len(source["parity_checks"]) != 2:
        raise ValueError("source model coverage changed")
    for entry in source["parity_checks"]:
        path = args.source_root / entry["path"]
        if (
            file_sha256(path) != entry["sha256"]
            or read_json(path)["maximum_parameter_difference"] != 0
        ):
            raise ValueError("baseline trainer parity was not verified")
    count = 0
    for model_id in ("chronos2", "timesfm2p5"):
        info, decisions, arrays = load_prepared_model(args.aligned_root, prep, model_id)
        point_path = args.accuracy_root / f"{model_id}_point_z.npy"
        if file_sha256(point_path) != accuracy["prediction_arrays"][point_path.name]:
            raise ValueError("source points changed")
        bank = np.load(point_path, mmap_mode="r")[
            :, [accuracy["action_orders"][model_id].index(name) for name in info["actions"]]
        ]
        vectors = decision_vectors(decisions, bank)
        gram, alignment = teacher_quadratics(vectors, arrays["direct_risk"][:, :7])
        for family in [*sorted(decisions.family_id.unique()), None]:
            train = np.flatnonzero(
                (decisions.split.to_numpy() == "train")
                & ((decisions.family_id.to_numpy() != family) if family is not None else True)
            )
            selected = decisions.iloc[train]
            if (family is not None and family in set(selected.family_id)) or selected.split.ne(
                "train"
            ).any():
                raise ValueError("source fitting used a held-out family or split")
            values = arrays["features"][train, :7, :33].astype(float)
            weights = _family_weights(selected)
            denominator = weights.sum() * 7
            mean = (values * weights[:, None, None]).sum((0, 1)) / denominator
            variance = ((values - mean) ** 2 * weights[:, None, None]).sum((0, 1)) / denominator
            scale = np.maximum(np.sqrt(variance), 1e-6)
            expected_entries = [
                row
                for row in source["checkpoints"]
                if row["model_id"] == model_id and row["held_family"] == family
            ]
            if [row["seed"] for row in expected_entries] != [5101, 5102, 5103]:
                raise ValueError("one of the three fixed seeds is missing")
            eval_ids = (
                np.flatnonzero(
                    (decisions.split.to_numpy() == "validation")
                    & (decisions.family_id.to_numpy() == family)
                )
                if family is not None
                else np.array([], dtype=int)
            )
            predictions = []
            for entry in expected_entries:
                path = args.source_root / entry["path"]
                if file_sha256(path) != entry["sha256"]:
                    raise ValueError("a source checkpoint changed")
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if (
                    saved["identity_sha256"] != source["identity_sha256"]
                    or saved["train_ids_sha256"] != hashlib.sha256(train.tobytes()).hexdigest()
                ):
                    raise ValueError("a source checkpoint used different training decisions")
                if set(saved["training_origins"]) != set(selected.origin_id) or set(
                    saved["training_families"]
                ) != set(selected.family_id):
                    raise ValueError("source training metadata differs from actual decisions")
                state = saved["state_dict"]
                np.testing.assert_array_equal(
                    state["feature_mean"].numpy().reshape(-1), mean.astype(np.float32)
                )
                np.testing.assert_array_equal(
                    state["feature_scale"].numpy().reshape(-1), scale.astype(np.float32)
                )
                if len(eval_ids):
                    predictions.append(
                        replay_network(
                            state, pack_gate_features(arrays["features"][eval_ids, :7, :33])
                        )
                    )
                count += 1
            fixed, _, _ = simplex_quadratic_weights(
                gram[train].mean(0)[None], alignment[train].mean(0)[None]
            )
            if family is None:
                np.testing.assert_array_equal(fixed[0], controls[model_id]["convex_weights"])
            else:
                directory = (args.source_root / expected_entries[0]["path"]).parent
                with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["decision_indices"], eval_ids)
                    np.testing.assert_array_equal(saved["seed_weights"], np.stack(predictions))
                    np.testing.assert_array_equal(
                        saved["mean_weights"], np.mean(predictions, axis=0)
                    )
                    expected = np.stack(
                        [
                            compose_forecasts(vectors[eval_ids], np.mean(predictions, axis=0)),
                            compose_forecasts(
                                vectors[eval_ids], np.repeat(fixed, len(eval_ids), axis=0)
                            ),
                            np.median(vectors[eval_ids], axis=1),
                        ]
                    )
                    np.testing.assert_array_equal(saved["point"], expected)
                    names = saved["methods"].tolist()
                truth_path = args.accuracy_root / "truth_z.npy"
                if file_sha256(truth_path) != accuracy["prediction_arrays"][truth_path.name]:
                    raise ValueError("source validation future bank changed")
                truth = decision_truth(decisions.iloc[eval_ids], np.load(truth_path, mmap_mode="r"))
                recorded = pd.read_parquet(directory / "scores.parquet")
                for index, method in enumerate(names):
                    part = (
                        recorded[recorded.method == method]
                        .set_index("episode_id")
                        .loc[decisions.iloc[eval_ids].episode_id]
                    )
                    np.testing.assert_allclose(
                        part.mae, abs(expected[index] - truth).mean(1), rtol=1e-12, atol=1e-12
                    )
                    np.testing.assert_allclose(
                        part.mse, ((expected[index] - truth) ** 2).mean(1), rtol=1e-12, atol=1e-12
                    )
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-root",
        "aligned-root",
        "accuracy-root",
        "policy-root",
        "policy-audit",
        "prepared-root",
        "comparison-root",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed weighting evaluations")
    output.mkdir(parents=True, exist_ok=True)
    source, audit = (
        read_json(args.source_root / "manifest.json"),
        read_json(args.policy_audit / "manifest.json"),
    )
    if (
        source["status"] != "completed"
        or audit["status"] != "completed"
        or file_sha256(args.source_root / "controls.json") != source["controls_sha256"]
    ):
        raise ValueError("source fitting and original policy audit must be complete")
    torch.set_num_threads(1)
    controls = read_json(args.source_root / "controls.json")
    checked = source_audit(args, source, controls)
    _write_json(
        output / "source_audit.json",
        {
            "status": "completed",
            "verified_checkpoints": checked,
            "source_manifest_sha256": file_sha256(args.source_root / "manifest.json"),
            "feature_normalization_unchanged": True,
            "baseline_parity_exact": True,
            "all_source_predictions_and_scores_replayed": True,
        },
    )
    prep = read_json(args.prepared_root / "manifest.json")
    scaler_path = args.prepared_root / "standardizers.json"
    if file_sha256(scaler_path) != prep["standardizers_sha256"]:
        raise ValueError("R6 scaling changed")
    scalers = {(row["dataset_id"], row["item_id"]): row for row in read_json(scaler_path)}
    names = [
        "origin_weighted_gate",
        "origin_weighted_fixed",
        "gate_source_fixed_convex",
        "forecast_median_guarded",
    ]
    banks, prediction_gap = [], 0.0
    for model_id in ("chronos2", "timesfm2p5"):
        root = args.policy_root / model_id
        audited = next(row for row in audit["models"] if row["model_id"] == model_id)
        if file_sha256(root / "manifest.json") != audited["manifest_sha256"]:
            raise ValueError("the original R6 policy manifest changed")
        original = read_json(root / "manifest.json")
        for entry in original["horizons"]:
            horizon = entry["horizon"]
            directory = root / entry["directory"]
            marker_path = directory / "predictions_frozen.json"
            if file_sha256(marker_path) != entry["marker_sha256"]:
                raise ValueError("the original policy inputs changed")
            marker = read_json(marker_path)
            for name in ("individual_features.parquet", "decisions.parquet"):
                if file_sha256(directory / name) != marker["files_sha256"][name]:
                    raise ValueError("an audited feature or decision table changed")
            individual, decisions = (
                pd.read_parquet(directory / "individual_features.parquet"),
                pd.read_parquet(directory / "decisions.parquet"),
            )
            actions = controls[model_id]["actions"]
            order = pd.MultiIndex.from_product(
                [decisions.episode_id, actions], names=["episode_id", "candidate_id"]
            )
            features = pack_gate_features(
                individual.set_index(["episode_id", "candidate_id"])
                .loc[order, list(FORECAST_FEATURES)]
                .to_numpy(np.float32)
                .reshape(len(decisions), 7, 33)
            )
            prediction_path = directory / "policy_predictions.npz"
            if file_sha256(prediction_path) != marker["prediction_sha256"]:
                raise ValueError("the original policy prediction bank changed")
            with np.load(prediction_path, allow_pickle=False) as saved:
                if saved["episode_ids"].tolist() != [row["episode_id"] for row in prep["episodes"]]:
                    raise ValueError("R6 evaluation order changed")
                methods, old_bank = saved["methods"].tolist(), saved["point_z"]
            candidate_bank = old_bank[:, [methods.index(name) for name in actions]]
            vectors = decision_vectors(decisions, candidate_bank)
            seed_weights = []
            for entry in controls[model_id]["models"]:
                path = args.source_root / entry["path"]
                if file_sha256(path) != entry["sha256"]:
                    raise ValueError("a complete source checkpoint changed")
                saved = torch.load(path, map_location="cpu", weights_only=True)
                model = SharedForecastGate()
                model.load_state_dict(saved["state_dict"])
                model.eval()
                weights = predict_weights(model, features)
                np.testing.assert_array_equal(
                    weights, replay_network(saved["state_dict"], features)
                )
                seed_weights.append(weights)
            learned, fixed = (
                np.mean(seed_weights, axis=0),
                np.repeat(
                    np.asarray(controls[model_id]["convex_weights"])[None], len(decisions), axis=0
                ),
            )
            values = []
            for weights in (learned, fixed):
                point = restore(
                    compose_forecasts(vectors, weights),
                    decisions,
                    len(prep["episodes"]),
                    horizon,
                    model_id == "chronos2",
                )
                direct = np.empty_like(point)
                for slot in (0, 1):
                    positions = np.flatnonzero(
                        decisions.target_slot.to_numpy() == (-1 if model_id == "chronos2" else slot)
                    )
                    indices = decisions.iloc[positions].episode_index.to_numpy(int)
                    direct[indices, :, slot] = np.einsum(
                        "na,nah->nh", weights[positions], candidate_bank[indices, :, :, slot]
                    )
                prediction_gap = max(prediction_gap, float(abs(point - direct).max()))
                np.testing.assert_allclose(point, direct, rtol=1e-12, atol=1e-12)
                values.append(point)
            values.extend(old_bank[:, methods.index(name)] for name in names[2:])
            path = output / model_id / f"h{horizon}_predictions.npz"
            _save_npz(
                path,
                point_z=np.stack(values, axis=1),
                methods=np.asarray(names),
                episode_ids=np.asarray([row["episode_id"] for row in prep["episodes"]]),
                mean_weights=learned,
            )
            banks.append(
                {
                    "model_id": model_id,
                    "horizon": horizon,
                    "path": str(path.relative_to(output)),
                    "sha256": file_sha256(path),
                }
            )
    _write_json(
        output / "prediction_freeze.json",
        {
            "banks": banks,
            "source_audit_sha256": file_sha256(output / "source_audit.json"),
            "evaluation_future_arrays_read": False,
        },
    )
    records, metric_gap = [], 0.0
    for entry in banks:
        model_id, horizon = entry["model_id"], entry["horizon"]
        with np.load(output / entry["path"], allow_pickle=False) as saved:
            bank = saved["point_z"]
        for index, row in enumerate(prep["episodes"]):
            path = args.prepared_root / row["path"]
            if file_sha256(path) != row["sha256"]:
                raise ValueError("an original R6 scoring input changed")
            with np.load(path, allow_pickle=False) as saved:
                future, observed = saved["future"][:horizon], saved["future_observed"][:horizon]
            scaler = scalers[(row["dataset_id"], row["item_id"])]
            mean, scale = np.asarray(scaler["mean"])[:2], np.asarray(scaler["scale"])[:2]
            errors, _ = observed_future_errors(
                bank[index] * scale + mean, future, observed, scale, minimum_observed=horizon // 2
            )
            truth = (future - mean) / scale
            direct_mae, direct_mse = [], []
            for slot in (0, 1):
                difference = (
                    bank[index, :, :, slot][:, observed[:, slot]]
                    - truth[observed[:, slot], slot][None]
                )
                direct_mae.append(abs(difference).mean(1))
                direct_mse.append((difference**2).mean(1))
            for metric, direct in (
                ("mae", np.mean(direct_mae, axis=0)),
                ("mse", np.mean(direct_mse, axis=0)),
            ):
                metric_gap = max(metric_gap, float(abs(direct - errors[metric].mean(1)).max()))
                np.testing.assert_allclose(direct, errors[metric].mean(1), rtol=1e-12, atol=1e-12)
            for position, method in enumerate(names):
                records.append(
                    {
                        **{
                            key: row[key]
                            for key in (
                                "episode_id",
                                "origin_id",
                                "dataset_id",
                                "family_id",
                                "item_id",
                                "panel",
                                "mechanism",
                                "missing_rate",
                                "mask_seed",
                            )
                        },
                        "model_id": model_id,
                        "horizon": horizon,
                        "method": method,
                        "native_missing_context": row["window"]["context_has_missing"],
                        **{name: float(value[position].mean()) for name, value in errors.items()},
                    }
                )
    scores = pd.DataFrame(records)
    scores.to_parquet(output / "episode_results.parquet", index=False)
    families, summaries = [], []
    for (_model, horizon), group in scores.groupby(["model_id", "horizon"]):
        for panel_name, panel in result_panels(group):
            _, family, summary = hierarchical_metrics(panel)
            families.append(family.assign(horizon=horizon, panel=panel_name))
            summaries.append(
                summary.assign(
                    horizon=horizon,
                    panel=panel_name,
                    families=panel.family_id.nunique(),
                    origins=panel.origin_id.nunique(),
                )
            )
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(output / "summary.csv", index=False)
    pd.concat(families, ignore_index=True).to_csv(output / "family_metrics.csv", index=False)
    original = pd.read_csv(
        args.comparison_root / "comparison_summary.csv", float_precision="round_trip"
    )
    keys = ["model_id", "horizon", "panel", "method"]
    controls = summary[summary.method.isin(names[2:])].set_index(keys)
    expected = original.set_index(keys).loc[controls.index]
    np.testing.assert_allclose(
        controls[["mae", "mse"]], expected[["mae", "mse"]], rtol=1e-12, atol=1e-12
    )
    pd.concat([original, summary[summary.method.isin(names[:2])]], ignore_index=True).to_csv(
        output / "comparison_summary.csv", index=False
    )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "source_audit_sha256": file_sha256(output / "source_audit.json"),
            "verified_source_checkpoints": checked,
            "maximum_prediction_reconstruction_difference": prediction_gap,
            "maximum_normalized_metric_difference": metric_gap,
            "score_rows": len(scores),
            "summary_sha256": file_sha256(output / "summary.csv"),
            "new_forecaster_calls": 0,
            "limits": "single source loss-weighting intervention; target R6 is previously used; old family weighting and all historical controls remain reported",
        },
    )


if __name__ == "__main__":
    main()
