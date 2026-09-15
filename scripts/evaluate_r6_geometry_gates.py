"""Apply fixed geometry gates to audited R6 and legacy-native prediction caches."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aligned_portfolio_io import decision_vectors  # noqa: E402
from apply_followup_policies import result_panels  # noqa: E402
from audit_shared_forecast_gate import replay_network  # noqa: E402
from evaluate_legacy_native_gates import check_future_fixed  # noqa: E402
from evaluate_r6_policies import restore  # noqa: E402
from geometry_forecast_gate import geometry_features  # noqa: E402
from r6_policy_inputs import decision_inputs, pack_gate_features  # noqa: E402
from run_native_confirmation import hierarchical_metrics  # noqa: E402
from train_shared_forecast_gate import predict_weights  # noqa: E402

from tsfm_fais.forecasting.observed_accuracy import observed_future_errors  # noqa: E402
from tsfm_fais.routing.forecast_gate import SharedForecastGate, compose_forecasts  # noqa: E402
from tsfm_fais.routing.forecast_response import FORECAST_FEATURES  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def r6_inputs(args, model_id, horizon, actions, prep):
    root = args.r6_policy / model_id
    audit = read_json(args.r6_audit / "manifest.json")
    expected = next(row for row in audit["models"] if row["model_id"] == model_id)
    if (
        audit["status"] != "completed"
        or file_sha256(root / "manifest.json") != expected["manifest_sha256"]
    ):
        raise ValueError("the original R6 policy bank changed")
    entry = next(
        row for row in read_json(root / "manifest.json")["horizons"] if row["horizon"] == horizon
    )
    directory = root / entry["directory"]
    marker_path = directory / "predictions_frozen.json"
    if file_sha256(marker_path) != entry["marker_sha256"]:
        raise ValueError("the R6 policy marker changed")
    marker = read_json(marker_path)
    for name in ("individual_features.parquet", "decisions.parquet"):
        if file_sha256(directory / name) != marker["files_sha256"][name]:
            raise ValueError("an audited R6 feature table changed")
    path = directory / "policy_predictions.npz"
    if file_sha256(path) != marker["prediction_sha256"]:
        raise ValueError("R6 predictions changed")
    with np.load(path, allow_pickle=False) as saved:
        if saved["episode_ids"].tolist() != [row["episode_id"] for row in prep["episodes"]]:
            raise ValueError("the R6 prediction order changed")
        names = saved["methods"].tolist()
        points = saved["point_z"][:, [names.index(name) for name in actions]]
        median = saved["point_z"][:, names.index("forecast_median_guarded")]
    individual = pd.read_parquet(directory / "individual_features.parquet")
    decisions = pd.read_parquet(directory / "decisions.parquet")
    order = pd.MultiIndex.from_product(
        [decisions.episode_id, actions], names=["episode_id", "candidate_id"]
    )
    base = pack_gate_features(
        individual.set_index(["episode_id", "candidate_id"])
        .loc[order, list(FORECAST_FEATURES)]
        .to_numpy(np.float32)
        .reshape(len(decisions), 7, 33)
    )
    return decisions, base, decision_vectors(decisions, points), points, median


def legacy_inputs(args, model_id, actions, prep, scalers):
    audited = read_json(args.legacy_results / "manifest.json")
    if audited["status"] != "completed" or audited["identity"][
        "input_manifest_sha256"
    ] != file_sha256(args.legacy_input / "prepared/manifest.json"):
        raise ValueError("legacy input provenance changed")
    binding = read_json(args.legacy_results / "prediction_freeze.json")
    entry = next(row for row in binding["banks"] if row["model_id"] == model_id)
    path = args.legacy_results / entry["path"]
    if file_sha256(path) != entry["sha256"]:
        raise ValueError("the audited legacy prediction bank changed")
    with np.load(path, allow_pickle=False) as saved:
        if saved["episode_ids"].tolist() != [row["episode_id"] for row in prep["episodes"]]:
            raise ValueError("legacy window ordering changed")
        names = saved["methods"].tolist()
        bank = saved["point_z"]
        median = bank[:, names.index("forecast_median_guarded")]
    bases, vectors, decisions, points = [], [], [], []
    for index, record in enumerate(prep["episodes"]):
        path = args.legacy_input / "prepared" / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("a legacy context changed")
        with np.load(path, allow_pickle=False) as saved:
            context, candidates = saved["context"], saved["candidate_values"]
            candidate_ids, coverage = saved["candidate_ids"].tolist(), saved["native_coverage"]
        queried = [*candidate_ids, "guarded_direct"]
        candidate_points = bank[index, [names.index(name) for name in queried]]
        scaler = scalers[(record["dataset_id"], record["item_id"])]
        inputs = decision_inputs(
            context,
            candidates,
            candidate_ids,
            coverage,
            candidate_points,
            np.asarray(scaler["mean"]),
            np.asarray(scaler["scale"]),
            joint=model_id == "chronos2",
            period=record["period"],
            metadata={
                **record,
                "model_id": model_id,
                "episode_index": index,
                "split": "legacy_native_replay",
            },
        )
        if list(inputs["actions"]) != actions:
            raise ValueError("legacy and source candidate order differ")
        bases.append(inputs["gate_features"])
        vectors.append(inputs["vectors"])
        decisions.append(inputs["decisions"])
        points.append(candidate_points[[queried.index(name) for name in actions]])
    return (
        pd.concat(decisions, ignore_index=True),
        pack_gate_features(np.concatenate(bases)),
        np.concatenate(vectors),
        np.stack(points),
        median,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "study-root",
        "study-audit",
        "r6-prepared",
        "r6-policy",
        "r6-audit",
        "r6-comparison",
        "legacy-input",
        "legacy-results",
        "future-cv",
        "future-cv-audit",
        "aligned-root",
        "accuracy-root",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed geometry transfer results")
    study, audit = (
        read_json(args.study_root / "manifest.json"),
        read_json(args.study_audit / "manifest.json"),
    )
    if (
        audit["status"] != "completed"
        or audit["study_sha256"] != file_sha256(args.study_root / "manifest.json")
        or audit["verified_checkpoints"] != 192
    ):
        raise ValueError("audit both source geometry conditions before transfer")
    if (
        file_sha256(ROOT / "scripts/geometry_forecast_gate.py")
        != study["identity"]["geometry_module_sha256"]
    ):
        raise ValueError("geometry input semantics changed")
    cv, cv_audit = (
        read_json(args.future_cv / "manifest.json"),
        read_json(args.future_cv_audit / "manifest.json"),
    )
    cv["manifest_sha256"] = file_sha256(args.future_cv / "manifest.json")
    future_controls = read_json(args.future_cv / "full_source_fixed_controls.json")
    cohorts = {
        "r6": (args.r6_prepared, read_json(args.r6_prepared / "manifest.json"), (96, 192)),
        "legacy_native": (
            args.legacy_input / "prepared",
            read_json(args.legacy_input / "prepared/manifest.json"),
            (96,),
        ),
    }
    scalers = {}
    for cohort, (root, prep, _) in cohorts.items():
        if file_sha256(root / "standardizers.json") != prep["standardizers_sha256"]:
            raise ValueError("a cohort's original prefix statistics changed")
        scalers[cohort] = {
            (row["dataset_id"], row["item_id"]): row
            for row in read_json(root / "standardizers.json")
        }
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "study_sha256": file_sha256(args.study_root / "manifest.json"),
        "study_audit_sha256": file_sha256(args.study_audit / "manifest.json"),
        "legacy_replay_sha256": file_sha256(args.legacy_results / "manifest.json"),
        "future_cv_sha256": cv["manifest_sha256"],
        "geometry_module_sha256": study["identity"]["geometry_module_sha256"],
    }
    _write_json(output / "identity.json", identity)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    names = [
        "geometry_full_gate",
        "geometry_diagonal_gate",
        "future_source_fixed_convex",
        "forecast_median_guarded",
    ]
    banks, prediction_difference, static_gap, checked_decisions = [], 0.0, 0.0, 0
    for model_id in ("chronos2", "timesfm2p5"):
        entries = [row for row in study["source_models"] if row["model_id"] == model_id]
        actions = entries[0]["actions"]
        if (
            any(row["actions"] != actions for row in entries)
            or future_controls[model_id]["actions"] != actions
        ):
            raise ValueError("the frozen source candidate order changed")
        static_gap = max(
            static_gap,
            check_future_fixed(
                model_id,
                future_controls[model_id],
                args.aligned_root,
                args.accuracy_root,
                cv,
                cv_audit,
            ),
        )
        for cohort, (_root, prep, horizons) in cohorts.items():
            for horizon in horizons:
                inputs = (
                    r6_inputs(args, model_id, horizon, actions, prep)
                    if cohort == "r6"
                    else legacy_inputs(args, model_id, actions, prep, scalers[cohort])
                )
                decisions, base, vectors, candidate_bank, median = inputs
                outputs, weights_to_save = [], {}
                for mode in ("full", "diagonal"):
                    features = geometry_features(base, vectors, mode=mode)
                    selected_models = [row for row in entries if row["mode"] == mode]
                    if [row["seed"] for row in selected_models] != [5101, 5102, 5103]:
                        raise ValueError("a geometry condition lost a source seed")
                    seed_weights = []
                    for entry in selected_models:
                        path = args.study_root / entry["path"]
                        if file_sha256(path) != entry["sha256"]:
                            raise ValueError("a full-source geometry model changed")
                        saved = torch.load(path, map_location="cpu", weights_only=True)
                        if (
                            saved["identity_sha256"] != study["identity_sha256"]
                            or len(set(saved["training_origins"])) != 165
                        ):
                            raise ValueError("a geometry model used different source histories")
                        if set(saved["training_origins"]) & {
                            row["origin_id"] for row in prep["episodes"]
                        }:
                            raise ValueError("a target history was used for source fitting")
                        model = SharedForecastGate(features=47).eval().requires_grad_(False)
                        model.load_state_dict(saved["state_dict"])
                        predicted = predict_weights(model, features)
                        np.testing.assert_array_equal(
                            predicted, replay_network(saved["state_dict"], features)
                        )
                        seed_weights.append(predicted)
                    average = np.mean(seed_weights, axis=0)
                    weights_to_save[mode] = average
                    value = restore(
                        compose_forecasts(vectors, average),
                        decisions,
                        len(prep["episodes"]),
                        horizon,
                        model_id == "chronos2",
                    )
                    direct = np.empty_like(value)
                    for slot in (0, 1):
                        positions = np.flatnonzero(
                            decisions.target_slot.to_numpy()
                            == (-1 if model_id == "chronos2" else slot)
                        )
                        indices = decisions.iloc[positions].episode_index.to_numpy(int)
                        direct[indices, :, slot] = np.einsum(
                            "na,nah->nh", average[positions], candidate_bank[indices, :, :, slot]
                        )
                    prediction_difference = max(
                        prediction_difference, float(abs(value - direct).max())
                    )
                    np.testing.assert_allclose(value, direct, rtol=1e-12, atol=1e-12)
                    outputs.append(value)
                    checked_decisions += len(decisions)
                fixed = np.repeat(
                    np.asarray(future_controls[model_id]["convex_weights"])[None],
                    len(decisions),
                    axis=0,
                )
                outputs.extend(
                    [
                        restore(
                            compose_forecasts(vectors, fixed),
                            decisions,
                            len(prep["episodes"]),
                            horizon,
                            model_id == "chronos2",
                        ),
                        median,
                    ]
                )
                bank = np.stack(outputs, axis=1)
                if (
                    bank.shape != (len(prep["episodes"]), 4, horizon, 2)
                    or not np.isfinite(bank).all()
                ):
                    raise ValueError("target prediction coverage or finiteness changed")
                path = output / cohort / model_id / f"h{horizon}_predictions.npz"
                _save_npz(
                    path,
                    point_z=bank,
                    methods=np.asarray(names),
                    episode_ids=np.asarray([row["episode_id"] for row in prep["episodes"]]),
                    **weights_to_save,
                )
                banks.append(
                    {
                        "cohort": cohort,
                        "model_id": model_id,
                        "horizon": horizon,
                        "path": str(path.relative_to(output)),
                        "sha256": file_sha256(path),
                    }
                )
    _write_json(
        output / "prediction_freeze.json",
        {"banks": banks, "identity": identity, "evaluation_future_arrays_read": False},
    )
    rows, metric_difference = [], 0.0
    for entry in banks:
        cohort, horizon = entry["cohort"], entry["horizon"]
        root, prep, _ = cohorts[cohort]
        with np.load(output / entry["path"], allow_pickle=False) as saved:
            bank = saved["point_z"]
        for index, record in enumerate(prep["episodes"]):
            path = root / record["path"]
            if file_sha256(path) != record["sha256"]:
                raise ValueError("an original scoring window changed")
            with np.load(path, allow_pickle=False) as saved:
                truth, observed = saved["future"][:horizon], saved["future_observed"][:horizon]
            scaler = scalers[cohort][(record["dataset_id"], record["item_id"])]
            mean, scale = np.asarray(scaler["mean"])[:2], np.asarray(scaler["scale"])[:2]
            raw = bank[index] * scale + mean
            scores, _ = observed_future_errors(
                raw, truth, observed, scale, minimum_observed=horizon // 2
            )
            direct_mae, direct_mse = [], []
            for slot in (0, 1):
                difference = (
                    raw[:, :, slot][:, observed[:, slot]] - truth[observed[:, slot], slot][None]
                ) / scale[slot]
                direct_mae.append(abs(difference).mean(1))
                direct_mse.append((difference**2).mean(1))
            for metric, direct in (
                ("mae", np.mean(direct_mae, axis=0)),
                ("mse", np.mean(direct_mse, axis=0)),
            ):
                metric_difference = max(
                    metric_difference, float(abs(direct - scores[metric].mean(1)).max())
                )
                np.testing.assert_allclose(direct, scores[metric].mean(1), rtol=1e-12, atol=1e-12)
            rows.extend(
                {
                    **{
                        key: record[key]
                        for key in ("episode_id", "origin_id", "family_id", "dataset_id", "item_id")
                    },
                    "cohort": cohort,
                    "model_id": entry["model_id"],
                    "horizon": horizon,
                    "method": method,
                    "panel": record.get("panel", "legacy_native"),
                    "native_missing_context": record["window"]["context_has_missing"],
                    **{name: float(value[position].mean()) for name, value in scores.items()},
                }
                for position, method in enumerate(names)
            )
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "episode_results.parquet", index=False)
    summaries, families = [], []
    for (cohort, _model, horizon), group in frame.groupby(["cohort", "model_id", "horizon"]):
        panels = (
            result_panels(group)
            if cohort == "r6"
            else (
                ("all_registered", group),
                ("naturally_missing", group[group.native_missing_context]),
                ("complete_context", group[~group.native_missing_context]),
            )
        )
        for panel_name, panel in panels:
            _, family, summary = hierarchical_metrics(panel)
            families.append(family.assign(cohort=cohort, horizon=horizon, panel=panel_name))
            summaries.append(
                summary.assign(
                    cohort=cohort,
                    horizon=horizon,
                    panel=panel_name,
                    families=panel.family_id.nunique(),
                )
            )
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(output / "summary.csv", index=False)
    pd.concat(families, ignore_index=True).to_csv(output / "family_metrics.csv", index=False)
    keys = ["cohort", "model_id", "horizon", "panel", "method"]
    original = pd.concat(
        [
            pd.read_csv(
                args.r6_comparison / "comparison_summary.csv", float_precision="round_trip"
            ).assign(cohort="r6"),
            pd.read_csv(args.legacy_results / "summary.csv", float_precision="round_trip").assign(
                cohort="legacy_native", horizon=96
            ),
        ],
        ignore_index=True,
    )
    current, previous = summary.set_index(keys), original.set_index(keys)
    common = current.index.intersection(previous.index)
    np.testing.assert_allclose(
        current.loc[common, ["mae", "mse"]],
        previous.loc[common, ["mae", "mse"]],
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    additional = current.loc[~current.index.isin(previous.index)].reset_index()
    pd.concat([original, additional], ignore_index=True).to_csv(
        output / "comparison_summary.csv", index=False
    )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "prediction_banks": banks,
            "gate_decisions_replayed": checked_decisions,
            "maximum_prediction_reconstruction_difference": prediction_difference,
            "maximum_normalized_metric_difference": metric_difference,
            "source_fixed_future_optimality_gap": static_gap,
            "score_rows": len(frame),
            "summary_sha256": file_sha256(output / "summary.csv"),
            "new_forecaster_calls": 0,
            "new_fits": 0,
            "limits": "both target cohorts previously used; retains all old baselines and failures; full geometry was declared primary before this readout",
        },
    )


if __name__ == "__main__":
    main()
