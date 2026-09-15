"""Retrospectively evaluate frozen gates on the audited 373-window native cohort."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aligned_portfolio_io import decision_truth, decision_vectors  # noqa: E402
from apply_followup_policies import load_frozen_selector  # noqa: E402
from audit_shared_forecast_gate import replay_network  # noqa: E402
from evaluate_r6_policies import restore, triple_vectors  # noqa: E402
from r6_policy_inputs import decision_inputs, pack_gate_features  # noqa: E402
from run_native_confirmation import hierarchical_metrics  # noqa: E402
from train_shared_forecast_gate import predict_weights  # noqa: E402

from tsfm_fais.forecasting.observed_accuracy import observed_future_errors  # noqa: E402
from tsfm_fais.routing.budgeted_portfolio import rank_pairwise_candidates  # noqa: E402
from tsfm_fais.routing.forecast_gate import SharedForecastGate, compose_forecasts  # noqa: E402
from tsfm_fais.routing.utility import _family_weights  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_future_fixed(model_id, controls, aligned_root, accuracy_root, cv, audit):
    if audit["study_sha256"] != cv["manifest_sha256"]:
        raise ValueError("the source-future study is not the independently audited version")
    prep, accuracy = (
        read_json(aligned_root / "manifest.json"),
        read_json(accuracy_root / "manifest.json"),
    )
    if cv["identity"]["aligned_sha256"] != file_sha256(aligned_root / "manifest.json") or cv[
        "identity"
    ]["accuracy_sha256"] != file_sha256(accuracy_root / "manifest.json"):
        raise ValueError("the fixed future-loss source inputs changed")
    entry = next(row for row in prep["models"] if row["model_id"] == model_id)
    path = aligned_root / entry["path"]
    if file_sha256(path) != entry["sha256"]:
        raise ValueError("aligned source metadata changed")
    info = read_json(path)
    path = aligned_root / info["decisions_path"]
    if file_sha256(path) != info["decisions_sha256"] or controls["actions"] != info["actions"]:
        raise ValueError("fixed source candidate order changed")
    decisions = pd.read_parquet(path)
    training = decisions[decisions.split == "train"]
    if (training.origin_id.nunique(), training.family_id.nunique()) != (165, 15):
        raise ValueError("fixed source population changed")
    path, truth_path = accuracy_root / f"{model_id}_point_z.npy", accuracy_root / "truth_z.npy"
    for file in (path, truth_path):
        if file_sha256(file) != accuracy["prediction_arrays"][file.name]:
            raise ValueError("a source forecasting or label bank changed")
    bank = np.load(path, mmap_mode="r")[
        :, [accuracy["action_orders"][model_id].index(name) for name in controls["actions"]]
    ]
    vectors = decision_vectors(training, bank)
    truth = decision_truth(training, np.load(truth_path, mmap_mode="r"))
    fixed = np.asarray(controls["convex_weights"])
    if fixed.min() < 0 or abs(fixed.sum() - 1) > 1e-10:
        raise ValueError("fixed source weights are outside the simplex")
    point = compose_forecasts(vectors, np.repeat(fixed[None], len(training), axis=0))
    weights = _family_weights(training)
    gradient = np.einsum(
        "n,na->a", weights / weights.sum(), 2 * np.mean(vectors * (point - truth)[:, None], axis=2)
    )
    gap = float((gradient @ fixed - gradient.min()) / max(1.0, abs(gradient).max()))
    if gap > 1e-7:
        raise ValueError("the full-source future-loss mixture failed direct optimality")
    return gap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "input-root",
        "legacy-audit",
        "legacy-readout",
        "legacy-bundle",
        "gate-bundle",
        "future-control",
        "origin-source",
        "origin-audit",
        "future-cv",
        "future-cv-audit",
        "aligned-root",
        "accuracy-root",
        "reference-forecasts",
        "protocol",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed retrospective evaluations")
    prep = read_json(args.input_root / "prepared/manifest.json")
    legacy_audit, legacy = (
        read_json(args.legacy_audit / "manifest.json"),
        read_json(args.legacy_bundle / "manifest.json"),
    )
    shared, future = (
        read_json(args.gate_bundle / "manifest.json"),
        read_json(args.future_control / "manifest.json"),
    )
    origin, origin_audit = (
        read_json(args.origin_source / "manifest.json"),
        read_json(args.origin_audit / "source_audit.json"),
    )
    cv, cv_audit = (
        read_json(args.future_cv / "manifest.json"),
        read_json(args.future_cv_audit / "manifest.json"),
    )
    if any(
        row["status"] != "completed"
        for row in (prep, legacy_audit, legacy, shared, future, origin, origin_audit, cv, cv_audit)
    ):
        raise ValueError("complete all prior input, source-model and metric audits")
    if origin_audit["source_manifest_sha256"] != file_sha256(args.origin_source / "manifest.json"):
        raise ValueError("the origin-weighted source bundle changed after its audit")
    if (
        len(prep["episodes"]) != 373
        or sum(row["window"]["context_has_missing"] for row in prep["episodes"]) != 164
    ):
        raise ValueError("the retrospective cohort changed")
    scaler_path = args.input_root / "prepared/standardizers.json"
    if file_sha256(scaler_path) != prep["standardizers_sha256"]:
        raise ValueError("original prefix scaling changed")
    scalers = {(row["dataset_id"], row["item_id"]): row for row in read_json(scaler_path)}
    shared_controls, origin_controls = (
        read_json(args.gate_bundle / "controls.json"),
        read_json(args.origin_source / "controls.json"),
    )
    if (
        file_sha256(args.gate_bundle / "controls.json") != shared["controls_sha256"]
        or file_sha256(args.origin_source / "controls.json") != origin["controls_sha256"]
    ):
        raise ValueError("a source-fixed control changed")
    future_controls = read_json(args.future_cv / "full_source_fixed_controls.json")
    cv["manifest_sha256"] = file_sha256(args.future_cv / "manifest.json")
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "protocol_sha256": file_sha256(args.protocol),
        "input_manifest_sha256": file_sha256(args.input_root / "prepared/manifest.json"),
        "legacy_audit_sha256": file_sha256(args.legacy_audit / "manifest.json"),
        "gate_source_sha256": file_sha256(args.gate_bundle / "manifest.json"),
        "future_source_sha256": file_sha256(args.future_control / "manifest.json"),
        "origin_source_sha256": file_sha256(args.origin_source / "manifest.json"),
        "future_cv_sha256": file_sha256(args.future_cv / "manifest.json"),
    }
    _write_json(output / "identity.json", identity)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    banks, old_replay_gap, static_gap, decision_count = [], 0.0, 0.0, 0
    for model_id in ("chronos2", "timesfm2p5"):
        root = args.input_root / model_id
        manifest_path = root / "manifest.json"
        if file_sha256(manifest_path) != legacy_audit["sources"][model_id]:
            raise ValueError("an audited original forecaster cache changed")
        manifest = read_json(manifest_path)
        if manifest["identity"]["prepared_manifest_sha256"] != identity["input_manifest_sha256"]:
            raise ValueError("the original forecaster cache uses different prepared inputs")
        if (
            manifest["parameter_sha256"]
            != read_json(args.reference_forecasts / model_id / "manifest.json")["parameter_sha256"]
        ):
            raise ValueError("the older and R6 forecasts use different backbone parameters")
        predictions = {row["episode_id"]: row for row in manifest["predictions"]}
        frames, decision_frames, features, vectors, original_points, errors = [], [], [], [], [], []
        actions = shared_controls[model_id]["actions"]
        names = None
        for index, record in enumerate(prep["episodes"]):
            path = args.input_root / "prepared" / record["path"]
            entry = predictions[record["episode_id"]]
            forecast_path = root / entry["path"]
            if (
                file_sha256(path) != record["sha256"]
                or file_sha256(forecast_path) != entry["sha256"]
            ):
                raise ValueError("an original input or prediction changed")
            with np.load(path, allow_pickle=False) as saved:
                context, candidate_values = saved["context"], saved["candidate_values"]
                candidate_ids, coverage = saved["candidate_ids"].tolist(), saved["native_coverage"]
            with np.load(forecast_path, allow_pickle=False) as saved:
                local_names, points = saved["methods"].tolist(), saved["point_z"]
                metadata = json.loads(str(saved["metadata"]))
                if str(saved["parameter_sha256"]) != manifest["parameter_sha256"]:
                    raise ValueError("an original window has different forecasting parameters")
            if names is None:
                names = local_names
            if local_names != names:
                raise ValueError("original method coverage differs across windows")
            queried = [*candidate_ids, "guarded_direct"]
            candidate_points = points[[local_names.index(name) for name in queried]]
            scaler = scalers[(record["dataset_id"], record["item_id"])]
            inputs = decision_inputs(
                context,
                candidate_values,
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
                raise ValueError("source and retrospective candidate order differ")
            frames.append(inputs["individual"])
            decision_frames.append(inputs["decisions"])
            features.append(inputs["gate_features"])
            vectors.append(inputs["vectors"])
            original_points.append(points)
            errors.append(metadata["errors"])
        individual, decisions = (
            pd.concat(frames, ignore_index=True),
            pd.concat(decision_frames, ignore_index=True),
        )
        x, vectors, original_points = (
            pack_gate_features(np.concatenate(features)),
            np.concatenate(vectors),
            np.stack(original_points),
        )
        methods = {name: original_points[:, index] for index, name in enumerate(names)}
        for objective, old_name in (
            ("clean_forecast_mse", "teacher_rank3"),
            ("future_mse", "future_supervised_rank3"),
        ):
            learner, _ = load_frozen_selector(
                args.legacy_bundle, legacy, model_id, objective, "objective"
            )
            ids, margins, pairs = learner.pair_scores(individual)
            ranks = rank_pairwise_candidates(
                margins, pairs, learner.candidate_ids, learner.baseline_id
            )
            choices = {
                identifier: [learner.candidate_ids[j] for j in row[:3]]
                for identifier, row in zip(ids, ranks, strict=True)
            }
            reproduced = restore(
                triple_vectors(vectors, decisions, choices, actions, named_portfolio=False),
                decisions,
                373,
                96,
                model_id == "chronos2",
            )
            old_replay_gap = max(old_replay_gap, float(abs(reproduced - methods[old_name]).max()))
            np.testing.assert_array_equal(reproduced, methods[old_name])
        groups = {
            "ensemble_gate": (
                args.gate_bundle,
                shared["identity_sha256"],
                [
                    row
                    for row in shared["models"]
                    if row["model_id"] == model_id and row["objective"] == "ensemble"
                ],
            ),
            "member_gate": (
                args.gate_bundle,
                shared["identity_sha256"],
                [
                    row
                    for row in shared["models"]
                    if row["model_id"] == model_id and row["objective"] == "member"
                ],
            ),
            "source_future_gate": (
                args.future_control,
                future["identity_sha256"],
                [row for row in future["models"] if row["model_id"] == model_id],
            ),
            "origin_weighted_gate": (
                args.origin_source,
                origin["identity_sha256"],
                origin_controls[model_id]["models"],
            ),
        }
        for name, (model_root, expected_identity, entries) in groups.items():
            if [row["seed"] for row in entries] != [5101, 5102, 5103]:
                raise ValueError("frozen seed coverage changed")
            seed_weights = []
            for entry in entries:
                path = model_root / entry["path"]
                if file_sha256(path) != entry["sha256"]:
                    raise ValueError("a source gate changed")
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if saved["identity_sha256"] != expected_identity:
                    raise ValueError("a source gate belongs to another fitting population")
                if len(set(saved["training_origins"])) != 165 or set(saved["training_origins"]) & {
                    row["origin_id"] for row in prep["episodes"]
                }:
                    raise ValueError("source and retrospective evaluation histories overlap")
                model = SharedForecastGate()
                model.load_state_dict(saved["state_dict"])
                model.eval()
                weight = predict_weights(model, x)
                np.testing.assert_array_equal(weight, replay_network(saved["state_dict"], x))
                seed_weights.append(weight)
            averaged = np.mean(seed_weights, axis=0)
            methods[name] = restore(
                compose_forecasts(vectors, averaged), decisions, 373, 96, model_id == "chronos2"
            )
            _save_npz(
                output / model_id / f"{name}_weights.npz",
                seed_weights=np.stack(seed_weights),
                mean_weights=averaged,
            )
            decision_count += len(decisions)
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
        for name, value in (
            ("gate_source_fixed_convex", shared_controls[model_id]),
            ("origin_weighted_fixed", origin_controls[model_id]),
            ("future_source_fixed_convex", future_controls[model_id]),
        ):
            if value["actions"] != actions:
                raise ValueError("a fixed control has different candidates")
            weights = np.repeat(np.asarray(value["convex_weights"])[None], len(decisions), axis=0)
            methods[name] = restore(
                compose_forecasts(vectors, weights), decisions, 373, 96, model_id == "chronos2"
            )
        new_names = [name for name in methods if name not in names]
        if len(new_names) != 7 or not all(np.isfinite(methods[name]).all() for name in new_names):
            raise ValueError("a new source policy is missing or nonfinite")
        path = output / model_id / "predictions.npz"
        _save_npz(
            path,
            point_z=np.stack(list(methods.values()), axis=1),
            methods=np.asarray(list(methods)),
            episode_ids=np.asarray([row["episode_id"] for row in prep["episodes"]]),
        )
        banks.append(
            {
                "model_id": model_id,
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
                "original_methods": names,
                "errors": errors,
            }
        )
    _write_json(
        output / "prediction_freeze.json",
        {
            "banks": banks,
            "old_ranking_prediction_difference": old_replay_gap,
            "evaluation_futures_read": False,
        },
    )
    rows, metric_gap = [], 0.0
    for entry in banks:
        with np.load(output / entry["path"], allow_pickle=False) as saved:
            bank, names = saved["point_z"], saved["methods"].tolist()
        for index, record in enumerate(prep["episodes"]):
            with np.load(
                args.input_root / "prepared" / record["path"], allow_pickle=False
            ) as saved:
                truth, observed = saved["future"], saved["future_observed"]
            scaler = scalers[(record["dataset_id"], record["item_id"])]
            mean, scale = np.asarray(scaler["mean"])[:2], np.asarray(scaler["scale"])[:2]
            for position, method in enumerate(names):
                if method in entry["errors"][index]:
                    scores = {name: np.nan for name in ("mae", "mse", "raw_mae", "raw_mse")}
                else:
                    raw = bank[index, position] * scale + mean
                    errors, _ = observed_future_errors(
                        raw[None], truth, observed, scale, minimum_observed=48
                    )
                    scores = {name: float(value.mean()) for name, value in errors.items()}
                    direct_mae, direct_mse = [], []
                    for slot in (0, 1):
                        delta = (
                            raw[observed[:, slot], slot] - truth[observed[:, slot], slot]
                        ) / scale[slot]
                        direct_mae.append(abs(delta).mean())
                        direct_mse.append((delta**2).mean())
                    direct = np.asarray([np.mean(direct_mae), np.mean(direct_mse)])
                    metric_gap = max(
                        metric_gap, float(abs(direct - [scores["mae"], scores["mse"]]).max())
                    )
                    np.testing.assert_allclose(
                        direct, [scores["mae"], scores["mse"]], rtol=1e-12, atol=1e-12
                    )
                rows.append(
                    {
                        **{
                            name: record[name]
                            for name in (
                                "episode_id",
                                "origin_id",
                                "dataset_id",
                                "family_id",
                                "item_id",
                            )
                        },
                        "model_id": entry["model_id"],
                        "method": method,
                        "native_missing_context": record["window"]["context_has_missing"],
                        **scores,
                    }
                )
    frame = pd.DataFrame(rows)
    output.mkdir(exist_ok=True)
    frame.to_parquet(output / "episode_results.parquet", index=False)
    summaries, families = [], []
    for panel_name, panel in (
        ("all_registered", frame),
        ("naturally_missing", frame[frame.native_missing_context]),
        ("complete_context", frame[~frame.native_missing_context]),
    ):
        _, family, summary = hierarchical_metrics(panel)
        families.append(family.assign(panel=panel_name))
        summaries.append(summary.assign(panel=panel_name, families=panel.family_id.nunique()))
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(output / "summary.csv", index=False)
    pd.concat(families, ignore_index=True).to_csv(output / "family_metrics.csv", index=False)
    original = pd.read_csv(args.legacy_readout / "summary.csv", float_precision="round_trip")
    keys = ["model_id", "method", "panel"]
    actual = summary.set_index(keys)
    expected = original.set_index(keys)
    np.testing.assert_allclose(
        actual.loc[expected.index, ["mae", "mse", "raw_mae", "raw_mse"]],
        expected[["mae", "mse", "raw_mae", "raw_mse"]],
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "input_windows": 373,
            "original_missing_windows": 164,
            "source_families": 9,
            "original_missing_families": frame[frame.native_missing_context].family_id.nunique(),
            "old_ranking_prediction_difference": old_replay_gap,
            "gate_decisions_replayed": decision_count,
            "fixed_future_optimality_gap": static_gap,
            "maximum_normalized_metric_difference": metric_gap,
            "score_rows": len(frame),
            "summary_sha256": file_sha256(output / "summary.csv"),
            "new_forecaster_calls": 0,
            "new_fits": 0,
            "limits": "retrospective used-data extension; preserves raw-native failures; longer-history control has additional information; no fresh confirmation claim",
        },
    )


if __name__ == "__main__":
    main()
