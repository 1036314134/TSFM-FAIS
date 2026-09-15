"""Replay shared gates, source-only normalization and all development metrics."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aligned_portfolio_io import decision_truth, decision_vectors, load_prepared_model  # noqa: E402

from tsfm_fais.routing.forecast_projection import (  # noqa: E402
    forecast_geometry,
    simplex_quadratic_weights,
)
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def replay_network(state, features):
    outputs = []
    with torch.no_grad():
        for batch in torch.as_tensor(features, dtype=torch.float32).split(512):
            values = ((batch - state["feature_mean"]) / state["feature_scale"]).clamp(-10, 10)
            hidden = F.relu(F.linear(values, state["encoder.0.weight"], state["encoder.0.bias"]))
            pooled = hidden.mean(1, keepdim=True).expand_as(hidden)
            mixed = F.relu(
                F.linear(
                    torch.cat([hidden, pooled], dim=2),
                    state["score.0.weight"],
                    state["score.0.bias"],
                )
            )
            logits = (
                F.linear(mixed, state["score.2.weight"], state["score.2.bias"])[..., 0]
                + state["action_bias"]
            )
            outputs.append(logits.softmax(1))
    result = torch.cat(outputs).numpy().astype(float)
    return result / result.sum(1, keepdims=True)


def mixture(points, weights):
    weights = weights / weights.sum(1, keepdims=True)
    return points[:, 0] + (weights[:, :, None] * (points - points[:, :1])).sum(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("aligned-root", "accuracy-root", "study-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed shared-gate audits")
    prep = json.loads((args.aligned_root / "manifest.json").read_text(encoding="utf-8"))
    accuracy = json.loads((args.accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    studies = {
        kind: json.loads((args.study_root / kind / "manifest.json").read_text(encoding="utf-8"))
        for kind in ("ensemble", "member")
    }
    if any(row["status"] != "completed" for row in studies.values()):
        raise ValueError("complete both matched gate objectives first")
    if (
        file_sha256(args.accuracy_root / "truth_z.npy")
        != accuracy["prediction_arrays"]["truth_z.npy"]
    ):
        raise ValueError("the original validation outcomes changed")
    settings = studies["ensemble"]["identity"]["settings"]
    if settings != studies["member"]["identity"]["settings"] or settings["seeds"] != [
        5101,
        5102,
        5103,
    ]:
        raise ValueError("the matched gate training settings differ")
    for study in studies.values():
        if study["identity"]["aligned_manifest_sha256"] != file_sha256(
            args.aligned_root / "manifest.json"
        ) or study["identity"]["accuracy_manifest_sha256"] != file_sha256(
            args.accuracy_root / "manifest.json"
        ):
            raise ValueError("the audited source banks differ from those used for training")
        if (
            study["identity"]["source_outcome_supervision"]
            or study["identity"]["trainable_parameters_per_seed"] != 1096
        ):
            raise ValueError("the registered supervision or capacity changed")
        for name, sha in study["identity"]["runtime_source_sha256"].items():
            if file_sha256(ROOT / name) != sha:
                raise ValueError("the trained gate's source definition changed")
    torch.set_num_threads(1)
    matched, summaries, checkpoints, decision_outputs = {}, [], 0, 0
    max_weight_diff, max_prediction_diff, max_metric_diff = 0.0, 0.0, 0.0
    for model in ("chronos2", "timesfm2p5"):
        info, decisions, arrays = load_prepared_model(args.aligned_root, prep, model)
        path = args.accuracy_root / f"{model}_point_z.npy"
        if file_sha256(path) != accuracy["prediction_arrays"][path.name]:
            raise ValueError("the source candidate forecasts changed")
        bank = np.load(path, mmap_mode="r")[
            :, [accuracy["action_orders"][model].index(name) for name in info["actions"]]
        ]
        vectors = decision_vectors(decisions, bank)
        _, _, energy, gram = forecast_geometry(vectors)
        alignment = (energy - arrays["direct_risk"][:, :7]) / 2
        for kind, study in studies.items():
            directory = args.study_root / kind
            covered = np.zeros(len(decisions), bool)
            score_parts = []
            for entry in study["folds"]:
                path = directory / entry["path"]
                if file_sha256(path) != entry["sha256"]:
                    raise ValueError("a source fold record changed")
                fold = json.loads(path.read_text(encoding="utf-8"))
                if fold["model_id"] != model:
                    continue
                for file in fold["files"]:
                    if file_sha256(directory / file["path"]) != file["sha256"]:
                        raise ValueError("a saved fold artifact changed")
                family = fold["held_family"]
                train_ids = np.flatnonzero(
                    (decisions.split.to_numpy() == "train")
                    & (decisions.family_id.to_numpy() != family)
                )
                eval_ids = np.flatnonzero(
                    (decisions.split.to_numpy() == "validation")
                    & (decisions.family_id.to_numpy() == family)
                )
                training, evaluation = decisions.iloc[train_ids], decisions.iloc[eval_ids]
                if (
                    sorted(training.origin_id.unique()) != fold["training_origins"]
                    or sorted(training.family_id.unique()) != fold["training_families"]
                ):
                    raise ValueError("a training population changed or includes a held-out family")
                if (
                    hashlib.sha256(train_ids.tobytes()).hexdigest() != fold["train_ids_sha256"]
                    or covered[eval_ids].any()
                ):
                    raise ValueError("the source training or held-out coverage changed")
                covered[eval_ids] = True
                counts = training.groupby(["family_id", "dataset_id"]).size()
                datasets = training.groupby("family_id").dataset_id.nunique()
                sample_weights = np.asarray(
                    [
                        1
                        / (
                            counts.loc[(row.family_id, row.dataset_id)]
                            * datasets.loc[row.family_id]
                        )
                        for row in training.itertuples(index=False)
                    ]
                )
                sample_weights /= sample_weights.mean()
                x_train = arrays["features"][train_ids, :7, :33].astype(float)
                denominator = sample_weights.sum() * 7
                expected_mean = (x_train * sample_weights[:, None, None]).sum(
                    axis=(0, 1)
                ) / denominator
                variance = ((x_train - expected_mean) ** 2 * sample_weights[:, None, None]).sum(
                    axis=(0, 1)
                ) / denominator
                expected_scale = np.maximum(np.sqrt(variance), 1e-6)
                seed_weights = []
                for seed in settings["seeds"]:
                    checkpoint = torch.load(
                        path.parent / f"seed_{seed}.pt", map_location="cpu", weights_only=True
                    )
                    if (
                        checkpoint["identity_sha256"] != study["identity_sha256"]
                        or checkpoint["train_ids_sha256"] != fold["train_ids_sha256"]
                        or len(checkpoint["training_history"]) != 25
                    ):
                        raise ValueError("a seed checkpoint changed identity or training duration")
                    state = checkpoint["state_dict"]
                    parameters = {
                        "action_bias",
                        "encoder.0.weight",
                        "encoder.0.bias",
                        "score.0.weight",
                        "score.0.bias",
                        "score.2.weight",
                        "score.2.bias",
                    }
                    if (
                        set(state) != parameters | {"feature_mean", "feature_scale"}
                        or sum(state[name].numel() for name in parameters) != 1096
                    ):
                        raise ValueError("the matched gate architecture changed")
                    np.testing.assert_array_equal(
                        state["feature_mean"].numpy().reshape(-1), expected_mean.astype(np.float32)
                    )
                    np.testing.assert_array_equal(
                        state["feature_scale"].numpy().reshape(-1),
                        expected_scale.astype(np.float32),
                    )
                    seed_weights.append(
                        replay_network(state, arrays["features"][eval_ids, :7, :33])
                    )
                    checkpoints += 1
                prediction_path = directory / fold["predictions_path"]
                with np.load(prediction_path, allow_pickle=False) as saved:
                    if (
                        not np.array_equal(saved["decision_indices"], eval_ids)
                        or str(saved["identity_sha256"]) != study["identity_sha256"]
                    ):
                        raise ValueError("saved prediction decision identities changed")
                    max_weight_diff = max(
                        max_weight_diff,
                        float(abs(saved["seed_weights"] - np.stack(seed_weights)).max()),
                    )
                    np.testing.assert_array_equal(saved["seed_weights"], np.stack(seed_weights))
                    average = np.mean(seed_weights, axis=0)
                    np.testing.assert_array_equal(saved["mean_weights"], average)
                    probability = sample_weights / sample_weights.sum()
                    mean_gram = np.einsum("n,nab->ab", probability, gram[train_ids])
                    mean_alignment = np.einsum("n,na->a", probability, alignment[train_ids])
                    fixed, _, _ = simplex_quadratic_weights(mean_gram[None], mean_alignment[None])
                    fixed_single = int((mean_gram.diagonal() - 2 * mean_alignment).argmin())
                    np.testing.assert_array_equal(saved["fixed_weights"], fixed)
                    if int(saved["fixed_single"]) != fixed_single:
                        raise ValueError("the source-fixed single changed")
                    controls = {
                        "source_fixed_convex": mixture(
                            vectors[eval_ids], np.repeat(fixed, len(eval_ids), axis=0)
                        ),
                        "source_fixed_single": vectors[eval_ids, fixed_single],
                        "forecast_mean_guarded": vectors[eval_ids].mean(axis=1),
                        "forecast_median_guarded": np.median(vectors[eval_ids], axis=1),
                    }
                    predictions = {kind + "_gate": mixture(vectors[eval_ids], average), **controls}
                    predictions.update(
                        {
                            f"{kind}_seed{seed}": mixture(vectors[eval_ids], weights)
                            for seed, weights in zip(settings["seeds"], seed_weights, strict=True)
                        }
                    )
                    rebuilt = np.stack([predictions[name] for name in saved["methods"].tolist()])
                    max_prediction_diff = max(
                        max_prediction_diff, float(abs(rebuilt - saved["point"]).max())
                    )
                    np.testing.assert_allclose(rebuilt, saved["point"], rtol=1e-12, atol=1e-12)
                key = (model, family)
                if key in matched:
                    for name, values in controls.items():
                        np.testing.assert_array_equal(values, matched[key][name])
                else:
                    matched[key] = controls
                truth = decision_truth(
                    evaluation, np.load(args.accuracy_root / "truth_z.npy", mmap_mode="r")
                )
                saved_scores = pd.read_parquet(directory / fold["scores_path"])
                for method, point in predictions.items():
                    expected = np.column_stack(
                        [abs(point - truth).mean(1), ((point - truth) ** 2).mean(1)]
                    )
                    actual = (
                        saved_scores[saved_scores.method == method]
                        .set_index("episode_id")
                        .loc[evaluation.episode_id]
                    )
                    max_metric_diff = max(
                        max_metric_diff,
                        float(abs(actual[["mae", "mse"]].to_numpy() - expected).max()),
                    )
                    np.testing.assert_allclose(
                        actual[["mae", "mse"]], expected, rtol=1e-12, atol=1e-12
                    )
                    decision_outputs += len(point)
                score_parts.append(saved_scores)
            if not np.array_equal(covered, decisions.split.to_numpy() == "validation"):
                raise ValueError("family-held-out validation is incomplete")
            scores = pd.concat(score_parts, ignore_index=True)
            keys = ["model_id", "method", "family_id", "source_episode_id"]
            family = scores.groupby(keys)[["mae", "mse"]].mean().groupby(keys[:-1]).mean()
            summary = family.groupby(["model_id", "method"]).mean()
            saved_summary = pd.read_csv(directory / "summary.csv").set_index(["model_id", "method"])
            np.testing.assert_allclose(
                saved_summary.loc[summary.index, ["mae", "mse"]], summary, rtol=1e-12, atol=1e-12
            )
            summaries.append(summary.reset_index().assign(objective=kind))
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(summaries, ignore_index=True).to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "study_manifest_sha256": {
                kind: file_sha256(args.study_root / kind / "manifest.json") for kind in studies
            },
            "verified_seed_checkpoints": checkpoints,
            "verified_decision_outputs": decision_outputs,
            "maximum_weight_difference": max_weight_diff,
            "maximum_prediction_difference": max_prediction_diff,
            "maximum_metric_difference": max_metric_diff,
            "matched_inputs_and_capacity": True,
            "limits": "source-development replay and metric audit; no independent confirmation of the changed gate",
        },
    )


if __name__ == "__main__":
    main()
