"""Check visible scales, matched model fits and independent bounded prediction replay."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from latent_source_inputs import ROOT, read_json
from position_objective_inputs import arguments, average_positions, checked_sources, load_inputs
from positional_forecast_portfolio import replay_offsets
from train_latent_source_gates import aggregate

from tsfm_fais.routing.utility import _family_weights
from tsfm_fais.utility_experiment import _write_json, file_sha256


def main():
    parser = arguments(__doc__)
    parser.set_defaults(protocol=ROOT / "docs/iclr2027/R18_DISAGREEMENT_SCALE_PROTOCOL.md")
    parser.add_argument(
        "--study-root", type=Path, default=ROOT / "artifacts/iclr27-r18/disagreement-scale-v001"
    )
    parser.add_argument(
        "--previous-study",
        type=Path,
        default=ROOT / "artifacts/iclr27-r17/position-objectives-v001",
    )
    parser.add_argument(
        "--previous-audit",
        type=Path,
        default=ROOT / "artifacts/iclr27-r17/position-objectives-audit-v002",
    )
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed scale audits")
    source = checked_sources(args)
    study = read_json(args.study_root / "manifest.json")
    audit = read_json(args.previous_audit / "manifest.json")
    if audit["status"] != "completed" or audit["study_sha256"] != file_sha256(
        args.previous_study / "manifest.json"
    ):
        raise ValueError("the previous study audit changed")
    for key, path in (
        ("script_sha256", ROOT / "scripts/train_disagreement_scale.py"),
        ("scale_module_sha256", ROOT / "scripts/disagreement_scale.py"),
        ("input_module_sha256", ROOT / "scripts/position_objective_inputs.py"),
        ("loss_module_sha256", ROOT / "scripts/position_objective.py"),
        ("model_module_sha256", ROOT / "scripts/positional_forecast_portfolio.py"),
        ("previous_sha256", args.previous_study / "manifest.json"),
        ("previous_audit_sha256", args.previous_audit / "manifest.json"),
        ("protocol_sha256", args.protocol),
    ):
        if study["identity"][key] != file_sha256(path):
            raise ValueError("a scale-study definition changed")
    if (
        study["status"] != "completed"
        or study["identity"]["settings"] != source["identity"]["settings"]
    ):
        raise ValueError("complete the registered training budget")
    freeze = read_json(args.study_root / "prediction_freeze.json")
    for key in ("checkpoints", "folds"):
        if freeze[key] != study[key]:
            raise ValueError("the scale model or prediction freeze changed")
    torch.set_num_threads(1)
    rows, verified, maximum_difference = [], 0, 0.0
    for fold in study["folds"]:
        model_id = fold["model_id"]
        frame, inputs, points, labels, _ = load_inputs(args, model_id)
        training = np.flatnonzero(frame.split.to_numpy() == "train")
        validation = np.flatnonzero(frame.split.to_numpy() == "validation")
        if not np.isnan(labels["future"][validation]).all():
            raise ValueError("validation outcomes entered training arrays")
        weight = _family_weights(frame.iloc[training])
        context, local = (
            inputs["context"][training].astype(float),
            inputs["local"][training].astype(float),
        )
        cm = (context * weight[:, None]).sum(0) / weight.sum()
        cv = ((context - cm) ** 2 * weight[:, None]).sum(0) / weight.sum()
        lm = (local * weight[:, None, None]).sum((0, 1)) / (weight.sum() * local.shape[1])
        lv = ((local - lm) ** 2 * weight[:, None, None]).sum((0, 1)) / (
            weight.sum() * local.shape[1]
        )
        statistics = {
            "context_mean": cm,
            "context_scale": np.maximum(np.sqrt(cv), 1e-6),
            "local_mean": lm,
            "local_scale": np.maximum(np.sqrt(lv), 1e-6),
        }
        path = args.previous_study / fold["previous_prediction_path"]
        if file_sha256(path) != fold["previous_prediction_sha256"]:
            raise ValueError("an original frozen prediction changed")
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["validation_indices"], validation)
            expected = dict(zip(saved["methods"].tolist(), saved["point"], strict=True))
        old_methods = set(expected)
        ordered = np.sort(points, axis=1)
        median = ordered[:, 3]
        for scale_kind in ("unit", "range", "mad"):
            scale = (
                np.ones_like(median)
                if scale_kind == "unit"
                else ordered[:, -1] - ordered[:, 0]
                if scale_kind == "range"
                else np.sort(abs(points - median[:, None]), axis=1)[:, 3]
            )
            for mode in ("local", "pooled"):
                entries = sorted(
                    [
                        row
                        for row in study["checkpoints"]
                        if row["model_id"] == model_id
                        and row["scale_kind"] == scale_kind
                        and row["mode"] == mode
                    ],
                    key=lambda row: row["seed"],
                )
                if [row["seed"] for row in entries] != [5101, 5102, 5103]:
                    raise ValueError("a matched scale seed is missing")
                predictions = []
                group = f"scale18_{scale_kind}_{mode}"
                for entry in entries:
                    path = args.study_root / entry["path"]
                    if file_sha256(path) != entry["sha256"]:
                        raise ValueError("a learned scale model changed")
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    metadata = {
                        key: value for key, value in entry.items() if key not in ("path", "sha256")
                    }
                    if (
                        saved["metadata"] != metadata
                        or metadata["train_indices_sha256"]
                        != hashlib.sha256(training.tobytes()).hexdigest()
                    ):
                        raise ValueError("a scale model used different training rows")
                    if metadata["training_origins"] != sorted(
                        frame.iloc[training].origin_id.unique()
                    ) or metadata["training_families"] != sorted(
                        frame.iloc[training].family_id.unique()
                    ):
                        raise ValueError("scale training history or family coverage changed")
                    if (
                        len(saved["history"]) != 25
                        or saved["initial_parameter_sha256"]
                        != source["initial_parameter_sha256"][str(entry["seed"])]
                    ):
                        raise ValueError("scale fitting budget or initialization changed")
                    state = saved["state_dict"]
                    for name, value in statistics.items():
                        np.testing.assert_array_equal(
                            state[name].numpy().ravel(), value.astype(np.float32)
                        )
                    if (
                        sum(
                            value.numel() for name, value in state.items() if name not in statistics
                        )
                        != 609
                    ):
                        raise ValueError("scaling changed model capacity")
                    chunks = []
                    for selection in np.array_split(
                        validation, max(1, (len(validation) + 255) // 256)
                    ):
                        offset = replay_offsets(
                            state, inputs["context"][selection], inputs["local"][selection], mode
                        )
                        estimate = torch.as_tensor(median[selection]) + torch.as_tensor(
                            scale[selection]
                        ) * torch.tanh(torch.as_tensor(offset))
                        prediction = torch.clamp(
                            estimate,
                            min=torch.as_tensor(ordered[selection, 0]),
                            max=torch.as_tensor(ordered[selection, -1]),
                        ).numpy()
                        zero = scale[selection] == 0
                        np.testing.assert_array_equal(prediction[zero], median[selection][zero])
                        chunks.append(prediction)
                    prediction = np.concatenate(chunks)
                    predictions.append(prediction)
                    expected[f"{group}_seed{entry['seed']}"] = prediction
                    verified += 1
                expected[group] = average_positions(predictions, inputs, validation)
        path = args.study_root / fold["prediction_path"]
        if file_sha256(path) != fold["prediction_sha256"]:
            raise ValueError("a scale-study validation bank changed")
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["validation_indices"], validation)
            predictions = dict(zip(saved["methods"].tolist(), saved["point"], strict=True))
        if set(predictions) != set(expected) or len(predictions) != 61:
            raise ValueError("a scale comparison output is missing")
        for method, value in expected.items():
            maximum_difference = max(
                maximum_difference, float(abs(value - predictions[method]).max())
            )
            if method in old_methods:
                np.testing.assert_array_equal(value, predictions[method])
            else:
                np.testing.assert_allclose(value, predictions[method], rtol=1e-12, atol=1e-12)
                if np.any(predictions[method] < inputs["lower"][validation]) or np.any(
                    predictions[method] > inputs["upper"][validation]
                ):
                    raise ValueError("a scale-study output left its bounds")
        _, _, _, held_labels, _ = load_inputs(args, model_id, validation=True)
        for method, point in predictions.items():
            error = point - held_labels["future"][validation]
            rows.append(
                frame.iloc[validation].assign(
                    model_id=model_id, method=method, mae=abs(error).mean(1), mse=(error**2).mean(1)
                )
            )
        print(f"{model_id}: all source scales and matched models audited", flush=True)
    scores = pd.concat(rows, ignore_index=True)
    keys = ["model_id", "method", "episode_id"]
    old_scores = pd.read_parquet(args.study_root / "decision_scores.parquet")
    pd.testing.assert_frame_equal(
        scores.sort_values(keys).reset_index(drop=True),
        old_scores.sort_values(keys).reset_index(drop=True),
        check_exact=True,
    )
    episodes, families, summary = aggregate(scores)
    pd.testing.assert_frame_equal(
        summary,
        pd.read_csv(args.study_root / "summary.csv", float_precision="round_trip"),
        check_exact=True,
    )
    if verified != 36 or len(scores) != 456768:
        raise ValueError("the scale-study audit population changed")
    output.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(output / "episode_scores.parquet", index=False)
    families.to_csv(output / "family_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "study_sha256": file_sha256(args.study_root / "manifest.json"),
            "verified_models": verified,
            "verified_scores": len(scores),
            "maximum_prediction_difference": maximum_difference,
            "all_original_predictions_exact": True,
            "new_forecaster_calls": 0,
            "limits": "full-source temporal validation; real-missing transfer still required",
        },
    )


if __name__ == "__main__":
    main()
