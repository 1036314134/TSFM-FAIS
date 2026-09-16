"""Independently replay the changed source loss and all newly repaired predictions."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from audit_matched_replay import errors
from latent_source_inputs import ROOT, read_json
from learned_patch_repair import PatchRepair, RepairHook
from mae_repair_inputs import evaluation_cases, load_evaluation_case
from patch_repair_inputs import load_case, source_population, training_weights
from patch_repair_runtime import differentiable_point
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster

from tsfm_fais.contracts import ForecastSpec
from tsfm_fais.utility_experiment import _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    training_root, evaluation_root, output = (
        args.training_root.resolve(),
        args.evaluation_root.resolve(),
        args.output_root.resolve(),
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed MAE-repair audit")
    trained, frozen = (
        read_json(training_root / "manifest.json"),
        read_json(evaluation_root / "predictions_frozen.json"),
    )
    if (
        trained["status"] != "completed"
        or frozen["status"] != "completed"
        or len(trained["models"]) != 1
    ):
        raise ValueError("complete the single model and fixed predictions")
    for mapping in (trained["identity"]["files"], frozen["identity"]):
        for name, digest in mapping.items():
            if file_sha256(ROOT / name) != digest:
                raise ValueError("a frozen loss, data or runtime definition changed")
    record = trained["models"][0]
    if (
        record["model_id"] != "chronos2"
        or record["condition"] != "fraction"
        or record["steps"] != (4 if args.smoke else 8046)
    ):
        raise ValueError("the controlled model, condition or training budget changed")
    previous_root = ROOT / "artifacts/iclr27-r24/repair-training-v001"
    previous = next(
        r
        for r in read_json(previous_root / "manifest.json")["models"]
        if r["model_id"] == "chronos2" and r["condition"] == "fraction"
    )
    first = torch.load(
        training_root / record["initial_path"], map_location="cpu", weights_only=True
    )
    original_initial = torch.load(
        previous_root / previous["initial_path"], map_location="cpu", weights_only=True
    )
    for key in first:
        if not torch.equal(first[key], original_initial[key]):
            raise ValueError("MAE and joint objectives did not share initialization")
    population, _ = source_population()
    rows = [r for r in population if r["split"] == "train"]
    validation = [r for r in population if r["split"] == "validation"]
    if {r["origin_id"] for r in rows} & {r["origin_id"] for r in validation}:
        raise ValueError("source training and validation overlap")
    weights = training_weights(rows)
    before = torch.load(
        training_root / record["last_step_before_path"], map_location="cpu", weights_only=True
    )
    epoch, position = divmod(before["step"], len(rows))
    index = int(np.random.default_rng(5101 + epoch).permutation(len(rows))[position])
    if index != before["row_index"] or before["step"] != record["steps"] - 1:
        raise ValueError("source step order changed")
    np.testing.assert_allclose(before["weight"], weights[index], rtol=1e-12, atol=1e-12)
    torch.set_num_threads(1)
    runner, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    repair = PatchRepair(record["width"], record["patch_size"], record["rank"]).to("cuda")
    repair.load_state_dict(before["model"])
    optimizer = torch.optim.AdamW(repair.parameters(), lr=0.001, weight_decay=0.001, foreach=False)
    optimizer.load_state_dict(before["optimizer"])
    raw, observed, truth = load_case(rows[index])
    values = torch.as_tensor(raw, device="cuda", dtype=torch.float32)
    target = torch.as_tensor(truth, device="cuda", dtype=torch.float64)
    scale = torch.as_tensor(rows[index]["scaler"]["scale"][:2], device="cuda", dtype=torch.float64)
    optimizer.zero_grad(set_to_none=True)
    with (
        torch.enable_grad(),
        RepairHook(backbone, "chronos2", observed, repair, "fraction") as hook,
    ):
        point = differentiable_point("chronos2", adapter, backbone, values)
        residual = (point.double() - target) / scale
        loss = before["weight"] * torch.sqrt(residual.square() + 1e-6).mean() + 0.001 * hook.penalty
        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(repair.parameters(), 1.0)
            optimizer.step()
    final = torch.load(
        training_root / record["checkpoint_path"], map_location="cpu", weights_only=True
    )
    optimizer_delta = max(
        float(abs(value.detach().cpu() - final[key]).max())
        for key, value in repair.state_dict().items()
    )
    if optimizer_delta > 1e-6 or any(p.grad is not None for p in backbone.parameters()):
        raise ValueError("the independent MAE-only optimizer replay failed")
    repair.load_state_dict(final)
    repair.eval().requires_grad_(False)
    if not args.smoke:
        study = read_json(evaluation_root / "manifest.json")
        if study["predictions"] != frozen["predictions"] or study["status"] != "completed":
            raise ValueError("complete the frozen readout before full audit")
        scores = pd.read_parquet(evaluation_root / "case_scores.parquet").set_index(
            ["case_id", "method"]
        )
        target_scores = pd.read_parquet(evaluation_root / "target_scores.parquet").set_index(
            ["case_id", "method", "target_slot"]
        )
    metadata = {r["case_id"]: r for r in evaluation_cases(args.smoke)}
    maximum_prediction, maximum_metric = 0.0, 0.0
    for entry in frozen["predictions"]:
        row = metadata[entry["case_id"]]
        if (
            file_sha256(evaluation_root / entry["path"]) != entry["sha256"]
            or file_sha256(Path(row["old_prediction_path"])) != row["old_prediction_sha256"]
        ):
            raise ValueError("a new prediction or old control changed")
        data = load_evaluation_case(row)
        mask = np.isfinite(data["context"])
        np.testing.assert_array_equal(data["base"][mask], data["context"][mask])
        spec = ForecastSpec(
            "chronos2",
            "joint_multivariate",
            96,
            context_length=row["context_length"],
            target_indices=[0, 1],
        )
        with RepairHook(backbone, "chronos2", mask, repair, "fraction"):
            actual = (
                runner.predict_missing(data["base"][None], spec).point[0] - data["mean"][:2]
            ) / data["scale"][:2]
        with (
            np.load(evaluation_root / entry["path"], allow_pickle=False) as predicted,
            np.load(row["old_prediction_path"], allow_pickle=False) as old,
        ):
            names = predicted["methods"].tolist()
            expected = predicted["points"][names.index("mae_repair")]
            maximum_prediction = max(maximum_prediction, float(abs(actual - expected).max()))
            np.testing.assert_array_equal(actual, expected)
            for name, value in zip(old["methods"].tolist(), old["points"], strict=True):
                np.testing.assert_array_equal(value, predicted["points"][names.index(name)])
            if not args.smoke:
                if file_sha256(Path(row["original_path"])) != row["original_sha256"]:
                    raise ValueError("the original future label file changed")
                with np.load(row["original_path"], allow_pickle=False) as original:
                    truth = original["future"][:96, :2]
                    if "future_observed" in original.files:
                        np.testing.assert_array_equal(
                            np.isfinite(truth), original["future_observed"][:96, :2]
                        )
                losses = errors(predicted["points"], truth, data["mean"], data["scale"])
                for i, method in enumerate(names):
                    actual_loss = scores.loc[(row["case_id"], method), ["mae", "mse"]].to_numpy(
                        float
                    )
                    maximum_metric = max(
                        maximum_metric, float(abs(actual_loss - losses[i].mean(0)).max())
                    )
                    np.testing.assert_allclose(
                        actual_loss, losses[i].mean(0), rtol=1e-10, atol=1e-10
                    )
                    for slot in (0, 1):
                        target_loss = target_scores.loc[(row["case_id"], method, slot)]
                        np.testing.assert_allclose(
                            target_loss[["mae", "mse"]].to_numpy(float),
                            losses[i, slot],
                            rtol=1e-10,
                            atol=1e-10,
                        )
                        if target_loss.observed_count != np.isfinite(truth[:, slot]).sum():
                            raise ValueError("an observed-future mask changed")
    if parameter_digest(backbone) != digest:
        raise ValueError("the frozen backbone changed")
    if not args.smoke:
        if len(scores) != 22999 or len(target_scores) != 45998:
            raise ValueError("the registered complete scoring coverage changed")
        frame = scores.reset_index()
        groups = pd.read_csv(evaluation_root / "groups.csv").set_index(
            ["panel", "context_length", "method", "group_id"]
        )
        for row in pd.read_csv(evaluation_root / "summary.csv").itertuples(index=False):
            subset = frame[
                (frame.panel == row.panel)
                & (frame.context_length == row.context_length)
                & (frame.method == row.method)
            ]
            means = []
            for group, grouped in subset.groupby("group_id"):
                dataset_means = []
                for _, dataset in grouped.groupby("dataset_id"):
                    series_means = []
                    for _, series in dataset.groupby("item_id"):
                        series_means.append(
                            np.mean(
                                [
                                    origin[["mae", "mse"]].to_numpy().mean(0)
                                    for _, origin in series.groupby("origin_id")
                                ],
                                axis=0,
                            )
                        )
                    dataset_means.append(np.mean(series_means, axis=0))
                value = np.mean(dataset_means, axis=0)
                np.testing.assert_allclose(
                    value,
                    groups.loc[(row.panel, row.context_length, row.method, group), ["mae", "mse"]],
                    rtol=1e-12,
                    atol=1e-12,
                )
                means.append(value)
            np.testing.assert_allclose(
                np.mean(means, axis=0), [row.mae, row.mse], rtol=1e-12, atol=1e-12
            )
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "smoke_only": args.smoke,
            "training_sha256": file_sha256(training_root / "manifest.json"),
            "study_sha256": None if args.smoke else file_sha256(evaluation_root / "manifest.json"),
            "optimizer_parameter_difference": optimizer_delta,
            "replayed_cases": len(frozen["predictions"]),
            "maximum_prediction_difference": maximum_prediction,
            "maximum_metric_difference": maximum_metric,
            "evaluation_future_read": not args.smoke,
        },
    )


if __name__ == "__main__":
    main()
