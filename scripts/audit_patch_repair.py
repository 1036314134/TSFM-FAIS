"""Verify source-only learning, the last optimizer step, latent repair algebra and native predictions."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from audit_matched_replay import errors
from latent_source_inputs import ROOT, read_json
from learned_patch_repair import PatchRepair, RepairHook
from patch_repair_eval_support import read_input
from patch_repair_inputs import load_case, source_population, training_weights
from patch_repair_runtime import differentiable_point
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster
from scipy.special import erf

from tsfm_fais.contracts import ForecastSpec
from tsfm_fais.utility_experiment import _write_json, file_sha256


def independent_repair(embedding, missing, state, condition):
    raw = np.asarray(embedding, dtype=float)
    active = missing.any(-1)
    result = raw.copy()
    if not active.any():
        return result
    centered = raw - raw.mean(-1, keepdims=True)
    normalized = centered / np.sqrt((centered**2).mean(-1, keepdims=True) + 1e-5)
    descriptor = missing.astype(float)
    if condition == "fraction":
        descriptor = np.repeat(descriptor.mean(-1, keepdims=True), descriptor.shape[-1], axis=-1)
    projected = normalized @ state["down.weight"].T + descriptor @ state["mask.weight"].T
    hidden = 0.5 * projected * (1 + erf(projected / np.sqrt(2)))
    change = hidden @ state["up.weight"].T
    result[active] += change[active]
    return result


def replay_last_step(model_id, adapter, backbone, record, training_root, rows, weights):
    path = training_root / record["last_step_before_path"]
    if file_sha256(path) != record["last_step_before_sha256"]:
        raise ValueError("the optimizer replay checkpoint changed")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    epoch, position = divmod(saved["step"], len(rows))
    expected_index = int(np.random.default_rng(5101 + epoch).permutation(len(rows))[position])
    if saved["row_index"] != expected_index or saved["step"] != record["steps"] - 1:
        raise ValueError("the final optimizer step used an unregistered source sample")
    np.testing.assert_allclose(saved["weight"], weights[expected_index], rtol=1e-12, atol=1e-12)
    repair = PatchRepair(record["width"], record["patch_size"], record["rank"]).to("cuda")
    repair.load_state_dict(saved["model"])
    optimizer = torch.optim.AdamW(repair.parameters(), lr=0.001, weight_decay=0.001, foreach=False)
    optimizer.load_state_dict(saved["optimizer"])
    if any(g["lr"] != 0.001 or g["weight_decay"] != 0.001 for g in optimizer.param_groups):
        raise ValueError("the registered optimizer settings changed")
    row = rows[expected_index]
    raw, observed, truth = load_case(row)
    value = torch.as_tensor(raw, device="cuda", dtype=torch.float32)
    target = torch.as_tensor(truth, device="cuda", dtype=torch.float64)
    scale = torch.as_tensor(row["scaler"]["scale"][:2], device="cuda", dtype=torch.float64)
    optimizer.zero_grad(set_to_none=True)
    with (
        torch.enable_grad(),
        RepairHook(backbone, model_id, observed, repair, record["condition"]) as hook,
    ):
        point = differentiable_point(model_id, adapter, backbone, value)
        residual = (point.double() - target) / scale
        loss = (
            saved["weight"]
            * 0.5
            * (residual.square() + torch.sqrt(residual.square() + 1e-6)).mean()
            + 0.001 * hook.penalty
        )
        if loss.requires_grad:
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(repair.parameters(), 1.0))
            if not np.isfinite(norm):
                raise ValueError("independently replayed gradients are nonfinite")
            optimizer.step()
    expected = torch.load(
        training_root / record["checkpoint_path"], map_location="cpu", weights_only=True
    )
    maximum = max(
        float(abs(value.detach().cpu() - expected[name]).max())
        for name, value in repair.state_dict().items()
    )
    if maximum > 1e-6 or any(p.grad is not None for p in backbone.parameters()):
        raise ValueError(f"adapter-only final-step replay failed: {maximum}")
    return maximum


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
        raise ValueError("preserve completed repair audits")
    trained = read_json(training_root / "manifest.json")
    frozen = read_json(evaluation_root / "predictions_frozen.json")
    if (
        trained["status"] != "completed"
        or frozen["status"] != "completed"
        or bool(trained["identity"]["smoke_only"]) != args.smoke
    ):
        raise ValueError("complete the matching source training and frozen predictions")
    for mapping in (trained["identity"]["files"], frozen["identity"]):
        for name, digest in mapping.items():
            if file_sha256(ROOT / name) != digest:
                raise ValueError("a frozen source or implementation definition changed")
    population, _ = source_population()
    rows = [r for r in population if r["split"] == "train"]
    validation = [r for r in population if r["split"] == "validation"]
    weights = training_weights(rows)
    recorded = read_json(training_root / "population.json")
    if [r["episode_id"] for r in rows] != [r["episode_id"] for r in recorded["training"]]:
        raise ValueError("the trained source population differs")
    if {r["origin_id"] for r in rows} & {r["origin_id"] for r in validation}:
        raise ValueError("source training and validation origins overlap")
    for key in {(r["dataset_id"], r["item_id"]) for r in rows}:
        train = [r for r in rows if (r["dataset_id"], r["item_id"]) == key]
        held = [r for r in validation if (r["dataset_id"], r["item_id"]) == key]
        if max(r["origin"] + 96 for r in train) > min(r["origin"] - 96 for r in held):
            raise ValueError("source label intervals overlap validation contexts")
    input_root = ROOT / "artifacts/iclr27-r24/repair-evaluation-inputs-v001"
    prepared = read_json(input_root / "manifest.json")
    metadata = {r["case_id"]: r for r in prepared["cases"]}
    prediction_map = {(r["model_id"], r["case_id"]): r for r in frozen["predictions"]}
    if not args.smoke:
        study = read_json(evaluation_root / "manifest.json")
        if study["status"] != "completed" or study["predictions"] != frozen["predictions"]:
            raise ValueError("complete the frozen evaluation readout before full audit")
        scores = pd.read_parquet(evaluation_root / "case_scores.parquet").set_index(
            ["model_id", "case_id", "method"]
        )
        target_scores = pd.read_parquet(evaluation_root / "target_scores.parquet").set_index(
            ["model_id", "case_id", "method", "target_slot"]
        )
    maximum_step, maximum_embedding, maximum_prediction, maximum_metric = 0.0, 0.0, 0.0, 0.0
    traced, replayed, cost = 0, 0, []
    torch.set_num_threads(1)
    for model_id in ("chronos2", "timesfm2p5"):
        runner, adapter, backbone, digest, joint = make_forecaster(
            model_id,
            ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
            ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
        )
        modules, states = {}, {}
        initial_states = []
        for condition in ("pattern", "fraction"):
            record = next(
                r
                for r in trained["models"]
                if r["model_id"] == model_id and r["condition"] == condition
            )
            if record["backbone_sha256"] != digest or record["steps"] != (
                4 if args.smoke else 8046
            ):
                raise ValueError("a trained model used different backbone parameters or steps")
            for key, digest_key in (
                ("checkpoint_path", "checkpoint_sha256"),
                ("initial_path", "initial_sha256"),
            ):
                if file_sha256(training_root / record[key]) != record[digest_key]:
                    raise ValueError("a trained or initial repair state changed")
            initial_states.append(
                torch.load(
                    training_root / record["initial_path"], map_location="cpu", weights_only=True
                )
            )
            log_path = (training_root / record["checkpoint_path"]).parent / "training_steps.jsonl"
            attempts = [json.loads(line) for line in log_path.read_text().splitlines()]
            completed = {r["step"]: r for r in attempts}
            if set(completed) != set(range(record["steps"])):
                raise ValueError("training step coverage is incomplete")
            orders = [
                np.random.default_rng(5101 + epoch).permutation(len(rows)) for epoch in range(3)
            ]
            for step, entry in completed.items():
                epoch, position = divmod(step, len(rows))
                if entry["row_index"] != int(orders[epoch][position]) or not np.isfinite(
                    entry["source_loss"]
                ):
                    raise ValueError("the source training order or finite-loss condition differs")
            maximum_step = max(
                maximum_step,
                replay_last_step(model_id, adapter, backbone, record, training_root, rows, weights),
            )
            model = PatchRepair(record["width"], record["patch_size"], record["rank"]).to("cuda")
            saved = torch.load(
                training_root / record["checkpoint_path"], map_location="cpu", weights_only=True
            )
            model.load_state_dict(saved)
            modules[condition] = model.eval().requires_grad_(False)
            states[condition] = {k: v.numpy().astype(float) for k, v in saved.items()}
            cost.append(
                {
                    "model_id": model_id,
                    "condition": condition,
                    "recorded_step_attempts": len(attempts),
                    "distinct_training_steps": len(completed),
                }
            )
        for name in initial_states[0]:
            if not torch.equal(initial_states[0][name], initial_states[1][name]):
                raise ValueError("pattern and fraction did not share initialization")
        for (model_name, case_id), entry in prediction_map.items():
            if model_name != model_id:
                continue
            row = metadata[case_id]
            if (
                file_sha256(input_root / row["path"]) != row["sha256"]
                or file_sha256(evaluation_root / entry["path"]) != entry["sha256"]
            ):
                raise ValueError("an evaluation input or prediction changed")
            data = read_input(input_root / row["path"])
            observed = np.isfinite(data["context"])
            np.testing.assert_array_equal(data["base"][observed], data["context"][observed])
            for trace in entry["traces"]:
                if file_sha256(evaluation_root / trace["path"]) != trace["sha256"]:
                    raise ValueError("an embedding trace changed")
                with np.load(evaluation_root / trace["path"], allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["observed"], observed)
                    real = observed.shape[1] if joint else 2
                    missing = ~observed if joint else ~observed[:, :2]
                    patch = modules[trace["condition"]].patch_size
                    for index in range(trace["calls"]):
                        before, after = saved[f"call_{index}_before"], saved[f"call_{index}_after"]
                        if joint and index == 1:
                            np.testing.assert_array_equal(before, after)
                            continue
                        pattern = np.zeros((*before.shape[:2], patch), bool)
                        pattern[:real] = missing.T.reshape(real, -1, patch)
                        active = pattern.any(-1)
                        np.testing.assert_array_equal(before[~active], after[~active])
                        expected = independent_repair(
                            before, pattern, states[trace["condition"]], trace["condition"]
                        )
                        maximum_embedding = max(
                            maximum_embedding, float(abs(expected - after).max())
                        )
                        np.testing.assert_allclose(expected, after, rtol=1e-5, atol=1e-6)
                traced += 1
            with np.load(evaluation_root / entry["path"], allow_pickle=False) as points:
                names = points["methods"].tolist()
                if row["panel"] == "native_development" or row["trace"] or args.smoke:
                    spec = ForecastSpec(
                        model_id,
                        "joint_multivariate" if joint else "independent_univariate",
                        96,
                        context_length=row["context_length"],
                        target_indices=[0, 1],
                    )
                    base = (
                        runner.predict_missing(data["base"][None], spec).point[0] - data["mean"][:2]
                    ) / data["scale"][:2]
                    np.testing.assert_allclose(
                        base, points["points"][names.index("base_seasonal")], rtol=0, atol=1e-6
                    )
                    for condition, module in modules.items():
                        with RepairHook(backbone, model_id, observed, module, condition):
                            actual = (
                                runner.predict_missing(data["base"][None], spec).point[0]
                                - data["mean"][:2]
                            ) / data["scale"][:2]
                        expected = points["points"][names.index(condition + "_repair")]
                        maximum_prediction = max(
                            maximum_prediction, float(abs(actual - expected).max())
                        )
                        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
                        if row["trace"]:
                            with RepairHook(
                                backbone, model_id, np.ones_like(observed), module, condition
                            ):
                                clean = (
                                    runner.predict_missing(data["base"][None], spec).point[0]
                                    - data["mean"][:2]
                                ) / data["scale"][:2]
                            np.testing.assert_array_equal(clean, base)
                    replayed += 1
                if not args.smoke:
                    if file_sha256(Path(row["original_path"])) != row["original_sha256"]:
                        raise ValueError("an original evaluation label file changed")
                    with np.load(row["original_path"], allow_pickle=False) as original:
                        truth = original["future"][:96, :2]
                        if "future_observed" in original.files:
                            np.testing.assert_array_equal(
                                np.isfinite(truth), original["future_observed"][:96, :2]
                            )
                    losses = errors(points["points"], truth, data["mean"], data["scale"])
                    for index, method in enumerate(names):
                        actual = scores.loc[(model_id, case_id, method), ["mae", "mse"]].to_numpy(
                            float
                        )
                        maximum_metric = max(
                            maximum_metric, float(abs(actual - losses[index].mean(0)).max())
                        )
                        np.testing.assert_allclose(
                            actual, losses[index].mean(0), rtol=1e-10, atol=1e-10
                        )
                        for slot in (0, 1):
                            target = target_scores.loc[(model_id, case_id, method, slot)]
                            np.testing.assert_allclose(
                                target[["mae", "mse"]].to_numpy(float),
                                losses[index, slot],
                                rtol=1e-10,
                                atol=1e-10,
                            )
                            if target.observed_count != np.isfinite(truth[:, slot]).sum():
                                raise ValueError("target scoring observations differ")
        if parameter_digest(backbone) != digest or any(
            p.grad is not None for p in backbone.parameters()
        ):
            raise ValueError("the backbone changed during independent replay")
        del runner, adapter, backbone, modules, model
        gc.collect()
        torch.cuda.empty_cache()
    if not args.smoke:
        if len(scores) != 33936 or len(target_scores) != 67872:
            raise ValueError("the registered full score coverage changed")
        frame = scores.reset_index()
        grouped = pd.read_csv(evaluation_root / "groups.csv").set_index(
            ["panel", "context_length", "model_id", "method", "group_id"]
        )
        for row in pd.read_csv(evaluation_root / "summary.csv").itertuples(index=False):
            subset = frame[
                (frame.panel == row.panel)
                & (frame.context_length == row.context_length)
                & (frame.model_id == row.model_id)
                & (frame.method == row.method)
            ]
            source_means = []
            for group, group_data in subset.groupby("group_id"):
                dataset_means = []
                for _, dataset in group_data.groupby("dataset_id"):
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
                    grouped.loc[
                        (row.panel, row.context_length, row.model_id, row.method, group),
                        ["mae", "mse"],
                    ],
                    rtol=1e-12,
                    atol=1e-12,
                )
                source_means.append(value)
            np.testing.assert_allclose(
                np.mean(source_means, axis=0), [row.mae, row.mse], rtol=1e-12, atol=1e-12
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
            "replayed_optimizer_models": 4,
            "maximum_optimizer_parameter_difference": maximum_step,
            "verified_embedding_traces": traced,
            "maximum_embedding_difference": maximum_embedding,
            "replayed_prediction_pairs": replayed,
            "maximum_prediction_difference": maximum_prediction,
            "maximum_metric_difference": maximum_metric,
            "training_step_accounting": cost,
            "evaluation_future_labels_read": not args.smoke,
        },
    )


if __name__ == "__main__":
    main()
