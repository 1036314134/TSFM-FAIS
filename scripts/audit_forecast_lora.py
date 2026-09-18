"""Replay LoRA source boundaries, final optimizer updates and every evaluation query."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from audit_forecast_correlation import check_metrics
from forecast_calibration_core import load_npz, read_json
from forecast_lora_core import MODES, ForecastLoRA, optimizer_for, predict_tensor, update_once
from forecast_lora_experiment import (
    PARENT,
    backbone_and_pipeline,
    source_examples,
    source_order,
)
from probe_differentiable_imputation import parameter_digest
from reliability_attention_experiment import case_truth, truth_sources

from tsfm_fais.utility_experiment import _write_json, file_sha256


def replay_last_update(base, prepared, backbone, pipeline, mode):
    directory = base / "adaptation-training-v001" / mode
    manifest = read_json(directory / "manifest.json")
    if manifest["status"] != "completed" or manifest["input_sha256"] != file_sha256(
        base / "adaptation-inputs-v001/manifest.json"
    ):
        raise ValueError("adapter training provenance changed")
    for name, key in (
        ("initial.npz", "initial_sha256"),
        ("final.npz", "final_sha256"),
        ("before-final-update.pt", "before_final_sha256"),
        ("training.csv", "training_log_sha256"),
    ):
        if file_sha256(directory / name) != manifest[key]:
            raise ValueError("a frozen training record changed")
    bank = ForecastLoRA(backbone).to(device=backbone.device, dtype=backbone.dtype)
    if (
        bank.projection_names != manifest["projection_names"]
        or sum(p.numel() for p in bank.parameters()) != manifest["trainable_parameters"]
    ):
        raise ValueError("registered adapter scope changed")
    initial = load_npz(directory / "initial.npz", manifest["initial_sha256"])
    for key, value in bank.state_dict().items():
        np.testing.assert_array_equal(value.detach().cpu().numpy(), initial[key])
    checkpoint = torch.load(
        directory / "before-final-update.pt", map_location="cpu", weights_only=True
    )
    if (
        checkpoint["case_id"] != prepared["training_order"][-1]
        or checkpoint["step"] != len(prepared["training_order"]) - 1
    ):
        raise ValueError("the final source update identity changed")
    bank.load_state_dict(checkpoint["adapter"], strict=True)
    optimizer = optimizer_for(bank)
    optimizer.load_state_dict(checkpoint["optimizer"])
    logs = pd.read_csv(directory / "training.csv")
    if logs.case_id.tolist() != prepared["training_order"] or len(logs) != manifest["updates"]:
        raise ValueError("training order, repetitions or budget changed")
    if (
        logs.step.tolist() != list(range(1, len(logs) + 1))
        or not np.isfinite(logs[["loss", "gradient_norm"]].to_numpy()).all()
    ):
        raise ValueError("missing, nonfinite or reordered training updates")
    source_row = next(r for r in prepared["source_cases"] if r["case_id"] == checkpoint["case_id"])
    example = load_npz(base / "adaptation-inputs-v001" / source_row["path"], source_row["sha256"])
    context = torch.tensor(
        example["natural" if mode == "natural_lora" else "corrupted"], device=backbone.device
    )
    with bank.installed():
        raw = predict_tensor(backbone, pipeline, context, 24)
        point = (
            raw[: example["future"].shape[1], pipeline.quantiles.index(0.5), :24]
            .T.detach()
            .cpu()
            .numpy()
        )
        reference_loss = []
        for column in range(point.shape[1]):
            valid = np.isfinite(example["future"][:, column])
            difference = np.abs(
                point[valid, column].astype(float)
                - example["future"][valid, column].astype(np.float32).astype(float)
            )
            reference_loss.append(
                np.where(difference < 0.01, difference**2 / 0.02, difference - 0.005).mean()
            )
        del raw
        loss, norm = update_once(bank, optimizer, backbone, pipeline, example, mode)
    np.testing.assert_allclose(loss, np.mean(reference_loss), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        [loss, norm], logs.iloc[-1][["loss", "gradient_norm"]].to_numpy(float), rtol=0, atol=1e-14
    )
    final = load_npz(directory / "final.npz", manifest["final_sha256"])
    for key, value in bank.state_dict().items():
        np.testing.assert_array_equal(value.detach().cpu().numpy(), final[key])
    if not any(not np.array_equal(initial[key], final[key]) for key in final):
        raise ValueError("the trained adapter did not update")
    return bank, manifest


def audit(base):
    started = perf_counter()
    inputs, forecasts, output = (
        base / "adaptation-inputs-v001",
        base / "adaptation-forecasts-v001",
        base / "adaptation-audit-v001",
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed adapter audits")
    prepared, fm = read_json(inputs / "manifest.json"), read_json(forecasts / "manifest.json")
    if fm["status"] != "completed" or fm["input_sha256"] != file_sha256(inputs / "manifest.json"):
        raise ValueError("incomplete adapter predictions or changed source inputs")
    for path, sha in prepared["identity"].items():
        if file_sha256(Path(path)) != sha:
            raise ValueError("a registered adaptation definition changed")
    if not fm["smoke"]:
        for row in read_json(base / "method_manifest.json")["files"]:
            if file_sha256(Path(row["path"])) != row["sha256"]:
                raise ValueError("a frozen adaptation runtime changed")
    source_metadata = {row["case_id"]: row for row in prepared["source_cases"]}
    verified, checked = [], 0
    for metadata, arrays in source_examples():
        row = source_metadata[metadata["case_id"]]
        for key, value in metadata.items():
            if row[key] != value:
                raise ValueError("source identity or label boundary changed")
        saved = load_npz(inputs / row["path"], row["sha256"])
        for key, value in arrays.items():
            np.testing.assert_array_equal(saved[key], value)
        retained = np.isfinite(saved["corrupted"])
        np.testing.assert_array_equal(saved["natural"][retained], saved["corrupted"][retained])
        if (np.isfinite(saved["future"]).sum(0) < 12).any():
            raise ValueError("training future support changed")
        if row["dataset"] == "hdb" and row["label_end"] > 336:
            raise ValueError("HDB training reached evaluation outcomes")
        if row["dataset"] == "beijing" and row["label_end"] > 7012:
            raise ValueError("Beijing training reached evaluation outcomes")
        verified.append(row)
        checked += 1
    if checked != len(source_metadata):
        raise ValueError("source population changed")
    expected_order = source_order(verified)
    if fm["smoke"]:
        expected_order = expected_order[:12]
    if prepared["training_order"] != expected_order:
        raise ValueError("balanced source sampling changed")
    actual_sources = [source_metadata[name]["dataset"] for name in expected_order]
    if actual_sources != ["beijing", "hdb"] * (len(expected_order) // 2):
        raise ValueError("source update balance changed")
    parent_inputs = {
        r["case_id"]: r for r in read_json(PARENT / "attention-inputs-v001/manifest.json")["cases"]
    }
    parent_forecasts = {
        r["case_id"]: r
        for r in read_json(PARENT / "attention-forecasts-v001/manifest.json")["cases"]
    }
    ids = {r["case_id"] for r in fm["cases"]}
    if ids != {r["case_id"] for r in prepared["evaluation_cases"]} or (
        not fm["smoke"] and ids != set(parent_inputs)
    ):
        raise ValueError("registered evaluation population changed")
    scores, records, hdb = None, None, None
    if not fm["smoke"]:
        result = base / "adaptation-results-v001"
        if read_json(result / "manifest.json")["forecast_sha256"] != file_sha256(
            forecasts / "manifest.json"
        ):
            raise ValueError("adapter score provenance changed")
        scores = pd.read_parquet(result / "target_scores.parquet").set_index(
            ["case_id", "method", "slot"]
        )
        if not scores.index.is_unique or len(scores) != 80925:
            raise ValueError("adapter target score population changed")
        records, hdb = truth_sources()
    backbone, pipeline, digest = backbone_and_pipeline()
    banks, manifests = {}, {}
    for mode in MODES:
        banks[mode], manifests[mode] = replay_last_update(base, prepared, backbone, pipeline, mode)
        if manifests[mode]["parameter_sha256"] != digest or fm["training"][mode] != file_sha256(
            base / "adaptation-training-v001" / mode / "manifest.json"
        ):
            raise ValueError("adapter training lineage changed")
    if manifests[MODES[0]]["initial_sha256"] != manifests[MODES[1]]["initial_sha256"]:
        first = load_npz(base / "adaptation-training-v001" / MODES[0] / "initial.npz")
        second = load_npz(base / "adaptation-training-v001" / MODES[1] / "initial.npz")
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
    zero = (
        ForecastLoRA(backbone).to(device=backbone.device, dtype=backbone.dtype)
        if fm["smoke"]
        else None
    )
    metadata = {r["case_id"]: r for r in prepared["evaluation_cases"]}
    mid, calls, ordinary, neutral, rebuilt_scores = pipeline.quantiles.index(0.5), 0, 0, 0, []
    for number, entry in enumerate(fm["cases"]):
        row = metadata[entry["case_id"]]
        old_input, old_output = parent_inputs[row["case_id"]], parent_forecasts[row["case_id"]]
        for key in (
            "dataset",
            "panel",
            "station",
            "origin",
            "horizon",
            "target_count",
            "source_column",
            "native_name",
        ):
            if row[key] != old_input[key]:
                raise ValueError("evaluation metadata changed")
        if (
            row["input_sha256"] != old_input["sha256"]
            or row["parent_sha256"] != old_output["sha256"]
        ):
            raise ValueError("evaluation input or old control lineage changed")
        data = load_npz(row["input_path"], row["input_sha256"])
        parent = load_npz(row["parent_path"], row["parent_sha256"])
        expected = dict(zip(parent["methods"].tolist(), parent["points"], strict=True))
        saved = load_npz(forecasts / entry["path"], entry["sha256"])
        requests = json.loads(str(saved["queries"]))
        if {q["name"] for q in requests} != set(MODES) | {"ordinary"} or len(requests) != 3:
            raise ValueError("a registered adapter query is missing")
        for request in requests:
            query = load_npz(forecasts / request["path"], request["sha256"])
            np.testing.assert_array_equal(query["context_z"], data["native"])
            context = torch.tensor(query["context_z"], device=backbone.device)
            name = request["name"]
            if name in MODES:
                with banks[name].installed(), torch.inference_mode():
                    raw = predict_tensor(backbone, pipeline, context, row["horizon"]).cpu().numpy()
                expected[name] = raw[: row["target_count"], mid, : row["horizon"]].T.astype(float)
                expected["half_var_" + name] = (
                    0.5 * expected[name] + 0.5 * expected["linear_var_direct"]
                )
                calls += 1
            else:
                with torch.inference_mode():
                    raw = predict_tensor(backbone, pipeline, context, row["horizon"]).cpu().numpy()
                np.testing.assert_array_equal(
                    raw[: row["target_count"], mid, : row["horizon"]].T.astype(float),
                    expected[row["native_name"]],
                )
                ordinary += 1
                if zero is not None:
                    with zero.installed(), torch.inference_mode():
                        initial = (
                            predict_tensor(backbone, pipeline, context, row["horizon"])
                            .cpu()
                            .numpy()
                        )
                    np.testing.assert_array_equal(initial, raw)
                    neutral += 1
            np.testing.assert_array_equal(raw, query["quantiles"])
        names = sorted(expected)
        np.testing.assert_array_equal(saved["methods"], np.asarray(names))
        np.testing.assert_array_equal(saved["points"], np.stack([expected[n] for n in names]))
        if scores is not None:
            truth = case_truth(row, records, hdb)
            for name, point in expected.items():
                values = []
                for slot in range(row["target_count"]):
                    valid = np.isfinite(truth[:, slot])
                    errors = (
                        point[valid, slot]
                        - (truth[valid, slot] - data["mean"][slot]) / data["scale"][slot]
                    )
                    metrics = [float(np.abs(errors).mean()), float(np.square(errors).mean())]
                    target = scores.loc[(row["case_id"], name, slot)]
                    np.testing.assert_allclose(
                        metrics, target[["mae", "mse"]].to_numpy(float), rtol=1e-12, atol=1e-12
                    )
                    if target["observed_count"] != valid.sum():
                        raise ValueError("target observation support changed")
                    values.append(metrics)
                mae, mse = np.mean(values, axis=0)
                rebuilt_scores.append(
                    {
                        **{k: row[k] for k in ("case_id", "panel", "station")},
                        "method": name,
                        "mae": mae,
                        "mse": mse,
                    }
                )
        if (number + 1) % 25 == 0:
            _write_json(
                output / "progress.json",
                {"cases_audited": number + 1, "adapter_queries_replayed": calls},
            )
            print(
                json.dumps({"cases_audited": number + 1, "adapter_queries_replayed": calls}),
                flush=True,
            )
    if rebuilt_scores:
        check_metrics(pd.DataFrame(rebuilt_scores), result)
    if (
        parameter_digest(backbone) != digest
        or digest != fm["parameter_sha256"]
        or calls != fm["adapter_calls"]
        or ordinary != fm["ordinary_calls"]
        or neutral != fm["neutral_checks"]
    ):
        raise ValueError("base model identity or query accounting changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": fm["smoke"],
            "source_cases_checked": checked,
            "optimizer_final_updates_replayed": 2,
            "optimizer_parameter_difference": 0,
            "evaluation_cases": len(ids),
            "adapter_queries_replayed": calls,
            "ordinary_restorations": ordinary,
            "neutral_checks": neutral,
            "prediction_difference": 0,
            "score_rows": len(rebuilt_scores),
            "heldout_value_analysis": False,
            "forecast_sha256": file_sha256(forecasts / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "wall_seconds": perf_counter() - started,
        },
    )
