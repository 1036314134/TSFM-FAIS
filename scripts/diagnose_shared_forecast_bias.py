"""Audit simple past-feedback bias controls using completed R19 caches only."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
from audit_matched_replay import errors
from evaluate_matched_replay import aggregate_groups, score_points
from latent_source_inputs import ROOT, read_json

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def fit_controls(point, truth):
    parameters = np.zeros((3, 2, 2))
    for slot in (0, 1):
        observed = np.isfinite(truth[..., slot])
        counts = observed.sum(1)
        if (counts < 48).any() or len(counts) != 8:
            raise ValueError("the original past target support changed")
        weight = np.broadcast_to((1 / (8 * counts))[:, None], observed.shape)[observed]
        p, r = point[..., slot][observed], (truth[..., slot] - point[..., slot])[observed]
        mean = weight @ r
        order = np.argsort(r, kind="stable")
        residual, mass = r[order], np.cumsum(weight[order])
        index = min(int(np.searchsorted(mass, 0.5, side="left")), len(r) - 1)
        median = residual[index]
        if abs(mass[index] - 0.5) <= 1e-12 and index + 1 < len(r):
            median = (median + residual[index + 1]) / 2
        design = np.column_stack([p, np.ones_like(p)])
        delta = np.linalg.solve(
            design.T @ (weight[:, None] * design) + np.eye(2), design.T @ (weight * r)
        )
        parameters[:, slot] = [[0, mean], [0, median], delta]
        np.testing.assert_allclose(np.sum(weight * (r - mean)), 0, atol=1e-10)
        if weight[r < median].sum() > 0.5 + 1e-12 or weight[r > median].sum() > 0.5 + 1e-12:
            raise ValueError("weighted median subgradient condition failed")
        gradient = (
            sum(w * x * (x @ delta - y) for w, x, y in zip(weight, design, r, strict=True)) + delta
        )
        np.testing.assert_allclose(gradient, 0, rtol=0, atol=1e-10)
    return parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed calibration diagnostics")
    then = perf_counter()
    base = ROOT / "artifacts/iclr27-r19"
    plan, prepared, forecast, old_study = [
        read_json(base / name / "manifest.json")
        for name in (
            "pilot-plan-v001",
            "replay-inputs-v001",
            "replay-forecasts-v001",
            "replay-results-v001",
        )
    ]
    audit = read_json(base / "replay-audit-v001/manifest.json")
    if audit["status"] != "completed" or audit["study_sha256"] != file_sha256(
        base / "replay-results-v001/manifest.json"
    ):
        raise ValueError("the R19 audit binding changed")
    identity = {
        str(p.relative_to(ROOT)): file_sha256(p)
        for p in (
            Path(__file__),
            ROOT / "docs/iclr2027/R22_SHARED_BIAS_DIAGNOSTIC_PLAN.md",
            *[
                base / n / "manifest.json"
                for n in (
                    "pilot-plan-v001",
                    "replay-inputs-v001",
                    "replay-forecasts-v001",
                    "replay-results-v001",
                    "replay-audit-v001",
                )
            ],
        )
    }
    inputs = {r["case_id"]: r for r in prepared["cases"]}
    metadata = {r["case_id"]: r for r in plan["cases"]}
    decisions = {(r["model_id"], r["case_id"]): r for r in old_study["decisions"]}
    retained = [
        "median8",
        "mean8",
        "source_single_mae",
        "source_fixed_mae",
        "source_fixed_joint",
        "native_long_raw",
        "native_long_prefix",
        "generic_erm",
        "shuffled_erm",
        "matched_erm",
    ]
    output.mkdir(parents=True, exist_ok=True)
    frozen = []
    for entry in forecast["cases"]:
        row, inp, old = (
            metadata[entry["case_id"]],
            inputs[entry["case_id"]],
            decisions[(entry["model_id"], entry["case_id"])],
        )
        paths = [
            base / "replay-inputs-v001" / inp["path"],
            base / "replay-forecasts-v001" / entry["path"],
            base / "replay-results-v001" / old["path"],
        ]
        for path, record in zip(paths, (inp, entry, old), strict=True):
            if file_sha256(path) != record["sha256"]:
                raise ValueError("a past input, prediction or old fixed decision changed")
        if len(row["selected_anchors"]) != 8 or any(
            u + 96 > row["origin"] for u in row["selected_anchors"]
        ):
            raise ValueError("past feedback is not fully available at the current origin")
        with (
            np.load(paths[0], allow_pickle=False) as data,
            np.load(paths[1], allow_pickle=False) as predicted,
            np.load(paths[2], allow_pickle=False) as selected,
        ):
            old_names = selected["methods"].tolist()
            methods = {name: selected["points"][old_names.index(name)] for name in retained}
            current = methods["median8"]
            past_truth = (data["historical_future"] - data["mean"][:2]) / data["scale"][:2]
            parameters = []
            for rule, bank in zip(
                ("generic", "shuffled", "matched"), predicted["historical"], strict=True
            ):
                past_point = np.median(bank, axis=1)
                fitted = fit_controls(past_point, past_truth)
                parameters.append(fitted)
                for name, delta in zip(
                    ("mean_bias", "median_bias", "affine_ridge"), fitted, strict=True
                ):
                    methods[f"{rule}_{name}"] = current + current * delta[:, 0] + delta[:, 1]
            if len(methods) != 19:
                raise ValueError("the registered diagnostic method count changed")
            path = output / "predictions" / entry["model_id"] / f"{entry['case_id']}.npz"
            _save_npz(
                path,
                methods=np.asarray(list(methods)),
                points=np.stack(list(methods.values())),
                parameters=np.stack(parameters),
            )
            frozen.append(
                {
                    "model_id": entry["model_id"],
                    "case_id": entry["case_id"],
                    "path": str(path.relative_to(output)),
                    "sha256": file_sha256(path),
                }
            )
    _write_json(
        output / "predictions_frozen.json",
        {"identity": identity, "predictions": frozen, "current_future_read": False},
    )
    scores, maximum_error = [], 0.0
    for entry in frozen:
        row, inp = metadata[entry["case_id"]], inputs[entry["case_id"]]
        if file_sha256(Path(row["current_artifact"])) != row["current_artifact_sha256"]:
            raise ValueError("an original current target file changed")
        with (
            np.load(row["current_artifact"], allow_pickle=False) as original,
            np.load(base / "replay-inputs-v001" / inp["path"], allow_pickle=False) as data,
            np.load(output / entry["path"], allow_pickle=False) as result,
        ):
            truth = original["future"][:96]
            np.testing.assert_array_equal(np.isfinite(truth), original["future_observed"][:96])
            normalized = (truth - data["mean"][:2]) / data["scale"][:2]
            mae, mse = score_points(result["points"], normalized)
            independent = errors(result["points"], truth, data["mean"], data["scale"])
            maximum_error = max(
                maximum_error,
                float(abs(independent[..., 0] - mae).max()),
                float(abs(independent[..., 1] - mse).max()),
            )
            np.testing.assert_allclose(independent[..., 0], mae, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(independent[..., 1], mse, rtol=1e-10, atol=1e-10)
            for index, method in enumerate(result["methods"].tolist()):
                record = {
                    name: row[name]
                    for name in (
                        "group_id",
                        "family_id",
                        "dataset_id",
                        "item_id",
                        "episode_id",
                        "origin_id",
                    )
                }
                scores.append(
                    {
                        **record,
                        "model_id": entry["model_id"],
                        "case_id": entry["case_id"],
                        "method": method,
                        "mae": float(mae[index].mean()),
                        "mse": float(mse[index].mean()),
                    }
                )
    frame = pd.DataFrame(scores)
    previous = pd.read_parquet(base / "replay-results-v001/case_scores.parquet").set_index(
        ["model_id", "case_id", "method"]
    )
    for name in retained:
        subset = frame[frame.method == name].set_index(["model_id", "case_id", "method"])
        np.testing.assert_array_equal(
            subset[["mae", "mse"]], previous.loc[subset.index, ["mae", "mse"]]
        )
    if len(frame) != 912:
        raise ValueError("the registered diagnostic scoring population changed")
    frame.to_parquet(output / "case_scores.parquet", index=False)
    for name, table in zip(
        ("series", "datasets", "groups", "summary"), aggregate_groups(frame), strict=True
    ):
        table.to_csv(output / f"{name}.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "predictions": frozen,
            "score_rows": len(frame),
            "maximum_metric_difference": maximum_error,
            "retained_controls_exact": True,
            "forecast_calls": 0,
            "neural_fits": 0,
            "wall_seconds": perf_counter() - then,
            "limits": "classical past-feedback calibration diagnostic on reused development data; no method novelty or independent confirmation claim",
        },
    )


if __name__ == "__main__":
    main()
