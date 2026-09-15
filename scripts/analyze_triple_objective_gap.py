"""Diagnose individual-versus-portfolio teacher objectives using frozen source forecasts."""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.routing.budgeted_portfolio import median_portfolio  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def best_option(points, costs, *, joint):
    """Minimize per-target costs while preserving the forecaster's decision scope."""
    points, costs = np.asarray(points, float), np.asarray(costs, float)
    if points.ndim != 4 or costs.shape != (points.shape[0], points.shape[1], points.shape[3]):
        raise ValueError("option forecasts and per-target costs must align")
    if not np.isfinite(points).all() or not np.isfinite(costs).all():
        raise ValueError("option forecasts and costs must be finite")
    choice = (
        np.repeat(costs.mean(axis=2).argmin(axis=1)[:, None], points.shape[3], axis=1)
        if joint
        else costs.argmin(axis=1)
    )
    selected = np.take_along_axis(points.transpose(0, 3, 1, 2), choice[:, :, None, None], axis=2)
    return selected[:, :, 0].transpose(0, 2, 1), choice


def teacher_top_three(points, teacher, *, joint):
    cost = np.square(points - teacher[:, None]).mean(axis=2)
    ranks = (
        np.repeat(
            np.argsort(cost.mean(axis=2), axis=1, kind="stable")[:, None], points.shape[3], axis=1
        )
        if joint
        else np.argsort(cost.transpose(0, 2, 1), axis=2, kind="stable")
    )
    return median_portfolio(points, ranks, 3, joint=joint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("accuracy-root", "teacher-root", "portfolio-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed objective diagnostics")
    accuracy = json.loads((args.accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    teachers = json.loads((args.teacher_root / "manifest.json").read_text(encoding="utf-8"))
    portfolio = json.loads((args.portfolio_root / "manifest.json").read_text(encoding="utf-8"))
    accuracy_sha = file_sha256(args.accuracy_root / "manifest.json")
    if (
        teachers["status"] != "completed"
        or portfolio["status"] != "completed"
        or any(
            obj["identity"]["accuracy_manifest_sha256"] != accuracy_sha
            for obj in (teachers, portfolio)
        )
    ):
        raise ValueError("complete the teacher and portfolio studies with the same forecast source")
    source_path = Path(accuracy["source_root"]) / "episodes_manifest.json"
    if file_sha256(source_path) != accuracy["source_episode_manifest_sha256"]:
        raise ValueError("source metadata changed")
    metadata = pd.DataFrame(json.loads(source_path.read_text(encoding="utf-8"))["episodes"])
    indices = np.flatnonzero(metadata.split.to_numpy() == "validation")
    metadata = metadata.iloc[indices].reset_index(drop=True)
    if len(indices) != 1872 or metadata.origin_id.nunique() != 52:
        raise ValueError("the registered development panel changed")
    truth_path = args.accuracy_root / "truth_z.npy"
    if file_sha256(truth_path) != accuracy["prediction_arrays"][truth_path.name]:
        raise ValueError("source validation outcomes changed")
    truth = np.load(truth_path, mmap_mode="r")[indices]
    output.mkdir(parents=True, exist_ok=True)
    rows, records = [], []
    roles = {
        "forecast_median_seven": "available candidate forecasts",
        "learned_teacher_rank3": "available candidate forecasts and frozen source-trained selector",
        "unavailable_individual_teacher_top3": "unavailable complete current history",
        "unavailable_joint_teacher_best3": "unavailable complete current history",
        "future_oracle_single_mae": "unavailable current future; MAE optimum in the single-candidate class",
        "future_oracle_single_mse": "unavailable current future; MSE optimum in the single-candidate class",
        "future_oracle_triple_mae": "unavailable current future; MAE optimum over three-candidate medians",
        "future_oracle_triple_mse": "unavailable current future; MSE optimum over three-candidate medians",
    }
    for model in ("chronos2", "timesfm2p5"):
        joint = model == "chronos2"
        directory = args.portfolio_root / model
        model_record = next(item for item in portfolio["models"] if item["model_id"] == model)
        if file_sha256(directory / "manifest.json") != model_record["sha256"]:
            raise ValueError("a recorded portfolio model changed")
        saved_model = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for entry in saved_model["files"]:
            if file_sha256(directory / entry["path"]) != entry["sha256"]:
                raise ValueError("a saved portfolio decision or score changed")
        actions = saved_model["actions"]
        point_path = args.accuracy_root / f"{model}_point_z.npy"
        if file_sha256(point_path) != accuracy["prediction_arrays"][point_path.name]:
            raise ValueError("candidate predictions changed")
        points = np.load(point_path, mmap_mode="r")[indices][
            :, [accuracy["action_orders"][model].index(a) for a in actions]
        ]
        teacher_record = next(item for item in teachers["models"] if item["model_id"] == model)
        teacher_path = args.teacher_root / teacher_record["teacher_file"]
        if file_sha256(teacher_path) != teacher_record["teacher_sha256"]:
            raise ValueError("complete-history predictions changed")
        teacher = np.load(teacher_path, mmap_mode="r")[indices]
        subsets = np.asarray(list(combinations(range(len(actions)), 3)))
        triples = np.stack([np.median(points[:, subset], axis=1) for subset in subsets], axis=1)
        teacher_costs = np.square(triples - teacher[:, None]).mean(axis=2)
        with np.load(directory / "student_rankings.npz", allow_pickle=False) as saved:
            learned = median_portfolio(points, saved["clean_forecast_mse"], 3, joint=joint)
        forecasts = {
            "forecast_median_seven": np.median(points, axis=1),
            "learned_teacher_rank3": learned,
            "unavailable_individual_teacher_top3": teacher_top_three(points, teacher, joint=joint),
        }
        forecasts["unavailable_joint_teacher_best3"], teacher_choice = best_option(
            triples, teacher_costs, joint=joint
        )
        choices = {"joint_teacher": teacher_choice}
        for kind, values in (("single", points), ("triple", triples)):
            for metric in ("mae", "mse"):
                residual = values - truth[:, None]
                cost = (np.abs(residual) if metric == "mae" else np.square(residual)).mean(axis=2)
                name = f"future_oracle_{kind}_{metric}"
                forecasts[name], choices[name] = best_option(values, cost, joint=joint)
        joint_loss = np.square(forecasts["unavailable_joint_teacher_best3"] - teacher).mean(
            axis=(1, 2)
        )
        individual_loss = np.square(
            forecasts["unavailable_individual_teacher_top3"] - teacher
        ).mean(axis=(1, 2))
        if np.any(joint_loss > individual_loss + 1e-10):
            raise ValueError("exhaustive triple teacher minimization failed")
        for name, prediction in forecasts.items():
            frame = metadata[["episode_id", "origin_id", "dataset_id", "family_id"]].copy()
            frame["model_id"], frame["method"] = model, name
            frame["mae"] = np.abs(prediction - truth).mean(axis=(1, 2))
            frame["mse"] = np.square(prediction - truth).mean(axis=(1, 2))
            frame["teacher_mse"] = np.square(prediction - teacher).mean(axis=(1, 2))
            rows.append(frame)
            if name == "learned_teacher_rank3":
                old = pd.read_parquet(directory / "episode_results.parquet")
                old = (
                    old[old.method == "clean_forecast_mse_rank3"]
                    .set_index("episode_id")
                    .loc[frame.episode_id]
                )
                np.testing.assert_allclose(
                    frame[["mae", "mse"]], old[["mae", "mse"]], rtol=0, atol=1e-10
                )
        path = output / f"{model}_diagnostic_forecasts.npz"
        _save_npz(
            path,
            methods=np.asarray(list(forecasts)),
            point_z=np.stack(list(forecasts.values())),
            episode_ids=metadata.episode_id.to_numpy(dtype=str),
            actions=np.asarray(actions),
            triple_indices=subsets,
            **choices,
        )
        records.append(
            {
                "model_id": model,
                "path": path.name,
                "sha256": file_sha256(path),
                "teacher_objective_strict_improvements": int(
                    (joint_loss < individual_loss - 1e-10).sum()
                ),
            }
        )
    frame = pd.concat(rows, ignore_index=True)
    family = (
        frame.groupby(["model_id", "method", "family_id"])[["mae", "mse", "teacher_mse"]]
        .mean()
        .reset_index()
    )
    summary = (
        family.groupby(["model_id", "method"])[["mae", "mse", "teacher_mse"]].mean().reset_index()
    )
    frame.to_parquet(output / "episode_metrics.parquet", index=False)
    family.to_csv(output / "family_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "models": records,
            "method_information": roles,
            "sources": {
                name: file_sha256(getattr(args, name) / "manifest.json")
                for name in ("accuracy_root", "teacher_root", "portfolio_root")
            },
            "script_sha256": file_sha256(Path(__file__)),
            "new_forecaster_calls": 0,
            "new_selector_fits": 0,
            "limits": "source development diagnostics only; teacher and future oracles are unavailable at deployment; no natural confirmation rerun",
        },
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
