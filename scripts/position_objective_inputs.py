"""Reuse the audited two-mask positional population without exposing validation targets."""

import argparse
from pathlib import Path

import numpy as np
from aligned_portfolio_io import decision_truth, decision_vectors, load_prepared_model
from latent_source_inputs import ROOT, read_json
from metric_source_gate import fit_fixed_metric
from positional_forecast_portfolio import position_inputs
from positional_portfolio_io import target_nodes

from tsfm_fais.routing.forecast_projection import (
    forecast_geometry,
    projection_targets,
    simplex_quadratic_weights,
)
from tsfm_fais.routing.utility import _family_weights
from tsfm_fais.utility_experiment import file_sha256


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    for name, path in {
        "aligned-root": "artifacts/iclr27-r5/aligned-portfolio-prepared-v001",
        "accuracy-root": "artifacts/iclr27-r4/accuracy-development-v002",
        "teacher-root": "artifacts/iclr27-r5/complete-history-projection-v001",
        "reference-study": "artifacts/iclr27-r6/positional-source-v001",
        "reference-audit": "artifacts/iclr27-r6/positional-source-audit-v001",
        "protocol": "docs/iclr2027/R17_POSITION_OBJECTIVE_PROTOCOL.md",
    }.items():
        parser.add_argument("--" + name, type=Path, default=ROOT / path)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def checked_sources(args):
    reference = read_json(args.reference_study / "manifest.json")
    audit = read_json(args.reference_audit / "manifest.json")
    if audit["status"] != "completed" or audit["study_sha256"] != file_sha256(
        args.reference_study / "manifest.json"
    ):
        raise ValueError("complete the original positional audit first")
    for key, path in (
        ("aligned_sha256", args.aligned_root / "manifest.json"),
        ("accuracy_sha256", args.accuracy_root / "manifest.json"),
        ("teacher_sha256", args.teacher_root / "manifest.json"),
        ("model_module_sha256", ROOT / "scripts/positional_forecast_portfolio.py"),
        ("io_module_sha256", ROOT / "scripts/positional_portfolio_io.py"),
    ):
        if reference["identity"][key] != file_sha256(path):
            raise ValueError("the original positional source definition changed")
    return reference


def load_inputs(args, model_id, *, validation=False):
    prep = read_json(args.aligned_root / "manifest.json")
    accuracy = read_json(args.accuracy_root / "manifest.json")
    teachers = read_json(args.teacher_root / "manifest.json")
    info, original, arrays = load_prepared_model(args.aligned_root, prep, model_id)
    point_path = args.accuracy_root / f"{model_id}_point_z.npy"
    if file_sha256(point_path) != accuracy["prediction_arrays"][point_path.name]:
        raise ValueError("the original candidate point bank changed")
    bank = np.load(point_path, mmap_mode="r")[
        :, [accuracy["action_orders"][model_id].index(name) for name in info["actions"]]
    ]
    vectors = decision_vectors(original, bank)
    frame, base, points = target_nodes(
        original, arrays["features"][:, :7, :33], vectors, joint=model_id == "chronos2"
    )
    inputs = position_inputs(base, points)
    training = np.flatnonzero(frame.split.to_numpy() == "train")
    held = np.flatnonzero(frame.split.to_numpy() == "validation")
    if (
        len(training) != 11880
        or len(held) != 3744
        or frame.iloc[training].origin_id.nunique() != 165
        or frame.iloc[held].origin_id.nunique() != 52
    ):
        raise ValueError("the original two-mask source population changed")
    if set(frame.iloc[training].origin_id) & set(frame.iloc[held].origin_id):
        raise ValueError("a history spans training and validation")
    entry = next(row for row in teachers["models"] if row["model_id"] == model_id)
    teacher_path = args.teacher_root / entry["teacher_file"]
    if file_sha256(teacher_path) != entry["teacher_sha256"]:
        raise ValueError("the original complete-history teacher changed")
    truth_path = args.accuracy_root / "truth_z.npy"
    if file_sha256(truth_path) != accuracy["prediction_arrays"][truth_path.name]:
        raise ValueError("the original source outcome changed")
    labels = {name: np.full(inputs["median"].shape, np.nan) for name in ("teacher", "future")}
    allowed = np.arange(len(frame)) if validation else training
    labels["teacher"][training] = decision_truth(
        frame.iloc[training], np.load(teacher_path, mmap_mode="r")
    )
    labels["future"][allowed] = decision_truth(
        frame.iloc[allowed], np.load(truth_path, mmap_mode="r")
    )
    return frame, inputs, points, labels, info["actions"]


def fixed_control(frame, points, future, indices, objective):
    weights = _family_weights(frame.iloc[indices])
    if objective == "joint":
        return fit_fixed_metric(points[indices], future[indices], weights, "joint")
    if objective != "mse":
        raise ValueError("unknown positional fixed objective")
    weights = weights / weights.sum()
    selected, target = points[indices], future[indices]
    g = np.einsum("n,nab->ab", weights, forecast_geometry(selected)[3])
    b = np.einsum("n,na->a", weights, projection_targets(selected, target)["raw_projection"])
    probability = simplex_quadratic_weights(g[None], b[None])[0][0]
    prediction = (selected * probability[None, :, None]).sum(1)
    error = prediction - target
    gradient = (
        2
        * np.einsum("naq,nq,n->a", selected - np.median(selected, axis=1)[:, None], error, weights)
        / selected.shape[2]
    )
    gap = float((gradient @ probability - gradient.min()) / max(abs(gradient).max(), 1.0))
    if gap > 1e-7:
        raise ValueError("fixed MSE optimality failed")
    singles = np.einsum("n,na->a", weights, ((selected - target[:, None]) ** 2).mean(2))
    return {
        "weights": probability.tolist(),
        "single_index": int(singles.argmin()),
        "single_objectives": singles.tolist(),
        "objective": float(weights @ (error**2).mean(1)),
        "optimality_gap": max(gap, 0.0),
    }


def average_positions(predictions, inputs, indices):
    median = inputs["median"][indices]
    return np.clip(
        median + np.mean(np.stack(predictions) - median[None], axis=0),
        inputs["lower"][indices],
        inputs["upper"][indices],
    )
