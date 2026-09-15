"""Load audited observed and known-process sources with validation labels withheld."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from aligned_portfolio_io import decision_truth
from latent_source_inputs import ROOT, read_json
from pool_gate_inputs import load_pool_inputs
from train_latent_source_gates import aggregate

from tsfm_fais.routing.forecast_projection import forecast_geometry
from tsfm_fais.utility_experiment import file_sha256


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    for name, path in {
        "pool-root": "artifacts/iclr27-r12/motm-pool-inputs-v001",
        "reference-root": "artifacts/iclr27-r12/motm-pool-source-v001",
        "reference-audit": "artifacts/iclr27-r12/motm-pool-audit-v001",
        "accuracy-root": "artifacts/iclr27-r4/accuracy-development-v002",
        "synthetic-root": "artifacts/iclr27-r16/conditional-source-v001",
        "synthetic-audit": "artifacts/iclr27-r16/conditional-source-audit-v001",
        "protocol": "docs/iclr2027/R16_TRAINING_EXECUTION_PROTOCOL.md",
    }.items():
        parser.add_argument("--" + name, type=Path, default=ROOT / path)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def checked_sources(args):
    original = read_json(args.reference_root / "manifest.json")
    audit = read_json(args.reference_audit / "manifest.json")
    synthetic = read_json(args.synthetic_root / "manifest.json")
    audit_synthetic = read_json(args.synthetic_audit / "manifest.json")
    accuracy = read_json(args.accuracy_root / "manifest.json")
    if (
        audit["status"] != "completed"
        or audit_synthetic["status"] != "completed"
        or audit["study_sha256"] != file_sha256(args.reference_root / "manifest.json")
        or audit_synthetic["collection_sha256"]
        != file_sha256(args.synthetic_root / "manifest.json")
        or original["identity"]["pool_sha256"] != file_sha256(args.pool_root / "manifest.json")
        or accuracy["prediction_arrays"]["truth_z.npy"]
        != file_sha256(args.accuracy_root / "truth_z.npy")
    ):
        raise ValueError("audited source bindings changed")
    for entry in synthetic["files"]:
        if file_sha256(args.synthetic_root / entry["path"]) != entry["sha256"]:
            raise ValueError("an audited synthetic source file changed")
    return original


def load_training_inputs(args, model_id, *, validation=False):
    manifest, old_frame, original = load_pool_inputs(args.pool_root, model_id)
    with np.load(
        args.pool_root / model_id / manifest["episodes"][0]["path"], allow_pickle=False
    ) as source:
        actions = source["actions"].tolist()
    root = args.synthetic_root / model_id
    frame = pd.read_parquet(root / "decisions.parquet")
    with np.load(root / "inputs.npz", allow_pickle=False) as source:
        x, p = source["features"], source["vectors"]
        if source["actions"].tolist() != actions:
            raise ValueError("original and synthetic candidate pools differ")
    old_truth = np.full((len(old_frame), p.shape[-1]), np.nan)
    old_indices = (
        np.arange(len(old_frame))
        if validation
        else np.flatnonzero(old_frame.split.to_numpy() == "train")
    )
    old_truth[old_indices] = decision_truth(
        old_frame.iloc[old_indices], np.load(args.accuracy_root / "truth_z.npy", mmap_mode="r")
    )
    new_labels = {}
    allowed = np.ones(len(frame), bool) if validation else frame.split.to_numpy() == "train"
    with np.load(root / "labels.npz", allow_pickle=False) as source:
        for name in ("future", "conditional_mean", "conditional_variance"):
            new_labels[name] = np.full_like(source[name], np.nan)
            new_labels[name][allowed] = source[name][allowed]
    combined = pd.concat(
        [
            old_frame.assign(source_population="original"),
            frame.assign(source_population="known_process"),
        ],
        ignore_index=True,
    )
    if combined.episode_id.duplicated().any() or combined.origin_id.nunique() != 457:
        raise ValueError("the registered source histories overlap or changed")
    data = {
        "features": np.concatenate([original["features"], x]),
        "points": np.concatenate([original["vectors"], p]),
        "truth": np.concatenate([old_truth, new_labels["future"]]),
        "mean": np.concatenate([old_truth.copy(), new_labels["conditional_mean"]]),
        "variance": np.concatenate([np.zeros_like(old_truth), new_labels["conditional_variance"]]),
        "simulated": np.r_[np.zeros(len(old_frame), bool), np.ones(len(frame), bool)],
    }
    data["gram"] = forecast_geometry(data["points"])[3]
    if not validation:
        held = combined.split.to_numpy() == "validation"
        if not np.isnan(data["truth"][held]).all() or not np.isnan(data["mean"][held]).all():
            raise ValueError("validation supervision entered training arrays")
    return combined, data, actions


def aggregate_panels(scores):
    expected = scores[scores.panel == "known_process"].copy()
    if len(expected):
        expected["mae"] = expected.expected_mae
        expected["mse"] = expected.expected_mse
        expected["panel"] = "known_process_expected"
        scores = pd.concat([scores, expected], ignore_index=True)
    tables = [[], [], []]
    for panel, group in scores.groupby("panel"):
        for parts, table in zip(tables, aggregate(group), strict=True):
            parts.append(table.assign(panel=panel))
    return tuple(pd.concat(parts, ignore_index=True) for parts in tables)
