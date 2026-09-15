"""Audited inputs and observed-target losses for the native-source extension."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aligned_portfolio_io import decision_truth, decision_vectors, load_prepared_model  # noqa: E402
from evaluate_r6_geometry_gates import legacy_inputs, r6_inputs  # noqa: E402
from r6_policy_inputs import pack_gate_features  # noqa: E402

from tsfm_fais.utility_experiment import file_sha256  # noqa: E402


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def holdout_group(family):
    return "singapore" if family in ("sg_pm25", "sg_weather") else family


def input_arguments(parser):
    defaults = {
        "aligned-root": "artifacts/iclr27-r5/aligned-portfolio-prepared-v001",
        "accuracy-root": "artifacts/iclr27-r4/accuracy-development-v002",
        "r6-prepared": "artifacts/iclr27-r6/confirmation-v001/prepared",
        "r6-policy": "artifacts/iclr27-r6/policy-results-v002",
        "r6-audit": "artifacts/iclr27-r6/policy-audit-v002",
        "legacy-input": "artifacts/iclr27-r5/native-confirmation-v001",
        "legacy-results": "artifacts/iclr27-r6/legacy-native-gates-v001",
        "future-control": "artifacts/iclr27-r6/source-future-control-v001",
        "future-cv": "artifacts/iclr27-r6/source-future-cv-v001",
        "comparison-root": "artifacts/iclr27-r6/positional-transfer-v001",
    }
    for name, path in defaults.items():
        parser.add_argument("--" + name, type=Path, default=ROOT / path)


def checked_source_bindings(args):
    result = {}
    for name in (
        "aligned_root",
        "accuracy_root",
        "r6_prepared",
        "r6_policy",
        "r6_audit",
        "legacy_input",
        "legacy_results",
        "future_control",
        "future_cv",
        "comparison_root",
    ):
        root = getattr(args, name)
        paths = [root / "manifest.json"]
        if name == "r6_policy":
            paths = [root / model / "manifest.json" for model in ("chronos2", "timesfm2p5")]
        elif name == "legacy_input":
            paths = [root / "prepared/manifest.json"]
        for path in paths:
            result[str(path.relative_to(ROOT))] = file_sha256(path)
    return result


def observed_weights(mask, *, joint, minimum=1):
    mask = np.asarray(mask, bool)
    targets = 2 if joint else 1
    if mask.ndim != 2 or mask.shape[1] % targets:
        raise ValueError("observation masks must preserve target coordinates")
    shaped = mask.reshape(len(mask), -1, targets)
    counts = shaped.sum(1)
    if (counts < minimum).any():
        raise ValueError("insufficient original future observations")
    return (shaped / counts[:, None] / targets).reshape(mask.shape)


def observed_errors(prediction, truth, mask, *, joint):
    prediction, truth, mask = (
        np.asarray(prediction, float),
        np.asarray(truth, float),
        np.asarray(mask, bool),
    )
    if prediction.shape != truth.shape or truth.shape != mask.shape:
        raise ValueError("predictions, labels and observation masks must align")
    if not np.isfinite(prediction).all() or not np.isfinite(truth[mask]).all():
        raise ValueError("predictions and observed labels must be finite")
    weight = observed_weights(mask, joint=joint)
    delta = np.where(mask, prediction - truth, 0.0)
    return (abs(delta) * weight).sum(1), (delta**2 * weight).sum(1)


def masked_geometry(points, truth, mask, *, joint):
    points, truth, mask = (
        np.asarray(points, float),
        np.asarray(truth, float),
        np.asarray(mask, bool),
    )
    if points.ndim != 3 or points.shape[::2] != truth.shape or truth.shape != mask.shape:
        raise ValueError("candidate trajectories must align with observed future labels")
    if not np.isfinite(points).all() or not np.isfinite(truth[mask]).all():
        raise ValueError("candidate trajectories and observed labels must be finite")
    weight = observed_weights(mask, joint=joint)
    median = np.median(points, axis=1)
    change = points - median[:, None]
    residual = np.where(mask, truth - median, 0.0)
    gram = np.einsum("naq,nbq,nq->nab", change, change, weight)
    alignment = np.einsum("naq,nq,nq->na", change, residual, weight)
    return gram, alignment


def fold_indices(frame, group):
    native = frame.cohort.to_numpy() != "source"
    evaluation = np.flatnonzero(native & (frame.holdout_group.to_numpy() == group))
    training = np.flatnonzero(~native | (frame.holdout_group.to_numpy() != group))
    if not len(evaluation) or set(frame.iloc[training].family_id) & set(
        frame.iloc[evaluation].family_id
    ):
        raise ValueError("the complete evaluation family must be excluded from training")
    if set(frame.iloc[training].origin_id) & set(frame.iloc[evaluation].origin_id):
        raise ValueError("an evaluation history entered fitting")
    return training, evaluation


def native_bank_path(args, model, cohort):
    if cohort == "r6":
        root = args.r6_policy / model
        entry = next(
            row for row in read_json(root / "manifest.json")["horizons"] if row["horizon"] == 96
        )
        return root / entry["directory"] / "policy_predictions.npz"
    entries = read_json(args.legacy_results / "prediction_freeze.json")["banks"]
    return args.legacy_results / next(row["path"] for row in entries if row["model_id"] == model)


def load_inputs(args, model):
    prep, accuracy = (
        read_json(args.aligned_root / "manifest.json"),
        read_json(args.accuracy_root / "manifest.json"),
    )
    if prep["identity"]["accuracy_manifest_sha256"] != file_sha256(
        args.accuracy_root / "manifest.json"
    ):
        raise ValueError("source feature and forecast provenance changed")
    info, decisions, arrays = load_prepared_model(args.aligned_root, prep, model)
    path = args.accuracy_root / f"{model}_point_z.npy"
    truth_path = args.accuracy_root / "truth_z.npy"
    for candidate in (path, truth_path):
        if file_sha256(candidate) != accuracy["prediction_arrays"][candidate.name]:
            raise ValueError("source predictions or future labels changed")
    complete_bank = np.load(path, mmap_mode="r")
    actions = info["actions"]
    selected = np.flatnonzero(decisions.split.to_numpy() == "train")
    source = decisions.iloc[selected].copy().reset_index(drop=True)
    source["cohort"] = "source"
    source["source_episode_id"] = source["source_episode_id"].astype(str)
    if (
        source.origin_id.nunique(),
        source.family_id.nunique(),
        source.source_episode_id.nunique(),
    ) != (165, 15, 5940):
        raise ValueError("the original source training population changed")
    source_points = decision_vectors(
        source, complete_bank[:, [accuracy["action_orders"][model].index(name) for name in actions]]
    )
    source_truth = decision_truth(source, np.load(truth_path, mmap_mode="r"))
    frames = [source]
    data = {
        "features": [pack_gate_features(arrays["features"][selected, :7, :33])],
        "points": [source_points],
        "truth": [source_truth],
        "observed": [np.ones_like(source_truth, bool)],
        "median8": [decision_truth(source, np.median(complete_bank, axis=1))],
    }
    for cohort, root in (
        ("legacy_native", args.legacy_input / "prepared"),
        ("r6", args.r6_prepared),
    ):
        native_prep = read_json(root / "manifest.json")
        scaler_path = root / "standardizers.json"
        if file_sha256(scaler_path) != native_prep["standardizers_sha256"]:
            raise ValueError("original native prefix statistics changed")
        scalers = {(row["dataset_id"], row["item_id"]): row for row in read_json(scaler_path)}
        loaded = (
            r6_inputs(args, model, 96, actions, native_prep)
            if cohort == "r6"
            else legacy_inputs(args, model, actions, native_prep, scalers)
        )
        native, features, points, _, _ = loaded
        selected_episodes = {
            index
            for index, row in enumerate(native_prep["episodes"])
            if row["window"]["context_has_missing"] and row.get("panel") != "new_synthetic"
        }
        chosen = np.flatnonzero(native.episode_index.isin(selected_episodes).to_numpy())
        frame = native.iloc[chosen].copy().reset_index(drop=True)
        frame["cohort"] = cohort
        frame["source_episode_id"] = [
            native_prep["episodes"][index]["episode_id"] for index in frame.episode_index
        ]
        frame["split"] = "native_group_cv"
        labels = np.full((len(native_prep["episodes"]), 96, 2), np.nan)
        observed = np.zeros(labels.shape, bool)
        for index in sorted(selected_episodes):
            row = native_prep["episodes"][index]
            episode_path = root / row["path"]
            if file_sha256(episode_path) != row["sha256"]:
                raise ValueError("an original native scoring window changed")
            with np.load(episode_path, allow_pickle=False) as saved:
                truth, mask = saved["future"][:96], saved["future_observed"][:96].astype(bool)
            scaler = scalers[(row["dataset_id"], row["item_id"])]
            mean, scale = np.asarray(scaler["mean"])[:2], np.asarray(scaler["scale"])[:2]
            labels[index] = np.where(mask, (truth - mean) / scale, np.nan)
            observed[index] = mask
        with np.load(native_bank_path(args, model, cohort), allow_pickle=False) as saved:
            median8 = saved["point_z"][
                :, saved["methods"].tolist().index("forecast_median_with_motm")
            ]
        frames.append(frame)
        for name, values in (
            ("features", features[chosen]),
            ("points", points[chosen]),
            ("truth", decision_truth(frame, labels)),
            ("observed", decision_truth(frame, observed).astype(bool)),
            ("median8", decision_truth(frame, median8)),
        ):
            data[name].append(values)
    columns = [
        "episode_id",
        "source_episode_id",
        "origin_id",
        "family_id",
        "dataset_id",
        "item_id",
        "target_slot",
        "episode_index",
        "cohort",
        "split",
    ]
    frame = pd.concat([part[columns] for part in frames], ignore_index=True)
    frame["holdout_group"] = frame.family_id.map(holdout_group)
    if frame.episode_id.duplicated().any():
        raise ValueError("duplicated source or native decision identities")
    native = frame[frame.cohort != "source"]
    if (native.origin_id.nunique(), native.family_id.nunique(), native.holdout_group.nunique()) != (
        358,
        9,
        8,
    ):
        raise ValueError("the registered native expansion population changed")
    if set(source.family_id) & set(native.family_id) or set(source.origin_id) & set(
        native.origin_id
    ):
        raise ValueError("the original source population overlaps the native extension")
    merged = {name: np.concatenate(values) for name, values in data.items()}
    merged["features"] = pack_gate_features(merged["features"])
    observed_weights(merged["observed"], joint=model == "chronos2", minimum=48)
    return frame, merged, actions


def summarize_scores(scores):
    keys = [
        "model_id",
        "method",
        "cohort",
        "source_episode_id",
        "origin_id",
        "family_id",
        "dataset_id",
        "item_id",
        "holdout_group",
    ]
    episodes = scores.groupby(keys)[["mae", "mse"]].mean().reset_index()
    items = (
        episodes.groupby(
            ["model_id", "method", "family_id", "dataset_id", "item_id", "holdout_group"]
        )[["mae", "mse"]]
        .mean()
        .reset_index()
    )
    datasets = (
        items.groupby(["model_id", "method", "family_id", "dataset_id", "holdout_group"])[
            ["mae", "mse"]
        ]
        .mean()
        .reset_index()
    )
    families = (
        datasets.groupby(["model_id", "method", "family_id", "holdout_group"])[["mae", "mse"]]
        .mean()
        .reset_index()
    )
    summary = families.groupby(["model_id", "method"])[["mae", "mse"]].mean().reset_index()
    groups = (
        families.groupby(["model_id", "method", "holdout_group"])[["mae", "mse"]]
        .mean()
        .reset_index()
    )
    group_summary = groups.groupby(["model_id", "method"])[["mae", "mse"]].mean().reset_index()
    return episodes, families, summary, group_summary
