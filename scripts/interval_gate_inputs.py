"""Restore the two existing quantile fields while preserving point-only inputs."""

import numpy as np
from aligned_portfolio_io import decision_vectors, load_prepared_model
from audit_source_quantiles import read_json

from tsfm_fais.routing.forecast_response import FORECAST_FEATURES
from tsfm_fais.utility_experiment import file_sha256

HAS_INDEX = FORECAST_FEATURES.index("response.has_quantiles")
WIDTH_INDEX = FORECAST_FEATURES.index("response.interval_width")


def add_intervals(base, decisions, joint_width, target_width):
    base = np.asarray(base, np.float32)
    if base.shape != (len(decisions), 7, 33) or np.any(base[:, :, [HAS_INDEX, WIDTH_INDEX]] != 0):
        raise ValueError("the comparison requires the original point-only feature fields")
    widths = []
    for row in decisions.itertuples(index=False):
        if row.target_slot == -1:
            widths.append(joint_width[row.episode_index])
        elif row.target_slot in (0, 1):
            widths.append(target_width[row.episode_index, :, row.target_slot])
        else:
            raise ValueError("unknown original target slot")
    widths = np.stack(widths)
    if not np.isfinite(widths).all():
        raise ValueError("quantile intervals are incomplete; do not drop source cases")
    result = np.array(base, copy=True, order="C")
    result[:, :, HAS_INDEX] = 1.0
    result[:, :, WIDTH_INDEX] = widths
    keep = [index for index in range(33) if index not in (HAS_INDEX, WIDTH_INDEX)]
    np.testing.assert_array_equal(result[:, :, keep], base[:, :, keep])
    return result


def load_interval_inputs(args, model_id):
    preparation = read_json(args.aligned_root / "manifest.json")
    accuracy = read_json(args.accuracy_root / "manifest.json")
    audit = read_json(args.quantile_root / "manifest.json")
    if audit["status"] != "completed" or audit["identity"][
        "accuracy_manifest_sha256"
    ] != file_sha256(args.accuracy_root / "manifest.json"):
        raise ValueError("source quantiles must be fully checked against the current point bank")
    info, decisions, arrays = load_prepared_model(args.aligned_root, preparation, model_id)
    record = next(row for row in audit["models"] if row["model_id"] == model_id)
    if (
        record["actions"] != info["actions"]
        or record["nonfinite_coordinates"] != 0
        or record["maximum_point_difference"] != 0
    ):
        raise ValueError("the checked quantiles do not cover the fixed candidate forecasts")
    for name, digest in record["files"].items():
        if file_sha256(args.quantile_root / name) != digest:
            raise ValueError("a checked source quantile array changed")
    joint = np.load(args.quantile_root / f"{model_id}_width_joint.npy", allow_pickle=False)
    target = np.load(args.quantile_root / f"{model_id}_width_by_target.npy", allow_pickle=False)
    quantiles = np.load(args.quantile_root / f"{model_id}_quantiles_z.npy", mmap_mode="r")
    spans = quantiles[..., 2] - quantiles[..., 0]
    reconstructed_joint = np.clip(spans.mean((2, 3)), -1e8, 1e8)
    reconstructed_target = np.clip(spans.mean(2), -1e8, 1e8)
    np.testing.assert_allclose(joint, reconstructed_joint, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(target, reconstructed_target, rtol=1e-12, atol=1e-12)
    width_difference = max(
        float(abs(joint - reconstructed_joint).max()),
        float(abs(target - reconstructed_target).max()),
    )
    base = np.ascontiguousarray(arrays["features"][:, :7, :33])
    features = add_intervals(base, decisions, joint, target)
    path = args.accuracy_root / f"{model_id}_point_z.npy"
    if file_sha256(path) != accuracy["prediction_arrays"][path.name]:
        raise ValueError("the source point bank changed")
    bank = np.load(path, mmap_mode="r")[
        :, [accuracy["action_orders"][model_id].index(name) for name in info["actions"]]
    ]
    return (
        info,
        decisions,
        arrays,
        base,
        features,
        decision_vectors(decisions, bank),
        width_difference,
    )
