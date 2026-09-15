"""Verify simulated source information boundaries, model queries and exact-risk labels."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from collect_conditional_supervision import visible_inputs
from conditional_future import condition_history, draw_history, psd_root, seasonal
from conditional_supervision import supervision_scenarios, supervision_seed
from latent_source_inputs import ROOT, read_json
from replay_preforecast_student import assemble_selected_context

from tsfm_fais.utility_experiment import _write_json, file_sha256


def independent_labels(model, clean, context, phase, seed, center, scale):
    mean, covariance = condition_history(model, context, phase, scalar=True)
    means, variances = [], []
    for _ in range(96):
        mean = model["a"] @ mean
        covariance = model["a"] @ covariance @ model["a"].T + model["q"]
        means.append(mean.copy())
        variances.append(np.diag(covariance).copy())
    seasonal_future = seasonal(model, np.arange(phase + 96, phase + 192))
    expected_mean = (np.asarray(means)[:, :2] + seasonal_future[:, :2] - center[:2]) / scale[:2]
    expected_variance = np.asarray(variances)[:, :2] / scale[:2] ** 2
    rng = np.random.default_rng(seed)
    rng.standard_normal((1, len(model["a"])))
    state = (clean[-1] - seasonal(model, [phase + 95])[0])[None]
    root = psd_root(model["q"])
    future = []
    for t in range(96):
        state = state @ model["a"].T + rng.standard_normal(state.shape) @ root.T
        future.append(state[0] + seasonal_future[t])
    future = (np.asarray(future)[:, :2] - center[:2]) / scale[:2]
    return future, expected_mean, expected_variance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, path in {
        "prepared-root": "artifacts/iclr27-r16/conditional-inputs-v001",
        "forecast-root": "artifacts/iclr27-r16/conditional-forecasts-v001",
        "collection-root": "artifacts/iclr27-r16/conditional-source-v001",
        "protocol": "docs/iclr2027/R16_CONDITIONAL_SUPERVISION_PROTOCOL.md",
    }.items():
        parser.add_argument("--" + name, type=Path, default=ROOT / path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed source audits")
    prep = read_json(args.prepared_root / "manifest.json")
    forecast = read_json(args.forecast_root / "manifest.json")
    collection = read_json(args.collection_root / "manifest.json")
    if any(item["status"] != "completed" for item in (prep, forecast, collection)):
        raise ValueError("finish all source collection stages first")
    checks = [
        (prep["identity"]["script_sha256"], ROOT / "scripts/prepare_conditional_supervision.py"),
        (prep["identity"]["process_module_sha256"], ROOT / "scripts/conditional_future.py"),
        (
            prep["identity"]["supervision_module_sha256"],
            ROOT / "scripts/conditional_supervision.py",
        ),
        (
            forecast["identity"]["script_sha256"],
            ROOT / "scripts/forecast_conditional_supervision.py",
        ),
        (forecast["identity"]["prepared_sha256"], args.prepared_root / "manifest.json"),
        (
            collection["identity"]["script_sha256"],
            ROOT / "scripts/collect_conditional_supervision.py",
        ),
        (collection["identity"]["prepared_sha256"], args.prepared_root / "manifest.json"),
        (collection["identity"]["forecast_sha256"], args.forecast_root / "manifest.json"),
        (collection["identity"]["features_module_sha256"], ROOT / "scripts/pool_gate_inputs.py"),
        (
            collection["identity"]["supervision_module_sha256"],
            ROOT / "scripts/conditional_supervision.py",
        ),
    ]
    checks.extend(
        (item["identity"]["protocol_sha256"], args.protocol)
        for item in (prep, forecast, collection)
    )
    if any(value != file_sha256(path) for value, path in checks):
        raise ValueError("a source definition or source artifact changed")
    for entry in collection["files"]:
        if file_sha256(args.collection_root / entry["path"]) != entry["sha256"]:
            raise ValueError("a source label or visible-input file changed")
    models = supervision_scenarios()
    serial = [
        {
            name: value.tolist() if isinstance(value, np.ndarray) else value
            for name, value in model.items()
        }
        for model in models
    ]
    if prep["identity"]["models"] != serial or len(prep["imputer_fits"]) != 24:
        raise ValueError("the source processes or imputer population changed")
    for generator, model in enumerate(models):
        entry = prep["prefixes"][generator]
        path = args.prepared_root / entry["path"]
        prefix = draw_history(model, 6144, 0, supervision_seed(0, generator))
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("a source calibration prefix changed")
        np.testing.assert_array_equal(np.load(path), prefix)
        np.testing.assert_allclose(prefix.mean(0), entry["mean"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(prefix.std(0), entry["scale"], rtol=1e-12, atol=1e-12)
    for entry in prep["imputer_fits"]:
        path = Path(entry["path"])
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("an imputer calibration record changed")
        fit = read_json(path)
        if fit["status"] != "fitted" or fit["training_windows"] != 64:
            raise ValueError("a prescribed calibration fit did not complete")
        for item in fit["files"]:
            if file_sha256(path.parent / entry["candidate_id"] / item["path"]) != item["sha256"]:
                raise ValueError("a calibrated imputer artifact changed")
    if len(prep["episodes"]) != 1200 or len({row["origin_id"] for row in prep["episodes"]}) != 240:
        raise ValueError("source histories are incomplete")
    queries, label_cache, factual_cache = set(), {}, {}
    maximum_label_difference = 0.0
    for model_entry in forecast["models"]:
        model_id = model_entry["model_id"]
        path = args.forecast_root / model_entry["path"]
        if file_sha256(path) != model_entry["sha256"]:
            raise ValueError("a source forecaster manifest changed")
        manifest = read_json(path)
        reference = read_json(
            ROOT / f"artifacts/iclr27-r14/conditional-forecasts-v001/{model_id}/manifest.json"
        )
        if manifest["parameter_sha256"] != reference["parameter_sha256"]:
            raise ValueError("source prediction model parameters differ from the frozen reference")
        forecasts = {row["episode_id"]: row for row in manifest["episodes"]}
        root = args.collection_root / model_id
        frame = pd.read_parquet(root / "decisions.parquet")
        with np.load(root / "inputs.npz", allow_pickle=False) as source:
            features, vectors, names = (
                source["features"],
                source["vectors"],
                source["actions"].tolist(),
            )
            if set(source.files) != {"features", "vectors", "actions"}:
                raise ValueError("labels or process parameters were added to visible inputs")
        with np.load(root / "labels.npz", allow_pickle=False) as source:
            labels = {name: source[name] for name in source.files}
        width = 1 if model_id == "chronos2" else 2
        if features.shape != (1200 * width, 8, 97) or frame.episode_id.duplicated().any():
            raise ValueError("the observed feature population changed")
        np.testing.assert_array_equal(features[:, :, 33:], np.zeros_like(features[:, :, 33:]))
        for split, histories in (("train", 192), ("validation", 48)):
            if frame[frame.split == split].origin_id.nunique() != histories:
                raise ValueError("training and validation histories changed")
        if set(frame[frame.split == "train"].origin_id) & set(
            frame[frame.split == "validation"].origin_id
        ):
            raise ValueError("training and validation share a history")
        for index, row in enumerate(prep["episodes"]):
            model = models[row["generator"]]
            expected_split = "train" if row["history"] < 16 else "validation"
            if row["split"] != expected_split or row["future_seed"] != supervision_seed(
                5, row["generator"], row["history"]
            ):
                raise ValueError("a label seed or source split changed")
            path = args.prepared_root / row["path"]
            if file_sha256(path) != row["sha256"]:
                raise ValueError("an observed input changed")
            point_record = forecasts[row["episode_id"]]
            point_path = args.forecast_root / model_id / point_record["path"]
            if file_sha256(point_path) != point_record["sha256"]:
                raise ValueError("a frozen candidate prediction changed")
            prefix = prep["prefixes"][row["generator"]]
            center, scale = np.asarray(prefix["mean"]), np.asarray(prefix["scale"])
            with (
                np.load(path, allow_pickle=False) as saved,
                np.load(point_path, allow_pickle=False) as prediction,
            ):
                if any(
                    key in saved.files
                    for key in ("future", "conditional_mean", "conditional_variance")
                ):
                    raise ValueError("future labels existed before the forecast freeze")
                clean = draw_history(
                    model, 96, row["phase"], supervision_seed(2, row["generator"], row["history"])
                )
                context = clean.copy()
                rng = np.random.default_rng(
                    supervision_seed(3, row["generator"], row["history"], row["condition"])
                )
                if row["mechanism"] == "random_point":
                    context[rng.random(context.shape) < row["missing_rate"]] = np.nan
                elif row["mechanism"] == "tail_block":
                    context[-int(round(96 * row["missing_rate"])) :] = np.nan
                np.testing.assert_array_equal(clean, saved["clean_context"])
                np.testing.assert_array_equal(context, saved["context"])
                mean, covariance = condition_history(model, context, row["phase"], scalar=True)
                np.testing.assert_allclose(mean, saved["posterior_mean"], rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(
                    covariance, saved["posterior_covariance"], rtol=1e-10, atol=1e-10
                )
                candidates = np.concatenate([saved["candidate_values"], saved["motm_values"][None]])
                actions = [*saved["candidate_ids"].tolist(), "motm_reference"]
                for candidate in candidates:
                    np.testing.assert_array_equal(
                        candidate[np.isfinite(context)], context[np.isfinite(context)]
                    )
                if prediction["actions"].tolist() != names:
                    raise ValueError("candidate ordering changed")
                for action, key, point in zip(
                    names, prediction["query_keys"].tolist(), prediction["point_z"], strict=True
                ):
                    effective = assemble_selected_context(
                        context,
                        candidates,
                        actions,
                        [action] if width == 1 else [action, action],
                        [0, 1],
                        joint=width == 1,
                    )
                    effective = np.asarray(
                        effective if width == 1 else effective[:, :2], np.float32
                    ).copy(order="C")
                    effective[np.isnan(effective)] = np.nan
                    if (
                        hashlib.sha256(
                            str(effective.shape).encode() + effective.tobytes()
                        ).hexdigest()
                        != key
                    ):
                        raise ValueError("a source query used a different visible input")
                    with np.load(
                        args.forecast_root / model_id / "queries" / f"{key}.npz", allow_pickle=False
                    ) as query:
                        np.testing.assert_array_equal(query["effective_input"], effective)
                        np.testing.assert_array_equal(
                            (query["point"] - center[:2]) / scale[:2], point
                        )
                        if str(query["parameter_sha256"]) != manifest["parameter_sha256"]:
                            raise ValueError("a query used changed prediction parameters")
                    queries.add((model_id, key))
                decisions, x, p = visible_inputs(
                    row,
                    index,
                    context,
                    saved["candidate_values"],
                    saved["candidate_ids"].tolist(),
                    saved["native_coverage"],
                    saved["motm_values"],
                    json.loads(str(saved["motm_diagnostics"])),
                    prediction["point_z"],
                    names,
                    center,
                    scale,
                    model_id,
                    model["period"],
                )
                if row["mechanism"] == "complete":
                    np.testing.assert_array_equal(
                        prediction["point_z"], np.repeat(prediction["point_z"][:1], 8, 0)
                    )
            selection = slice(index * width, (index + 1) * width)
            pd.testing.assert_frame_equal(
                decisions, frame.iloc[selection].reset_index(drop=True), check_dtype=False
            )
            np.testing.assert_array_equal(x, features[selection])
            np.testing.assert_array_equal(p, vectors[selection])
            if row["episode_id"] not in label_cache:
                label_cache[row["episode_id"]] = independent_labels(
                    model, clean, context, row["phase"], row["future_seed"], center, scale
                )
            expected = label_cache[row["episode_id"]]
            if row["origin_id"] in factual_cache:
                np.testing.assert_array_equal(expected[0], factual_cache[row["origin_id"]])
            else:
                factual_cache[row["origin_id"]] = expected[0]
            for name, values in zip(
                ("future", "conditional_mean", "conditional_variance"), expected, strict=True
            ):
                rebuilt = np.stack(
                    [
                        values.reshape(-1) if slot == -1 else values[:, slot]
                        for slot in decisions.target_slot
                    ]
                )
                maximum_label_difference = max(
                    maximum_label_difference, float(abs(rebuilt - labels[name][selection]).max())
                )
                np.testing.assert_allclose(rebuilt, labels[name][selection], rtol=1e-10, atol=1e-10)
        print(f"{model_id}: all source inputs, labels and query boundaries verified", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "collection_sha256": file_sha256(args.collection_root / "manifest.json"),
            "verified_observed_inputs": 1200,
            "verified_decisions": 3600,
            "training_histories": 192,
            "validation_histories": 48,
            "verified_factual_future_groups": len(factual_cache),
            "verified_effective_forecast_inputs": len(queries),
            "maximum_label_reconstruction_difference": maximum_label_difference,
            "new_forecaster_calls": 0,
            "limits": "new known-process source collection only; no learned-policy accuracy or novelty claim",
        },
    )


if __name__ == "__main__":
    main()
