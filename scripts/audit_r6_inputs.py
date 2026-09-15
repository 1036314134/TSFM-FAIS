"""Validate R6 input values, source-prefix fitting and both future horizons."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort-root", "prepared-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if (args.output_root / "manifest.json").exists():
        raise ValueError("preserve the completed R6 input audit")
    cohort = json.loads((args.cohort_root / "manifest.json").read_text(encoding="utf-8"))
    prep = json.loads((args.prepared_root / "manifest.json").read_text(encoding="utf-8"))
    if prep["status"] != "completed" or prep["identity"]["cohort_sha256"] != file_sha256(
        args.cohort_root / "manifest.json"
    ):
        raise ValueError("the prepared inputs and cohort disagree")
    expected = {row["episode_id"]: row for row in cohort["tasks"]}
    if len(prep["episodes"]) != len(expected) or {
        row["episode_id"] for row in prep["episodes"]
    } != set(expected):
        raise ValueError("R6 input coverage changed")
    scaler_path = args.prepared_root / "standardizers.json"
    if file_sha256(scaler_path) != prep["standardizers_sha256"]:
        raise ValueError("scoring standardizers changed")
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(scaler_path.read_text(encoding="utf-8"))
    }
    checked, source_arrays, prefixes = 0, {}, {}
    for source in cohort["sources"]:
        key = (source["dataset_id"], source["item_id"])
        path = Path(source["path"])
        if file_sha256(path) != source["sha256"]:
            raise ValueError("an original source array changed")
        values = np.load(path, mmap_mode="r")
        source_arrays[key], prefixes[key] = values, source["prefix_end"]
        prefix = values[: source["prefix_end"]]
        observed = np.isfinite(prefix)
        mean = np.array([prefix[observed[:, i], i].mean() for i in range(values.shape[1])])
        std = np.array([prefix[observed[:, i], i].std(ddof=0) for i in range(values.shape[1])])
        scaler = scalers[key]
        np.testing.assert_allclose(scaler["mean"], mean, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            scaler["scale"], np.where(std <= 1e-12, 1.0, std), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_array_equal(scaler["observed_count"], observed.sum(0))
        np.testing.assert_array_equal(scaler["constant"], std <= 1e-12)
        if scaler["prefix_end"] != source["prefix_end"]:
            raise ValueError("a standardizer used a different prefix")
    failed_fits = []
    for dataset in prep["datasets"]:
        directory = args.prepared_root / "imputers" / dataset["dataset_id"]
        batch_path = directory / "training_batch.npz"
        if file_sha256(batch_path) != dataset["training_batch_sha256"]:
            raise ValueError("a historical training batch changed")
        with np.load(batch_path, allow_pickle=False) as batch:
            if len(batch["values"]) > 64 or batch["values"].shape[1] != 96:
                raise ValueError("the registered fitting budget changed")
            for index, identifier in enumerate(batch["window_ids"].tolist()):
                item, rest = identifier.rsplit("@", 1)
                start = int(rest.split("|", 1)[0])
                key = (dataset["dataset_id"], item)
                if key not in prefixes or start + 96 > prefixes[key]:
                    raise ValueError("a fitting window extends past the prefix")
                original = source_arrays[key][start : start + 96]
                mask = batch["observed"][index]
                np.testing.assert_array_equal(batch["values"][index][mask], original[mask])
        for record in dataset["imputers"]:
            path = Path(record["path"])
            if file_sha256(path) != record["sha256"]:
                raise ValueError("a fitted imputer record changed")
            fit = json.loads(path.read_text(encoding="utf-8"))
            if fit["status"] != "fitted":
                failed_fits.append({"dataset_id": dataset["dataset_id"], **record})
            else:
                for file in fit["files"]:
                    if (
                        file_sha256(directory / record["candidate_id"] / file["path"])
                        != file["sha256"]
                    ):
                        raise ValueError("a fitted imputer changed")
    for record in prep["episodes"]:
        task = expected[record["episode_id"]]
        path = args.prepared_root / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("a prepared input artifact changed")
        original = source_arrays[(record["dataset_id"], record["item_id"])]
        origin = task["window"]["origin"]
        with np.load(path, allow_pickle=False) as saved:
            context = saved["context"]
            visible = np.isfinite(context)
            raw_context = original[origin - 96 : origin]
            np.testing.assert_array_equal(context[visible], raw_context[visible])
            if task["mechanism"] == "native":
                np.testing.assert_array_equal(context, raw_context)
            elif not np.isfinite(raw_context).all():
                raise ValueError("a synthetic task has unobserved original context truth")
            future = original[origin : origin + 192, :2]
            np.testing.assert_array_equal(saved["future"], future)
            np.testing.assert_array_equal(saved["future_observed"], np.isfinite(future))
            for horizon in (96, 192):
                counts = np.isfinite(future[:horizon]).sum(0)
                np.testing.assert_array_equal(
                    counts, task["window"]["future_observed_by_horizon"][str(horizon)]
                )
                if (counts < horizon // 2).any():
                    raise ValueError("a horizon has insufficient actual future observations")
            candidates, motm = saved["candidate_values"], saved["motm_values"]
            if (
                candidates.shape != (6, 96, original.shape[1])
                or motm.shape != context.shape
                or not np.isfinite(candidates).all()
                or not np.isfinite(motm).all()
            ):
                raise ValueError("candidate filling is incomplete or mis-shaped")
            if saved["candidate_ids"].tolist() != prep["identity"]["candidate_ids"]:
                raise ValueError("the imputer order changed")
            for values in [*candidates, motm]:
                np.testing.assert_array_equal(values[visible], context[visible])
        checked += 1
    _write_json(
        args.output_root / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "cohort_sha256": file_sha256(args.cohort_root / "manifest.json"),
            "prepared_sha256": file_sha256(args.prepared_root / "manifest.json"),
            "verified_inputs": checked,
            "verified_standardizers": len(scalers),
            "verified_horizons": [96, 192],
            "failed_imputer_fits": failed_fits,
            "limits": "input, prefix-fitting and future-observation audit; no new forecasting accuracy has been measured",
        },
    )


if __name__ == "__main__":
    main()
