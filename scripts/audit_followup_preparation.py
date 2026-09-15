"""Check actual follow-up inputs against frozen raw trajectories and prefix boundaries."""

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
        raise ValueError("preserve the completed input audit")
    cohort = json.loads((args.cohort_root / "manifest.json").read_text(encoding="utf-8"))
    prep = json.loads((args.prepared_root / "manifest.json").read_text(encoding="utf-8"))
    if prep["status"] != "completed" or prep["identity"]["cohort_sha256"] != file_sha256(
        args.cohort_root / "manifest.json"
    ):
        raise ValueError("input preparation does not match the frozen cohort")
    expected = {row["episode_id"]: row for row in cohort["tasks"]}
    if len(prep["episodes"]) != len(expected) or {
        row["episode_id"] for row in prep["episodes"]
    } != set(expected):
        raise ValueError("follow-up task coverage changed")
    scaler_path = args.prepared_root / "standardizers.json"
    if file_sha256(scaler_path) != prep["standardizers_sha256"]:
        raise ValueError("a standardizer record changed")
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(scaler_path.read_text(encoding="utf-8"))
    }
    dataset_records = {row["dataset_id"]: row for row in prep["datasets"]}
    checked_datasets, cases, failed_fits = set(), 0, []
    for source in cohort["sources"]:
        key = (source["dataset_id"], source["item_id"])
        raw_path = args.cohort_root / source["path"]
        if file_sha256(raw_path) != source["sha256"]:
            raise ValueError("a raw trajectory changed")
        raw = np.load(raw_path, mmap_mode="r")
        prefix = raw[: source["prefix_end"]]
        observed = np.isfinite(prefix)
        mean = np.array([prefix[observed[:, i], i].mean() for i in range(raw.shape[1])])
        std = np.array([prefix[observed[:, i], i].std(ddof=0) for i in range(raw.shape[1])])
        constants = std <= 1e-12
        scale = np.where(constants, 1.0, std)
        scaler = scalers[key]
        np.testing.assert_allclose(scaler["mean"], mean, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(scaler["scale"], scale, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(scaler["constant"], constants)
        np.testing.assert_array_equal(scaler["observed_count"], observed.sum(axis=0))
        if scaler["prefix_end"] != source["prefix_end"]:
            raise ValueError("a standardizer used a different prefix")
        if key[0] not in checked_datasets:
            dataset = dataset_records[key[0]]
            directory = args.prepared_root / "imputers" / key[0]
            if file_sha256(directory / "training_batch.npz") != dataset["training_batch_sha256"]:
                raise ValueError("an imputer training batch changed")
            prefixes = {
                row["item_id"]: row["prefix_end"]
                for row in cohort["sources"]
                if row["dataset_id"] == key[0]
            }
            with np.load(directory / "training_batch.npz", allow_pickle=False) as saved:
                if len(saved["values"]) > 64 or saved["values"].shape[1] != 96:
                    raise ValueError("imputer training exceeded the registered window budget")
                for identifier in saved["window_ids"].tolist():
                    item, start = identifier.rsplit("@", 1)
                    if item not in prefixes or int(start.split("|", 1)[0]) + 96 > prefixes[item]:
                        raise ValueError("a training window leaves the historical prefix")
            for record in dataset["imputers"]:
                path = Path(record["path"])
                if file_sha256(path) != record["sha256"]:
                    raise ValueError("a fitted-imputer record changed")
                fit = json.loads(path.read_text(encoding="utf-8"))
                if fit["status"] != "fitted":
                    failed_fits.append({"dataset_id": key[0], **record})
                else:
                    for file in fit["files"]:
                        if (
                            file_sha256(directory / record["candidate_id"] / file["path"])
                            != file["sha256"]
                        ):
                            raise ValueError("a fitted-imputer file changed")
            checked_datasets.add(key[0])
        for record in prep["episodes"]:
            if (record["dataset_id"], record["item_id"]) != key:
                continue
            task = expected[record["episode_id"]]
            path = args.prepared_root / record["path"]
            if file_sha256(path) != record["sha256"]:
                raise ValueError("a completed input artifact changed")
            origin = task["window"]["origin"]
            with np.load(path, allow_pickle=False) as saved:
                context, candidates = saved["context"], saved["candidate_values"]
                clean = raw[origin - 96 : origin]
                visible = np.isfinite(context)
                np.testing.assert_array_equal(context[visible], clean[visible])
                if task["mechanism"] == "native":
                    np.testing.assert_array_equal(context, clean)
                elif not np.isfinite(clean).all():
                    raise ValueError("a synthetic history contained original missing measurements")
                np.testing.assert_array_equal(saved["future"], raw[origin : origin + 96, :2])
                np.testing.assert_array_equal(
                    saved["future_observed"], np.isfinite(raw[origin : origin + 96, :2])
                )
                if not np.isfinite(candidates).all() or candidates.shape != (6, 96, raw.shape[1]):
                    raise ValueError("the six completed candidates changed shape or support")
                for candidate in candidates:
                    np.testing.assert_array_equal(candidate[visible], context[visible])
                if saved["candidate_ids"].tolist() != prep["identity"]["candidate_ids"]:
                    raise ValueError("candidate order changed")
            cases += 1
    _write_json(
        args.output_root / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "cohort_sha256": file_sha256(args.cohort_root / "manifest.json"),
            "prepared_sha256": file_sha256(args.prepared_root / "manifest.json"),
            "verified_episodes": cases,
            "verified_prefix_standardizers": len(scalers),
            "verified_fit_datasets": len(checked_datasets),
            "failed_imputer_fits": failed_fits,
            "limits": "input identity, future observability, prefix statistics, fit-window boundaries and observed-value preservation; stochastic mask replay is separate",
        },
    )


if __name__ == "__main__":
    main()
