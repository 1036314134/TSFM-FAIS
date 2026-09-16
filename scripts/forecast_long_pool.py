"""Freeze Chronos L192 full-pool forecasts and the two unchanged R24 repairs."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.dataset  # noqa: F401
import torch
from latent_source_inputs import ROOT, read_json
from learned_patch_repair import PatchRepair, RepairHook
from patch_repair_eval_support import fixed_controls
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster
from replay_preforecast_student import assemble_selected_context

from tsfm_fais.contracts import ForecastSpec
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed long-pool predictions")
    input_root = ROOT / "artifacts/iclr27-r25/long-inputs-v001"
    prepared = read_json(input_root / ("smoke.json" if args.smoke else "manifest.json"))
    train_root = ROOT / "artifacts/iclr27-r24/repair-training-v001"
    trained = read_json(train_root / "manifest.json")
    old_path = ROOT / "artifacts/iclr27-r19/replay-results-v001/source_defaults.json"
    matched_path = ROOT / "artifacts/iclr27-r24/repair-evaluation-inputs-v001/matched_controls.json"
    old, matched = (
        read_json(old_path)["models"]["chronos2"],
        read_json(matched_path)["models"]["chronos2"],
    )
    identity = {
        str(path.relative_to(ROOT)): file_sha256(path)
        for path in (
            Path(__file__),
            input_root / "identity.json",
            ROOT / "scripts/learned_patch_repair.py",
            ROOT / "scripts/patch_repair_eval_support.py",
            ROOT / "docs/iclr2027/R25_LONG_POOL_PROTOCOL.md",
            train_root / "manifest.json",
            old_path,
            matched_path,
        )
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial long-pool forecast definitions changed")
    _write_json(output / "identity.json", identity)
    identity_sha = file_sha256(output / "identity.json")
    torch.set_num_threads(1)
    started = perf_counter()
    runner, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    modules, module_shas = {}, {}
    for condition in ("fraction", "pattern"):
        entry = next(
            r
            for r in trained["models"]
            if r["model_id"] == "chronos2" and r["condition"] == condition
        )
        path = train_root / entry["checkpoint_path"]
        if file_sha256(path) != entry["checkpoint_sha256"]:
            raise ValueError("an original R24 repair changed")
        repair = PatchRepair(entry["width"], entry["patch_size"], entry["rank"]).to("cuda")
        repair.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        modules[condition], module_shas[condition] = (
            repair.eval().requires_grad_(False),
            entry["checkpoint_sha256"],
        )
    spec = ForecastSpec(
        "chronos2", "joint_multivariate", 96, context_length=192, target_indices=[0, 1]
    )
    counters = {
        "logical_requests": 0,
        "new_queries": 0,
        "cache_hits": 0,
        "repeats": 0,
        "forward_batches": 0,
    }

    def before(module, positional, keyword):
        counters["forward_batches"] += 1

    handle = backbone.register_forward_pre_hook(before, with_kwargs=True)
    keys, sentinels = set(), {}

    def execute(raw, observed, mode):
        if mode == "ordinary":
            return runner.predict_missing(raw[None], spec).point[0]
        with RepairHook(backbone, "chronos2", observed, modules[mode], mode):
            return runner.predict_missing(raw[None], spec).point[0]

    def query(raw, context, mode, row):
        counters["logical_requests"] += 1
        values = np.array(raw, np.float32, order="C")
        values[np.isnan(values)] = np.nan
        observed = np.isfinite(context)
        binding = {
            "identity_sha256": identity_sha,
            "case_id": row["case_id"],
            "input_sha256": row["sha256"],
            "model_id": "chronos2",
            "parameter_sha256": digest,
            "mode": mode,
            "repair_sha256": module_shas.get(mode),
            "origin": row["origin"],
            "context_length": 192,
            "horizon": 96,
            "observation_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
        }
        encoded = json.dumps(binding, sort_keys=True)
        key = hashlib.sha256(encoded.encode() + values.tobytes()).hexdigest()
        path = output / "queries" / f"{key}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["input"], values)
                if str(saved["binding"]) != encoded:
                    raise ValueError("the cached long-pool identity changed")
                point = saved["point"]
            counters["cache_hits"] += 1
        else:
            point = execute(values, observed, mode)
            if counters["new_queries"] % 64 == 0:
                np.testing.assert_array_equal(point, execute(values, observed, mode))
                counters["repeats"] += 1
            _save_npz(
                path, input=values, observed=observed, point=point, binding=np.asarray(encoded)
            )
            counters["new_queries"] += 1
        keys.add(key)
        sentinels.setdefault(mode, key)
        return point, key

    records = []
    previous_root = ROOT / "artifacts/iclr27-r24/repair-evaluation-v001"
    previous = {
        r["case_id"]: r
        for r in read_json(previous_root / "predictions_frozen.json")["predictions"]
        if r["model_id"] == "chronos2"
    }
    replay_count, replay_maximum = 0, 0.0
    for row in prepared["cases"]:
        if file_sha256(input_root / row["path"]) != row["sha256"]:
            raise ValueError("a prepared L192 candidate changed")
        with np.load(input_root / row["path"], allow_pickle=False) as data:
            context, candidates, ids = (
                data["context"],
                data["candidate_values"],
                data["candidate_ids"].tolist(),
            )
            actions = sorted([*ids, "guarded_direct"])
            bank, queries = [], []
            for action in actions:
                raw = assemble_selected_context(
                    context, candidates, ids, [action], [0, 1], joint=True
                )
                point, key = query(raw, context, "ordinary", row)
                bank.append((point - data["mean"][:2]) / data["scale"][:2])
                queries.append({"mode": "ordinary", "action": action, "key": key})
            bank = np.stack(bank)
            methods = fixed_controls(bank, actions, old, matched)
            for condition in ("fraction", "pattern"):
                point, key = query(candidates[ids.index("seasonal_lag")], context, condition, row)
                methods[condition + "_repair"] = (point - data["mean"][:2]) / data["scale"][:2]
                queries.append({"mode": condition, "action": "seasonal_lag", "key": key})
            if len(methods) != 18:
                raise ValueError("the registered long-pool method count changed")
            old_case = f"native_{row['case_id']}_l192"
            if old_case in previous:
                record = previous[old_case]
                if file_sha256(previous_root / record["path"]) != record["sha256"]:
                    raise ValueError("a previous pilot prediction changed")
                with np.load(previous_root / record["path"], allow_pickle=False) as saved:
                    names = saved["methods"].tolist()
                    for current, old_name in (
                        ("seasonal_lag", "base_seasonal"),
                        ("guarded_direct", "native192"),
                        ("fraction_repair", "fraction_repair"),
                        ("pattern_repair", "pattern_repair"),
                    ):
                        expected = saved["points"][names.index(old_name)]
                        replay_maximum = max(
                            replay_maximum, float(abs(methods[current] - expected).max())
                        )
                        np.testing.assert_allclose(methods[current], expected, rtol=0, atol=1e-6)
                replay_count += 1
        path = output / "predictions" / f"{row['case_id']}.npz"
        arrays = {
            "methods": np.asarray(list(methods)),
            "points": np.stack(list(methods.values())),
            "bank": bank,
            "actions": np.asarray(actions),
            "queries": np.asarray(json.dumps(queries)),
        }
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                for name, value in arrays.items():
                    np.testing.assert_array_equal(saved[name], value)
        else:
            _save_npz(path, **arrays)
        records.append(
            {
                "case_id": row["case_id"],
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
            }
        )
        if len(records) % 25 == 0:
            print(f"frozen {len(records)} long-pool cases", flush=True)
    for mode, key in sentinels.items():
        with np.load(output / "queries" / f"{key}.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(
                saved["point"], execute(saved["input"], saved["observed"], mode)
            )
            counters["repeats"] += 1
    if parameter_digest(backbone) != digest:
        raise ValueError("the fixed Chronos backbone changed")
    handle.remove()
    if not args.smoke and (len(records) != 301 or replay_count != 23):
        raise ValueError("the expanded or old pilot population changed")
    _write_json(
        output / ("smoke.json" if args.smoke else "manifest.json"),
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "cases": records,
            "counters": {**counters, "distinct_queries": len(keys)},
            "repair_shas": module_shas,
            "pilot_replay_cases": replay_count,
            "maximum_pilot_replay_difference": replay_maximum,
            "wall_seconds": perf_counter() - started,
            "evaluation_future_read": False,
        },
    )


if __name__ == "__main__":
    main()
