"""Cache frozen current and historical forecasts with measured execution batches."""

import argparse
import gc
import hashlib
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from latent_source_inputs import ROOT, read_json
from matched_replay_core import RULES
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster
from replay_preforecast_student import assemble_selected_context

from tsfm_fais.contracts import ForecastSpec
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


class QueryCache:
    def __init__(self, root, model_id, identity_sha):
        self.root, self.model_id, self.identity_sha = root, model_id, identity_sha
        started = perf_counter()
        self.runner, self.adapter, self.backbone, self.digest, self.joint = make_forecaster(
            model_id,
            ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
            ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
        )
        self.load_seconds = perf_counter() - started
        self.logical = self.hits = self.new = self.repeats = self.batches = self.adapter_calls = 0
        self.seconds = 0.0
        self.shapes = Counter()
        self.keys = set()
        self.first_short = None

        def before_forward(module, positional, keywords):
            tensors = [value for value in positional if isinstance(value, torch.Tensor)]
            tensors.extend(value for value in keywords.values() if isinstance(value, torch.Tensor))
            self.batches += 1
            self.shapes[str([list(value.shape) for value in tensors])] += 1

        self.hook = self.backbone.register_forward_pre_hook(before_forward, with_kwargs=True)
        old_predict = self.adapter._predict

        def counted_predict(*args, **kwargs):
            self.adapter_calls += 1
            return old_predict(*args, **kwargs)

        self.adapter._predict = counted_predict

    def execute(self, effective):
        spec = ForecastSpec(
            self.model_id,
            "joint_multivariate" if self.joint else "independent_univariate",
            96,
            context_length=len(effective),
            target_indices=[0, 1],
        )
        started = perf_counter()
        point = self.runner.predict_missing(effective[None], spec).point[0]
        self.seconds += perf_counter() - started
        if point.shape != (96, 2) or not np.isfinite(point).all():
            raise ValueError("a frozen replay forecast is invalid")
        return point

    def query(self, raw, source_mask, mean, scale, fit_binding, *, normalized=False):
        self.logical += 1
        values = (raw - mean) / scale if normalized else raw
        effective = np.asarray(values if self.joint else values[:, :2], np.float32).copy(order="C")
        effective[np.isnan(effective)] = np.nan
        binding = {
            "identity_sha256": self.identity_sha,
            "model_id": self.model_id,
            "parameter_sha256": self.digest,
            "fit_binding": fit_binding,
            "input_normalized": normalized,
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "context_length": len(effective),
            "horizon": 96,
            "mask_sha256": hashlib.sha256(
                np.ascontiguousarray(source_mask, bool).tobytes()
            ).hexdigest(),
        }
        key = hashlib.sha256(
            json.dumps(binding, sort_keys=True).encode()
            + str(effective.shape).encode()
            + effective.tobytes()
        ).hexdigest()
        path = self.root / "queries" / f"{key}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["effective_input"], effective)
                if str(saved["binding"]) != json.dumps(binding, sort_keys=True):
                    raise ValueError("a forecast cache binding changed")
                point = saved["point"]
            self.hits += 1
        else:
            point = self.execute(effective)
            if self.new % 128 == 0:
                np.testing.assert_array_equal(point, self.execute(effective))
                self.repeats += 1
            _save_npz(
                path,
                effective_input=effective,
                point=point,
                source_mask=source_mask,
                binding=np.asarray(json.dumps(binding, sort_keys=True)),
            )
            self.new += 1
        self.keys.add(key)
        if len(effective) == 96 and self.first_short is None:
            self.first_short = key
        return (point if normalized else (point - mean[:2]) / scale[:2]), key

    def finish(self):
        if self.first_short is not None:
            with np.load(
                self.root / "queries" / f"{self.first_short}.npz", allow_pickle=False
            ) as saved:
                np.testing.assert_array_equal(
                    saved["point"], self.execute(saved["effective_input"])
                )
                self.repeats += 1
        if parameter_digest(self.backbone) != self.digest:
            raise ValueError("the frozen predictor parameters changed")
        if self.new and not self.batches:
            raise ValueError("model execution batches were not observable")
        self.hook.remove()
        return {
            "model_id": self.model_id,
            "parameter_sha256": self.digest,
            "logical_requests": self.logical,
            "distinct_query_keys": len(self.keys),
            "cache_hits": self.hits,
            "new_effective_inputs": self.new,
            "numerical_repeat_requests": self.repeats,
            "adapter_calls": self.adapter_calls,
            "model_forward_batches": self.batches,
            "forward_batch_shapes": dict(self.shapes),
            "model_load_seconds": self.load_seconds,
            "forecast_seconds": self.seconds,
        }


def guarded_long(context, defaults, joint):
    filled = pd.DataFrame(context).ffill().to_numpy()
    filled = np.where(np.isfinite(filled), filled, defaults[None])
    empty = ~np.isfinite(context).any(0)
    if joint:
        return filled if empty.any() else context.copy()
    result = context.copy()
    for slot in (0, 1):
        if empty[slot]:
            result[:, slot] = filled[:, slot]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    plan_root = ROOT / "artifacts/iclr27-r19/pilot-plan-v001"
    prepared_root = ROOT / "artifacts/iclr27-r19/replay-inputs-v001"
    plan = read_json(plan_root / "manifest.json")
    prepared = read_json(
        prepared_root / ("smoke_preparation.json" if args.smoke else "manifest.json")
    )
    if prepared["status"] != "completed" or prepared["identity"]["plan_sha256"] != file_sha256(
        plan_root / "manifest.json"
    ):
        raise ValueError("complete the registered replay inputs first")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "plan_sha256": file_sha256(plan_root / "manifest.json"),
        "preparation_identity_sha256": file_sha256(prepared_root / "identity.json"),
        "runtime_sha256": file_sha256(ROOT / "scripts/r6_runtime.py"),
        "protocol_sha256": file_sha256(ROOT / "docs/iclr2027/R19_MATCHED_REPLAY_PROTOCOL.md"),
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed replay forecasts")
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial replay forecast definitions changed")
    _write_json(output / "identity.json", identity)
    identity_sha = file_sha256(output / "identity.json")
    torch.set_num_threads(1)
    records, costs = [], []
    case_metadata = {row["case_id"]: row for row in plan["cases"]}
    for model_id in ("chronos2", "timesfm2p5"):
        cache = QueryCache(output / model_id, model_id, identity_sha)
        for record in prepared["cases"]:
            metadata = case_metadata[record["case_id"]]
            path = prepared_root / record["path"]
            if file_sha256(path) != record["sha256"]:
                raise ValueError("a prepared replay case changed")
            target = output / model_id / "cases" / f"{record['case_id']}.npz"
            if target.exists() and not args.smoke:
                with np.load(target, allow_pickle=False) as saved:
                    if (
                        str(saved["input_sha256"]) != record["sha256"]
                        or str(saved["identity_sha256"]) != identity_sha
                    ):
                        raise ValueError("a resumed replay forecast case changed")
                records.append(
                    {
                        "model_id": model_id,
                        "case_id": record["case_id"],
                        "path": str(target.relative_to(output)),
                        "sha256": file_sha256(target),
                    }
                )
                continue
            with np.load(path, allow_pickle=False) as data:
                mean, scale = data["mean"], data["scale"]
                ids = data["candidate_ids"].tolist()
                actions = sorted([*ids, "guarded_direct"])
                fit_binding = {
                    "neural": record["frozen_neural_imputers"],
                    "prefix_end": metadata["prefix_end"],
                    "dataset_id": metadata["dataset_id"],
                    "item_id": metadata["item_id"],
                    "motm_reference": "f824b1d183ec39b3c471101a25cb1a039107af53d6618d47cad945b6344e8274",
                }
                current, current_keys = [], []
                for action in actions:
                    effective = assemble_selected_context(
                        data["context"],
                        data["current_candidates"],
                        ids,
                        [action] if cache.joint else [action, action],
                        [0, 1],
                        joint=cache.joint,
                    )
                    point, key = cache.query(
                        effective, ~np.isfinite(data["context"]), mean, scale, fit_binding
                    )
                    current.append(point)
                    current_keys.append(key)
                historical = np.full((3, 8, 8, 96, 2), np.nan)
                historical_keys = np.full((3, 8, 8), "", dtype="U64")
                for rule in range(3):
                    for anchor in range(1 if args.smoke else 8):
                        for action_index, action in enumerate(
                            actions[:1] if args.smoke else actions
                        ):
                            context = data["historical_contexts"][rule, anchor]
                            effective = assemble_selected_context(
                                context,
                                data["historical_candidates"][rule, anchor],
                                ids,
                                [action] if cache.joint else [action, action],
                                [0, 1],
                                joint=cache.joint,
                            )
                            point, key = cache.query(
                                effective, ~np.isfinite(context), mean, scale, fit_binding
                            )
                            historical[rule, anchor, action_index] = point
                            historical_keys[rule, anchor, action_index] = key
                extras, extra_keys = {}, {}
                if not args.smoke:
                    effective = assemble_selected_context(
                        data["context"],
                        data["current_candidates"],
                        ids,
                        ["guarded_direct"] if cache.joint else ["guarded_direct", "guarded_direct"],
                        [0, 1],
                        joint=cache.joint,
                    )
                    extras["native_prefix96"], extra_keys["native_prefix96"] = cache.query(
                        effective,
                        ~np.isfinite(data["context"]),
                        mean,
                        scale,
                        fit_binding,
                        normalized=True,
                    )
                long = guarded_long(data["long_context"], data["defaults"], cache.joint)
                for name, normalized in (("native_long_raw", False), ("native_long_prefix", True)):
                    extras[name], extra_keys[name] = cache.query(
                        long,
                        ~np.isfinite(data["long_context"]),
                        mean,
                        scale,
                        fit_binding,
                        normalized=normalized,
                    )
                if not args.smoke:
                    _save_npz(
                        target,
                        current=np.stack(current),
                        current_keys=np.asarray(current_keys),
                        historical=historical,
                        historical_keys=historical_keys,
                        actions=np.asarray(actions),
                        extra_names=np.asarray(list(extras)),
                        extras=np.stack(list(extras.values())),
                        extra_keys=np.asarray(list(extra_keys.values())),
                        input_sha256=np.asarray(record["sha256"]),
                        identity_sha256=np.asarray(identity_sha),
                    )
                    records.append(
                        {
                            "model_id": model_id,
                            "case_id": record["case_id"],
                            "path": str(target.relative_to(output)),
                            "sha256": file_sha256(target),
                        }
                    )
            print(
                f"{model_id}: {'throughput' if args.smoke else 'forecasts'} {record['case_id']}",
                flush=True,
            )
        cost = cache.finish()
        costs.append(cost)
        del cache
        gc.collect()
        torch.cuda.empty_cache()
    if not args.smoke and len(records) != 48:
        raise ValueError("the two-model replay forecast population changed")
    _write_json(
        output / ("smoke.json" if args.smoke else "manifest.json"),
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "cases": records,
            "costs": costs,
            "current_future_values_read": False,
            "rules": list(RULES),
            "limits": "frozen current and past predictions; current decisions and outcomes not yet evaluated",
        },
    )


if __name__ == "__main__":
    main()
