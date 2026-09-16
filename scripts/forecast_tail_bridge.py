"""Freeze time-aligned tail forecasts and equal-output-budget original-origin controls."""

import argparse
import gc
import hashlib
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.dataset  # noqa: F401
import torch
from forecast_matched_replay import guarded_long
from latent_source_inputs import ROOT, read_json
from prepare_tail_bridge import smoke_cases
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster
from replay_preforecast_student import assemble_selected_context
from tail_bridge_core import bridge_slice

from tsfm_fais.contracts import ForecastSpec
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


class TailCache:
    def __init__(self, root, model_id, identity_sha):
        self.root, self.model_id, self.identity_sha = root, model_id, identity_sha
        then = perf_counter()
        self.runner, self.adapter, self.backbone, self.digest, self.joint = make_forecaster(
            model_id,
            ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
            ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
        )
        self.load_seconds = perf_counter() - then
        self.logical = self.hits = self.new = self.repeats = self.batches = self.adapter_calls = 0
        self.seconds = 0.0
        self.shapes = Counter()
        self.keys = set()
        self.first = None

        def before(module, args, kwargs):
            tensors = [
                value for value in [*args, *kwargs.values()] if isinstance(value, torch.Tensor)
            ]
            self.batches += 1
            self.shapes[str([list(value.shape) for value in tensors])] += 1

        self.hook = self.backbone.register_forward_pre_hook(before, with_kwargs=True)
        original = self.adapter._predict

        def predict(*args, **kwargs):
            self.adapter_calls += 1
            return original(*args, **kwargs)

        self.adapter._predict = predict

    def execute(self, effective, horizon):
        targets = [0, 1] if self.joint else [0]
        spec = ForecastSpec(
            self.model_id,
            "joint_multivariate" if self.joint else "independent_univariate",
            int(horizon),
            context_length=len(effective),
            target_indices=targets,
        )
        then = perf_counter()
        point = self.runner.predict_missing(effective[None], spec).point[0]
        self.seconds += perf_counter() - then
        if point.shape != (horizon, len(targets)) or not np.isfinite(point).all():
            raise ValueError("the requested bridge forecast shape or support changed")
        return point

    def query(self, raw, source_mask, mean, scale, provenance, origin, horizon, slot):
        self.logical += 1
        effective = np.asarray(raw if self.joint else raw[:, slot : slot + 1], np.float32).copy(
            order="C"
        )
        effective[np.isnan(effective)] = np.nan
        selected_mean = mean[:2] if self.joint else mean[slot : slot + 1]
        selected_scale = scale[:2] if self.joint else scale[slot : slot + 1]
        binding = {
            "identity_sha256": self.identity_sha,
            "model_id": self.model_id,
            "parameter_sha256": self.digest,
            "provenance": provenance,
            "forecast_origin": origin,
            "history_start": origin - len(raw),
            "horizon": horizon,
            "target_slot": slot,
            "input_shape": list(effective.shape),
            "mean": selected_mean.tolist(),
            "scale": selected_scale.tolist(),
            "source_mask_sha256": hashlib.sha256(
                np.ascontiguousarray(source_mask, bool).tobytes()
            ).hexdigest(),
        }
        text = json.dumps(binding, sort_keys=True)
        key = hashlib.sha256(text.encode() + effective.tobytes()).hexdigest()
        path = self.root / "queries" / f"{key}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["effective_input"], effective)
                if str(saved["binding"]) != text:
                    raise ValueError("a tail forecast cache binding changed")
                point = saved["point"]
            self.hits += 1
        else:
            point = self.execute(effective, horizon)
            if self.new % 64 == 0:
                np.testing.assert_array_equal(point, self.execute(effective, horizon))
                self.repeats += 1
            _save_npz(
                path,
                effective_input=effective,
                source_mask=source_mask,
                point=point,
                binding=np.asarray(text),
            )
            self.new += 1
        self.keys.add(key)
        if self.first is None:
            self.first = key
        return (point - selected_mean) / selected_scale, key

    def finish(self):
        if self.first:
            with np.load(self.root / "queries" / f"{self.first}.npz", allow_pickle=False) as saved:
                horizon = json.loads(str(saved["binding"]))["horizon"]
                np.testing.assert_array_equal(
                    saved["point"], self.execute(saved["effective_input"], horizon)
                )
                self.repeats += 1
        if parameter_digest(self.backbone) != self.digest:
            raise ValueError("frozen forecasting parameters changed")
        self.hook.remove()
        return {
            "model_id": self.model_id,
            "parameter_sha256": self.digest,
            "logical_scope_requests": self.logical,
            "distinct_query_keys": len(self.keys),
            "cache_hits": self.hits,
            "new_effective_inputs": self.new,
            "repeat_requests": self.repeats,
            "adapter_calls": self.adapter_calls,
            "forward_batches": self.batches,
            "forward_shapes": dict(self.shapes),
            "model_load_seconds": self.load_seconds,
            "forecast_seconds": self.seconds,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    plan_root, prepared_root = (
        ROOT / "artifacts/iclr27-r20/tail-plan-v001",
        ROOT / "artifacts/iclr27-r20/tail-inputs-v001",
    )
    plan = read_json(plan_root / "manifest.json")
    prepared = read_json(
        prepared_root / ("smoke_preparation.json" if args.smoke else "manifest.json")
    )
    if prepared["status"] != "completed" or prepared["identity"]["plan_sha256"] != file_sha256(
        plan_root / "manifest.json"
    ):
        raise ValueError("complete the frozen tail inputs first")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "plan_sha256": file_sha256(plan_root / "manifest.json"),
        "preparation_identity_sha256": file_sha256(prepared_root / "identity.json"),
        "core_module_sha256": file_sha256(ROOT / "scripts/tail_bridge_core.py"),
        "protocol_sha256": file_sha256(ROOT / "docs/iclr2027/R20_TAIL_BRIDGE_PROTOCOL.md"),
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed tail forecasts")
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial tail forecasting definitions changed")
    _write_json(output / "identity.json", identity)
    identity_sha = file_sha256(output / "identity.json")
    cases = smoke_cases(plan["cases"]) if args.smoke else plan["cases"]
    inputs = {row["case_id"]: row for row in prepared["cases"]}
    torch.set_num_threads(1)
    records, costs = [], []
    for model_id in ("chronos2", "timesfm2p5"):
        cache = TailCache(output / model_id, model_id, identity_sha)
        for row in cases:
            if model_id not in row["models"]:
                continue
            entry = inputs[row["case_id"]]
            path = prepared_root / entry["path"]
            if file_sha256(path) != entry["sha256"]:
                raise ValueError("a prepared tail input changed")
            target = output / model_id / "cases" / f"{row['case_id']}.npz"
            with np.load(path, allow_pickle=False) as data:
                context, candidates = data["context"], data["candidate_values"]
                ids = data["candidate_ids"].tolist()
                actions = sorted([*ids, "guarded_direct"])
                mean, scale = data["mean"], data["scale"]
                provenance = {
                    "cohort": row["cohort"],
                    "dataset_id": row["dataset_id"],
                    "item_id": row["item_id"],
                    "prefix_end": row["prefix_end"],
                    "imputers": entry["frozen_neural_imputers"],
                }
                normal, budget = np.empty((8, 96, 2)), np.empty((8, 96, 2))
                bridge = np.empty((96, 2))
                long_points = np.empty((2, 96, 2))
                query_records = []
                slots = [-1] if cache.joint else [0, 1]
                for slot in slots:
                    gap = row["gaps"][model_id][0 if cache.joint else slot]
                    destination = slice(None) if cache.joint else slice(slot, slot + 1)
                    for action_index, action in enumerate(actions):
                        effective = assemble_selected_context(
                            context,
                            candidates,
                            ids,
                            [action] if cache.joint else [action, action],
                            [0, 1],
                            joint=cache.joint,
                        )
                        for role, horizon, bank in (
                            ("normal", 96, normal),
                            ("budget", 96 + gap, budget),
                        ):
                            point, key = cache.query(
                                effective,
                                ~np.isfinite(context),
                                mean,
                                scale,
                                provenance,
                                row["origin"],
                                horizon,
                                slot,
                            )
                            bank[action_index, :, destination] = point[:96]
                            query_records.append(
                                {
                                    "role": role,
                                    "action": action,
                                    "slot": slot,
                                    "key": key,
                                    "gap": gap,
                                    "slice_start": 0,
                                }
                            )
                    length = 96 - gap
                    effective = guarded_long(context[:length], data["defaults"], cache.joint)
                    point, key = cache.query(
                        effective,
                        ~np.isfinite(context[:length]),
                        mean,
                        scale,
                        provenance,
                        row["origin"] - gap,
                        96 + gap,
                        slot,
                    )
                    bridge[:, destination] = bridge_slice(point, gap)
                    query_records.append(
                        {"role": "bridge", "slot": slot, "key": key, "gap": gap, "slice_start": gap}
                    )
                    if gap == 0:
                        np.testing.assert_allclose(
                            bridge[:, destination],
                            normal[actions.index("guarded_direct"), :, destination],
                            rtol=1e-6,
                            atol=1e-6,
                        )
                    for position, length in enumerate((1024, 4096)):
                        long = data[f"long{length}"]
                        effective = guarded_long(long, data["defaults"], cache.joint)
                        point, key = cache.query(
                            effective,
                            ~np.isfinite(long),
                            mean,
                            scale,
                            provenance,
                            row["origin"],
                            96,
                            slot,
                        )
                        long_points[position, :, destination] = point
                        query_records.append(
                            {
                                "role": f"long{length}",
                                "slot": slot,
                                "key": key,
                                "gap": 0,
                                "slice_start": 0,
                            }
                        )
                if not all(
                    np.isfinite(value).all() for value in (normal, budget, bridge, long_points)
                ):
                    raise ValueError("the tail forecast bank is incomplete")
                if not args.smoke:
                    _save_npz(
                        target,
                        normal=normal,
                        budget=budget,
                        bridge=bridge,
                        long=long_points,
                        actions=np.asarray(actions),
                        query_records=np.asarray(json.dumps(query_records)),
                        input_sha256=np.asarray(entry["sha256"]),
                        identity_sha256=np.asarray(identity_sha),
                    )
                    records.append(
                        {
                            "model_id": model_id,
                            "case_id": row["case_id"],
                            "path": str(target.relative_to(output)),
                            "sha256": file_sha256(target),
                        }
                    )
            print(
                f"{model_id}: {'throughput' if args.smoke else 'forecasts'} {row['case_id']}",
                flush=True,
            )
        costs.append(cache.finish())
        del cache
        gc.collect()
        torch.cuda.empty_cache()
    if not args.smoke and len(records) != 100:
        raise ValueError("the registered natural and synthetic forecast population changed")
    _write_json(
        output / ("smoke.json" if args.smoke else "manifest.json"),
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "cases": records,
            "costs": costs,
            "current_future_values_read": False,
            "limits": "fixed origin and budget controls; current future outcomes not yet scored",
        },
    )


if __name__ == "__main__":
    main()
