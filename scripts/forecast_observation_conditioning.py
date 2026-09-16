"""Freeze an older H192 prior and two L192 original-origin forecasting controls."""

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
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster

from tsfm_fais.contracts import ForecastSpec
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


class PriorCache:
    def __init__(self, root, model_id, identity):
        self.root, self.model_id, self.identity = root, model_id, identity
        then = perf_counter()
        self.runner, self.adapter, self.backbone, self.digest, self.joint = make_forecaster(
            model_id,
            ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
            ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
        )
        self.load_seconds = perf_counter() - then
        self.logical = self.new = self.hits = self.repeats = self.forward = self.calls = 0
        self.seconds, self.first = 0.0, None
        self.keys, self.shapes = set(), Counter()

        def before(module, args, kwargs):
            self.forward += 1
            self.shapes[
                str(
                    [
                        list(x.shape)
                        for x in [*args, *kwargs.values()]
                        if isinstance(x, torch.Tensor)
                    ]
                )
            ] += 1

        self.hook = self.backbone.register_forward_pre_hook(before, with_kwargs=True)

    def execute(self, raw, horizon):
        spec = ForecastSpec(
            self.model_id,
            "joint_multivariate" if self.joint else "independent_univariate",
            horizon,
            context_length=len(raw),
            target_indices=[0, 1] if self.joint else [0],
        )
        then = perf_counter()
        result = self.runner.predict_missing(raw[None], spec)
        self.seconds += perf_counter() - then
        self.calls += 1
        point, quantiles = result.point[0], result.quantiles[0]
        if point.shape != (horizon, 2 if self.joint else 1) or quantiles.shape != (*point.shape, 3):
            raise ValueError("the registered point or quantile support changed")
        return point, quantiles

    def query(self, raw, row, role, origin, horizon, slot):
        self.logical += 1
        effective = np.array(
            raw if self.joint else raw[:, slot : slot + 1], dtype=np.float32, order="C"
        )
        effective[np.isnan(effective)] = np.nan
        binding = {
            "identity_sha256": self.identity,
            "model_id": self.model_id,
            "parameter_sha256": self.digest,
            "case_id": row["case_id"],
            "input_sha256": row["sha256"],
            "role": role,
            "forecast_origin": origin,
            "history_start": origin - len(raw),
            "horizon": horizon,
            "target_slot": slot,
            "quantile_levels": [0.1, 0.5, 0.9],
        }
        encoded = json.dumps(binding, sort_keys=True)
        key = hashlib.sha256(encoded.encode() + effective.tobytes()).hexdigest()
        path = self.root / "queries" / f"{key}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["effective"], effective)
                if str(saved["binding"]) != encoded:
                    raise ValueError("a prior query binding changed")
                point, quantiles = saved["point"], saved["quantiles"]
            self.hits += 1
        else:
            point, quantiles = self.execute(effective, horizon)
            if self.new % 32 == 0:
                repeat_point, repeat_quantiles = self.execute(effective, horizon)
                np.testing.assert_array_equal(point, repeat_point)
                np.testing.assert_array_equal(quantiles, repeat_quantiles)
                self.repeats += 1
            _save_npz(
                path,
                point=point,
                quantiles=quantiles,
                effective=effective,
                binding=np.asarray(encoded),
            )
            self.new += 1
        self.keys.add(key)
        if self.first is None:
            self.first = key
        return point, quantiles, key

    def finish(self):
        with np.load(self.root / "queries" / f"{self.first}.npz", allow_pickle=False) as saved:
            point, quantiles = self.execute(
                saved["effective"], json.loads(str(saved["binding"]))["horizon"]
            )
            np.testing.assert_array_equal(point, saved["point"])
            np.testing.assert_array_equal(quantiles, saved["quantiles"])
            self.repeats += 1
        if parameter_digest(self.backbone) != self.digest:
            raise ValueError("frozen forecasting parameters changed")
        self.hook.remove()
        return {
            "model_id": self.model_id,
            "parameter_sha256": self.digest,
            "logical_scope_requests": self.logical,
            "distinct_queries": len(self.keys),
            "new_queries": self.new,
            "cache_hits": self.hits,
            "repeat_requests": self.repeats,
            "adapter_calls": self.calls,
            "forward_batches": self.forward,
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
    input_root = ROOT / "artifacts/iclr27-r23/conditioning-inputs-v001"
    prepared = read_json(input_root / "manifest.json")
    if prepared["status"] != "completed" or (output / "manifest.json").exists():
        raise ValueError("inputs must complete and full forecasts must be new")
    identity = {
        str(path.relative_to(ROOT)): file_sha256(path)
        for path in (
            Path(__file__),
            input_root / "manifest.json",
            ROOT / "scripts/observation_conditioning.py",
            ROOT / "docs/iclr2027/R23_OBSERVATION_CONDITIONING_PROTOCOL.md",
            ROOT / "scripts/r6_runtime.py",
        )
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial conditioning forecast definitions changed")
    _write_json(output / "identity.json", identity)
    identity_sha = file_sha256(output / "identity.json")
    cases = prepared["cases"][:1] if args.smoke else prepared["cases"]
    records, costs = [], []
    torch.set_num_threads(1)
    for model_id in ("chronos2", "timesfm2p5"):
        cache = PriorCache(output / model_id, model_id, identity_sha)
        for row in cases:
            path = input_root / row["path"]
            if file_sha256(path) != row["sha256"]:
                raise ValueError("a registered conditioning input changed")
            with np.load(path, allow_pickle=False) as data:
                prior, prior_quantiles = np.empty((192, 2)), np.empty((192, 2, 3))
                native, budget = np.empty((96, 2)), np.empty((96, 2))
                queries = []
                for slot in [-1] if cache.joint else [0, 1]:
                    location = slice(None) if cache.joint else slice(slot, slot + 1)
                    mean = data["mean"][:2] if cache.joint else data["mean"][slot : slot + 1]
                    scale = data["scale"][:2] if cache.joint else data["scale"][slot : slot + 1]
                    for role, raw, origin, horizon in (
                        ("prior", data["prior_context"], row["origin"] - 96, 192),
                        ("native192", data["direct_context"], row["origin"], 96),
                        ("budget_native192", data["direct_context"], row["origin"], 192),
                    ):
                        effective = guarded_long(raw, data["defaults"], cache.joint)
                        point, quantiles, key = cache.query(
                            effective, row, role, origin, horizon, slot
                        )
                        normalized = (point - mean) / scale
                        if role == "prior":
                            prior[:, location] = normalized
                            prior_quantiles[:, location, :] = (
                                quantiles - mean[None, :, None]
                            ) / scale[None, :, None]
                        elif role == "native192":
                            native[:, location] = normalized
                        else:
                            budget[:, location] = normalized[:96]
                        queries.append({"role": role, "slot": slot, "key": key})
            target = output / model_id / "cases" / f"{row['case_id']}.npz"
            arrays = {
                "prior": prior,
                "prior_quantiles": prior_quantiles,
                "native192": native,
                "budget_native192": budget,
                "queries": np.asarray(json.dumps(queries)),
                "identity_sha256": np.asarray(identity_sha),
            }
            if target.exists():
                with np.load(target, allow_pickle=False) as previous:
                    for name, value in arrays.items():
                        np.testing.assert_array_equal(previous[name], value)
            else:
                _save_npz(target, **arrays)
            records.append(
                {
                    "model_id": model_id,
                    "case_id": row["case_id"],
                    "path": str(target.relative_to(output)),
                    "sha256": file_sha256(target),
                }
            )
            print(f"frozen {model_id} {row['case_id']}", flush=True)
        costs.append(cache.finish())
        del cache
        gc.collect()
        torch.cuda.empty_cache()
    _write_json(
        output / ("smoke.json" if args.smoke else "manifest.json"),
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "cases": records,
            "costs": costs,
            "current_future_values_read": False,
        },
    )


if __name__ == "__main__":
    main()
