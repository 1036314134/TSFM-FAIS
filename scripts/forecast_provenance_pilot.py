"""Freeze value-matched ordinary, provenance and shifted-marker forecast banks."""

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
from latent_source_inputs import ROOT, read_json
from probe_differentiable_imputation import parameter_digest
from provenance_marker import ProvenanceMarker
from r6_runtime import make_forecaster
from replay_preforecast_student import assemble_selected_context

from tsfm_fais.contracts import ForecastSpec
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256

ROLES = ("ordinary", "provenance", "shifted")


class MarkerCache:
    def __init__(self, root, model_id, identity):
        self.root, self.model_id, self.identity = root, model_id, identity
        start = perf_counter()
        self.runner, self.adapter, self.backbone, self.digest, self.joint = make_forecaster(
            model_id,
            ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
            ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
        )
        self.load_seconds = perf_counter() - start
        self.logical = self.hits = self.new = self.repeats = self.forward = self.physical = 0
        self.seconds = 0.0
        self.keys = set()
        self.shapes = Counter()
        self.first = None

        def before(module, args, kwargs):
            self.forward += 1
            self.shapes[
                str(
                    [
                        list(value.shape)
                        for value in [*args, *kwargs.values()]
                        if isinstance(value, torch.Tensor)
                    ]
                )
            ] += 1

        self.hook = self.backbone.register_forward_pre_hook(before, with_kwargs=True)

    def execute(self, effective, marker, role):
        spec = ForecastSpec(
            self.model_id,
            "joint_multivariate" if self.joint else "independent_univariate",
            96,
            context_length=96,
            target_indices=[0, 1] if self.joint else [0],
        )
        then = perf_counter()
        if np.isfinite(effective).all():
            with ProvenanceMarker(
                self.backbone, self.model_id, marker, role != "ordinary"
            ) as intervention:
                point = self.runner.predict_missing(effective[None], spec).point[0]
            calls = intervention.calls
        else:
            if role != "ordinary":
                raise ValueError("only the unchanged native baseline accepts incomplete inputs")
            point = self.runner.predict_missing(effective[None], spec).point[0]
            calls = []
        self.seconds += perf_counter() - then
        self.physical += 1
        if point.shape != (96, 2 if self.joint else 1) or not np.isfinite(point).all():
            raise ValueError("the frozen forecast shape or finiteness changed")
        return point, calls

    def query(self, raw, observed, role, row, slot):
        self.logical += 1
        effective = np.array(raw if self.joint else raw[:, slot : slot + 1], np.float32, order="C")
        effective[np.isnan(effective)] = np.nan
        marker = observed if self.joint else observed[:, slot : slot + 1]
        marker = np.array(
            np.roll(marker, 37, axis=0) if role == "shifted" else marker, bool, order="C"
        )
        binding = {
            "identity_sha256": self.identity,
            "model_id": self.model_id,
            "parameter_sha256": self.digest,
            "case_id": row["case_id"],
            "input_sha256": row["sha256"],
            "origin": row["origin"],
            "context_length": 96,
            "horizon": 96,
            "target_slot": slot,
            "role": role,
            "marker_sha256": hashlib.sha256(marker.tobytes()).hexdigest(),
        }
        encoded = json.dumps(binding, sort_keys=True)
        key = hashlib.sha256(encoded.encode() + effective.tobytes()).hexdigest()
        path = self.root / "queries" / f"{key}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if str(saved["binding"]) != encoded:
                    raise ValueError("a cached marker-query identity changed")
                np.testing.assert_array_equal(saved["effective"], effective)
                np.testing.assert_array_equal(saved["marker"], marker)
                point = saved["point"]
            self.hits += 1
        else:
            point, calls = self.execute(effective, marker, role)
            if self.new % 64 == 0:
                repeated, repeated_calls = self.execute(effective, marker, role)
                np.testing.assert_array_equal(point, repeated)
                for a, b in zip(calls, repeated_calls, strict=True):
                    np.testing.assert_array_equal(a["after"], b["after"])
                self.repeats += 1
            arrays = {
                f"call_{i}_{field}": call[field]
                for i, call in enumerate(calls)
                for field in ("before", "after")
            }
            _save_npz(
                path,
                effective=effective,
                marker=marker,
                point=point,
                binding=np.asarray(encoded),
                tokenizer_calls=np.asarray(len(calls)),
                **arrays,
            )
            self.new += 1
        if self.first is None:
            self.first = key
        self.keys.add(key)
        return point, key

    def finish(self):
        if self.first:
            with np.load(self.root / "queries" / f"{self.first}.npz", allow_pickle=False) as saved:
                binding = json.loads(str(saved["binding"]))
                point, _ = self.execute(saved["effective"], saved["marker"], binding["role"])
                np.testing.assert_array_equal(saved["point"], point)
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
            "adapter_calls": self.physical,
            "forward_batches": self.forward,
            "forward_shapes": dict(self.shapes),
            "model_load_seconds": self.load_seconds,
            "forecast_seconds": self.seconds,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    input_root = ROOT / "artifacts/iclr27-r21/provenance-inputs-v001"
    prepared = read_json(input_root / "manifest.json")
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed marker forecasts")
    identity = {
        str(path.relative_to(ROOT)): file_sha256(path)
        for path in (
            Path(__file__),
            ROOT / "scripts/provenance_marker.py",
            input_root / "manifest.json",
            ROOT / "docs/iclr2027/R21_PROVENANCE_PILOT_PROTOCOL.md",
            ROOT / "scripts/r6_runtime.py",
        )
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial marker forecast definitions changed")
    _write_json(output / "identity.json", identity)
    identity_sha = file_sha256(output / "identity.json")
    cases = prepared["cases"][:1] if args.smoke else prepared["cases"]
    records, costs = [], []
    torch.set_num_threads(1)
    for model_id in ("chronos2", "timesfm2p5"):
        cache = MarkerCache(output / model_id, model_id, identity_sha)
        for row in cases:
            path = input_root / row["path"]
            if file_sha256(path) != row["sha256"]:
                raise ValueError("a registered marker-pilot input changed")
            with np.load(path, allow_pickle=False) as data:
                context, candidates, ids = (
                    data["context"],
                    data["candidate_values"],
                    data["candidate_ids"].tolist(),
                )
                actions = sorted([*ids, "guarded_direct"])
                bank, queries = np.empty((3, 8, 96, 2)), []
                for slot in [-1] if cache.joint else [0, 1]:
                    destination = slice(None) if cache.joint else slice(slot, slot + 1)
                    for index, action in enumerate(actions):
                        raw = assemble_selected_context(
                            context,
                            candidates,
                            ids,
                            [action] if cache.joint else [action, action],
                            [0, 1],
                            joint=cache.joint,
                        )
                        previous = None
                        for role_index, role in enumerate(ROLES):
                            if action == "guarded_direct" and role_index:
                                point, key = previous
                            else:
                                point, key = cache.query(raw, np.isfinite(context), role, row, slot)
                                previous = (point, key)
                            selected_mean = (
                                data["mean"][:2] if cache.joint else data["mean"][slot : slot + 1]
                            )
                            selected_scale = (
                                data["scale"][:2] if cache.joint else data["scale"][slot : slot + 1]
                            )
                            bank[role_index, index, :, destination] = (
                                point - selected_mean
                            ) / selected_scale
                            queries.append(
                                {"role": role, "action": action, "slot": slot, "key": key}
                            )
            target = output / model_id / "cases" / f"{row['case_id']}.npz"
            _save_npz(
                target,
                bank=bank,
                actions=np.asarray(actions),
                queries=np.asarray(json.dumps(queries)),
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
