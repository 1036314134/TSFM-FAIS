"""Read point-only controls and observe latent repairs without reading evaluation targets."""

import json
from pathlib import Path

import numpy as np
from latent_source_inputs import ROOT, read_json
from learned_patch_repair import RepairHook

from tsfm_fais.utility_experiment import file_sha256

POOL = ROOT / "artifacts/iclr27-r12/motm-pool-inputs-v001"


def pool_catalog(model_id):
    parent = read_json(POOL / "manifest.json")
    record = next(r for r in parent["models"] if r["model_id"] == model_id)
    path = POOL / record["path"]
    if file_sha256(path) != record["sha256"]:
        raise ValueError("the audited source pool changed")
    return {row["episode_id"]: row for row in read_json(path)["episodes"]}


def source_bank(row, model_id, catalog):
    record = catalog[row["episode_id"]]
    path = POOL / model_id / record["path"]
    if file_sha256(path) != record["sha256"] or record["source_index"] != row["episode_index"]:
        raise ValueError("a source point bank or episode index changed")
    with np.load(path, allow_pickle=False) as saved:
        decisions = json.loads(str(saved["decisions"]))
        if any(r["source_episode_id"] != row["episode_id"] for r in decisions):
            raise ValueError("source point decisions are not aligned")
        points = (
            saved["vectors"][0].reshape(8, 96, 2)
            if model_id == "chronos2"
            else saved["vectors"].transpose(1, 2, 0)
        )
        return points, saved["actions"].tolist()


def fixed_controls(points, actions, old, matched):
    if actions != old["actions"] or actions != matched["actions"]:
        raise ValueError("the source control action orders differ")
    result = {name: points[i] for i, name in enumerate(actions)}
    result.update(mean8=points.mean(0), median8=np.median(points, axis=0))
    for prefix, control in (("source_", old), ("matched_source_", matched)):
        result[prefix + "single_mae"] = points[control["single_index"]]
        result[prefix + "fixed_mae"] = (
            points * np.asarray(control["fixed_mae"]["weights"])[:, None, None]
        ).sum(0)
        joint = control.get("fixed_joint_weights", control.get("fixed_joint", {}).get("weights"))
        result[prefix + "fixed_joint"] = (points * np.asarray(joint)[:, None, None]).sum(0)
    return result


class TracedRepair:
    def __init__(self, backbone, model_id, observed, repair, condition, trace=False):
        self.host = backbone.input_patch_embedding if model_id == "chronos2" else backbone.tokenizer
        self.repair = RepairHook(backbone, model_id, observed, repair, condition)
        self.trace = trace
        self.records = []

    def __enter__(self):
        if self.trace:
            self.before_handle = self.host.register_forward_hook(self.before)
        self.repair.__enter__()
        if self.trace:
            self.after_handle = self.host.register_forward_hook(self.after)
        return self

    def before(self, module, args, value):
        self.original = value.detach().cpu().numpy().copy()

    def after(self, module, args, value):
        self.records.append({"before": self.original, "after": value.detach().cpu().numpy().copy()})

    def __exit__(self, exc_type, exc_value, traceback):
        if self.trace:
            self.before_handle.remove()
            self.after_handle.remove()
        return self.repair.__exit__(exc_type, exc_value, traceback)


def read_input(path: Path):
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in ("context", "base", "mean", "scale")}
