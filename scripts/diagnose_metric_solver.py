"""Replay the observed fixed-control failure and verify same-objective precision repair."""

import ast
import importlib.util
from pathlib import Path

import numpy as np
import psutil
import torch
from audit_metric_source_gates import direct_control
from latent_source_inputs import ROOT, load_source_inputs
from metric_source_gate import fit_fixed_metric
from threadpoolctl import threadpool_limits

from tsfm_fais.routing.utility import _family_weights
from tsfm_fais.utility_experiment import _write_json, file_sha256


def main():
    output = ROOT / "artifacts/iclr27-r10/solver-precision-diagnostic-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve a completed solver precision diagnostic")
    previous = ROOT / "artifacts/iclr27-r10/metric-source-v001/failure_snapshot"
    old_tree = ast.parse((previous / "metric_source_gate.py").read_text(encoding="utf-8"))
    new_tree = ast.parse((ROOT / "scripts/metric_source_gate.py").read_text(encoding="utf-8"))
    for name in ("metric_objective", "fixed_objective"):
        old = next(
            node
            for node in old_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        new = next(
            node
            for node in new_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        if ast.dump(old, include_attributes=False) != ast.dump(new, include_attributes=False):
            raise ValueError("a training objective or convex problem definition changed")
    if file_sha256(previous / "train_metric_source_gates.py") != file_sha256(
        ROOT / "scripts/train_metric_source_gates.py"
    ):
        raise ValueError("the source training script changed")
    process = psutil.Process()
    process.cpu_affinity(process.cpu_affinity()[-1:])
    torch.set_num_threads(1)
    _, frame, arrays = load_source_inputs(
        ROOT / "artifacts/iclr27-r7/latent-source-v001", "chronos2"
    )
    indices = np.flatnonzero(
        (frame.split.to_numpy() == "train") & (frame.family_id.to_numpy() != "ett")
    )
    points, target, weight = (
        arrays["vectors"][indices],
        arrays["teacher"][indices],
        _family_weights(frame.iloc[indices]),
    )
    spec = importlib.util.spec_from_file_location(
        "metric_source_gate_original", previous / "metric_source_gate.py"
    )
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    with threadpool_limits(limits=1):
        try:
            original.fit_fixed_metric(points, target, weight, "mae")
        except ValueError as error:
            previous_error = str(error)
        else:
            raise ValueError("the reported solver failure did not reproduce")
        result = fit_fixed_metric(points, target, weight, "mae")
        value, gradient = direct_control(
            points, target, weight, np.asarray(result["weights"]), "mae"
        )
    probability = np.asarray(result["weights"])
    gap = float((gradient @ probability - gradient.min()) / max(abs(gradient).max(), 1.0))
    if gap > 1e-7 or not result["polishing_steps"] or value > result["initial_objective"] + 1e-12:
        raise ValueError("the repaired fixed control did not meet the unchanged precision checks")
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "status": "completed",
        "script_sha256": file_sha256(Path(__file__)),
        "module_sha256": file_sha256(ROOT / "scripts/metric_source_gate.py"),
        "condition": "chronos2/ett/mae_teacher",
        "previous_error": previous_error,
        "repaired": result,
        "independent_optimality_gap": gap,
        "unchanged_objective_definitions": True,
        "training_script_unchanged": True,
        "new_forecaster_calls": 0,
        "new_model_fits": 0,
    }
    _write_json(output / "manifest.json", record)
    print(
        {
            "initial_gap": result["initial_optimality_gap"],
            "repaired_gap": result["optimality_gap"],
            "independent_gap": gap,
            "precision_steps": result["polishing_steps"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
