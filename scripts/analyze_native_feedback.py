"""Audit a two-action historical selector using existing forecast caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsfm_fais.routing.recent_feedback import select_feedback, validate_feedback_end  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def source_fixed_choice(references, model, held_out_family, finite_actions):
    training = references[
        (references.model_id == model)
        & (references.split == "train")
        & (references.target_slot == -1)
        & (references.family_id != held_out_family)
        & references.candidate_id.isin(finite_actions)
    ]
    if training.empty or set(training.candidate_id) != set(finite_actions):
        raise ValueError("every finite candidate needs other-family source training evidence")
    if not np.isfinite(training[["mae", "mse"]].to_numpy()).all():
        raise ValueError("source training errors must be finite")
    macro = (
        training.groupby(["candidate_id", "family_id", "dataset_id"])[["mae", "mse"]]
        .mean()
        .groupby(level=["candidate_id", "family_id"])
        .mean()
        .groupby(level="candidate_id")
        .mean()
    )
    scales = {metric: max(float(macro.loc["locf", metric]), 1e-12) for metric in ("mae", "mse")}
    risk = 0.5 * (macro.mae / scales["mae"] + macro.mse / scales["mse"])
    selected = min(risk.index, key=lambda action: (risk[action], action != "locf", action))
    return selected, scales


def restricted_choice(history, fixed_action, normalizers, *, per_target):
    """Read completed historical risks only; ties or missing feedback retain native input."""
    actions = ["guarded_direct", fixed_action]
    selected = history[history.candidate_id.isin(actions)]
    if selected.duplicated(["candidate_id", "target_slot"]).any():
        raise ValueError("duplicate historical risk rows")
    if set(selected.candidate_id) != set(actions):
        raise ValueError("the two historical candidates are not both present")
    slots = sorted(selected.target_slot.unique())
    if slots != list(range(len(slots))):
        raise ValueError("target slots must be contiguous and start at zero")
    counts = selected.pivot(index="candidate_id", columns="target_slot", values="observed_count")
    counts = counts.reindex(index=actions, columns=slots).to_numpy()
    if not np.isfinite(counts).all() or not np.array_equal(counts[0], counts[1]):
        raise ValueError("both candidates must use exactly the same observed feedback cells")
    risks = {
        metric: selected.pivot(
            index="candidate_id", columns="target_slot", values="historical_" + metric
        )
        .reindex(index=actions, columns=slots)
        .to_numpy()
        for metric in ("mae", "mse")
    }
    choice = select_feedback(risks, "joint", normalizers, 0, per_target=per_target)
    positions = choice if per_target else (choice,) * len(slots)
    return tuple(actions[position] for position in positions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--accuracy-root", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed native-feedback evidence")
    paths = {
        "source": args.source_root / "episodes_manifest.json",
        "accuracy": args.accuracy_root / "manifest.json",
        "plan": args.probe_root / "plan.json",
        "analysis": args.probe_root / "analysis-v001/manifest.json",
    }
    payload = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    digests = {name: file_sha256(path) for name, path in paths.items()}
    source, accuracy, plan, analysis = [payload[name] for name in paths]
    if (
        accuracy["source_episode_manifest_sha256"] != digests["source"]
        or plan["source_manifest_sha256"] != digests["source"]
        or analysis["status"] != "completed"
        or analysis["accuracy_manifest_sha256"] != digests["accuracy"]
        or analysis["plan_sha256"] != digests["plan"]
    ):
        raise ValueError("historical risks and evaluation forecasts have different identities")
    if (
        file_sha256(args.probe_root / "prepared_manifest.json")
        != analysis["prepared_manifest_sha256"]
    ):
        raise ValueError("observed historical inputs changed")
    probes = {row["probe_id"]: row for row in plan["probes"]}
    for link in plan["links"]:
        probe = probes[link["probe_id"]]
        validate_feedback_end(probe["origin"], probe["horizon"], link["origin"])
    risk_path = args.probe_root / "analysis-v001/historical_risks.parquet"
    history = pd.read_parquet(risk_path)
    references = pd.read_parquet(
        args.accuracy_root / "candidate_accuracy.parquet",
        filters=[("split", "==", "train"), ("target_slot", "==", -1)],
        columns=[
            "model_id",
            "split",
            "target_slot",
            "family_id",
            "dataset_id",
            "candidate_id",
            "mae",
            "mse",
        ],
    )
    finite = source["identity"]["config"]["candidate_ids"]
    priors, decisions = {}, []
    for (model, episode, horizon, count), group in history.groupby(
        ["model_id", "episode_id", "probe_horizon", "probe_count"]
    ):
        if horizon not in plan["horizons"] or count not in (1, 2):
            raise ValueError("unexpected historical horizon or probe count")
        family = group.family_id.unique()
        if len(family) != 1:
            raise ValueError("a decision spans multiple families")
        key = (model, family[0])
        if key not in priors:
            priors[key] = source_fixed_choice(references, model, family[0], finite)
        fixed, scales = priors[key]
        for metric in ("mae", "mse"):
            if not np.allclose(group["source_" + metric + "_normalizer"], scales[metric]):
                raise ValueError(
                    "source-only joint objective does not match historical risk export"
                )
        for per_target in (False, True) if model == "timesfm2p5" else (False,):
            choices = restricted_choice(group, fixed, scales, per_target=per_target)
            decisions.append(
                {
                    "model_id": model,
                    "episode_id": episode,
                    "probe_horizon": horizon,
                    "probe_count": count,
                    "method": "native_fixed_" + ("target" if per_target else "sequence"),
                    "source_fixed_action": fixed,
                    "selected_action_ids": json.dumps(choices),
                }
            )
    # Complete the decision trace before opening any current future labels.
    decisions = pd.DataFrame(decisions)
    expected = set(plan["decision_episode_ids"])
    for _, group in decisions.groupby(["model_id", "method", "probe_count", "probe_horizon"]):
        if set(group.episode_id) != expected or group.episode_id.duplicated().any():
            raise ValueError("every selector must cover exactly the registered development tasks")
    records = {row["episode_id"]: (index, row) for index, row in enumerate(source["episodes"])}
    truth = np.load(args.accuracy_root / "truth_z.npy", mmap_mode="r")
    results = []
    for model, group in decisions.groupby("model_id"):
        if (
            file_sha256(args.probe_root / model / "manifest.json")
            != analysis["forecast_manifests"][model]
        ):
            raise ValueError("historical forecast manifest changed")
        point = np.load(args.accuracy_root / f"{model}_point_z.npy", mmap_mode="r")
        actions = accuracy["action_orders"][model]
        positions = [actions.index(action) for action in finite + ["guarded_direct"]]
        for episode, choices in group.groupby("episode_id"):
            index, meta = records[episode]
            fixed, scales = priors[(model, meta["family_id"])]
            forecasts = point[index]

            def record(
                method,
                count,
                prediction,
                native_fraction=np.nan,
                *,
                index=index,
                model=model,
                episode=episode,
                meta=meta,
            ):
                error = prediction - truth[index]
                if not np.isfinite(error).all():
                    raise ValueError("current forecast errors must be finite")
                results.append(
                    {
                        "model_id": model,
                        "episode_id": episode,
                        "family_id": meta["family_id"],
                        "dataset_id": meta["dataset_id"],
                        "mechanism": meta["mechanism"],
                        "missing_rate": meta["missing_rate"],
                        "method": method,
                        "probe_count": count,
                        "mae": float(np.abs(error).mean()),
                        "mse": float((error**2).mean()),
                        "native_fraction": native_fraction,
                    }
                )

            record("forecast_median_guarded", 0, np.median(forecasts[positions], axis=0))
            record("source_fixed", 0, forecasts[actions.index(fixed)], 0.0)
            record("guarded_direct", 0, forecasts[actions.index("guarded_direct")], 1.0)
            for choice in choices.itertuples():
                selected = json.loads(choice.selected_action_ids)
                prediction = np.column_stack(
                    [
                        forecasts[actions.index(action), :, slot]
                        for slot, action in enumerate(selected)
                    ]
                )
                record(
                    choice.method,
                    choice.probe_count,
                    prediction,
                    float(np.mean(np.array(selected) == "guarded_direct")),
                )
    data = pd.DataFrame(results)
    keys = ["model_id", "method", "probe_count"]
    family = data.groupby(keys + ["family_id", "dataset_id"])[
        ["mae", "mse", "native_fraction"]
    ].mean()
    family = family.groupby(level=keys + ["family_id"]).mean().reset_index()
    summary = family.groupby(keys)[["mae", "mse", "native_fraction"]].mean().reset_index()
    output.mkdir(parents=True, exist_ok=True)
    decisions.to_parquet(output / "decisions.parquet", index=False)
    data.to_parquet(output / "episode_results.parquet", index=False)
    family.to_csv(output / "family_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "evidence_role": "development_action_pool_ablation",
            "source_manifests": digests,
            "historical_risks_sha256": file_sha256(risk_path),
            "script_sha256": file_sha256(Path(__file__)),
            "decision_count": len(expected),
            "training": "finite action chosen from other-family source training labels only",
            "information": "routing reads only completed, originally observed historical errors; current future is evaluation-only",
            "input_protocol": "original R4 cached model inputs; downstream errors use shared training-prefix units; these are not the input-standardized R5 composer runs",
            "native_fraction_definition": "fraction selecting guarded_direct, including its declared all-missing fallback; not the fraction of native SDK executions",
            "interpretation": "two-action simplification of the existing historical selector, not a new learned composer or independent confirmation; no new model inference",
        },
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
