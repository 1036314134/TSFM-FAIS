"""Build explicit synthetic histories and frozen-budget imputation candidates."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset  # noqa: F401 - initialize Arrow before Torch on Windows.
import torch
from conditional_future import condition_history, draw_history, scenarios
from latent_source_inputs import ROOT, read_json
from prepare_followup_inputs import fit_deep
from prepare_native_confirmation import ACTIONS, complete_candidates

from tsfm_fais.contracts import SeriesBatch
from tsfm_fais.forecasting.accuracy import PrefixStandardizer
from tsfm_fais.imputers import DEFAULT_REGISTRY, CandidateRunner
from tsfm_fais.imputers.motm import MOTMReference
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "docs/iclr2027/R14_CONDITIONAL_RISK_PROTOCOL.md"
    )
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed controlled histories")
    definitions = scenarios()
    serial = [
        {
            name: (value.tolist() if isinstance(value, np.ndarray) else value)
            for name, value in model.items()
        }
        for model in definitions
    ]
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "process_module_sha256": file_sha256(ROOT / "scripts/conditional_future.py"),
        "protocol_sha256": file_sha256(args.protocol),
        "models": serial,
        "prefix_length": 6144,
        "histories_per_process": 12,
        "history_length": 96,
        "horizon": 96,
        "mask_conditions": [
            ["complete", 0.0],
            ["random_point", 0.3],
            ["random_point", 0.6],
            ["tail_block", 0.3],
            ["tail_block", 0.6],
        ],
        "seed_scheme": "prefix 14400+g; history 14500+100g+i; mask 14600+1000g+10i+c; future 14700+1000g+10i+c",
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("controlled preparation definitions changed")
    _write_json(output / "identity.json", identity)
    identity_sha = file_sha256(output / "identity.json")
    torch.set_num_threads(1)
    reference = ROOT / "artifacts/iclr27-r5/motm-reference-v001"
    runtime = ROOT / "artifacts/iclr27-r5/motm-runtime-v001"
    imputer = MOTMReference(reference, runtime, device="cuda", ridge=0.5, batch_size=32)
    runner = CandidateRunner()
    records, prefixes, fitting = [], [], []
    for generator, model in enumerate(definitions):
        prefix = draw_history(model, 6144, 0, 14400 + generator)
        prefix_path = output / model["name"] / "prefix.npy"
        prefix_path.parent.mkdir(parents=True, exist_ok=True)
        if prefix_path.exists():
            np.testing.assert_array_equal(np.load(prefix_path), prefix)
        else:
            np.save(prefix_path, prefix, allow_pickle=False)
        scaler = PrefixStandardizer.fit(prefix)
        defaults = np.median(prefix, axis=0)
        starts = np.linspace(0, len(prefix) - 96, 64, dtype=int)
        windows = []
        for index, start in enumerate(starts):
            values = prefix[start : start + 96].copy()
            rng = np.random.default_rng(14450 + generator * 100 + index)
            values[rng.random(values.shape) < (0.1 + 0.1 * (index % 5))] = np.nan
            windows.append(values)
        values = np.stack(windows)
        batch = SeriesBatch(
            values,
            np.isfinite(values),
            item_ids=tuple(f"{model['name']}@{start}" for start in starts),
            metadata={"period": model["period"]},
        )
        artifacts, failures, fit_records = fit_deep(
            runner,
            batch,
            output / model["name"] / "imputers",
            identity_sha,
            model["period"],
            model["name"],
        )
        if failures:
            raise ValueError(f"a controlled prefix imputer failed: {failures}")
        fitting.extend(fit_records)
        prefix_batch = SeriesBatch(
            prefix[None], np.ones_like(prefix[None], bool), metadata={"period": model["period"]}
        )
        params = {"seasonal_lag": {"period": model["period"]}}
        for action in ACTIONS:
            if action not in artifacts and (
                action == "seasonal_lag" or DEFAULT_REGISTRY.get_spec(action).fit_scope == "dataset"
            ):
                artifacts[action] = runner.fit(
                    action, prefix_batch, {"period": model["period"]}, params=params.get(action)
                )
        prefixes.append(
            {
                "generator": generator,
                "name": model["name"],
                "path": str(prefix_path.relative_to(output)),
                "sha256": file_sha256(prefix_path),
                "mean": scaler.mean.tolist(),
                "scale": scaler.scale.tolist(),
            }
        )
        for origin in range(12):
            phase = (5 * origin) % model["period"]
            clean = draw_history(model, 96, phase, 14500 + 100 * generator + origin)
            for condition, (mechanism, rate) in enumerate(identity["mask_conditions"]):
                context = clean.copy()
                rng = np.random.default_rng(14600 + 1000 * generator + 10 * origin + condition)
                if mechanism == "random_point":
                    context[rng.random(context.shape) < rate] = np.nan
                elif mechanism == "tail_block":
                    context[-int(round(96 * rate)) :] = np.nan
                mean, covariance = condition_history(model, context, phase)
                check_mean, check_cov = condition_history(model, context, phase, scalar=True)
                np.testing.assert_allclose(mean, check_mean, rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(covariance, check_cov, rtol=1e-10, atol=1e-10)
                episode_id = f"{model['name']}|{origin}|{mechanism}|{rate}"
                path = (
                    output
                    / "episodes"
                    / (hashlib.sha256(episode_id.encode()).hexdigest()[:24] + ".npz")
                )
                if not path.exists():
                    local = dict(artifacts)
                    seasonal = dict(local["seasonal_lag"])
                    seasonal["profiles"] = np.roll(
                        seasonal["profiles"], -phase % model["period"], axis=1
                    )
                    local["seasonal_lag"] = seasonal
                    batch = SeriesBatch(
                        context[None],
                        np.isfinite(context[None]),
                        metadata={"period": model["period"]},
                    )
                    completed = runner.run_many(
                        ACTIONS,
                        batch,
                        local,
                        seed=14800 + 1000 * generator + 10 * origin + condition,
                        params=params,
                    )
                    candidates, coverage, statuses = complete_candidates(
                        context, completed, defaults, ACTIONS
                    )
                    if any(row["status"] != "success" for row in statuses):
                        raise ValueError(
                            "a controlled candidate failed; preserve the complete pool"
                        )
                    motm, diagnostics = imputer.impute(context, candidates[ACTIONS.index("locf")])
                    np.testing.assert_array_equal(
                        motm[np.isfinite(context)], context[np.isfinite(context)]
                    )
                    _save_npz(
                        path,
                        context=context,
                        clean_context=clean,
                        candidate_values=candidates,
                        candidate_ids=np.asarray(ACTIONS),
                        native_coverage=coverage,
                        motm_values=motm,
                        motm_diagnostics=np.asarray(json.dumps(diagnostics)),
                        posterior_mean=mean,
                        posterior_covariance=covariance,
                        identity_sha256=np.asarray(identity_sha),
                    )
                with np.load(path, allow_pickle=False) as saved:
                    if str(saved["identity_sha256"]) != identity_sha:
                        raise ValueError("a resumed controlled input changed")
                    np.testing.assert_array_equal(saved["context"], context)
                    np.testing.assert_allclose(saved["posterior_mean"], mean, rtol=0, atol=0)
                records.append(
                    {
                        "episode_id": episode_id,
                        "origin_id": f"{model['name']}|{origin}",
                        "generator": generator,
                        "history": origin,
                        "phase": phase,
                        "mechanism": mechanism,
                        "missing_rate": rate,
                        "condition": condition,
                        "future_seed": 14700 + 1000 * generator + 10 * origin + condition,
                        "path": str(path.relative_to(output)),
                        "sha256": file_sha256(path),
                    }
                )
        print(f"{model['name']}: independent prefix and 60 observed histories prepared", flush=True)
        del artifacts
        gc.collect()
    imputer.verify_frozen()
    if len(records) != 180 or len({row["origin_id"] for row in records}) != 36 or len(fitting) != 6:
        raise ValueError("controlled diagnostic population changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "prefixes": prefixes,
            "imputer_fits": fitting,
            "episodes": records,
            "motm_parameters_unchanged": True,
            "future_samples_generated": False,
        },
    )


if __name__ == "__main__":
    main()
