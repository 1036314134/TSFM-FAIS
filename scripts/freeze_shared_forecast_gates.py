"""Fit the unchanged small gates on source training data before transfer diagnostics."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aligned_portfolio_io import decision_vectors, load_prepared_model  # noqa: E402
from train_shared_forecast_gate import SETTINGS, fit_gate  # noqa: E402

from tsfm_fais.routing.forecast_gate import teacher_quadratics  # noqa: E402
from tsfm_fais.routing.forecast_projection import simplex_quadratic_weights  # noqa: E402
from tsfm_fais.routing.utility import _family_weights  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("aligned-root", "accuracy-root", "audit-root", "diagnostic-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve the completed shared-gate source freeze")
    if args.diagnostic_root.exists() and any(args.diagnostic_root.rglob("*predictions*.npz")):
        raise ValueError("do not refit after this changed-gate diagnostic has started")
    prep = json.loads((args.aligned_root / "manifest.json").read_text(encoding="utf-8"))
    accuracy = json.loads((args.accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((args.audit_root / "manifest.json").read_text(encoding="utf-8"))
    if audit["status"] != "completed" or audit["verified_seed_checkpoints"] != 180:
        raise ValueError("audit the matched source-development study before final fitting")
    if prep["identity"]["accuracy_manifest_sha256"] != file_sha256(
        args.accuracy_root / "manifest.json"
    ):
        raise ValueError("source representations and forecast vectors disagree")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "settings": SETTINGS,
        "aligned_manifest_sha256": file_sha256(args.aligned_root / "manifest.json"),
        "accuracy_manifest_sha256": file_sha256(args.accuracy_root / "manifest.json"),
        "audit_sha256": file_sha256(args.audit_root / "manifest.json"),
        "module_sha256": file_sha256(ROOT / "src/tsfm_fais/routing/forecast_gate.py"),
        "trainer_sha256": file_sha256(ROOT / "scripts/train_shared_forecast_gate.py"),
        "source_split": "train only",
        "source_outcome_supervision": False,
        "diagnostic_root": str(args.diagnostic_root.resolve()),
        "target_evaluation_role": "previously used follow-up data; descriptive transfer diagnostic only",
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("partial source-gate freeze changed identity")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    models, controls = [], {}
    for model_id in ("chronos2", "timesfm2p5"):
        info, decisions, arrays = load_prepared_model(args.aligned_root, prep, model_id)
        indices = np.flatnonzero(decisions.split.to_numpy() == "train")
        training = decisions.iloc[indices]
        if (
            training.origin_id.nunique() != 165
            or training.family_id.nunique() != 15
            or training.source_episode_id.nunique() != 5940
        ):
            raise ValueError("the original source fitting population changed")
        path = args.accuracy_root / f"{model_id}_point_z.npy"
        if file_sha256(path) != accuracy["prediction_arrays"][path.name]:
            raise ValueError("source candidate predictions changed")
        bank = np.load(path, mmap_mode="r")[
            :, [accuracy["action_orders"][model_id].index(name) for name in info["actions"]]
        ]
        vectors = decision_vectors(training, bank)
        gram, alignment = teacher_quadratics(vectors, arrays["direct_risk"][indices, :7])
        features, weights = arrays["features"][indices, :7, :33], _family_weights(training)
        probability = weights / weights.sum()
        mean_gram = np.einsum("n,nab->ab", probability, gram)
        mean_alignment = np.einsum("n,na->a", probability, alignment)
        fixed, gap, _ = simplex_quadratic_weights(mean_gram[None], mean_alignment[None])
        controls[model_id] = {
            "actions": info["actions"],
            "convex_weights": fixed[0].tolist(),
            "single_index": int((mean_gram.diagonal() - 2 * mean_alignment).argmin()),
            "optimality_gap": float(gap[0]),
        }
        train_ids_sha = hashlib.sha256(indices.tobytes()).hexdigest()
        for objective in ("ensemble", "member"):
            for seed in SETTINGS["seeds"]:
                path = output / f"{model_id}_{objective}_{seed}.pt"
                if path.exists():
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    if (
                        saved["identity_sha256"] != identity_sha
                        or saved["train_ids_sha256"] != train_ids_sha
                        or saved["seed"] != seed
                    ):
                        raise ValueError("a frozen seed checkpoint changed identity")
                else:
                    model, history = fit_gate(
                        features, gram, alignment, weights, kind=objective, seed=seed
                    )
                    saved = {
                        "state_dict": model.state_dict(),
                        "training_history": history,
                        "seed": seed,
                        "identity_sha256": identity_sha,
                        "train_ids_sha256": train_ids_sha,
                        "training_origins": sorted(training.origin_id.unique()),
                        "training_families": sorted(training.family_id.unique()),
                    }
                    temporary = path.with_suffix(".tmp")
                    torch.save(saved, temporary)
                    temporary.replace(path)
                models.append(
                    {
                        "model_id": model_id,
                        "objective": objective,
                        "seed": seed,
                        "path": path.name,
                        "sha256": file_sha256(path),
                    }
                )
                print(json.dumps({"completed_models": len(models), "total_models": 12}), flush=True)
    _write_json(output / "controls.json", controls)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "models": models,
            "controls_sha256": file_sha256(output / "controls.json"),
            "new_forecaster_calls": 0,
            "target_features_read": False,
            "limits": "source-only model fitting; subsequent reuse of the 823-task cohort is not independent confirmation",
        },
    )


if __name__ == "__main__":
    main()
