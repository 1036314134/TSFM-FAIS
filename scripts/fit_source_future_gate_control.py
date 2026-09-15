"""Freeze a matched gate trained with actual source future labels as a comparator."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aligned_portfolio_io import decision_truth, decision_vectors, load_prepared_model  # noqa: E402
from train_shared_forecast_gate import SETTINGS, fit_gate  # noqa: E402

from tsfm_fais.routing.forecast_projection import (  # noqa: E402
    forecast_geometry,
    projection_targets,
)
from tsfm_fais.routing.utility import _family_weights  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "aligned-root",
        "accuracy-root",
        "teacher-bundle",
        "method-freeze",
        "protocol",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve the completed source-future control")
    binding = json.loads(args.method_freeze.read_text(encoding="utf-8"))
    confirmation = Path(binding["identity"]["confirmation_root"])
    if confirmation.exists() and any(confirmation.rglob("*predictions*.npz")):
        raise ValueError("new confirmation forecasts exist; do not refit the source control")
    prep = json.loads((args.aligned_root / "manifest.json").read_text(encoding="utf-8"))
    accuracy = json.loads((args.accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    teacher_bundle = json.loads((args.teacher_bundle / "manifest.json").read_text(encoding="utf-8"))
    if teacher_bundle["identity"]["settings"] != SETTINGS or teacher_bundle["identity"][
        "trainer_sha256"
    ] != file_sha256(ROOT / "scripts/train_shared_forecast_gate.py"):
        raise ValueError("the matched teacher-gate settings changed")
    if prep["identity"]["accuracy_manifest_sha256"] != file_sha256(
        args.accuracy_root / "manifest.json"
    ):
        raise ValueError("source feature and forecasting inputs differ")
    truth_path = args.accuracy_root / "truth_z.npy"
    if file_sha256(truth_path) != accuracy["prediction_arrays"][truth_path.name]:
        raise ValueError("the source future labels changed")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "settings": SETTINGS,
        "protocol_sha256": file_sha256(args.protocol),
        "method_freeze_sha256": file_sha256(args.method_freeze),
        "teacher_bundle_sha256": file_sha256(args.teacher_bundle / "manifest.json"),
        "aligned_sha256": file_sha256(args.aligned_root / "manifest.json"),
        "accuracy_sha256": file_sha256(args.accuracy_root / "manifest.json"),
        "source_outcome_supervision": True,
        "target_cohort_features_or_outcomes_read": False,
        "role": "matched actual-source-future supervision comparator; primary method unchanged",
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("a partial source control changed identity")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    models = []
    for model_id in ("chronos2", "timesfm2p5"):
        info, decisions, arrays = load_prepared_model(args.aligned_root, prep, model_id)
        indices = np.flatnonzero(decisions.split.to_numpy() == "train")
        training = decisions.iloc[indices]
        if (
            training.origin_id.nunique() != 165
            or training.family_id.nunique() != 15
            or training.source_episode_id.nunique() != 5940
        ):
            raise ValueError("the matched source training population changed")
        path = args.accuracy_root / f"{model_id}_point_z.npy"
        if file_sha256(path) != accuracy["prediction_arrays"][path.name]:
            raise ValueError("source candidate forecasts changed")
        bank = np.load(path, mmap_mode="r")[
            :, [accuracy["action_orders"][model_id].index(name) for name in info["actions"]]
        ]
        vectors = decision_vectors(training, bank)
        future = decision_truth(training, np.load(truth_path, mmap_mode="r"))
        _, _, _, gram = forecast_geometry(vectors)
        alignment = projection_targets(vectors, future)["raw_projection"]
        features, weights = arrays["features"][indices, :7, :33], _family_weights(training)
        train_ids_sha = hashlib.sha256(indices.tobytes()).hexdigest()
        for seed in SETTINGS["seeds"]:
            path = output / f"{model_id}_future_{seed}.pt"
            if path.exists():
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if (
                    saved["identity_sha256"] != identity_sha
                    or saved["train_ids_sha256"] != train_ids_sha
                ):
                    raise ValueError("a partial control model changed identity")
            else:
                model, history = fit_gate(
                    features, gram, alignment, weights, kind="ensemble", seed=seed
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
            matched = next(
                row
                for row in teacher_bundle["models"]
                if row["model_id"] == model_id
                and row["objective"] == "ensemble"
                and row["seed"] == seed
            )
            teacher_path = args.teacher_bundle / matched["path"]
            if file_sha256(teacher_path) != matched["sha256"]:
                raise ValueError("the comparison teacher gate changed")
            teacher = torch.load(teacher_path, map_location="cpu", weights_only=True)
            if (
                teacher["training_origins"] != saved["training_origins"]
                or teacher["training_families"] != saved["training_families"]
            ):
                raise ValueError("the matched controls used different training populations")
            for name in ("feature_mean", "feature_scale"):
                if not torch.equal(teacher["state_dict"][name], saved["state_dict"][name]):
                    raise ValueError("the matched controls used different feature normalization")
            models.append(
                {
                    "model_id": model_id,
                    "seed": seed,
                    "path": path.name,
                    "sha256": file_sha256(path),
                    "actions": info["actions"],
                }
            )
            print(json.dumps({"completed_models": len(models), "total_models": 6}), flush=True)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "models": models,
            "normalization_and_training_population_match": True,
            "new_forecaster_calls": 0,
            "limits": "source-supervised comparator, not a change to the primary frozen teacher gate",
        },
    )


if __name__ == "__main__":
    main()
