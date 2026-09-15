"""Compare more training histories with an equal-update original-source control."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from aligned_portfolio_io import decision_truth
from audit_shared_forecast_gate import replay_network
from latent_source_inputs import ROOT, read_json
from source_expansion_inputs import load_expanded_inputs
from train_calibrated_source_gates import probability_from_state
from train_latent_source_gates import aggregate
from train_shared_forecast_gate import SETTINGS

from tsfm_fais.routing.forecast_gate import SharedForecastGate, compose_forecasts, gate_objective
from tsfm_fais.routing.forecast_projection import (
    forecast_geometry,
    projection_targets,
    simplex_quadratic_weights,
)
from tsfm_fais.routing.utility import _family_weights
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    for name, path in {
        "base-root": "artifacts/iclr27-r7/latent-source-v001",
        "supplement-root": "artifacts/iclr27-r9/source-supplement-forecasts-v001",
        "accuracy-root": "artifacts/iclr27-r4/accuracy-development-v002",
        "reference-root": "artifacts/iclr27-r7/latent-source-gates-v001",
        "reference-audit": "artifacts/iclr27-r7/latent-source-audit-v001",
        "prepared-root": "artifacts/iclr27-r9/source-supplement-inputs-v001",
        "inventory": "artifacts/iclr27-r9/source-expansion-inventory-v001/manifest.json",
        "protocol": "docs/iclr2027/R9_SOURCE_EXPANSION_PROTOCOL.md",
    }.items():
        parser.add_argument("--" + name, type=Path, default=ROOT / path)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def fit_updates(frame, features, gram, alignment, indices, seed, updates):
    torch.manual_seed(seed)
    model = SharedForecastGate(features=97)
    weights = _family_weights(frame.iloc[indices])
    model.fit_normalization(features[indices], weights)
    initial = hashlib.sha256(
        b"".join(value.detach().numpy().tobytes() for value in model.parameters())
    ).hexdigest()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=SETTINGS["learning_rate"], weight_decay=SETTINGS["weight_decay"]
    )
    x, g, b, w = [
        torch.as_tensor(np.asarray(value), dtype=torch.float32)
        for value in (features[indices], gram[indices], alignment[indices], weights)
    ]
    generator = torch.Generator().manual_seed(seed + 100000)
    completed, epochs, history = 0, 0, []
    while completed < updates:
        epochs += 1
        for batch in torch.randperm(len(indices), generator=generator).split(128):
            loss = (
                gate_objective(model(x[batch]), g[batch], b[batch], "ensemble") * w[batch]
            ).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite source expansion objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(norm)):
                raise ValueError("nonfinite source expansion gradient")
            optimizer.step()
            completed += 1
            if completed % 25 == 0 or completed == updates:
                history.append({"update": completed, "relative_batch_loss": float(loss.detach())})
            if completed == updates:
                break
    return {
        "state_dict": model.state_dict(),
        "initial_parameter_sha256": initial,
        "train_indices_sha256": hashlib.sha256(np.asarray(indices, np.int64).tobytes()).hexdigest(),
        "training_origins": sorted(frame.iloc[indices].origin_id.unique()),
        "training_families": sorted(frame.iloc[indices].family_id.unique()),
        "updates": completed,
        "epochs_started": epochs,
        "history": history,
    }


def main():
    args = arguments(__doc__).parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed source expansion learning")
    reference = read_json(args.reference_root / "manifest.json")
    audit = read_json(args.reference_audit / "manifest.json")
    accuracy = read_json(args.accuracy_root / "manifest.json")
    if (
        audit["status"] != "completed"
        or audit["study_sha256"] != file_sha256(args.reference_root / "manifest.json")
        or accuracy["prediction_arrays"]["truth_z.npy"]
        != file_sha256(args.accuracy_root / "truth_z.npy")
    ):
        raise ValueError("original source references changed")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "input_module_sha256": file_sha256(ROOT / "scripts/source_expansion_inputs.py"),
        "protocol_sha256": file_sha256(args.protocol),
        "supplement_sha256": file_sha256(args.supplement_root / "manifest.json"),
        "base_sha256": file_sha256(args.base_root / "manifest.json"),
        "reference_sha256": file_sha256(args.reference_root / "manifest.json"),
        "reference_audit_sha256": file_sha256(args.reference_audit / "manifest.json"),
        "settings": SETTINGS,
        "primary": "expanded_future",
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial source expansion learning definitions changed")
    _write_json(output / "identity.json", identity)
    identity_sha = file_sha256(output / "identity.json")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoints, folds = [], []
    for model_id in ("chronos2", "timesfm2p5"):
        frame, arrays = load_expanded_inputs(
            args.base_root, args.supplement_root, args.accuracy_root, model_id
        )
        vectors, features = arrays["vectors"], arrays["features"]
        _, _, _, gram = forecast_geometry(vectors)
        train = np.flatnonzero(frame.split.to_numpy() == "train")
        labels = {
            "teacher": projection_targets(vectors, arrays["teacher"])["raw_projection"],
            "future": np.full(vectors.shape[:2], np.nan),
        }
        labels["future"][train] = projection_targets(vectors[train], arrays["truth"][train])[
            "raw_projection"
        ]
        for family in [None, *sorted(frame.family_id.unique())]:
            allowed = (
                train
                if family is None
                else np.flatnonzero(
                    (frame.split.to_numpy() == "train") & (frame.family_id.to_numpy() != family)
                )
            )
            original = allowed[frame.iloc[allowed].source_population.to_numpy() == "original"]
            validation = (
                np.array([], np.int64)
                if family is None
                else np.flatnonzero(
                    (frame.split.to_numpy() == "validation")
                    & (frame.family_id.to_numpy() == family)
                )
            )
            if family in set(frame.iloc[allowed].family_id) or set(
                frame.iloc[allowed].origin_id
            ) & set(frame.iloc[validation].origin_id):
                raise ValueError("an evaluation family or origin entered training")
            updates = int(np.ceil(len(allowed) / 128)) * 25
            directory = output / model_id / (family or "full_source")
            directory.mkdir(parents=True, exist_ok=True)
            predictions, controls = {}, {}
            if len(validation):
                old_fold = next(
                    row
                    for row in reference["folds"]
                    if row["model_id"] == model_id and row["held_family"] == family
                )
                path = args.reference_root / old_fold["prediction_path"]
                if file_sha256(path) != old_fold["prediction_sha256"]:
                    raise ValueError("a frozen original validation prediction changed")
                with np.load(path, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["validation_indices"], validation)
                    predictions.update(zip(saved["methods"].tolist(), saved["point"], strict=True))
            for label, alignment in labels.items():
                weights = _family_weights(frame.iloc[allowed])
                weights /= weights.sum()
                g = np.einsum("n,nab->ab", weights, gram[allowed])
                b = np.einsum("n,na->a", weights, alignment[allowed])
                fixed, gap, _ = simplex_quadratic_weights(g[None], b[None])
                single = int((g.diagonal() - 2 * b).argmin())
                controls[label] = {
                    "weights": fixed[0].tolist(),
                    "single_index": single,
                    "optimality_gap": float(gap[0]),
                }
                if len(validation):
                    predictions[f"expanded_fixed_{label}"] = compose_forecasts(
                        vectors[validation], np.broadcast_to(fixed[0], (len(validation), 7))
                    )
                    predictions[f"expanded_single_{label}"] = vectors[validation, single]
                for regime, indices in (("expanded", allowed), ("steps_matched", original)):
                    method = f"{regime}_{label}"
                    seed_weights = []
                    for seed in (5101, 5102, 5103):
                        metadata = {
                            "identity_sha256": identity_sha,
                            "model_id": model_id,
                            "held_family": family,
                            "regime": regime,
                            "label": label,
                            "seed": seed,
                            "updates": updates,
                        }
                        path = directory / f"{method}_{seed}.pt"
                        if not path.exists():
                            saved = {
                                **fit_updates(
                                    frame, features, gram, alignment, indices, seed, updates
                                ),
                                "metadata": metadata,
                            }
                            temporary = path.with_suffix(".tmp")
                            torch.save(saved, temporary)
                            temporary.replace(path)
                        saved = torch.load(path, map_location="cpu", weights_only=True)
                        if (
                            saved["metadata"] != metadata
                            or saved["initial_parameter_sha256"]
                            != reference["initial_parameters"][str(seed)]
                        ):
                            raise ValueError("a resumed model or matched initialization changed")
                        checkpoints.append(
                            {
                                "path": str(path.relative_to(output)),
                                "sha256": file_sha256(path),
                                **metadata,
                            }
                        )
                        if len(validation):
                            probability = probability_from_state(
                                saved["state_dict"], features[validation]
                            )
                            np.testing.assert_array_equal(
                                probability,
                                replay_network(saved["state_dict"], features[validation]),
                            )
                            seed_weights.append(probability)
                            predictions[f"{method}_seed{seed}"] = compose_forecasts(
                                vectors[validation], probability
                            )
                    if len(validation):
                        predictions[method] = compose_forecasts(
                            vectors[validation], np.mean(seed_weights, axis=0)
                        )
            control_path = directory / "controls.json"
            _write_json(control_path, controls)
            record = {
                "model_id": model_id,
                "held_family": family,
                "allowed": allowed.tolist(),
                "original": original.tolist(),
                "updates": updates,
                "controls_path": str(control_path.relative_to(output)),
                "controls_sha256": file_sha256(control_path),
            }
            if len(validation):
                path = directory / "predictions.npz"
                _save_npz(
                    path,
                    validation_indices=validation,
                    methods=np.asarray(list(predictions)),
                    point=np.stack(list(predictions.values())),
                )
                record.update(
                    prediction_path=str(path.relative_to(output)),
                    prediction_sha256=file_sha256(path),
                )
            folds.append(record)
            print(
                f"{model_id} {family or 'full_source'}: two populations and two labels complete",
                flush=True,
            )
    if len(checkpoints) != 384 or len(folds) != 32:
        raise ValueError("the source learning curve is incomplete")
    _write_json(
        output / "prediction_freeze.json",
        {
            "identity_sha256": identity_sha,
            "checkpoints": checkpoints,
            "folds": folds,
            "outer_validation_outcomes_read": False,
        },
    )
    rows = []
    for model_id in ("chronos2", "timesfm2p5"):
        frame, _ = load_expanded_inputs(
            args.base_root, args.supplement_root, args.accuracy_root, model_id
        )
        for fold in (
            row for row in folds if row["model_id"] == model_id and row["held_family"] is not None
        ):
            with np.load(output / fold["prediction_path"], allow_pickle=False) as saved:
                evaluation = frame.iloc[saved["validation_indices"]]
                truth = decision_truth(
                    evaluation, np.load(args.accuracy_root / "truth_z.npy", mmap_mode="r")
                )
                for method, point in zip(saved["methods"].tolist(), saved["point"], strict=True):
                    rows.append(
                        evaluation.assign(
                            model_id=model_id,
                            method=method,
                            mae=abs(point - truth).mean(1),
                            mse=((point - truth) ** 2).mean(1),
                        )
                    )
    scores = pd.concat(rows, ignore_index=True)
    if len(scores) != 117936 or scores.groupby("model_id").method.nunique().ne(42).any():
        raise ValueError("source learning-curve scoring is incomplete")
    episodes, families, summary = aggregate(scores)
    scores.to_parquet(output / "decision_scores.parquet", index=False)
    episodes.to_parquet(output / "episode_scores.parquet", index=False)
    families.to_csv(output / "family_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "checkpoints": checkpoints,
            "folds": folds,
            "score_rows": len(scores),
            "new_forecaster_calls": 0,
            "limits": "source development learning curve; independent audit required; eight families have additional eligible histories",
        },
    )


if __name__ == "__main__":
    main()
