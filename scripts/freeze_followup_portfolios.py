"""Fit the matched portfolio rules on all source training histories before follow-up evaluation."""

import argparse
import json
import sys
from pathlib import Path
from time import monotonic

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aligned_portfolio_io import load_prepared_model  # noqa: E402
from train_pairwise_portfolio import triple_labels, triple_rows  # noqa: E402

from tsfm_fais.routing.pairwise_utility import PairwiseUtilitySelector  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def assert_unstarted_confirmation(directory):
    if directory.exists() and any(
        "predictions" in path.relative_to(directory).parts for path in directory.rglob("*.npz")
    ):
        raise ValueError("follow-up forecasts already exist; do not refit the frozen method")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-root", "audit-root", "confirmation-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve the completed source freeze")
    assert_unstarted_confirmation(args.confirmation_root)
    prep = json.loads((args.prepared_root / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((args.audit_root / "manifest.json").read_text(encoding="utf-8"))
    if (
        prep["status"] != "completed"
        or audit["status"] != "completed"
        or audit["verified_comparators"] != 35700
        or audit["maximum_metric_difference"] != 0
    ):
        raise ValueError("complete and audit the matched source study first")
    for name, expected in prep["identity"]["source_sha256"].items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError("the source feature recipe changed after development")
    identity = {
        "prepared_manifest_sha256": file_sha256(args.prepared_root / "manifest.json"),
        "development_audit_sha256": file_sha256(args.audit_root / "manifest.json"),
        "primary_label_kind": "median_risk",
        "control_label_kind": "member_risk",
        "source_split": "train only",
        "confirmation_root": str(args.confirmation_root.resolve()),
        "script_sha256": file_sha256(Path(__file__)),
        "source_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "scripts/train_pairwise_portfolio.py",
                "scripts/aligned_portfolio_io.py",
                "src/tsfm_fais/routing/pairwise_utility.py",
                "src/tsfm_fais/routing/aligned_portfolio.py",
            )
        },
        "candidate_count": 7,
        "portfolio_count": 35,
        "aggregation_size": 3,
        "context_length": 96,
        "horizon": 96,
        "target_indices": [0, 1],
        "metrics": "training-prefix-standardized downstream MAE and MSE",
        "new_confirmation_status": "not_started",
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "identity.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != identity:
        raise ValueError("the source freeze identity changed")
    _write_json(path, identity)
    identity_sha = file_sha256(path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    records = []
    for model in ("chronos2", "timesfm2p5"):
        info, decisions, arrays = load_prepared_model(args.prepared_root, prep, model)
        indices = np.flatnonzero(decisions.split.to_numpy() == "train")
        training = decisions.iloc[indices]
        if (
            training.origin_id.nunique() != 165
            or training.family_id.nunique() != 15
            or training.source_episode_id.nunique() != 5940
        ):
            raise ValueError("the final source fitting population changed")
        for kind in ("median_risk", "member_risk"):
            marker = output / f"{model}_{kind}.json"
            model_path = output / f"{model}_{kind}.joblib"
            if marker.exists():
                record = json.loads(marker.read_text(encoding="utf-8"))
                if (
                    record["identity_sha256"] != identity_sha
                    or file_sha256(model_path) != record["model_sha256"]
                ):
                    raise ValueError("a partially frozen model changed")
            else:
                assert_unstarted_confirmation(args.confirmation_root)
                started = monotonic()
                frame = triple_rows(decisions, arrays["features"], info, indices)
                labels = triple_labels(arrays["direct_risk"], info["members"], kind)
                frame["loss"] = labels[indices].reshape(-1)
                learner = PairwiseUtilitySelector(
                    feature_prefixes=("static.", "response.", "member.")
                ).fit(frame)
                if len(learner.models) != 595 or set(learner.feature_names) != set(
                    info["feature_names"]
                ):
                    raise ValueError("the final model changed capacity or input features")
                joblib.dump(learner, model_path, compress=3)
                record = {
                    "status": "completed",
                    "identity_sha256": identity_sha,
                    "model_id": model,
                    "label_kind": kind,
                    "model_path": model_path.name,
                    "model_sha256": file_sha256(model_path),
                    "training_origins": sorted(training.origin_id.unique()),
                    "training_families": sorted(training.family_id.unique()),
                    "feature_names": list(learner.feature_names),
                    "candidate_ids": list(learner.candidate_ids),
                    "actions": info["actions"],
                    "option_names": info["option_names"],
                    "members": info["members"],
                    "source_outcome_supervision": False,
                    "seconds": monotonic() - started,
                }
                _write_json(marker, record)
            records.append(
                {
                    "model_id": model,
                    "label_kind": kind,
                    "path": marker.name,
                    "sha256": file_sha256(marker),
                }
            )
            _write_json(
                output / "progress.json",
                {"status": "fitting", "completed_models": len(records), "total_models": 4},
            )
            print(
                json.dumps({"model": model, "label": kind, "seconds": record["seconds"]}),
                flush=True,
            )
    assert_unstarted_confirmation(args.confirmation_root)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "models": records,
            "new_forecaster_calls": 0,
            "information_boundary": "all 15 source training families only; no validation or confirmation examples used for fitting",
            "limits": "source model freeze only; no follow-up performance has been evaluated",
        },
    )


if __name__ == "__main__":
    main()
