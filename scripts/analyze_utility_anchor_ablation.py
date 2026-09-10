"""Compare fixed utility references on cached development forecasts only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tsfm_fais.routing.utility import (  # noqa: E402
    UtilitySelector,
    _family_weights,
    family_macro,
    response_features,
)
from tsfm_fais.utility_experiment import file_sha256  # noqa: E402


class BoundedUtilityModel:
    """Scale a bounded conditional target by known forecast disagreement."""

    def __init__(self, model):
        self.model = model

    def fit(self, matrix, delta, sample_weight):
        distance = matrix["response.mean_change"].to_numpy()
        if np.any(np.abs(delta) > distance + 1e-7):
            raise ValueError("MASE differences exceed their forecast-distance bound")
        target = np.divide(delta, distance, out=np.zeros_like(delta), where=distance > 0)
        self.model.fit(matrix, np.clip(target, -1.0, 1.0), sample_weight=sample_weight)
        return self

    def predict(self, matrix):
        return (
            np.clip(self.model.predict(matrix), -1.0, 1.0)
            * matrix["response.mean_change"].to_numpy()
        )


def anchored_fit(
    frame: pd.DataFrame, anchor_id: str, use_response: bool, objective: str
) -> UtilitySelector:
    """Keep the reference fixed; retain the original learner and weighting."""
    selector = UtilitySelector(use_response=use_response).fit(frame)
    baseline = frame[frame.candidate_id == anchor_id].set_index("episode_id").loss
    target = frame.loss.to_numpy() - frame.episode_id.map(baseline).to_numpy()
    selector.baseline_id = anchor_id
    selector.model.set_params(
        objective="regression" if objective == "bounded_regression" else objective
    )
    if objective == "bounded_regression" and use_response:
        selector.model = BoundedUtilityModel(selector.model)
    selector.model.fit(selector._matrix(frame), target, sample_weight=_family_weights(frame))
    return selector


def reanchor_features(
    root: Path, frame: pd.DataFrame, model_id: str, anchor_id: str
) -> pd.DataFrame:
    manifest = json.loads((root / "episodes_manifest.json").read_text(encoding="utf-8"))
    targets = manifest["identity"]["config"]["target_indices"]
    result = frame.copy().set_index(["episode_id", "candidate_id"])
    available_actions = set(frame.candidate_id)
    for record in manifest["episodes"]:
        episode_path = root / record["path"]
        with (
            np.load(episode_path, allow_pickle=False) as episode,
            np.load(
                root / model_id / "predictions" / episode_path.name, allow_pickle=False
            ) as forecast,
        ):
            ids = episode["candidate_ids"].tolist()
            point = forecast["point"]
            if len(point) == len(ids) + 1:
                ids.append("native_missing")
            anchor = point[ids.index(anchor_id)]
            pool = np.median(
                point[[ids.index(action) for action in sorted(available_actions)]], axis=0
            )
            last = episode["candidate_values"][ids.index("locf"), -1, targets]
            for index, candidate_id in enumerate(ids):
                if candidate_id not in available_actions:
                    continue
                features = response_features(
                    point[index],
                    anchor,
                    pool,
                    last,
                    episode["mase_scales"][targets],
                    forecast["quantiles"][index],
                )
                for key, value in features.items():
                    result.loc[(record["episode_id"], candidate_id), key] = value
    return result.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--models", default="chronos2,timesfm2p5")
    parser.add_argument(
        "--objective",
        choices=("regression_l1", "regression", "bounded_regression"),
        default="regression_l1",
    )
    args = parser.parse_args()
    output = args.run_root / f"analysis-anchor-{args.objective}-v002"
    output.mkdir(parents=True, exist_ok=True)
    rows, folds = [], []
    for model_id in args.models.split(","):
        original = pd.read_parquet(args.run_root / model_id / "utility_rows.parquet")
        anchors = (
            ["locf", "native_missing"]
            if "native_missing" in set(original.candidate_id)
            else ["locf"]
        )
        for anchor_id in anchors:
            frame = reanchor_features(args.run_root, original, model_id, anchor_id)
            for held_family in sorted(frame.family_id.unique()):
                training = frame[(frame.family_id != held_family) & (frame.split == "train")]
                calibration = frame[
                    (frame.family_id != held_family) & (frame.split == "validation")
                ]
                validation = frame[(frame.family_id == held_family) & (frame.split == "validation")]
                selected = {"fixed_anchor": validation[validation.candidate_id == anchor_id]}
                for name, use_response in [("static", False), ("response", True)]:
                    selector = anchored_fit(
                        training, anchor_id, use_response, args.objective
                    ).calibrate(calibration)
                    selected[name] = selector.select(validation)
                    selected[name + "_gated"] = selector.select(validation, gated=True)
                    folds.append(
                        {
                            "model_id": model_id,
                            "anchor_id": anchor_id,
                            "held_family": held_family,
                            "method": name,
                            "calibration": selector.calibration,
                        }
                    )
                selected["forecast_medoid"] = validation.sort_values(
                    ["episode_id", "response.pool_distance", "candidate_id"]
                ).drop_duplicates("episode_id")
                for name, choice in selected.items():
                    scores = choice[
                        [
                            "episode_id",
                            "origin_id",
                            "family_id",
                            "dataset_id",
                            "candidate_id",
                            "loss",
                        ]
                    ].copy()
                    scores["model_id"] = model_id
                    scores["anchor_id"] = anchor_id
                    scores["method"] = name
                    rows.append(scores)
            print(
                json.dumps({"model": model_id, "anchor": anchor_id, "status": "completed"}),
                flush=True,
            )
    results = pd.concat(rows, ignore_index=True)
    summary = []
    for (model_id, anchor, method), group in results.groupby(["model_id", "anchor_id", "method"]):
        summary.append(
            {
                "model_id": model_id,
                "anchor_id": anchor,
                "method": method,
                "family_macro_mase": family_macro(group),
                "episode_count": int(group.episode_id.nunique()),
            }
        )
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
    results.to_csv(output / "episode_results.csv", index=False)
    (output / "folds.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "evidence_role": "development",
                "regression_objective": args.objective,
                "bounded_response": "for bounded_regression, response targets are delta divided by mean scaled forecast distance; predictions are clipped to [-1,1] and rescaled; static control uses ordinary squared-loss regression",
                "script_sha256": file_sha256(Path(__file__)),
                "source_episode_manifest_sha256": file_sha256(
                    args.run_root / "episodes_manifest.json"
                ),
                "selection_rule": "fixed reference losses; response features reanchored to the same reference; unchanged learner and margin grid",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
