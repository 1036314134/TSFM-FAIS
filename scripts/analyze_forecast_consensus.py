"""Evaluate label-free forecast-consensus candidate selectors on saved episodes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from tsfm_fais.config import load_config
from tsfm_fais.data import stable_seed
from tsfm_fais.evaluation import (
    _build_forecast_runner,
    _forecast_spec,
    _load_saved_candidates,
    _set_forecast_seed,
)
from tsfm_fais.imputers import DEFAULT_REGISTRY

TRUSTED_CANDIDATES = (
    "locf",
    "linear_interp",
    "seasonal_lag",
    "kalman_local_trend",
    "kalman_ar",
    "stl_kalman",
    "knn_multivariate",
    "mice",
    "missforest",
    "softimpute",
)
CLASSICAL_CANDIDATES = TRUSTED_CANDIDATES[:6] + ("gp_rbf",)
ANCHOR_CANDIDATES = ("locf", "linear_interp", "seasonal_lag", "kalman_ar")
FULL_COVERAGE_CANDIDATES = (
    "knn_multivariate",
    "mice",
    "missforest",
    "softimpute",
    "gpvae",
    "saits",
    "imputeformer",
    "helix",
    "timemixerpp",
    "totem",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--impute-artifact", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--forecaster-id", required=True)
    parser.add_argument("--forecaster-artifact", required=True)
    parser.add_argument(
        "--forecast-seed-scope",
        default="routing_forecast_consensus",
        help="stable-seed namespace used by the routing forecast call",
    )
    return parser.parse_args()


def _records(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _valid_candidates(archive: np.lib.npyio.NpzFile) -> dict[str, dict[str, object]]:
    observed = np.asarray(archive["observed_mask"], dtype=bool)
    candidates = _load_saved_candidates(archive, observed)
    has_tail_missing = bool((~observed[-1]).any())
    period = int(np.asarray(archive.get("period", 1)).reshape(-1)[0])
    return {
        candidate_id: metadata
        for candidate_id, metadata in candidates.items()
        if bool(metadata["native_valid"])
        and (not has_tail_missing or DEFAULT_REGISTRY.get_spec(candidate_id).supports_tail)
        and (not DEFAULT_REGISTRY.get_spec(candidate_id).requires_period or period >= 2)
    }


def _medoid(
    candidate_ids: tuple[str, ...],
    predictions: np.ndarray,
    scale: np.ndarray,
    allowed: tuple[str, ...] | None = None,
) -> str:
    indices = [
        index
        for index, candidate_id in enumerate(candidate_ids)
        if allowed is None or candidate_id in allowed
    ]
    if not indices:
        indices = list(range(len(candidate_ids)))
    normalized = predictions[indices] / scale[None, None, :]
    center = np.median(normalized, axis=0)
    scores = np.mean(np.abs(normalized - center[None, ...]), axis=(1, 2))
    return min(
        (float(scores[offset]), candidate_ids[index]) for offset, index in enumerate(indices)
    )[1]


def _blockwise_value_medoid(
    candidate_ids: tuple[str, ...],
    candidates: dict[str, dict[str, object]],
    observed_mask: np.ndarray,
    *,
    allowed: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    eligible = tuple(
        candidate_id for candidate_id in candidate_ids if allowed is None or candidate_id in allowed
    )
    if not eligible:
        eligible = candidate_ids
    values = np.stack(
        [np.asarray(candidates[candidate_id]["values"], dtype=float) for candidate_id in eligible]
    )
    completed = values[0].copy()
    selected: Counter[str] = Counter()
    for channel in range(observed_mask.shape[1]):
        missing = ~observed_mask[:, channel]
        padded = np.pad(missing.astype(np.int8), (1, 1))
        changes = np.flatnonzero(np.diff(padded))
        for start, end in changes.reshape(-1, 2):
            block_values = values[:, start:end, channel]
            center = np.median(block_values, axis=0)
            scores = np.mean(np.abs(block_values - center[None, :]), axis=1)
            index = min(
                range(len(eligible)),
                key=lambda offset: (float(scores[offset]), eligible[offset]),
            )
            completed[start:end, channel] = block_values[index]
            selected[eligible[index]] += 1
    completed[observed_mask] = values[0][observed_mask]
    return completed, dict(selected)


def main() -> int:
    args = _arguments()
    config = load_config(args.config)
    impute_root = Path(args.impute_artifact).resolve()
    metrics = pd.read_csv(args.evaluation)
    metrics = metrics[
        metrics["metric_eligible"].astype(bool)
        & metrics["method_role"].isin(("baseline", "missing_anchor"))
    ]
    losses = metrics.set_index(["episode_id", "method"])["mase"].to_dict()
    runner = _build_forecast_runner(
        args.forecaster_id,
        Path(args.forecaster_artifact),
        device=config.runtime.device,
        batch_size=config.experiment.forecast_batch_size,
    )
    selections: list[dict[str, object]] = []
    for record in _records(impute_root / "routing_assignments.jsonl"):
        episode_id = str(record["episode_id"])
        with np.load(
            impute_root / "imputations" / str(record["file"]), allow_pickle=False
        ) as archive:
            candidates = _valid_candidates(archive)
            candidate_ids = tuple(sorted(candidates))
            contexts = np.stack(
                [
                    np.asarray(candidates[candidate_id]["values"], dtype=float)
                    for candidate_id in candidate_ids
                ]
            )
            spec = _forecast_spec(config, args.forecaster_id, contexts.shape[2])
            forecast_seed = stable_seed(
                config.seed,
                args.forecaster_id,
                episode_id,
                args.forecast_seed_scope,
            )
            _set_forecast_seed(forecast_seed)
            predictions = runner.predict(contexts, spec).point
            scale = np.asarray(archive["mase_scale"], dtype=float)[list(spec.target_indices)]
            observed_mask = np.asarray(archive["observed_mask"], dtype=bool)
            clean_future = np.asarray(archive["clean_future"], dtype=float)
        available_ids = tuple(
            candidate_id for candidate_id in candidate_ids if (episode_id, candidate_id) in losses
        )
        if not available_ids:
            continue
        available_indices = [candidate_ids.index(candidate_id) for candidate_id in available_ids]
        available_predictions = predictions[available_indices]
        all_medoid = _medoid(available_ids, available_predictions, scale)
        trusted_medoid = _medoid(
            available_ids,
            available_predictions,
            scale,
            TRUSTED_CANDIDATES,
        )
        classical_medoid = _medoid(
            available_ids,
            available_predictions,
            scale,
            CLASSICAL_CANDIDATES,
        )
        anchor_medoid = _medoid(
            available_ids,
            available_predictions,
            scale,
            ANCHOR_CANDIDATES,
        )
        eligible_anchor_ids = tuple(
            candidate_id for candidate_id in available_ids if candidate_id in ANCHOR_CANDIDATES
        )
        shortlist_ids = tuple(
            candidate_id
            for candidate_id in record.get("shortlist", ())
            if candidate_id in available_ids
        )
        shortlist_medoid = _medoid(
            available_ids,
            available_predictions,
            scale,
            shortlist_ids,
        )
        anchor_or_shortlist_medoid = _medoid(
            available_ids,
            available_predictions,
            scale,
            eligible_anchor_ids or shortlist_ids,
        )
        full_coverage_medoid = _medoid(
            available_ids,
            available_predictions,
            scale,
            FULL_COVERAGE_CANDIDATES,
        )
        blockwise_contexts: list[np.ndarray] = []
        blockwise_counts: dict[str, dict[str, int]] = {}
        for policy, allowed in (
            ("blockwise_all", None),
            ("blockwise_trusted", TRUSTED_CANDIDATES),
            ("blockwise_shortlist", shortlist_ids),
        ):
            context, counts = _blockwise_value_medoid(
                available_ids,
                candidates,
                observed_mask,
                allowed=allowed,
            )
            blockwise_contexts.append(context)
            blockwise_counts[policy] = counts
        _set_forecast_seed(forecast_seed)
        blockwise_predictions = runner.predict(np.stack(blockwise_contexts), spec).point
        future = clean_future[:, list(spec.target_indices)]
        blockwise_losses = np.mean(
            np.mean(np.abs(blockwise_predictions - future[None, ...]), axis=1) / scale[None, :],
            axis=1,
        )
        episode_losses = {
            candidate_id: float(losses[(episode_id, candidate_id)])
            for candidate_id in available_ids
        }
        selections.append(
            {
                "dataset_id": record["dataset_id"],
                "episode_id": episode_id,
                "all_medoid": episode_losses[all_medoid],
                "trusted_medoid": episode_losses[trusted_medoid],
                "classical_medoid": episode_losses[classical_medoid],
                "anchor_medoid": episode_losses[anchor_medoid],
                "shortlist_medoid": episode_losses[shortlist_medoid],
                "anchor_or_shortlist_medoid": episode_losses[anchor_or_shortlist_medoid],
                "full_coverage_medoid": episode_losses[full_coverage_medoid],
                "blockwise_all": float(blockwise_losses[0]),
                "blockwise_trusted": float(blockwise_losses[1]),
                "blockwise_shortlist": float(blockwise_losses[2]),
                "oracle": min(episode_losses.values()),
                "all_medoid_candidate": all_medoid,
                "trusted_medoid_candidate": trusted_medoid,
                "classical_medoid_candidate": classical_medoid,
                "anchor_medoid_candidate": anchor_medoid,
                "shortlist_medoid_candidate": shortlist_medoid,
                "anchor_or_shortlist_medoid_candidate": anchor_or_shortlist_medoid,
                "full_coverage_medoid_candidate": full_coverage_medoid,
                "blockwise_all_candidates": blockwise_counts["blockwise_all"],
                "blockwise_trusted_candidates": blockwise_counts["blockwise_trusted"],
                "blockwise_shortlist_candidates": blockwise_counts["blockwise_shortlist"],
            }
        )
    frame = pd.DataFrame(selections)
    policy_columns = [
        "all_medoid",
        "trusted_medoid",
        "classical_medoid",
        "anchor_medoid",
        "shortlist_medoid",
        "anchor_or_shortlist_medoid",
        "full_coverage_medoid",
        "blockwise_all",
        "blockwise_trusted",
        "blockwise_shortlist",
        "oracle",
    ]
    summary = frame.groupby("dataset_id")[policy_columns].mean()
    single = metrics.groupby(["dataset_id", "method"])["mase"].mean()
    summary["best_single"] = single.groupby(level=0).min()
    summary["best_single_method"] = single.groupby(level=0).idxmin().map(lambda item: item[1])
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))
    print("\nmacro")
    print(summary[[*policy_columns, "best_single"]].mean())
    print("\nselection counts")
    print(
        frame.groupby("dataset_id")[
            [
                "all_medoid_candidate",
                "trusted_medoid_candidate",
                "classical_medoid_candidate",
                "anchor_medoid_candidate",
                "shortlist_medoid_candidate",
                "anchor_or_shortlist_medoid_candidate",
                "full_coverage_medoid_candidate",
            ]
        ]
        .agg(lambda values: dict(pd.Series(values).value_counts()))
        .to_string()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
