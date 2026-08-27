"""Pure risk scoring and whole-episode fallback rules for B-FAIS R2.

The functions in this module have no file-system or forecasting dependencies.
They operate only on deployment-visible imputation evidence and explicit metric
records supplied by a caller.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cmp_to_key
from math import ceil, isfinite
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from tsfm_fais.contracts import MissingBlock

TOLERANCE = 1e-12
ForecastMode = Literal["joint_multivariate", "independent_univariate"]
BlockScoreStatus = Literal["direct", "safety", "unavailable", "irrelevant"]
ThresholdKind = Literal["always", "finite", "disabled"]


@dataclass(frozen=True)
class RiskScoreProtocol:
    """Deployment-visible inputs used to score one imputation episode."""

    forecast_mode: ForecastMode
    target_indices: tuple[int, ...] = ()
    max_pseudo_blocks: int = 8
    tolerance: float = TOLERANCE

    def __post_init__(self) -> None:
        if self.forecast_mode not in {"joint_multivariate", "independent_univariate"}:
            raise ValueError("unsupported forecast mode")
        if any(index < 0 for index in self.target_indices):
            raise ValueError("target indices must be non-negative")
        if len(set(self.target_indices)) != len(self.target_indices):
            raise ValueError("target indices must be unique")
        if self.forecast_mode == "independent_univariate" and not self.target_indices:
            raise ValueError("independent mode requires target indices")
        if self.max_pseudo_blocks < 1:
            raise ValueError("max_pseudo_blocks must be positive")
        if not isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be finite and non-negative")


@dataclass(frozen=True)
class BlockRiskScore:
    block_id: str
    channel: int
    visible: bool
    status: BlockScoreStatus
    selected_candidate_id: str | None
    eligible_candidate_ids: tuple[str, ...]
    selected_proxy_mae: float | None
    best_proxy_mae: float | None
    worst_proxy_mae: float | None
    risk: float | None
    reason: str


@dataclass(frozen=True)
class EpisodeRiskScore:
    score: float | None
    status: Literal["available", "unavailable"]
    scored_block_ids: tuple[str, ...]
    unavailable_block_ids: tuple[str, ...]


@dataclass(frozen=True)
class RiskThreshold:
    kind: ThresholdKind
    value: float | None = None

    def __post_init__(self) -> None:
        if self.kind == "finite":
            if self.value is None or not isfinite(float(self.value)):
                raise ValueError("a finite threshold requires a finite value")
        elif self.value is not None:
            raise ValueError("non-finite threshold kinds cannot carry a value")

    @property
    def identifier(self) -> str:
        if self.kind != "finite":
            return self.kind
        assert self.value is not None
        return f"finite:{float(self.value).hex()}"


@dataclass(frozen=True)
class RiskAction:
    values: NDArray[np.float64]
    switched: bool
    reason: str
    robust_native_valid: bool


@dataclass(frozen=True)
class CandidateDatasetStatistics:
    dataset_id: str
    episode_denominator: int
    row_count: int
    native_available_count: int
    missing_row_count: int
    status_counts: tuple[tuple[str, int], ...]
    availability: float
    mean_loss: float | None
    cvar90_loss: float | None


@dataclass(frozen=True)
class RobustCandidateStatistics:
    candidate_id: str
    eligible: bool
    ineligibility_reasons: tuple[str, ...]
    datasets: tuple[CandidateDatasetStatistics, ...]
    global_native_available_count: int
    dataset_equal_mean_loss: float | None
    dataset_equal_unavailability: float
    dataset_equal_cvar90_loss: float | None


@dataclass(frozen=True)
class RobustSelection:
    status: Literal["selected", "no_eligible_candidate"]
    selected_candidate_id: str | None
    candidates: tuple[RobustCandidateStatistics, ...]
    availability_floor: float
    tolerance: float


@dataclass(frozen=True)
class ThresholdEpisode:
    episode_id: str
    forecaster_id: str
    dataset_id: str
    score: float | None
    action_ready: bool
    b_fais_mase: float
    robust_mase: float
    clean_mase: float


@dataclass(frozen=True)
class ThresholdCellStatistics:
    forecaster_id: str
    dataset_id: str
    episode_count: int
    switched_count: int
    mean_mase: float
    cvar90_mase: float
    cvar90_delta_from_clean: float


@dataclass(frozen=True)
class ThresholdCandidateStatistics:
    threshold: RiskThreshold
    eligible: bool
    ineligibility_reasons: tuple[str, ...]
    switched_count: int
    switched_fraction: float
    mean_mase: float
    cvar90_mase: float
    cvar90_delta_from_clean: float
    cells: tuple[ThresholdCellStatistics, ...]


@dataclass(frozen=True)
class ThresholdSelection:
    selected_threshold: RiskThreshold
    candidates: tuple[ThresholdCandidateStatistics, ...]
    disabled_mean_mase: float
    disabled_cvar90_mase: float
    delta_ett: float
    cell_count: int
    episode_count: int


def deterministic_pseudo_observed_mask(
    observed_mask: NDArray[np.bool_],
    seed: int,
    *,
    max_blocks: int = 8,
    target_blocks: Sequence[MissingBlock] = (),
    priority_channels: Sequence[int] = (),
) -> NDArray[np.bool_]:
    """Reproduce the fixed route pseudo-mask procedure without constructing a batch."""

    mask = np.asarray(observed_mask, dtype=bool)
    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError("observed_mask must have shape [1,L,D]")
    if max_blocks < 1:
        return mask.copy()
    result = mask.copy()
    length = result.shape[1]
    if not np.any(result[0]):
        return result
    rng = np.random.default_rng(seed)
    default_length = max(1, min(length // 20, 8))
    maximum_length = max(default_length, min(max(1, length // 8), 12))
    templates = list(target_blocks)
    if templates:
        priority = tuple(dict.fromkeys(int(value) for value in priority_channels))
        ordered: list[MissingBlock] = []
        seen_channels: set[int] = set()
        for channel in priority:
            match = next((block for block in templates if block.channel == channel), None)
            if match is not None:
                ordered.append(match)
                seen_channels.add(channel)
        for block in templates:
            if block.channel not in seen_channels:
                ordered.append(block)
                seen_channels.add(block.channel)
        ordered.extend(block for block in templates if block not in ordered)
        templates = ordered

    occupied_time = np.zeros(length, dtype=bool)
    placed = 0
    attempts = 0
    maximum_attempts = max(32, max_blocks * 8)
    while placed < max_blocks and attempts < maximum_attempts:
        attempts += 1
        template = templates[placed] if placed < len(templates) else None
        if template is None:
            channel = int(rng.integers(0, result.shape[2]))
            desired_start = int(rng.integers(0, max(1, length - default_length + 1)))
            desired_length = default_length
        else:
            channel = int(template.channel)
            if not 0 <= channel < result.shape[2]:
                raise ValueError("target block channel is outside the mask")
            desired_start = int(template.start)
            desired_length = min(maximum_length, max(1, int(template.length)))
        candidate_lengths = tuple(dict.fromkeys((desired_length, default_length, 1)))
        selected: tuple[int, int] | None = None
        for block_length in candidate_lengths:
            starts = [
                start
                for start in range(0, length - block_length + 1)
                if result[0, start : start + block_length, channel].all()
                and not occupied_time[start : start + block_length].any()
            ]
            if not starts:
                continue
            distances = np.abs(np.asarray(starts, dtype=int) - desired_start)
            nearest = np.flatnonzero(distances == np.min(distances))
            chosen = int(starts[int(rng.choice(nearest))])
            selected = chosen, block_length
            break
        if selected is None:
            if template is not None:
                templates.pop(placed)
            continue
        start, block_length = selected
        result[0, start : start + block_length, channel] = False
        occupied_time[start : start + block_length] = True
        placed += 1
    return result


def newly_hidden_mask(
    observed_mask: NDArray[np.bool_], pseudo_observed_mask: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    original = np.asarray(observed_mask, dtype=bool)
    pseudo = np.asarray(pseudo_observed_mask, dtype=bool)
    if original.shape != pseudo.shape:
        raise ValueError("original and pseudo masks must have the same shape")
    if np.any(pseudo & ~original):
        raise ValueError("a pseudo mask cannot restore an originally missing value")
    return original & ~pseudo


def channel_proxy_mae(
    truth: NDArray[np.float64],
    candidate_values: NDArray[np.float64],
    candidate_native_valid: NDArray[np.bool_],
    observed_mask: NDArray[np.bool_],
    pseudo_observed_mask: NDArray[np.bool_],
    *,
    channel: int,
    candidate_status: str,
) -> float | None:
    """Return native finite MAE on newly hidden positions for one channel."""

    truth_array = np.asarray(truth, dtype=float)
    values = np.asarray(candidate_values, dtype=float)
    native = np.asarray(candidate_native_valid, dtype=bool)
    original = np.asarray(observed_mask, dtype=bool)
    pseudo = np.asarray(pseudo_observed_mask, dtype=bool)
    if not (truth_array.shape == values.shape == native.shape == original.shape == pseudo.shape):
        raise ValueError("proxy arrays and masks must have the same shape")
    if not 0 <= channel < truth_array.shape[-1]:
        raise ValueError("proxy channel is outside the variate axis")
    if str(candidate_status).lower() in {"failed", "unavailable"}:
        return None
    hidden = newly_hidden_mask(original, pseudo)
    selected = np.zeros_like(hidden, dtype=bool)
    selected[..., channel] = hidden[..., channel]
    if not np.any(selected):
        return None
    if not native[selected].all():
        return None
    if not np.isfinite(values[selected]).all() or not np.isfinite(truth_array[selected]).all():
        return None
    return float(np.mean(np.abs(values[selected] - truth_array[selected])))


def block_is_visible(block: MissingBlock, protocol: RiskScoreProtocol) -> bool:
    return protocol.forecast_mode == "joint_multivariate" or block.channel in set(
        protocol.target_indices
    )


def normalized_proxy_regret(
    selected_proxy_mae: float,
    eligible_proxy_mae: Sequence[float],
    *,
    tolerance: float = TOLERANCE,
) -> tuple[float, float, float]:
    selected = float(selected_proxy_mae)
    eligible = np.asarray(tuple(float(value) for value in eligible_proxy_mae), dtype=float)
    if eligible.size < 1 or not np.isfinite(eligible).all() or not isfinite(selected):
        raise ValueError("proxy regrets require finite evidence")
    best = float(np.min(eligible))
    worst = float(np.max(eligible))
    if worst - best <= tolerance:
        return 0.0, best, worst
    risk = (selected - best) / (worst - best)
    if risk < -tolerance or risk > 1.0 + tolerance:
        raise ValueError("selected proxy error is outside the eligible candidate range")
    return float(np.clip(risk, 0.0, 1.0)), best, worst


def score_block(
    block: MissingBlock,
    protocol: RiskScoreProtocol,
    *,
    selected_candidate_id: str | None,
    shortlist: Sequence[str],
    proxy_mae_by_candidate: Mapping[str, float | None],
    direct_native_valid: bool,
    safety_result_used: bool,
) -> BlockRiskScore:
    visible = block_is_visible(block, protocol)
    if not visible:
        return BlockRiskScore(
            block.block_id,
            block.channel,
            False,
            "irrelevant",
            selected_candidate_id,
            (),
            None,
            None,
            None,
            None,
            "block is not visible to the downstream forecast interface",
        )
    if safety_result_used:
        return BlockRiskScore(
            block.block_id,
            block.channel,
            True,
            "safety",
            selected_candidate_id,
            (),
            None,
            None,
            None,
            1.0,
            "recorded safety result",
        )
    eligible_values: list[str] = []
    for candidate_id in shortlist:
        proxy_value = proxy_mae_by_candidate.get(candidate_id)
        if proxy_value is not None and isfinite(float(proxy_value)):
            eligible_values.append(candidate_id)
    eligible = tuple(eligible_values)
    selected_error = proxy_mae_by_candidate.get(selected_candidate_id or "")
    if (
        not direct_native_valid
        or selected_candidate_id not in eligible
        or selected_error is None
        or not eligible
    ):
        return BlockRiskScore(
            block.block_id,
            block.channel,
            True,
            "unavailable",
            selected_candidate_id,
            eligible,
            None,
            None,
            None,
            None,
            "direct native shortlisted proxy evidence is unavailable",
        )
    error_values: list[float] = []
    for candidate_id in eligible:
        proxy_value = proxy_mae_by_candidate[candidate_id]
        assert proxy_value is not None
        error_values.append(float(proxy_value))
    errors = tuple(error_values)
    risk, best, worst = normalized_proxy_regret(
        float(selected_error), errors, tolerance=protocol.tolerance
    )
    return BlockRiskScore(
        block.block_id,
        block.channel,
        True,
        "direct",
        selected_candidate_id,
        eligible,
        float(selected_error),
        best,
        worst,
        risk,
        "normalized native proxy regret",
    )


def score_episode(block_scores: Sequence[BlockRiskScore]) -> EpisodeRiskScore:
    identifiers = [score.block_id for score in block_scores]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("block score identifiers must be unique")
    scored = tuple(
        score.block_id
        for score in block_scores
        if score.visible and score.risk is not None and isfinite(float(score.risk))
    )
    unavailable = tuple(
        score.block_id for score in block_scores if score.visible and score.status == "unavailable"
    )
    if not scored:
        return EpisodeRiskScore(None, "unavailable", (), unavailable)
    scored_set = set(scored)
    risks: list[float] = []
    for score in block_scores:
        if score.block_id in scored_set:
            assert score.risk is not None
            risks.append(float(score.risk))
    result = float(max(risks))
    if not 0.0 <= result <= 1.0:
        raise ValueError("episode risk must lie in [0,1]")
    return EpisodeRiskScore(result, "available", scored, unavailable)


def threshold_switches(score: float | None, threshold: RiskThreshold) -> bool:
    if score is None or not isfinite(float(score)) or threshold.kind == "disabled":
        return False
    if threshold.kind == "always":
        return True
    assert threshold.value is not None
    return float(score) > float(threshold.value)


def apply_whole_episode_fallback(
    source_values: NDArray[np.float64],
    observed_mask: NDArray[np.bool_],
    robust_values: NDArray[np.float64] | None,
    robust_native_valid: NDArray[np.bool_] | None,
    *,
    score: float | None,
    threshold: RiskThreshold,
) -> RiskAction:
    source = np.asarray(source_values, dtype=float)
    observed = np.asarray(observed_mask, dtype=bool)
    if source.shape != observed.shape:
        raise ValueError("source values and observed mask must have the same shape")
    if not np.isfinite(source).all():
        raise ValueError("source assembled values must be finite")
    if not threshold_switches(score, threshold):
        reason = "score_unavailable" if score is None else "threshold_not_exceeded"
        if threshold.kind == "disabled":
            reason = "disabled"
        return RiskAction(source.copy(), False, reason, False)
    if robust_values is None or robust_native_valid is None:
        return RiskAction(source.copy(), False, "robust_candidate_absent", False)
    robust = np.asarray(robust_values, dtype=float)
    native = np.asarray(robust_native_valid, dtype=bool)
    if robust.shape != source.shape or native.shape != source.shape:
        raise ValueError("robust candidate arrays must match the source shape")
    missing = ~observed
    available = bool(missing.any() and native[missing].all() and np.isfinite(robust[missing]).all())
    if not available:
        return RiskAction(source.copy(), False, "robust_candidate_not_native_finite", False)
    result = source.copy()
    result[missing] = robust[missing]
    if not np.array_equal(result[observed], source[observed]):
        raise AssertionError("observed values changed during fallback")
    if not np.array_equal(result[missing], robust[missing]):
        raise AssertionError("fallback values differ from the robust candidate")
    return RiskAction(result, True, "score_exceeded", True)


def source_routing_actual_ids(
    shortlist: Sequence[str],
    fallback_records: Mapping[str, Mapping[str, Any]],
    valid_candidate_ids: Iterable[str],
) -> tuple[str, ...]:
    valid = set(valid_candidate_ids)
    actual = {candidate_id for candidate_id in shortlist if candidate_id in valid}
    for record in fallback_records.values():
        attempts = record.get("attempts", ())
        if isinstance(attempts, Sequence) and not isinstance(attempts, (str, bytes)):
            actual.update(str(candidate_id) for candidate_id in attempts if candidate_id in valid)
    return tuple(sorted(actual))


def derived_pipeline_runtime(
    source_runtime_seconds: float,
    *,
    switched: bool,
    robust_candidate_id: str | None,
    robust_candidate_runtime_seconds: float | None,
    source_actual_candidate_ids: Iterable[str],
) -> float:
    source_runtime = float(source_runtime_seconds)
    if not isfinite(source_runtime) or source_runtime < 0:
        raise ValueError("source runtime must be finite and non-negative")
    if not switched:
        return source_runtime
    if not robust_candidate_id or robust_candidate_runtime_seconds is None:
        raise ValueError("a switch requires a robust candidate runtime")
    robust_runtime = float(robust_candidate_runtime_seconds)
    if not isfinite(robust_runtime) or robust_runtime < 0:
        raise ValueError("robust candidate runtime must be finite and non-negative")
    if robust_candidate_id in set(source_actual_candidate_ids):
        return source_runtime
    return source_runtime + robust_runtime


def discrete_cvar(values: Sequence[float], quantile: float = 0.9) -> float:
    array = np.asarray(tuple(float(value) for value in values), dtype=float)
    if array.size < 1 or not np.isfinite(array).all():
        raise ValueError("CVaR requires at least one finite observation")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between zero and one")
    tail_count = max(1, ceil((1.0 - quantile) * array.size))
    return float(np.mean(np.sort(array)[-tail_count:]))


def weighted_empirical_quantile(
    values: Sequence[float], weights: Sequence[float], quantile: float
) -> float:
    observations = np.asarray(tuple(float(value) for value in values), dtype=float)
    masses = np.asarray(tuple(float(value) for value in weights), dtype=float)
    if observations.size < 1 or observations.shape != masses.shape:
        raise ValueError("values and weights must have the same non-empty shape")
    if not np.isfinite(observations).all() or not np.isfinite(masses).all():
        raise ValueError("weighted quantile inputs must be finite")
    if np.any(masses < 0) or float(np.sum(masses)) <= 0:
        raise ValueError("weights must be non-negative with positive total mass")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0,1]")
    order = np.argsort(observations, kind="stable")
    sorted_values = observations[order]
    cumulative = np.cumsum(masses[order]) / float(np.sum(masses))
    index = int(np.searchsorted(cumulative, quantile, side="left"))
    return float(sorted_values[min(index, sorted_values.size - 1)])


def _mapping_value(row: Mapping[str, Any], field: str) -> Any:
    if field not in row:
        raise ValueError(f"reconstruction row is missing {field!r}")
    return row[field]


def select_robust_candidate(
    rows: Sequence[Mapping[str, Any]],
    dataset_episode_ids: Mapping[str, Sequence[str]],
    candidate_ids: Sequence[str],
    *,
    availability_floor: float = 0.95,
    tolerance: float = TOLERANCE,
) -> RobustSelection:
    """Apply the preregistered dataset-equal reconstruction ranking."""

    if not 0.0 <= availability_floor <= 1.0:
        raise ValueError("availability floor must lie in [0,1]")
    datasets = tuple(sorted(dataset_episode_ids))
    candidates = tuple(sorted(set(candidate_ids)))
    if not datasets or not candidates:
        raise ValueError("robust selection requires datasets and candidates")
    expected: dict[str, set[str]] = {}
    for dataset_id in datasets:
        episode_ids = tuple(str(value) for value in dataset_episode_ids[dataset_id])
        if not episode_ids or len(set(episode_ids)) != len(episode_ids):
            raise ValueError("each dataset plan must contain unique episode IDs")
        expected[dataset_id] = set(episode_ids)

    indexed: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        dataset_id = str(_mapping_value(row, "dataset_id"))
        episode_id = str(_mapping_value(row, "episode_id"))
        candidate_id = str(_mapping_value(row, "candidate_id"))
        if dataset_id not in expected or episode_id not in expected[dataset_id]:
            raise ValueError("reconstruction row is outside the deterministic plan")
        if candidate_id not in candidates:
            raise ValueError("reconstruction row contains an undeclared candidate")
        key = dataset_id, episode_id, candidate_id
        if key in indexed:
            raise ValueError("reconstruction rows contain a duplicate whole-series key")
        loss = float(_mapping_value(row, "imputation_loss"))
        if not isfinite(loss) or loss < -tolerance or loss > 1.0 + tolerance:
            raise ValueError("imputation loss must be finite and lie in [0,1]")
        indexed[key] = row

    summaries: list[RobustCandidateStatistics] = []
    for candidate_id in candidates:
        dataset_summaries: list[CandidateDatasetStatistics] = []
        reasons: list[str] = []
        all_means: list[float] = []
        all_cvars: list[float] = []
        global_available = 0
        for dataset_id in datasets:
            planned_episode_ids = sorted(expected[dataset_id])
            present_rows = [
                indexed[(dataset_id, episode_id, candidate_id)]
                for episode_id in planned_episode_ids
                if (dataset_id, episode_id, candidate_id) in indexed
            ]
            status_counts = Counter(
                str(_mapping_value(row, "candidate_status")).lower() for row in present_rows
            )
            available_rows = [
                row
                for row in present_rows
                if bool(_mapping_value(row, "native_valid"))
                and str(_mapping_value(row, "candidate_status")).lower()
                not in {"failed", "unavailable"}
            ]
            denominator = len(planned_episode_ids)
            available_count = len(available_rows)
            availability = available_count / denominator
            losses = [float(row["imputation_loss"]) for row in available_rows]
            mean_loss = float(np.mean(losses)) if losses else None
            cvar = discrete_cvar(losses) if losses else None
            if availability + tolerance < availability_floor:
                reasons.append(
                    f"{dataset_id}:availability={availability:.17g}<floor={availability_floor:.17g}"
                )
            if mean_loss is None or cvar is None:
                reasons.append(f"{dataset_id}:no_native_available_loss")
            else:
                all_means.append(mean_loss)
                all_cvars.append(cvar)
            global_available += available_count
            dataset_summaries.append(
                CandidateDatasetStatistics(
                    dataset_id=dataset_id,
                    episode_denominator=denominator,
                    row_count=len(present_rows),
                    native_available_count=available_count,
                    missing_row_count=denominator - len(present_rows),
                    status_counts=tuple(sorted(status_counts.items())),
                    availability=availability,
                    mean_loss=mean_loss,
                    cvar90_loss=cvar,
                )
            )
        eligible = not reasons and len(all_means) == len(datasets)
        summaries.append(
            RobustCandidateStatistics(
                candidate_id=candidate_id,
                eligible=eligible,
                ineligibility_reasons=tuple(reasons),
                datasets=tuple(dataset_summaries),
                global_native_available_count=global_available,
                dataset_equal_mean_loss=(float(np.mean(all_means)) if all_means else None),
                dataset_equal_unavailability=float(
                    np.mean([1.0 - item.availability for item in dataset_summaries])
                ),
                dataset_equal_cvar90_loss=(float(np.mean(all_cvars)) if all_cvars else None),
            )
        )

    eligible_summaries = [summary for summary in summaries if summary.eligible]

    def compare(left: RobustCandidateStatistics, right: RobustCandidateStatistics) -> int:
        metrics = (
            (left.dataset_equal_mean_loss, right.dataset_equal_mean_loss),
            (left.dataset_equal_unavailability, right.dataset_equal_unavailability),
            (left.dataset_equal_cvar90_loss, right.dataset_equal_cvar90_loss),
        )
        for left_value, right_value in metrics:
            assert left_value is not None and right_value is not None
            if abs(left_value - right_value) > tolerance:
                return -1 if left_value < right_value else 1
        return (
            -1
            if left.candidate_id < right.candidate_id
            else (left.candidate_id > right.candidate_id)
        )

    ranked = sorted(eligible_summaries, key=cmp_to_key(compare))
    selected = ranked[0].candidate_id if ranked else None
    return RobustSelection(
        status="selected" if selected is not None else "no_eligible_candidate",
        selected_candidate_id=selected,
        candidates=tuple(summaries),
        availability_floor=availability_floor,
        tolerance=tolerance,
    )


def enumerate_thresholds(scores: Iterable[float | None]) -> tuple[RiskThreshold, ...]:
    finite_scores = sorted(
        {float(score) for score in scores if score is not None and isfinite(float(score))}
    )
    return (
        RiskThreshold("always"),
        *(RiskThreshold("finite", score) for score in finite_scores),
        RiskThreshold("disabled"),
    )


def _validate_threshold_episodes(
    episodes: Sequence[ThresholdEpisode], expected_cells: Iterable[tuple[str, str]] | None
) -> tuple[tuple[str, str], ...]:
    if not episodes:
        raise ValueError("threshold selection requires episode records")
    keys: set[tuple[str, str, str]] = set()
    cells: set[tuple[str, str]] = set()
    for episode in episodes:
        key = episode.forecaster_id, episode.dataset_id, episode.episode_id
        if key in keys:
            raise ValueError("threshold records contain a duplicate episode key")
        keys.add(key)
        cells.add((episode.forecaster_id, episode.dataset_id))
        for value in (episode.b_fais_mase, episode.robust_mase, episode.clean_mase):
            if not isfinite(float(value)):
                raise ValueError("threshold metrics must be finite")
        if episode.score is not None and not isfinite(float(episode.score)):
            raise ValueError("episode risk scores must be finite or unavailable")
    expected = set(expected_cells) if expected_cells is not None else cells
    if cells != expected:
        raise ValueError("threshold records do not match the expected cells")
    if not cells:
        raise ValueError("threshold records contain no cells")
    return tuple(sorted(cells))


def _threshold_statistics(
    episodes: Sequence[ThresholdEpisode],
    cells: Sequence[tuple[str, str]],
    threshold: RiskThreshold,
    *,
    disabled_mean: float | None,
    disabled_cvar: float | None,
    tolerance: float,
) -> ThresholdCandidateStatistics:
    cell_summaries: list[ThresholdCellStatistics] = []
    switched_total = 0
    for forecaster_id, dataset_id in cells:
        group = [
            episode
            for episode in episodes
            if episode.forecaster_id == forecaster_id and episode.dataset_id == dataset_id
        ]
        if not group:
            raise ValueError("each expected threshold cell must contain episodes")
        switches = [
            episode.action_ready and threshold_switches(episode.score, threshold)
            for episode in group
        ]
        method = [
            episode.robust_mase if switch else episode.b_fais_mase
            for episode, switch in zip(group, switches, strict=True)
        ]
        deltas = [value - episode.clean_mase for value, episode in zip(method, group, strict=True)]
        switched_count = sum(switches)
        switched_total += switched_count
        cell_summaries.append(
            ThresholdCellStatistics(
                forecaster_id=forecaster_id,
                dataset_id=dataset_id,
                episode_count=len(group),
                switched_count=switched_count,
                mean_mase=float(np.mean(method)),
                cvar90_mase=discrete_cvar(method),
                cvar90_delta_from_clean=discrete_cvar(deltas),
            )
        )
    mean_mase = float(np.mean([cell.mean_mase for cell in cell_summaries]))
    cvar_mase = float(np.mean([cell.cvar90_mase for cell in cell_summaries]))
    cvar_delta = float(np.mean([cell.cvar90_delta_from_clean for cell in cell_summaries]))
    reasons: list[str] = []
    eligible = threshold.kind == "disabled"
    if threshold.kind != "disabled" and disabled_mean is not None and disabled_cvar is not None:
        if mean_mase > 1.02 * disabled_mean + tolerance:
            reasons.append("mean_mase_exceeds_1.02_times_disabled")
        if not cvar_mase < disabled_cvar - tolerance:
            reasons.append("cvar90_not_strictly_lower_than_disabled")
        eligible = not reasons
    return ThresholdCandidateStatistics(
        threshold=threshold,
        eligible=eligible,
        ineligibility_reasons=tuple(reasons),
        switched_count=switched_total,
        switched_fraction=switched_total / len(episodes),
        mean_mase=mean_mase,
        cvar90_mase=cvar_mase,
        cvar90_delta_from_clean=cvar_delta,
        cells=tuple(cell_summaries),
    )


def select_threshold(
    episodes: Sequence[ThresholdEpisode],
    *,
    expected_cells: Iterable[tuple[str, str]] | None = None,
    tolerance: float = TOLERANCE,
) -> ThresholdSelection:
    """Select one shared threshold using cell-equal ETT MASE statistics."""

    cells = _validate_threshold_episodes(episodes, expected_cells)
    thresholds = enumerate_thresholds(episode.score for episode in episodes)
    disabled_threshold = next(item for item in thresholds if item.kind == "disabled")
    disabled = _threshold_statistics(
        episodes,
        cells,
        disabled_threshold,
        disabled_mean=None,
        disabled_cvar=None,
        tolerance=tolerance,
    )
    candidates = tuple(
        _threshold_statistics(
            episodes,
            cells,
            threshold,
            disabled_mean=disabled.mean_mase,
            disabled_cvar=disabled.cvar90_mase,
            tolerance=tolerance,
        )
        for threshold in thresholds
    )
    active = [
        candidate
        for candidate in candidates
        if candidate.threshold.kind != "disabled" and candidate.eligible
    ]

    def compare(left: ThresholdCandidateStatistics, right: ThresholdCandidateStatistics) -> int:
        lower_metrics = (
            (left.cvar90_mase, right.cvar90_mase),
            (left.mean_mase, right.mean_mase),
            (left.cvar90_delta_from_clean, right.cvar90_delta_from_clean),
            (left.switched_fraction, right.switched_fraction),
        )
        for left_value, right_value in lower_metrics:
            if abs(left_value - right_value) > tolerance:
                return -1 if left_value < right_value else 1
        left_finite = left.threshold.kind == "finite"
        right_finite = right.threshold.kind == "finite"
        if left_finite and right_finite:
            assert left.threshold.value is not None and right.threshold.value is not None
            if abs(left.threshold.value - right.threshold.value) > tolerance:
                return -1 if left.threshold.value > right.threshold.value else 1
        if left_finite != right_finite:
            return -1 if left_finite else 1
        return (left.threshold.identifier > right.threshold.identifier) - (
            left.threshold.identifier < right.threshold.identifier
        )

    selected = sorted(active, key=cmp_to_key(compare))[0] if active else disabled
    weights: list[float] = []
    deltas: list[float] = []
    for forecaster_id, dataset_id in cells:
        group = [
            episode
            for episode in episodes
            if episode.forecaster_id == forecaster_id and episode.dataset_id == dataset_id
        ]
        cell_weight = 1.0 / len(cells) / len(group)
        for episode in group:
            deltas.append(episode.b_fais_mase - episode.clean_mase)
            weights.append(cell_weight)
    delta_ett = max(0.0, weighted_empirical_quantile(deltas, weights, 0.9))
    return ThresholdSelection(
        selected_threshold=selected.threshold,
        candidates=candidates,
        disabled_mean_mase=disabled.mean_mase,
        disabled_cvar90_mase=disabled.cvar90_mase,
        delta_ett=delta_ett,
        cell_count=len(cells),
        episode_count=len(episodes),
    )


__all__ = [
    "TOLERANCE",
    "BlockRiskScore",
    "CandidateDatasetStatistics",
    "EpisodeRiskScore",
    "RiskAction",
    "RiskScoreProtocol",
    "RiskThreshold",
    "RobustCandidateStatistics",
    "RobustSelection",
    "ThresholdCandidateStatistics",
    "ThresholdCellStatistics",
    "ThresholdEpisode",
    "ThresholdSelection",
    "apply_whole_episode_fallback",
    "block_is_visible",
    "channel_proxy_mae",
    "derived_pipeline_runtime",
    "deterministic_pseudo_observed_mask",
    "discrete_cvar",
    "enumerate_thresholds",
    "newly_hidden_mask",
    "normalized_proxy_regret",
    "score_block",
    "score_episode",
    "select_robust_candidate",
    "select_threshold",
    "source_routing_actual_ids",
    "threshold_switches",
    "weighted_empirical_quantile",
]
