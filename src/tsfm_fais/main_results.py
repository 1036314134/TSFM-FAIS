"""Deterministic aggregation of multi-forecaster evaluation artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FORECAST_METRICS: tuple[str, ...] = ("mase", "mae", "rmse")
IMPUTATION_METRICS: tuple[str, ...] = ("imputation_mae", "imputation_rmse")
PAIRED_METRICS: tuple[str, ...] = (*FORECAST_METRICS, *IMPUTATION_METRICS)
GROUP_FIELDS: tuple[str, ...] = (
    "forecaster_id",
    "mechanism",
    "missing_rate",
    "family_id",
)
SUMMARY_SCOPES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("overall", ()),
    ("forecaster", ("forecaster_id",)),
    ("mechanism", ("mechanism",)),
    ("missing_rate", ("missing_rate",)),
    ("family", ("family_id",)),
)
DEFAULT_BOOTSTRAP_REPLICATES = 2_000
DEFAULT_BOOTSTRAP_SEED = 20260710
SMALL_SAMPLE_THRESHOLD = 30
_STUDY_SIGNATURE_FIELDS: tuple[str, ...] = (
    "context_length",
    "horizon",
    "target_indices",
    "forecast_num_samples",
    "forecast_batch_size",
    "seed",
    "mask_protocol",
    "forecast_call_protocol",
)
_DEFAULT_PRIMARY_COMPARATOR_ROLES: tuple[str, ...] = (
    "baseline",
    "missing_anchor",
    "selector_baseline",
)


EpisodeKey = tuple[str, str, str]


@dataclass(frozen=True)
class _Source:
    metrics_path: Path
    sha256: str
    row_count: int
    manifest_path: Path | None
    manifest_status: str | None
    manifest_forecaster_id: str | None
    study_signature: Mapping[str, Any] | None

    def payload(self) -> dict[str, Any]:
        return {
            "metrics_path": str(self.metrics_path),
            "sha256": self.sha256,
            "row_count": self.row_count,
            "manifest_path": (None if self.manifest_path is None else str(self.manifest_path)),
            "manifest_status": self.manifest_status,
            "manifest_forecaster_id": self.manifest_forecaster_id,
            "study_signature": (
                None if self.study_signature is None else dict(self.study_signature)
            ),
        }


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    if not rows:
        raise ValueError(f"cannot write an empty result table: {path.name}")
    fieldnames = tuple(rows[0])
    if any(tuple(row) != fieldnames for row in rows):
        raise ValueError("result rows do not share a stable CSV schema")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_source(
    value: str | Path,
) -> tuple[
    Path,
    Path,
    str,
    str | None,
    int | None,
    dict[str, Any],
    str,
]:
    source = Path(value).resolve()
    if source.is_dir():
        root = source
        metrics = source / "episode_metrics.jsonl"
    else:
        if source.name != "episode_metrics.jsonl":
            raise ValueError(
                "summarize-main only accepts evaluation directories or their "
                "episode_metrics.jsonl files"
            )
        root = source.parent
        metrics = source
    if not metrics.is_file():
        raise FileNotFoundError(f"evaluation metrics do not exist: {metrics}")
    manifest_path = root / "evaluation_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            "summarize-main requires a completed evaluation_manifest.json beside "
            f"the metrics file: {metrics}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read evaluation manifest: {manifest_path}") from error
    if not isinstance(manifest, Mapping):
        raise ValueError(f"evaluation manifest must be an object: {manifest_path}")
    status = str(manifest.get("status", ""))
    if status != "completed":
        raise ValueError(
            f"evaluation manifest is not completed ({status or 'missing status'}): {manifest_path}"
        )
    forecaster_id = manifest.get("forecaster_id")
    total_rows = manifest.get("total_rows")
    if total_rows is not None:
        try:
            total_rows = int(total_rows)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"evaluation manifest has an invalid total_rows: {manifest_path}"
            ) from error
        if total_rows < 0:
            raise ValueError(f"evaluation manifest has a negative total_rows: {manifest_path}")
    signature = manifest.get("evaluation_signature")
    if not isinstance(signature, Mapping):
        raise ValueError(
            f"completed evaluation manifest has no evaluation_signature: {manifest_path}"
        )
    missing_signature_fields = sorted(
        {"forecaster_id", *_STUDY_SIGNATURE_FIELDS} - signature.keys()
    )
    if missing_signature_fields:
        raise ValueError(
            f"evaluation_signature is incomplete at {manifest_path}: "
            + ", ".join(missing_signature_fields)
        )
    if forecaster_id is None or str(signature["forecaster_id"]) != str(forecaster_id):
        raise ValueError(
            f"evaluation manifest and signature forecaster IDs differ: {manifest_path}"
        )
    study_signature = {field: signature[field] for field in _STUDY_SIGNATURE_FIELDS}

    recorded_sha256 = manifest.get("episode_metrics_jsonl_sha256")
    if not isinstance(recorded_sha256, str):
        raise ValueError(f"completed evaluation manifest has no metrics SHA-256: {manifest_path}")
    normalized_sha256 = recorded_sha256.strip().lower()
    if len(normalized_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in normalized_sha256
    ):
        raise ValueError(f"evaluation manifest has an invalid metrics SHA-256: {manifest_path}")
    actual_sha256 = _sha256(metrics)
    if actual_sha256 != normalized_sha256:
        raise ValueError(f"evaluation metrics SHA-256 does not match its manifest: {metrics}")
    recorded_size = manifest.get("episode_metrics_jsonl_size_bytes")
    if recorded_size is not None:
        try:
            expected_size = int(recorded_size)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"evaluation manifest has an invalid metrics size: {manifest_path}"
            ) from error
        if expected_size < 0 or metrics.stat().st_size != expected_size:
            raise ValueError(f"evaluation metrics size does not match its manifest: {metrics}")
    return (
        metrics,
        manifest_path,
        status,
        None if forecaster_id is None else str(forecaster_id),
        total_rows,
        study_signature,
        actual_sha256,
    )


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _metric_eligible(row: Mapping[str, Any]) -> bool:
    eligible = bool(row.get("metric_eligible", row.get("native_valid", True)))
    return eligible and all(
        _finite_number(row.get(metric)) is not None for metric in FORECAST_METRICS
    )


def _metric_available(row: Mapping[str, Any], metric: str) -> bool:
    return bool(row.get("metric_eligible", row.get("native_valid", True))) and (
        _finite_number(row.get(metric)) is not None
    )


def _episode_key(row: Mapping[str, Any]) -> EpisodeKey:
    return (
        str(row.get("forecaster_id", "")),
        str(row.get("dataset_id", "")),
        str(row.get("episode_id", "")),
    )


_SHARED_METHOD_ROLES = frozenset({"reference", "baseline", "missing_anchor", "oracle"})
_SHARED_ROW_FIELDS = (
    "schema_version",
    "episode_id",
    "dataset_id",
    "family_id",
    "forecaster_id",
    "routing_forecaster_id",
    "routing_artifact_forecaster_id",
    "mechanism",
    "missing_rate",
    "item_id",
    "forecast_origin",
    "seed",
    "mask_protocol",
    "mask_seed",
    "mask_realization_id",
    "target_missing_rate",
    "global_missing_rate",
    "local_missing_rate",
    "contains_missing",
    "mase_scale_lag",
    "method",
    "method_role",
    "oracle_source",
    "oracle_eligible",
    "native_valid",
    "metric_eligible",
    "ineligibility_reason",
    "candidate_status",
    "mase",
    "mae",
    "rmse",
    "imputation_mae",
    "imputation_rmse",
    "degradation_vs_clean_mase",
    "relative_degradation_vs_clean",
    "regret_mase",
    "relative_regret",
    "runtime_seconds",
    "rss_delta_bytes",
    "runtime_scope",
    "forecast_seed",
)


def _shared_row_conflicts(
    existing: Mapping[str, Any],
    incoming: Mapping[str, Any],
) -> tuple[str, ...]:
    return tuple(
        field for field in _SHARED_ROW_FIELDS if existing.get(field) != incoming.get(field)
    )


def _read_sources(
    inputs: Sequence[str | Path],
) -> tuple[
    dict[EpisodeKey, dict[str, dict[str, Any]]],
    dict[EpisodeKey, dict[str, Any]],
    list[_Source],
    dict[str, str],
]:
    if not inputs:
        raise ValueError("at least one evaluation input is required")
    resolved = [_resolve_source(value) for value in inputs]
    paths = [entry[0] for entry in resolved]
    if len(set(paths)) != len(paths):
        raise ValueError("evaluation inputs must resolve to unique metrics files")

    episodes: dict[EpisodeKey, dict[str, dict[str, Any]]] = {}
    metadata: dict[EpisodeKey, dict[str, Any]] = {}
    method_roles: dict[str, str] = {}
    sources: list[_Source] = []
    required = {
        "episode_id",
        "dataset_id",
        "family_id",
        "forecaster_id",
        "mechanism",
        "missing_rate",
        "method",
        "method_role",
    }
    metadata_fields = (
        "episode_id",
        "dataset_id",
        "family_id",
        "forecaster_id",
        "mechanism",
        "missing_rate",
        "item_id",
        "mask_protocol",
        "mask_realization_id",
        "contains_missing",
    )
    reference_signature: dict[str, Any] | None = None
    for (
        metrics_path,
        manifest_path,
        manifest_status,
        manifest_forecaster_id,
        manifest_total_rows,
        study_signature,
        manifest_metrics_sha256,
    ) in sorted(resolved, key=lambda entry: str(entry[0]).casefold()):
        row_count = 0
        source_forecasters: set[str] = set()
        source_episode_methods: set[tuple[EpisodeKey, str]] = set()
        with metrics_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {metrics_path}:{line_number}") from error
                missing = sorted(required - raw.keys())
                if missing:
                    raise ValueError(
                        f"missing fields at {metrics_path}:{line_number}: " + ", ".join(missing)
                    )
                row = dict(raw)
                if row.get("mask_protocol") != "sequence_mask_v2":
                    raise ValueError(
                        f"row does not use sequence_mask_v2 at {metrics_path}:{line_number}"
                    )
                key = _episode_key(row)
                if not all(key):
                    raise ValueError(f"empty episode identity at {metrics_path}:{line_number}")
                method = str(row["method"])
                source_forecasters.add(key[0])
                role = str(row["method_role"])
                if not method or not role:
                    raise ValueError(f"empty method identity at {metrics_path}:{line_number}")
                previous_role = method_roles.setdefault(method, role)
                if previous_role != role:
                    raise ValueError(
                        f"method {method!r} has conflicting roles: {previous_role!r} and {role!r}"
                    )
                source_identity = (key, method)
                if source_identity in source_episode_methods:
                    raise ValueError(
                        f"duplicate forecaster/episode/method row: {key[0]}/{key[2]}/{method}"
                    )
                source_episode_methods.add(source_identity)
                current_metadata = {field: row.get(field) for field in metadata_fields}
                if key in metadata and metadata[key] != current_metadata:
                    raise ValueError(f"inconsistent metadata within episode {key[0]}/{key[2]}")
                metadata[key] = current_metadata
                episode_rows = episodes.setdefault(key, {})
                if method in episode_rows:
                    existing = episode_rows[method]
                    if role not in _SHARED_METHOD_ROLES:
                        raise ValueError(
                            f"duplicate forecaster/episode/method row: {key[0]}/{key[2]}/{method}"
                        )
                    conflicts = _shared_row_conflicts(existing, row)
                    if conflicts:
                        raise ValueError(
                            "conflicting shared forecaster/episode/method row: "
                            f"{key[0]}/{key[2]}/{method}; fields: " + ", ".join(conflicts)
                        )
                else:
                    episode_rows[method] = row
                row_count += 1
        if row_count == 0:
            raise ValueError(f"evaluation metrics file is empty: {metrics_path}")
        if manifest_total_rows is not None and manifest_total_rows != row_count:
            raise ValueError(
                f"evaluation row count does not match its manifest at {metrics_path}: "
                f"{row_count} != {manifest_total_rows}"
            )
        if manifest_forecaster_id is not None and source_forecasters != {manifest_forecaster_id}:
            raise ValueError(
                f"evaluation forecaster rows do not match the manifest at {metrics_path}"
            )
        if study_signature is not None:
            if reference_signature is None:
                reference_signature = study_signature
            elif study_signature != reference_signature:
                raise ValueError(
                    "evaluation study signatures differ in context, horizon, targets, "
                    "sampling, seed, or forecast-call protocol"
                )
        final_metrics_sha256 = _sha256(metrics_path)
        if final_metrics_sha256 != manifest_metrics_sha256:
            raise ValueError(
                f"evaluation metrics changed while summarize-main was reading: {metrics_path}"
            )
        sources.append(
            _Source(
                metrics_path=metrics_path,
                sha256=final_metrics_sha256,
                row_count=row_count,
                manifest_path=manifest_path,
                manifest_status=manifest_status,
                manifest_forecaster_id=manifest_forecaster_id,
                study_signature=study_signature,
            )
        )

    for key, rows in episodes.items():
        absent = sorted({"clean", "b_fais", "oracle"} - rows.keys())
        if absent:
            raise ValueError(
                f"episode {key[0]}/{key[2]} is incomplete; missing: " + ", ".join(absent)
            )
        for method in ("clean", "b_fais", "oracle"):
            if not _metric_eligible(rows[method]):
                raise ValueError(
                    f"episode {key[0]}/{key[2]} has invalid required method {method!r}"
                )
    return episodes, metadata, sources, method_roles


def _scope_groups(
    episode_keys: Sequence[EpisodeKey],
    metadata: Mapping[EpisodeKey, Mapping[str, Any]],
) -> Iterable[tuple[str, tuple[str, ...], tuple[Any, ...], list[EpisodeKey]]]:
    for scope, fields in SUMMARY_SCOPES:
        partitions: dict[tuple[Any, ...], list[EpisodeKey]] = defaultdict(list)
        for episode_key in episode_keys:
            group_key = tuple(metadata[episode_key].get(field) for field in fields)
            partitions[group_key].append(episode_key)
        for group_key in sorted(
            partitions, key=lambda values: tuple(str(value) for value in values)
        ):
            yield scope, fields, group_key, partitions[group_key]


def _group_prefix(scope: str, fields: Sequence[str], values: Sequence[Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "scope": scope,
        "group_value": "all" if not fields else " | ".join(map(str, values)),
        **{field: None for field in GROUP_FIELDS},
    }
    for field, value in zip(fields, values, strict=True):
        result[field] = value
    return result


def _method_order(method_roles: Mapping[str, str]) -> tuple[str, ...]:
    selector_methods = sorted(
        method for method, role in method_roles.items() if role == "selector_baseline"
    )
    candidate_methods = sorted(
        method
        for method, role in method_roles.items()
        if role in {"baseline", "missing_anchor"} and method not in {"locf", "linear_interp"}
    )
    ordered = [
        "b_fais",
        *selector_methods,
        "clean",
        "locf",
        "linear_interp",
        *candidate_methods,
        "oracle",
    ]
    return tuple(method for method in ordered if method in method_roles)


def _comparator_order(method_roles: Mapping[str, str]) -> tuple[str, ...]:
    methods = _method_order(method_roles)
    return tuple(method for method in methods if method != "b_fais")


def _average_tie_ranks(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    result: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        for method, _ in ordered[index:end]:
            result[method] = average_rank
        index = end
    return result


def _episode_ranks(
    episodes: Mapping[EpisodeKey, Mapping[str, Mapping[str, Any]]],
) -> tuple[dict[EpisodeKey, dict[str, float]], dict[EpisodeKey, int]]:
    ranks: dict[EpisodeKey, dict[str, float]] = {}
    pool_sizes: dict[EpisodeKey, int] = {}
    for episode_key, rows in episodes.items():
        values: dict[str, float] = {}
        for method, row in rows.items():
            role = str(row["method_role"])
            if method != "b_fais" and role not in {
                "selector_baseline",
                "baseline",
                "missing_anchor",
            }:
                continue
            if _metric_eligible(row):
                value = _finite_number(row.get("mase"))
                assert value is not None
                values[method] = value
        ranks[episode_key] = _average_tie_ranks(values)
        pool_sizes[episode_key] = len(values)
    return ranks, pool_sizes


def _summary(values: Sequence[float]) -> tuple[int, float | None, float | None, float | None]:
    if not values:
        return 0, None, None, None
    array = np.asarray(values, dtype=float)
    return (
        int(array.size),
        float(np.mean(array)),
        float(np.median(array)),
        float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
    )


def _method_summaries(
    episodes: Mapping[EpisodeKey, Mapping[str, Mapping[str, Any]]],
    metadata: Mapping[EpisodeKey, Mapping[str, Any]],
    method_roles: Mapping[str, str],
    ranks: Mapping[EpisodeKey, Mapping[str, float]],
    pool_sizes: Mapping[EpisodeKey, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    episode_keys = sorted(episodes)
    methods = _method_order(method_roles)
    for scope, fields, values, group in _scope_groups(episode_keys, metadata):
        for method in methods:
            recorded = [episodes[key].get(method) for key in group]
            valid = [row for row in recorded if row is not None and _metric_eligible(row)]
            rank_values = [ranks[key][method] for key in group if method in ranks[key]]
            rank_pool_values = [float(pool_sizes[key]) for key in group if method in ranks[key]]
            result: dict[str, Any] = {
                **_group_prefix(scope, fields, values),
                "method": method,
                "method_role": method_roles[method],
                "episode_count": len(group),
                "recorded_count": sum(row is not None for row in recorded),
                "valid_count": len(valid),
                "valid_rate": len(valid) / len(group),
                "small_sample": len(valid) < SMALL_SAMPLE_THRESHOLD,
                "rank_count": len(rank_values),
                "average_rank_mase": (None if not rank_values else float(np.mean(rank_values))),
                "rank_pool_size_mean": (
                    None if not rank_pool_values else float(np.mean(rank_pool_values))
                ),
            }
            for metric in PAIRED_METRICS:
                metric_values = [
                    float(row[metric])
                    for row in recorded
                    if row is not None and _metric_available(row, metric)
                ]
                count, mean, median, standard_deviation = _summary(metric_values)
                result[f"{metric}_count"] = count
                result[f"{metric}_small_sample"] = count < SMALL_SAMPLE_THRESHOLD
                result[f"{metric}_mean"] = mean
                result[f"{metric}_median"] = median
                result[f"{metric}_std"] = standard_deviation
            for metric in (
                "regret_mase",
                "degradation_vs_clean_mase",
                "runtime_seconds",
                "rss_delta_bytes",
            ):
                metric_values = [
                    number
                    for row in (
                        valid if metric not in {"runtime_seconds", "rss_delta_bytes"} else recorded
                    )
                    if row is not None and (number := _finite_number(row.get(metric))) is not None
                ]
                count, mean, median, standard_deviation = _summary(metric_values)
                result[f"{metric}_count"] = count
                result[f"{metric}_mean"] = mean
                result[f"{metric}_median"] = median
                result[f"{metric}_std"] = standard_deviation
                if metric == "runtime_seconds":
                    result["runtime_seconds_sum"] = (
                        None if not metric_values else float(np.sum(metric_values))
                    )
            rows.append(result)
    return rows


def _derived_seed(seed: int, *parts: Any) -> int:
    payload = json.dumps([int(seed), *parts], ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def _bootstrap_mean_intervals(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
    chunk_size: int = 32,
) -> np.ndarray:
    """Return percentile intervals for column means under paired row resampling."""

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("bootstrap values must have shape [episodes, statistics]")
    if not np.isfinite(matrix).all():
        raise ValueError("bootstrap values must be finite")
    if replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    episode_count = matrix.shape[0]
    if episode_count < 2:
        return np.full((matrix.shape[1], 2), np.nan, dtype=float)
    rng = np.random.default_rng(seed)
    probability = np.full(episode_count, 1.0 / episode_count, dtype=float)
    means = np.empty((replicates, matrix.shape[1]), dtype=float)
    for start in range(0, replicates, chunk_size):
        stop = min(replicates, start + chunk_size)
        counts = rng.multinomial(episode_count, probability, size=stop - start)
        means[start:stop] = (counts @ matrix) / episode_count
    quantiles = np.quantile(means, (0.025, 0.975), axis=0)
    return np.column_stack((quantiles[0], quantiles[1]))


def _hierarchical_family_interval(
    values: Mapping[EpisodeKey, float],
    metadata: Mapping[EpisodeKey, Mapping[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> tuple[float | None, float | None]:
    """Bootstrap family, dataset, and item/mask-realization clusters in order."""

    hierarchy: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for key, value in values.items():
        entry = metadata[key]
        family = str(entry.get("family_id", ""))
        dataset = str(entry.get("dataset_id", ""))
        cluster = "|".join(
            (
                str(entry.get("item_id", "")),
                str(entry.get("mask_realization_id", "")),
            )
        )
        hierarchy[family][dataset][cluster].append(float(value))
    families = tuple(sorted(hierarchy))
    if len(families) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    statistics = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled_family_names = rng.choice(families, size=len(families), replace=True)
        sampled_family_values: list[float] = []
        for family in sampled_family_names:
            datasets = hierarchy[str(family)]
            dataset_names = tuple(sorted(datasets))
            sampled_dataset_names = rng.choice(dataset_names, size=len(dataset_names), replace=True)
            sampled_dataset_values: list[float] = []
            for dataset in sampled_dataset_names:
                clusters = datasets[str(dataset)]
                cluster_names = tuple(sorted(clusters))
                sampled_clusters = rng.choice(cluster_names, size=len(cluster_names), replace=True)
                sampled_dataset_values.append(
                    float(
                        np.mean([np.mean(clusters[str(cluster)]) for cluster in sampled_clusters])
                    )
                )
            sampled_family_values.append(float(np.mean(sampled_dataset_values)))
        statistics[replicate] = float(np.mean(sampled_family_values))
    low, high = np.quantile(statistics, (0.025, 0.975))
    return float(low), float(high)


def _family_macro_primary(
    episodes: Mapping[EpisodeKey, Mapping[str, Mapping[str, Any]]],
    metadata: Mapping[EpisodeKey, Mapping[str, Any]],
    method_roles: Mapping[str, str],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    comparator_roles: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Build the primary family-equal MASE comparison table."""

    primary_scopes = SUMMARY_SCOPES[:4]
    allowed_roles = None if comparator_roles is None else frozenset(comparator_roles)
    comparators = tuple(
        comparator
        for comparator in _comparator_order(method_roles)
        if allowed_roles is None or method_roles[comparator] in allowed_roles
    )
    results: list[dict[str, Any]] = []
    all_keys = sorted(episodes)
    for view, view_keys in (
        ("all_windows", all_keys),
        (
            "windows_with_missing",
            [key for key in all_keys if bool(metadata[key].get("contains_missing"))],
        ),
    ):
        for scope, fields in primary_scopes:
            partitions: dict[tuple[Any, ...], list[EpisodeKey]] = defaultdict(list)
            for key in view_keys:
                partitions[tuple(metadata[key].get(field) for field in fields)].append(key)
            for group_values in sorted(partitions, key=lambda values: tuple(map(str, values))):
                group = partitions[group_values]
                for comparator in comparators:
                    pair_keys = [
                        key
                        for key in group
                        if comparator in episodes[key]
                        and _metric_available(episodes[key]["b_fais"], "mase")
                        and _metric_available(episodes[key][comparator], "mase")
                    ]
                    if not pair_keys:
                        continue
                    b_values = {key: float(episodes[key]["b_fais"]["mase"]) for key in pair_keys}
                    comparator_values = {
                        key: float(episodes[key][comparator]["mase"]) for key in pair_keys
                    }
                    deltas = {key: b_values[key] - comparator_values[key] for key in pair_keys}
                    families: dict[str, list[EpisodeKey]] = defaultdict(list)
                    for key in pair_keys:
                        families[str(metadata[key]["family_id"])].append(key)
                    family_b = {
                        family: float(np.mean([b_values[key] for key in keys]))
                        for family, keys in families.items()
                    }
                    family_comparator = {
                        family: float(np.mean([comparator_values[key] for key in keys]))
                        for family, keys in families.items()
                    }
                    family_delta = np.asarray(
                        [
                            family_b[family] - family_comparator[family]
                            for family in sorted(families)
                        ],
                        dtype=float,
                    )
                    ci_low, ci_high = _hierarchical_family_interval(
                        deltas,
                        metadata,
                        replicates=bootstrap_replicates,
                        seed=_derived_seed(
                            bootstrap_seed,
                            "family_macro",
                            view,
                            scope,
                            group_values,
                            comparator,
                        ),
                    )
                    p_value: float | None = None
                    if len(family_delta) >= 2:
                        if np.allclose(family_delta, 0.0):
                            p_value = 1.0
                        else:
                            from scipy.stats import wilcoxon

                            p_value = float(
                                wilcoxon(
                                    family_delta,
                                    alternative="two-sided",
                                    zero_method="pratt",
                                ).pvalue
                            )
                    result = {
                        "view": view,
                        **_group_prefix(scope, fields, group_values),
                        "comparator": comparator,
                        "comparator_role": method_roles[comparator],
                        "family_count": len(families),
                        "dataset_count": len(
                            {str(metadata[key]["dataset_id"]) for key in pair_keys}
                        ),
                        "pair_count": len(pair_keys),
                        "b_fais_mase_family_macro": float(np.mean(tuple(family_b.values()))),
                        "comparator_mase_family_macro": float(
                            np.mean(tuple(family_comparator.values()))
                        ),
                        "mase_family_macro_delta": float(np.mean(family_delta)),
                        "mase_family_macro_delta_ci95_low": ci_low,
                        "mase_family_macro_delta_ci95_high": ci_high,
                        "family_win_rate": float(np.mean(family_delta < 0)),
                        "family_tie_rate": float(np.mean(np.isclose(family_delta, 0.0))),
                        "wilcoxon_p_value": p_value,
                        "holm_adjusted_p_value": None,
                    }
                    results.append(result)

    correction_groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(results):
        if row["wilcoxon_p_value"] is not None:
            correction_groups[
                (str(row["view"]), str(row["scope"]), str(row["group_value"]))
            ].append(index)
    for indices in correction_groups.values():
        ordered = sorted(indices, key=lambda index: results[index]["wilcoxon_p_value"])
        previous = 0.0
        count = len(ordered)
        for rank, index in enumerate(ordered):
            adjusted = min(
                1.0,
                max(
                    previous,
                    (count - rank) * float(results[index]["wilcoxon_p_value"]),
                ),
            )
            results[index]["holm_adjusted_p_value"] = adjusted
            previous = adjusted
    return results


def _comparison_summaries(
    episodes: Mapping[EpisodeKey, Mapping[str, Mapping[str, Any]]],
    metadata: Mapping[EpisodeKey, Mapping[str, Any]],
    method_roles: Mapping[str, str],
    ranks: Mapping[EpisodeKey, Mapping[str, float]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    episode_keys = sorted(episodes)
    comparators = _comparator_order(method_roles)
    for scope, fields, values, group in _scope_groups(episode_keys, metadata):
        paired: dict[tuple[str, str], dict[str, Any]] = {}
        for comparator in comparators:
            for metric in PAIRED_METRICS:
                pair_keys = [
                    key
                    for key in group
                    if comparator in episodes[key]
                    and _metric_available(episodes[key]["b_fais"], metric)
                    and _metric_available(episodes[key][comparator], metric)
                ]
                b_values = np.asarray(
                    [float(episodes[key]["b_fais"][metric]) for key in pair_keys],
                    dtype=float,
                )
                comparator_values = np.asarray(
                    [float(episodes[key][comparator][metric]) for key in pair_keys],
                    dtype=float,
                )
                paired[(comparator, metric)] = {
                    "keys": pair_keys,
                    "b_values": b_values,
                    "comparator_values": comparator_values,
                    "delta": b_values - comparator_values,
                }

        masks: dict[tuple[EpisodeKey, ...], list[tuple[str, str]]] = defaultdict(list)
        for comparison_key, values_by_metric in paired.items():
            masks[tuple(values_by_metric["keys"])].append(comparison_key)
        intervals: dict[tuple[str, str], np.ndarray] = {}
        for mask_keys, comparison_keys in masks.items():
            if len(mask_keys) < 2:
                for comparison_key in comparison_keys:
                    intervals[comparison_key] = np.full(2, np.nan)
                continue
            ordered_comparisons = sorted(comparison_keys)
            matrix = np.column_stack(
                [paired[comparison_key]["delta"] for comparison_key in ordered_comparisons]
            )
            interval_matrix = _bootstrap_mean_intervals(
                matrix,
                replicates=bootstrap_replicates,
                seed=_derived_seed(
                    bootstrap_seed,
                    scope,
                    list(values),
                    ordered_comparisons,
                    list(mask_keys),
                ),
            )
            for comparison_index, comparison_key in enumerate(ordered_comparisons):
                intervals[comparison_key] = interval_matrix[comparison_index]

        for comparator in comparators:
            mase_pairs = paired[(comparator, "mase")]["keys"]
            comparator_recorded = sum(comparator in episodes[key] for key in group)
            comparator_valid = sum(
                comparator in episodes[key] and _metric_eligible(episodes[key][comparator])
                for key in group
            )
            b_valid = sum(_metric_eligible(episodes[key]["b_fais"]) for key in group)
            result: dict[str, Any] = {
                **_group_prefix(scope, fields, values),
                "comparator": comparator,
                "comparator_role": method_roles[comparator],
                "episode_count": len(group),
                "comparator_recorded_count": comparator_recorded,
                "comparator_valid_count": comparator_valid,
                "comparator_valid_rate": comparator_valid / len(group),
                "b_fais_valid_count": b_valid,
                "b_fais_valid_rate": b_valid / len(group),
                "pair_count": len(mase_pairs),
                "paired_valid_rate": len(mase_pairs) / len(group),
                "small_sample": len(mase_pairs) < SMALL_SAMPLE_THRESHOLD,
                "bootstrap_replicates": (bootstrap_replicates if len(mase_pairs) >= 2 else 0),
            }
            for metric in PAIRED_METRICS:
                metric_pair = paired[(comparator, metric)]
                pair_keys = metric_pair["keys"]
                b_values = metric_pair["b_values"]
                comparator_values = metric_pair["comparator_values"]
                delta = metric_pair["delta"]
                if not pair_keys:
                    b_mean = comparator_mean = mean_delta = win_rate = tie_rate = None
                    ci_low = ci_high = None
                else:
                    tied = np.isclose(b_values, comparator_values, rtol=1e-12, atol=1e-12)
                    b_mean = float(np.mean(b_values))
                    comparator_mean = float(np.mean(comparator_values))
                    mean_delta = float(np.mean(delta))
                    win_rate = float(np.mean((b_values < comparator_values) & ~tied))
                    tie_rate = float(np.mean(tied))
                    interval = intervals[(comparator, metric)]
                    ci_low = None if np.isnan(interval[0]) else float(interval[0])
                    ci_high = None if np.isnan(interval[1]) else float(interval[1])
                result[f"{metric}_pair_count"] = len(pair_keys)
                result[f"{metric}_paired_valid_rate"] = len(pair_keys) / len(group)
                result[f"{metric}_small_sample"] = len(pair_keys) < SMALL_SAMPLE_THRESHOLD
                result[f"{metric}_bootstrap_replicates"] = (
                    bootstrap_replicates if len(pair_keys) >= 2 else 0
                )
                result[f"b_fais_{metric}_mean"] = b_mean
                result[f"comparator_{metric}_mean"] = comparator_mean
                result[f"{metric}_mean_delta"] = mean_delta
                result[f"{metric}_mean_delta_ci95_low"] = ci_low
                result[f"{metric}_mean_delta_ci95_high"] = ci_high
                result[f"{metric}_win_rate"] = win_rate
                result[f"{metric}_tie_rate"] = tie_rate

            rank_pairs = [
                key for key in mase_pairs if "b_fais" in ranks[key] and comparator in ranks[key]
            ]
            result["rank_pair_count"] = len(rank_pairs)
            result["b_fais_average_rank_mase"] = (
                None
                if not rank_pairs
                else float(np.mean([ranks[key]["b_fais"] for key in rank_pairs]))
            )
            result["comparator_average_rank_mase"] = (
                None
                if not rank_pairs
                else float(np.mean([ranks[key][comparator] for key in rank_pairs]))
            )
            for metric in ("regret_mase", "runtime_seconds"):
                b_numbers = [
                    number
                    for key in mase_pairs
                    if (number := _finite_number(episodes[key]["b_fais"].get(metric))) is not None
                ]
                comparator_numbers = [
                    number
                    for key in mase_pairs
                    if (number := _finite_number(episodes[key][comparator].get(metric))) is not None
                ]
                result[f"b_fais_{metric}_mean"] = (
                    None if not b_numbers else float(np.mean(b_numbers))
                )
                result[f"comparator_{metric}_mean"] = (
                    None if not comparator_numbers else float(np.mean(comparator_numbers))
                )
            results.append(result)
    return results


def _format_number(value: Any, digits: int = 4) -> str:
    number = _finite_number(value)
    return "NA" if number is None else f"{number:.{digits}f}"


def _markdown_table(headers: Sequence[str], body: Sequence[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in body:
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_markdown(
    path: Path,
    *,
    sources: Sequence[_Source],
    method_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    primary_rows: Sequence[Mapping[str, Any]],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> Path:
    overall_methods = [row for row in method_rows if row["scope"] == "overall"]
    overall_comparisons = [row for row in comparison_rows if row["scope"] == "overall"]
    breakdown = [
        row for row in method_rows if row["method"] == "b_fais" and row["scope"] != "overall"
    ]
    key_breakdown = [
        row
        for row in comparison_rows
        if row["scope"] != "overall"
        and row["comparator"] in {"clean", "locf", "linear_interp", "oracle"}
    ]
    small_count = sum(bool(row["small_sample"]) for row in comparison_rows)
    primary_overall = [
        row for row in primary_rows if row["scope"] == "overall" and row["view"] == "all_windows"
    ]
    lines = [
        "# TSFM-FAIS multi-forecaster summary",
        "",
        "This report is generated from completed episode-level evaluation rows. "
        "It is descriptive and does not claim statistical significance.",
        "",
        "## Statistical definitions",
        "",
        "- Every paired comparison uses the same forecaster/episode on both sides. "
        "Losses are lower-is-better.",
        "- `mean delta` is B-FAIS minus the comparator. Negative values favor B-FAIS.",
        "- `win rate` is the strict fraction with lower B-FAIS loss; ties are reported separately.",
        "- The 95% interval is a percentile paired bootstrap interval for the episode-level "
        f"mean delta ({bootstrap_replicates} resamples, seed {bootstrap_seed}). An interval "
        "is unavailable when fewer than two pairs exist.",
        "- MASE average rank uses midranks among B-FAIS, selector baselines, and all "
        "natively valid evaluated single-imputer candidates in each episode. Clean and "
        "oracle are excluded from that ranking universe.",
        "- Oracle is the valid single imputer selected by minimum MASE in each episode. Its "
        "MAE and RMSE are copied from that MASE-selected method and are not metric-specific "
        "oracles. It is also not an imputation-error oracle.",
        "- Clean has zero imputation error by construction and is shown only as a complete-"
        "context reference. It is not executable when the input contains missing values.",
        "- Overall rows weight each forecaster/episode evaluation equally; they are not "
        "reweighted to give every model or data family equal mass.",
        "- Episode bootstrap intervals do not adjust for dependence among episodes that "
        "share an item, forecast origin, data family, or imputed context.",
        "- Runtime is end-to-end imputation for B-FAIS and selector baselines, and one "
        "imputer invocation for a candidate, so the scopes differ.",
        f"- `{small_count}` comparison strata contain fewer than "
        f"{SMALL_SAMPLE_THRESHOLD} paired episodes and are marked as small samples.",
        "",
        "## Inputs",
        "",
    ]
    lines[4:4] = [
        "## Primary family-macro MASE comparisons",
        "",
        "Families receive equal weight. Intervals resample family, dataset, then "
        "item/mask-realization clusters. Holm-adjusted values correct all comparators "
        "within the displayed stratum.",
        "",
        *_markdown_table(
            (
                "comparator",
                "families",
                "pairs",
                "MASE delta [95% CI]",
                "family win",
                "Holm p",
            ),
            [
                (
                    row["comparator"],
                    row["family_count"],
                    row["pair_count"],
                    f"{_format_number(row['mase_family_macro_delta'])} "
                    f"[{_format_number(row['mase_family_macro_delta_ci95_low'])}, "
                    f"{_format_number(row['mase_family_macro_delta_ci95_high'])}]",
                    _format_number(row["family_win_rate"]),
                    _format_number(row["holm_adjusted_p_value"]),
                )
                for row in primary_overall
            ],
        ),
        "",
    ]
    lines.extend(
        _markdown_table(
            ("metrics", "rows", "SHA-256", "manifest"),
            [
                (
                    source.metrics_path,
                    source.row_count,
                    source.sha256,
                    source.manifest_status or "not provided",
                )
                for source in sources
            ],
        )
    )
    lines.extend(["", "## Overall method performance", ""])
    lines.extend(
        _markdown_table(
            (
                "method",
                "valid",
                "MASE",
                "MAE",
                "RMSE",
                "imputation MAE",
                "imputation RMSE",
                "MASE rank",
                "MASE regret",
                "runtime (s)",
            ),
            [
                (
                    row["method"],
                    f"{row['valid_count']}/{row['episode_count']} "
                    f"({_format_number(row['valid_rate'])})",
                    _format_number(row["mase_mean"]),
                    _format_number(row["mae_mean"]),
                    _format_number(row["rmse_mean"]),
                    f"{_format_number(row['imputation_mae_mean'])} "
                    f"(n={row['imputation_mae_count']})",
                    f"{_format_number(row['imputation_rmse_mean'])} "
                    f"(n={row['imputation_rmse_count']})",
                    _format_number(row["average_rank_mase"]),
                    _format_number(row["regret_mase_mean"]),
                    _format_number(row["runtime_seconds_mean"]),
                )
                for row in overall_methods
            ],
        )
    )
    lines.extend(["", "## Overall paired B-FAIS comparisons", ""])
    lines.extend(
        _markdown_table(
            (
                "comparator",
                "pairs",
                "MASE delta [95% CI]",
                "MASE win",
                "MAE delta [95% CI]",
                "RMSE delta [95% CI]",
                "small sample",
            ),
            [
                (
                    row["comparator"],
                    f"{row['pair_count']}/{row['episode_count']}",
                    f"{_format_number(row['mase_mean_delta'])} "
                    f"[{_format_number(row['mase_mean_delta_ci95_low'])}, "
                    f"{_format_number(row['mase_mean_delta_ci95_high'])}]",
                    _format_number(row["mase_win_rate"]),
                    f"{_format_number(row['mae_mean_delta'])} "
                    f"[{_format_number(row['mae_mean_delta_ci95_low'])}, "
                    f"{_format_number(row['mae_mean_delta_ci95_high'])}]",
                    f"{_format_number(row['rmse_mean_delta'])} "
                    f"[{_format_number(row['rmse_mean_delta_ci95_low'])}, "
                    f"{_format_number(row['rmse_mean_delta_ci95_high'])}]",
                    "yes" if row["small_sample"] else "no",
                )
                for row in overall_comparisons
            ],
        )
    )
    lines.extend(["", "## Overall paired imputation-error comparisons", ""])
    lines.extend(
        _markdown_table(
            (
                "comparator",
                "imputation MAE pairs",
                "imputation MAE delta [95% CI]",
                "imputation MAE win",
                "imputation RMSE pairs",
                "imputation RMSE delta [95% CI]",
                "imputation RMSE win",
            ),
            [
                (
                    row["comparator"],
                    f"{row['imputation_mae_pair_count']}/{row['episode_count']}",
                    f"{_format_number(row['imputation_mae_mean_delta'])} "
                    f"[{_format_number(row['imputation_mae_mean_delta_ci95_low'])}, "
                    f"{_format_number(row['imputation_mae_mean_delta_ci95_high'])}]",
                    _format_number(row["imputation_mae_win_rate"]),
                    f"{row['imputation_rmse_pair_count']}/{row['episode_count']}",
                    f"{_format_number(row['imputation_rmse_mean_delta'])} "
                    f"[{_format_number(row['imputation_rmse_mean_delta_ci95_low'])}, "
                    f"{_format_number(row['imputation_rmse_mean_delta_ci95_high'])}]",
                    _format_number(row["imputation_rmse_win_rate"]),
                )
                for row in overall_comparisons
            ],
        )
    )
    lines.extend(["", "## B-FAIS breakdown", ""])
    lines.extend(
        _markdown_table(
            (
                "scope",
                "group",
                "episodes",
                "MASE",
                "MAE",
                "RMSE",
                "imputation MAE",
                "imputation RMSE",
                "MASE rank",
                "MASE regret",
                "runtime (s)",
            ),
            [
                (
                    row["scope"],
                    row["group_value"],
                    row["valid_count"],
                    _format_number(row["mase_mean"]),
                    _format_number(row["mae_mean"]),
                    _format_number(row["rmse_mean"]),
                    f"{_format_number(row['imputation_mae_mean'])} "
                    f"(n={row['imputation_mae_count']})",
                    f"{_format_number(row['imputation_rmse_mean'])} "
                    f"(n={row['imputation_rmse_count']})",
                    _format_number(row["average_rank_mase"]),
                    _format_number(row["regret_mase_mean"]),
                    _format_number(row["runtime_seconds_mean"]),
                )
                for row in breakdown
            ],
        )
    )
    lines.extend(["", "## Key paired imputation comparisons by breakdown", ""])
    lines.extend(
        _markdown_table(
            (
                "scope",
                "group",
                "comparator",
                "imputation MAE pairs",
                "imputation MAE delta [95% CI]",
                "imputation MAE win",
                "imputation RMSE pairs",
                "imputation RMSE delta [95% CI]",
                "imputation RMSE win",
            ),
            [
                (
                    row["scope"],
                    row["group_value"],
                    row["comparator"],
                    row["imputation_mae_pair_count"],
                    f"{_format_number(row['imputation_mae_mean_delta'])} "
                    f"[{_format_number(row['imputation_mae_mean_delta_ci95_low'])}, "
                    f"{_format_number(row['imputation_mae_mean_delta_ci95_high'])}]",
                    _format_number(row["imputation_mae_win_rate"]),
                    row["imputation_rmse_pair_count"],
                    f"{_format_number(row['imputation_rmse_mean_delta'])} "
                    f"[{_format_number(row['imputation_rmse_mean_delta_ci95_low'])}, "
                    f"{_format_number(row['imputation_rmse_mean_delta_ci95_high'])}]",
                    _format_number(row["imputation_rmse_win_rate"]),
                )
                for row in key_breakdown
            ],
        )
    )
    lines.extend(["", "## Key paired forecast comparisons by breakdown", ""])
    lines.extend(
        _markdown_table(
            (
                "scope",
                "group",
                "comparator",
                "pairs",
                "MASE delta [95% CI]",
                "MASE win",
                "valid rate",
                "small sample",
            ),
            [
                (
                    row["scope"],
                    row["group_value"],
                    row["comparator"],
                    row["pair_count"],
                    f"{_format_number(row['mase_mean_delta'])} "
                    f"[{_format_number(row['mase_mean_delta_ci95_low'])}, "
                    f"{_format_number(row['mase_mean_delta_ci95_high'])}]",
                    _format_number(row["mase_win_rate"]),
                    _format_number(row["comparator_valid_rate"]),
                    "yes" if row["small_sample"] else "no",
                )
                for row in key_breakdown
            ],
        )
    )
    lines.extend(
        [
            "",
            "All method-by-method and grouped values are available in "
            "`method_summary.csv`, `comparison_summary.csv`, and `main_summary.json`.",
            "",
        ]
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
    return path


def summarize_multi_forecaster(
    *,
    evaluation_inputs: Sequence[str | Path],
    output_dir: str | Path,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    primary_comparator_roles: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Aggregate evaluation directories into paired, grouped main-result tables."""

    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be non-negative")
    normalized_primary_roles = (
        _DEFAULT_PRIMARY_COMPARATOR_ROLES
        if primary_comparator_roles is None
        else tuple(dict.fromkeys(str(role) for role in primary_comparator_roles))
    )
    valid_primary_roles = set(_DEFAULT_PRIMARY_COMPARATOR_ROLES)
    if not normalized_primary_roles:
        raise ValueError("primary_comparator_roles must not be empty")
    unknown_roles = set(normalized_primary_roles) - valid_primary_roles
    if unknown_roles:
        raise ValueError(
            "unsupported primary comparator roles: " + ", ".join(sorted(unknown_roles))
        )
    episodes, metadata, sources, method_roles = _read_sources(evaluation_inputs)
    ranks, pool_sizes = _episode_ranks(episodes)
    method_rows = _method_summaries(episodes, metadata, method_roles, ranks, pool_sizes)
    comparison_rows = _comparison_summaries(
        episodes,
        metadata,
        method_roles,
        ranks,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    primary_rows = _family_macro_primary(
        episodes,
        metadata,
        method_roles,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        comparator_roles=normalized_primary_roles,
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "main_summary.json"
    method_csv = output / "method_summary.csv"
    comparison_csv = output / "comparison_summary.csv"
    primary_csv = output / "family_macro_comparison_summary.csv"
    markdown_path = output / "report.md"
    small_method_strata = sum(bool(row["small_sample"]) for row in method_rows)
    small_comparison_strata = sum(bool(row["small_sample"]) for row in comparison_rows)
    payload = {
        "schema_version": 2,
        "sources": [source.payload() for source in sources],
        "episode_count": len(episodes),
        "forecaster_count": len({key[0] for key in episodes}),
        "evaluated_candidate_ids": sorted(
            method
            for method, role in method_roles.items()
            if role in {"baseline", "missing_anchor"}
        ),
        "evaluated_selector_ids": sorted(
            method for method, role in method_roles.items() if role == "selector_baseline"
        ),
        "scopes": [scope for scope, _ in SUMMARY_SCOPES],
        "bootstrap": {
            "unit": "forecaster/dataset/episode",
            "method": "paired nonparametric percentile bootstrap of the mean delta",
            "confidence_level": 0.95,
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "minimum_pairs": 2,
        },
        "primary_analysis": {
            "metric": "mase",
            "weighting": "family_macro",
            "views": ["all_windows", "windows_with_missing"],
            "interval": "family/dataset/item-mask hierarchical percentile bootstrap",
            "multiple_comparison_adjustment": "Holm within view and stratum",
            "comparator_roles": list(normalized_primary_roles),
        },
        "definitions": {
            "loss_direction": "lower is better",
            "mean_delta": "B-FAIS metric minus comparator metric on paired episodes",
            "win_rate": (
                "strict fraction of paired episodes where B-FAIS has lower loss; "
                "ties are reported separately"
            ),
            "valid_rate": (
                "metric-eligible recorded rows divided by all episodes in the stratum; "
                "an absent method counts as unavailable"
            ),
            "average_rank_mase": (
                "episode midrank among B-FAIS, selector baselines, and natively valid "
                "single-imputer candidates; clean and oracle are excluded"
            ),
            "oracle": (
                "the valid single imputer with minimum MASE in each episode; its MAE "
                "and RMSE are not metric-specific oracle selections, and it is not an "
                "imputation-error oracle"
            ),
            "imputation_mae": "MAE on synthetically hidden context entries only",
            "imputation_rmse": "RMSE on synthetically hidden context entries only",
            "clean": (
                "complete-context reference with zero imputation error by construction; "
                "not executable for an input that contains missing values"
            ),
            "regret_mase": "method MASE minus the per-episode MASE oracle",
            "runtime_seconds": (
                "end-to-end imputation for B-FAIS and selector baselines, and one "
                "invocation for a candidate"
            ),
            "small_sample": f"fewer than {SMALL_SAMPLE_THRESHOLD} valid episodes or pairs",
            "overall_weighting": (
                "method_summary and comparison_summary are episode-weighted diagnostics; "
                "family_macro_comparison_summary is the primary family-equal analysis"
            ),
            "bootstrap_dependence_caveat": (
                "episode bootstrap intervals do not adjust for dependence among rows that "
                "share an item, forecast origin, family, or imputed context"
            ),
            "inference": (
                "the primary table reports two-sided family-level Wilcoxon p-values with "
                "Holm adjustment; interpretation must also use effect sizes and intervals"
            ),
        },
        "warnings": {
            "small_method_strata": small_method_strata,
            "small_method_strata_by_metric": {
                metric: sum(bool(row[f"{metric}_small_sample"]) for row in method_rows)
                for metric in PAIRED_METRICS
            },
            "small_comparison_strata": small_comparison_strata,
            "small_comparison_strata_by_metric": {
                metric: sum(bool(row[f"{metric}_small_sample"]) for row in comparison_rows)
                for metric in PAIRED_METRICS
            },
            "missing_manifests": sum(source.manifest_path is None for source in sources),
            "missing_study_signatures": sum(source.study_signature is None for source in sources),
        },
        "method_summary": method_rows,
        "comparison_summary": comparison_rows,
        "family_macro_comparison_summary": primary_rows,
    }
    _write_json(json_path, payload)
    _write_csv(method_csv, method_rows)
    _write_csv(comparison_csv, comparison_rows)
    _write_csv(primary_csv, primary_rows)
    _write_markdown(
        markdown_path,
        sources=sources,
        method_rows=method_rows,
        comparison_rows=comparison_rows,
        primary_rows=primary_rows,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        "status": "completed",
        "episode_count": len(episodes),
        "forecaster_count": payload["forecaster_count"],
        "method_group_count": len(method_rows),
        "comparison_group_count": len(comparison_rows),
        "family_macro_comparison_count": len(primary_rows),
        "main_summary_json": str(json_path),
        "method_summary_csv": str(method_csv),
        "comparison_summary_csv": str(comparison_csv),
        "family_macro_comparison_summary_csv": str(primary_csv),
        "report_markdown": str(markdown_path),
    }


__all__ = [
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_BOOTSTRAP_SEED",
    "FORECAST_METRICS",
    "IMPUTATION_METRICS",
    "PAIRED_METRICS",
    "SUMMARY_SCOPES",
    "summarize_multi_forecaster",
]
