"""Reanalyse the completed seven-selector evaluation after excluding ETT.

The script is deliberately CPU-only.  It reads completed episode-level JSONL
artifacts, verifies their manifests, and writes new summaries without touching
the source evaluations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FORECASTERS = ("chronos2", "timesfm2p5")
ANALYSIS_VERSION = "p0-0-legacy-non-ett-v001"
SELECTOR_METHODS = (
    "b_fais",
    "alors",
    "dselect1",
    "hybrid_lstm",
    "metaod",
    "neuralucb",
    "random_valid_series",
)
DIRECTORY_METHODS = {
    "b_fais": "bfais",
    "alors": "alors",
    "dselect1": "dselect1",
    "hybrid_lstm": "hybrid_lstm",
    "metaod": "metaod",
    "neuralucb": "neuralucb",
    "random_valid_series": "random_valid_block",
}
COMPARISON_ORDER = (
    "strongest_selector",
    "train_selected_robust_fixed",
    "test_posthoc_best_fixed_diagnostic",
    "episode_oracle_diagnostic",
)
KEY_COLUMNS = ["forecaster_id", "episode_id"]
META_COLUMNS = ["dataset_id", "family_id", "item_id"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_state(repository_root: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "repository_root": str(repository_root.resolve()),
        "repository_commit": git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
    }


def _default_evaluations(artifacts_root: Path) -> dict[tuple[str, str], Path]:
    return {
        (method, forecaster): artifacts_root
        / (
            f"main-selector-{DIRECTORY_METHODS[method]}-sequence-eval-"
            f"{forecaster}-v2"
        )
        for method in SELECTOR_METHODS
        for forecaster in FORECASTERS
    }


def _is_finite_number(value: object) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _read_evaluation_rows(
    directory: Path,
    expected_forecaster: str,
    expected_method: str,
    excluded_family: str,
    *,
    include_candidates: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = directory / "evaluation_manifest.json"
    metrics_path = directory / "episode_metrics.jsonl"
    if not manifest_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"incomplete evaluation directory: {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(f"evaluation is not completed: {directory}")
    if manifest.get("forecaster_id") != expected_forecaster:
        raise ValueError(f"forecaster mismatch in {manifest_path}")
    actual_hash = _sha256(metrics_path)
    recorded_hash = manifest.get("episode_metrics_jsonl_sha256")
    if recorded_hash and recorded_hash != actual_hash:
        raise ValueError(f"episode metrics hash mismatch: {metrics_path}")

    method_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    total_rows = excluded_rows = 0
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {metrics_path}:{line_number}") from error
            family_id = str(row.get("family_id", ""))
            if family_id.casefold() == excluded_family.casefold():
                excluded_rows += 1
                continue
            if row.get("forecaster_id") != expected_forecaster:
                raise ValueError(f"row forecaster mismatch at {metrics_path}:{line_number}")
            if row.get("method") == expected_method:
                method_rows.append(row)
            if include_candidates and row.get("method_role") in {"baseline", "missing_anchor"}:
                candidate_rows.append(row)

    source = {
        "directory": str(directory.resolve()),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
        "metrics": str(metrics_path.resolve()),
        "metrics_sha256": actual_hash,
        "total_rows": total_rows,
        "excluded_ett_rows": excluded_rows,
        "selected_method_rows": len(method_rows),
        "candidate_rows": len(candidate_rows),
    }
    return method_rows, candidate_rows, source


def _normalise_rows(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    required = {*KEY_COLUMNS, *META_COLUMNS, "method", "metric_eligible", "mase"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"evaluation rows are missing columns: {missing}")
    frame = frame.copy()
    frame["valid"] = frame.apply(
        lambda row: bool(row["metric_eligible"]) and _is_finite_number(row["mase"]), axis=1
    )
    frame["mase"] = pd.to_numeric(frame["mase"], errors="coerce")
    duplicate = frame.duplicated(KEY_COLUMNS + ["method"], keep=False)
    if duplicate.any():
        examples = frame.loc[duplicate, KEY_COLUMNS + ["method"]].head(5).to_dict("records")
        raise ValueError(f"duplicate episode/method rows: {examples}")
    return frame.sort_values(KEY_COLUMNS + ["method"]).reset_index(drop=True)


def _top_tail_mean(values: Iterable[float], quantile: float) -> float:
    array = np.asarray(tuple(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan")
    count = max(1, int(math.ceil((1.0 - quantile) * len(array))))
    return float(np.mean(np.sort(array)[-count:]))


def _quantile(values: Iterable[float], quantile: float) -> float:
    array = np.asarray(tuple(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, quantile)) if len(array) else float("nan")


def _family_group_columns(scope: str) -> list[str]:
    return ["forecaster_id", "family_id"] if scope == "combined" else ["family_id"]


def _method_summary(frame: pd.DataFrame, scope: str, forecaster: str | None) -> dict[str, Any]:
    selected = frame if forecaster is None else frame[frame["forecaster_id"] == forecaster]
    valid = selected[selected["valid"]]
    group_columns = _family_group_columns(scope)
    family_means = valid.groupby(group_columns, sort=True)["mase"].mean()
    return {
        "scope": scope,
        "forecaster_id": forecaster,
        "method": str(selected["method"].iloc[0]),
        "recorded_count": int(len(selected)),
        "valid_count": int(len(valid)),
        "valid_rate": float(len(valid) / len(selected)) if len(selected) else float("nan"),
        "family_stratum_count": int(len(family_means)),
        "family_macro_mase": float(family_means.mean()),
        "episode_mean_mase": float(valid["mase"].mean()),
    }


def _load_teacher_candidate_losses(
    path: Path,
    excluded_family: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    values: dict[tuple[str, str, str], list[float]] = {}
    metadata: dict[tuple[str, str], tuple[str, str]] = {}
    input_rows = excluded_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            input_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if str(row.get("family_id", "")).casefold() == excluded_family.casefold():
                excluded_rows += 1
                continue
            loss = row.get("full_candidate_loss")
            if not _is_finite_number(loss):
                continue
            forecaster = str(row["forecaster_id"])
            episode = str(row["episode_id"])
            candidate = str(row["candidate_id"])
            key = (forecaster, episode, candidate)
            values.setdefault(key, []).append(float(loss))
            metadata[(forecaster, episode)] = (str(row["family_id"]), str(row["dataset_id"]))

    inconsistent = 0
    records: list[dict[str, Any]] = []
    for (forecaster, episode, candidate), losses in sorted(values.items()):
        if max(losses) - min(losses) > 1e-10:
            inconsistent += 1
        family, dataset = metadata[(forecaster, episode)]
        records.append(
            {
                "forecaster_id": forecaster,
                "episode_id": episode,
                "family_id": family,
                "dataset_id": dataset,
                "candidate_id": candidate,
                "full_candidate_loss": float(np.median(losses)),
            }
        )
    frame = pd.DataFrame(records)
    sibling_manifest = path.parent / "labels_manifest.json"
    details = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "sibling_manifest": (
            str(sibling_manifest.resolve()) if sibling_manifest.is_file() else None
        ),
        "sibling_manifest_sha256": (
            _sha256(sibling_manifest) if sibling_manifest.is_file() else None
        ),
        "sibling_manifest_size_bytes": (
            sibling_manifest.stat().st_size if sibling_manifest.is_file() else None
        ),
        "input_rows": input_rows,
        "excluded_ett_rows": excluded_rows,
        "deduplicated_rows": len(frame),
        "inconsistent_duplicate_loss_groups": inconsistent,
    }
    return frame, details


def _select_robust_fixed(
    teacher: pd.DataFrame,
) -> tuple[dict[str, str], pd.DataFrame]:
    selections: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for forecaster in FORECASTERS:
        subset = teacher[teacher["forecaster_id"] == forecaster]
        episodes = subset["episode_id"].nunique()
        if not episodes:
            raise ValueError(f"teacher labels contain no rows for {forecaster}")
        for candidate, candidate_rows in subset.groupby("candidate_id", sort=True):
            support = candidate_rows["episode_id"].nunique()
            coverage = support / episodes
            family_mean = candidate_rows.groupby("family_id", sort=True)[
                "full_candidate_loss"
            ].mean()
            family_cvar = candidate_rows.groupby("family_id", sort=True)[
                "full_candidate_loss"
            ].apply(lambda values: _top_tail_mean(values, 0.90))
            records.append(
                {
                    "forecaster_id": forecaster,
                    "candidate_id": candidate,
                    "support": int(support),
                    "teacher_episode_count": int(episodes),
                    "coverage": float(coverage),
                    "family_macro_mean_loss": float(family_mean.mean()),
                    "family_macro_cvar90": float(family_cvar.mean()),
                    "eligible_for_selection": bool(support == episodes),
                }
            )
        eligible = [record for record in records if record["forecaster_id"] == forecaster and record["eligible_for_selection"]]
        if not eligible:
            raise ValueError(f"no fully covered fixed candidate for {forecaster}")
        winner = min(
            eligible,
            key=lambda record: (
                record["family_macro_cvar90"],
                record["family_macro_mean_loss"],
                record["candidate_id"],
            ),
        )
        selections[forecaster] = str(winner["candidate_id"])
    return selections, pd.DataFrame(records).sort_values(["forecaster_id", "candidate_id"])


def _select_test_best_fixed(candidate_frame: pd.DataFrame) -> dict[str, str]:
    selections: dict[str, str] = {}
    for forecaster in FORECASTERS:
        subset = candidate_frame[candidate_frame["forecaster_id"] == forecaster]
        expected = subset["episode_id"].nunique()
        scores: list[tuple[float, str]] = []
        for candidate, rows in subset.groupby("method", sort=True):
            valid = rows[rows["valid"]]
            if len(valid) != expected:
                continue
            family_macro = valid.groupby("family_id", sort=True)["mase"].mean().mean()
            scores.append((float(family_macro), str(candidate)))
        if not scores:
            raise ValueError(f"no fully valid evaluation candidate for {forecaster}")
        selections[forecaster] = min(scores)[1]
    return selections


def _lookup_rows(
    selector_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    forecaster: str,
    method: str,
) -> pd.DataFrame:
    source = selector_frame if method in SELECTOR_METHODS else candidate_frame
    return source[(source["forecaster_id"] == forecaster) & (source["method"] == method)].copy()


def _paired_statistics(
    b_fais: pd.DataFrame,
    comparator: pd.DataFrame,
    scope: str,
    forecaster: str | None,
    comparator_type: str,
    comparator_label: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    left = b_fais if forecaster is None else b_fais[b_fais["forecaster_id"] == forecaster]
    right = comparator if forecaster is None else comparator[comparator["forecaster_id"] == forecaster]
    left = left[left["valid"]]
    right = right[right["valid"]]
    joined = left.merge(
        right,
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("_b_fais", "_comparator"),
        validate="one_to_one",
    )
    if not len(joined):
        raise ValueError(f"no valid pairs for {scope}/{comparator_type}")
    if not (joined["family_id_b_fais"] == joined["family_id_comparator"]).all():
        raise ValueError(f"family mismatch for {scope}/{comparator_type}")
    joined["family_id"] = joined["family_id_b_fais"]
    joined["delta"] = joined["mase_b_fais"] - joined["mase_comparator"]
    group_columns = _family_group_columns(scope)
    family = (
        joined.groupby(group_columns, sort=True)
        .agg(
            pair_count=("delta", "size"),
            b_fais_mean_mase=("mase_b_fais", "mean"),
            comparator_mean_mase=("mase_comparator", "mean"),
            mean_delta=("delta", "mean"),
            p90_delta=("delta", lambda values: _quantile(values, 0.90)),
            p95_delta=("delta", lambda values: _quantile(values, 0.95)),
            cvar90_delta=("delta", lambda values: _top_tail_mean(values, 0.90)),
            cvar95_delta=("delta", lambda values: _top_tail_mean(values, 0.95)),
            max_degradation=("delta", "max"),
            worse_fraction=("delta", lambda values: float(np.mean(np.asarray(values) > 0.0))),
        )
        .reset_index()
    )
    family.insert(0, "comparator_label", comparator_label)
    family.insert(0, "comparator_type", comparator_type)
    family.insert(0, "scope", scope)

    values = joined["delta"].to_numpy(dtype=float)
    summary = {
        "scope": scope,
        "forecaster_id": forecaster,
        "comparator_type": comparator_type,
        "comparator_label": comparator_label,
        "pair_count": int(len(joined)),
        "family_stratum_count": int(len(family)),
        "b_fais_family_macro_mase": float(family["b_fais_mean_mase"].mean()),
        "comparator_family_macro_mase": float(family["comparator_mean_mase"].mean()),
        "family_macro_mean_delta": float(family["mean_delta"].mean()),
        "episode_mean_delta": float(values.mean()),
        "p90_delta": _quantile(values, 0.90),
        "p95_delta": _quantile(values, 0.95),
        "cvar90_delta": _top_tail_mean(values, 0.90),
        "cvar95_delta": _top_tail_mean(values, 0.95),
        "max_degradation": float(values.max()),
        "worse_fraction": float(np.mean(values > 0.0)),
        "b_fais_win_fraction": float(np.mean(values < 0.0)),
        "tie_fraction": float(np.mean(np.isclose(values, 0.0, atol=1e-12, rtol=0.0))),
    }
    return summary, family


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _format_float(value: object) -> str:
    return "NA" if not _is_finite_number(value) else f"{float(value):.4f}"


def _write_report(
    path: Path,
    comparisons: pd.DataFrame,
    method_summary: pd.DataFrame,
    strongest_selector: str,
    robust_fixed: Mapping[str, str],
    test_best: Mapping[str, str],
    excluded_family: str,
) -> None:
    lines = [
        "# Legacy non-ETT reanalysis",
        "",
        f"This descriptive reanalysis excludes `family_id={excluded_family}` from all evaluation rows.",
        "Every data family receives equal weight in the primary mean. Positive deltas mean that B-FAIS is worse.",
        "P90/P95 use NumPy linear quantiles. CVaR90/CVaR95 are the means of the largest ceil(10%)/ceil(5%) paired deltas.",
        "",
        "## Comparator selection",
        "",
        f"The strongest complete selector on the combined non-ETT legacy table is `{strongest_selector}`.",
        "The training-selected fixed candidates minimize family-equal CVaR90 among candidates present in every available teacher-label episode: "
        + ", ".join(f"`{model}={candidate}`" for model, candidate in robust_fixed.items())
        + ".",
        "The test-selected fixed candidates are diagnostic only: "
        + ", ".join(f"`{model}={candidate}`" for model, candidate in test_best.items())
        + ".",
        "The stored episode oracle is also diagnostic because it selects the minimum-MASE candidate after observing each test outcome.",
        "",
        "## Paired MASE differences",
        "",
        "| scope | comparator | pairs | family-macro delta | P90 | P95 | CVaR90 | CVaR95 | max | worse fraction |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    ordered = comparisons.copy()
    ordered["comparison_order"] = ordered["comparator_type"].map(
        {name: index for index, name in enumerate(COMPARISON_ORDER)}
    )
    ordered["scope_order"] = ordered["scope"].map(
        {"chronos2": 0, "timesfm2p5": 1, "combined": 2}
    )
    for _, row in ordered.sort_values(["scope_order", "comparison_order"]).iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scope"]),
                    str(row["comparator_label"]),
                    str(int(row["pair_count"])),
                    _format_float(row["family_macro_mean_delta"]),
                    _format_float(row["p90_delta"]),
                    _format_float(row["p95_delta"]),
                    _format_float(row["cvar90_delta"]),
                    _format_float(row["cvar95_delta"]),
                    _format_float(row["max_degradation"]),
                    _format_float(row["worse_fraction"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Selector family-macro MASE",
            "",
            "| scope | method | valid rate | family-macro MASE |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for _, row in method_summary.sort_values(["scope", "family_macro_mase", "method"]).iterrows():
        lines.append(
            f"| {row['scope']} | {row['method']} | {_format_float(row['valid_rate'])} | "
            f"{_format_float(row['family_macro_mase'])} |"
        )
    lines.extend(
        [
            "",
            "## Required caveats",
            "",
            "The seven selectors use their archived training protocols; this pass changes only the evaluation filter and aggregation.",
            "The strongest-selector label is selected descriptively from this legacy test table and has no selection-adjusted interval.",
            "The robust fixed comparator uses training-origin teacher labels, whose six-candidate shortlist gives unequal support to many candidates. Only candidates with complete teacher-label coverage are eligible.",
            "Tail summaries are paired episode descriptions. Episodes sharing a family, item, mask realization, or imputed context are dependent.",
            "No additional TSFM inference was run.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_reanalysis(
    *,
    artifacts_root: Path,
    teacher_labels: Path,
    output_dir: Path,
    excluded_family: str = "ett",
    evaluation_dirs: Mapping[tuple[str, str], Path] | None = None,
) -> dict[str, Any]:
    evaluations = dict(evaluation_dirs or _default_evaluations(artifacts_root))
    expected_keys = {(method, model) for method in SELECTOR_METHODS for model in FORECASTERS}
    if set(evaluations) != expected_keys:
        raise ValueError("evaluation mapping must contain all seven methods for both forecasters")

    selector_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for method in SELECTOR_METHODS:
        for forecaster in FORECASTERS:
            rows, candidates, source = _read_evaluation_rows(
                evaluations[(method, forecaster)],
                forecaster,
                method,
                excluded_family,
                include_candidates=method == "b_fais",
            )
            selector_rows.extend(rows)
            candidate_rows.extend(candidates)
            source.update({"method": method, "forecaster_id": forecaster})
            sources.append(source)

    selector_frame = _normalise_rows(selector_rows)
    candidate_frame = _normalise_rows(candidate_rows)
    reference_keys = set(
        map(tuple, selector_frame[selector_frame["method"] == "b_fais"][KEY_COLUMNS].to_numpy())
    )
    for method in SELECTOR_METHODS:
        keys = set(map(tuple, selector_frame[selector_frame["method"] == method][KEY_COLUMNS].to_numpy()))
        if keys != reference_keys:
            raise ValueError(f"selector episode universe mismatch for {method}")

    method_records: list[dict[str, Any]] = []
    for method in SELECTOR_METHODS:
        frame = selector_frame[selector_frame["method"] == method]
        for forecaster in FORECASTERS:
            method_records.append(_method_summary(frame, forecaster, forecaster))
        method_records.append(_method_summary(frame, "combined", None))
    method_summary = pd.DataFrame(method_records)
    complete_selectors = method_summary[
        (method_summary["scope"] == "combined")
        & (method_summary["method"] != "b_fais")
        & np.isclose(method_summary["valid_rate"], 1.0)
    ]
    if complete_selectors.empty:
        raise ValueError("no selector baseline has complete non-ETT evaluation coverage")
    strongest_selector = str(
        complete_selectors.sort_values(["family_macro_mase", "method"]).iloc[0]["method"]
    )

    teacher_frame, teacher_source = _load_teacher_candidate_losses(
        teacher_labels, excluded_family
    )
    if teacher_source["inconsistent_duplicate_loss_groups"]:
        raise ValueError("full_candidate_loss differs across blocks for an episode/candidate")
    robust_fixed, training_selection = _select_robust_fixed(teacher_frame)
    test_best = _select_test_best_fixed(candidate_frame)

    b_fais = selector_frame[selector_frame["method"] == "b_fais"]
    comparator_specs = [
        ("strongest_selector", {model: strongest_selector for model in FORECASTERS}),
        ("train_selected_robust_fixed", robust_fixed),
        ("test_posthoc_best_fixed_diagnostic", test_best),
        ("episode_oracle_diagnostic", {model: "oracle" for model in FORECASTERS}),
    ]
    oracle_rows: list[dict[str, Any]] = []
    for forecaster in FORECASTERS:
        directory = evaluations[("b_fais", forecaster)]
        metrics = directory / "episode_metrics.jsonl"
        with metrics.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if (
                    str(row.get("family_id", "")).casefold() != excluded_family.casefold()
                    and row.get("method") == "oracle"
                ):
                    oracle_rows.append(row)
    oracle_frame = _normalise_rows(oracle_rows)
    candidate_frame = pd.concat([candidate_frame, oracle_frame], ignore_index=True)

    comparison_records: list[dict[str, Any]] = []
    family_records: list[pd.DataFrame] = []
    for comparator_type, methods in comparator_specs:
        parts = [
            _lookup_rows(selector_frame, candidate_frame, model, methods[model])
            for model in FORECASTERS
        ]
        comparator = pd.concat(parts, ignore_index=True)
        label = (
            strongest_selector
            if comparator_type == "strongest_selector"
            else ";".join(f"{model}={methods[model]}" for model in FORECASTERS)
        )
        for forecaster in FORECASTERS:
            summary, family = _paired_statistics(
                b_fais,
                comparator,
                forecaster,
                forecaster,
                comparator_type,
                methods[forecaster],
            )
            comparison_records.append(summary)
            family_records.append(family)
        summary, family = _paired_statistics(
            b_fais,
            comparator,
            "combined",
            None,
            comparator_type,
            label,
        )
        comparison_records.append(summary)
        family_records.append(family)

    comparisons = pd.DataFrame(comparison_records)
    family_comparisons = pd.concat(family_records, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    method_summary.to_csv(output_dir / "selector_family_macro.csv", index=False, float_format="%.12g")
    comparisons.to_csv(output_dir / "paired_tail_summary.csv", index=False, float_format="%.12g")
    family_comparisons.to_csv(
        output_dir / "family_paired_summary.csv", index=False, float_format="%.12g"
    )
    training_selection.to_csv(
        output_dir / "training_fixed_candidate_selection.csv",
        index=False,
        float_format="%.12g",
    )
    selection = {
        "strongest_selector": strongest_selector,
        "strongest_selector_protocol": (
            "minimum combined non-ETT family-macro MASE among selector baselines with "
            "complete metric coverage"
        ),
        "train_selected_robust_fixed": robust_fixed,
        "train_selected_robust_fixed_protocol": (
            "minimum family-equal training CVaR90, then mean loss, among candidates with "
            "complete teacher-label episode coverage"
        ),
        "test_posthoc_best_fixed_diagnostic": test_best,
        "test_posthoc_best_fixed_protocol": (
            "minimum per-model non-ETT family-macro MASE among fixed candidates with "
            "complete evaluation coverage; diagnostic only"
        ),
        "episode_oracle_protocol": "stored minimum-MASE candidate per test episode; diagnostic only",
    }
    _write_json(output_dir / "comparator_selection.json", selection)
    _write_report(
        output_dir / "report.md",
        comparisons,
        method_summary,
        strongest_selector,
        robust_fixed,
        test_best,
        excluded_family,
    )
    script_path = Path(__file__).resolve()
    repository_state = _repository_state(script_path.parents[1])
    output_names = [
        "comparator_selection.json",
        "family_paired_summary.csv",
        "paired_tail_summary.csv",
        "report.md",
        "selector_family_macro.csv",
        "training_fixed_candidate_selection.csv",
    ]
    output_files = [
        {
            "path": name,
            "size_bytes": (output_dir / name).stat().st_size,
            "sha256": _sha256(output_dir / name),
        }
        for name in output_names
    ]
    manifest = {
        "schema_version": 1,
        "analysis_version": ANALYSIS_VERSION,
        "artifact_type": "legacy_non_ett_reanalysis",
        "verification_status": "legacy_descriptive_reanalysis",
        "excluded_family": excluded_family,
        "forecasters": list(FORECASTERS),
        "selector_methods": list(SELECTOR_METHODS),
        "episode_count_after_filter": len(reference_keys),
        "family_ids_after_filter": sorted(selector_frame["family_id"].unique().tolist()),
        "source_evaluations": sorted(
            sources, key=lambda row: (row["method"], row["forecaster_id"])
        ),
        "teacher_labels": teacher_source,
        "script": str(script_path),
        "script_sha256": _sha256(script_path),
        **repository_state,
        "outputs": output_names,
        "output_files": output_files,
        "limitations": [
            "The strongest selector is selected descriptively on the legacy non-ETT test table.",
            "Teacher labels shortlist candidates unequally; robust fixed selection requires complete label coverage.",
            "Tail metrics do not treat correlated episodes as independent inferential samples.",
            "The test-selected fixed candidate and episode oracle are diagnostic only.",
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return {
        "output_dir": str(output_dir.resolve()),
        "episode_count": len(reference_keys),
        "family_count": selector_frame["family_id"].nunique(),
        "strongest_selector": strongest_selector,
        "robust_fixed": robust_fixed,
        "test_best_fixed": test_best,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--teacher-labels",
        type=Path,
        default=Path(
            "artifacts/main-seq96-opt23-labels-merged-rolling-b128-v10/teacher_labels.jsonl"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--excluded-family", default="ett")
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_reanalysis(
        artifacts_root=args.artifacts_root,
        teacher_labels=args.teacher_labels,
        output_dir=args.output_dir,
        excluded_family=args.excluded_family,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
