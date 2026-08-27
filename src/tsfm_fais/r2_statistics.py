"""Pre-registered statistical utilities for the B-FAIS R2 study."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .main_results import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    EpisodeKey,
    _hierarchical_family_interval,
    _hierarchical_family_means,
)

GLOBAL_HOLM_TEST_IDS: tuple[str, ...] = (
    "H1_forecast_vs_recon",
    "H2_block_vs_sequence",
    "H3_structured_vs_independent",
    "H4_core_vs_external",
    "H5_lofo_core_vs_sequence",
    "H6_sundial_transfer",
)

GLOBAL_HOLM_COMPONENT_IDS: Mapping[str, tuple[str, str]] = {
    test_id: ("chronos2", "timesfm2p5") for test_id in GLOBAL_HOLM_TEST_IDS[:5]
} | {
    "H6_sundial_transfer": ("fixed_imputer", "seq_forecast"),
}


def paired_family_component(
    *,
    test_id: str,
    component_id: str,
    tested_method: str,
    reference_method: str,
    tested_values: Mapping[EpisodeKey, float],
    reference_values: Mapping[EpisodeKey, float],
    metadata: Mapping[EpisodeKey, Mapping[str, Any]],
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute one strict paired component of the fixed six-test family."""

    if test_id not in GLOBAL_HOLM_TEST_IDS:
        raise ValueError(f"unknown global Holm test ID: {test_id}")
    if component_id not in GLOBAL_HOLM_COMPONENT_IDS[test_id]:
        raise ValueError(f"unexpected component {component_id!r} for {test_id}")
    tested_keys = set(tested_values)
    reference_keys = set(reference_values)
    if tested_keys != reference_keys:
        missing_tested = sorted(map(str, reference_keys - tested_keys))
        missing_reference = sorted(map(str, tested_keys - reference_keys))
        raise ValueError(
            "paired component keys differ; missing tested="
            f"{missing_tested}; missing reference={missing_reference}"
        )
    if not tested_keys:
        raise ValueError("paired component must contain at least one pair")
    missing_metadata = sorted(map(str, tested_keys - set(metadata)))
    if missing_metadata:
        raise ValueError(f"paired component metadata is incomplete: {missing_metadata}")

    deltas: dict[EpisodeKey, float] = {}
    for key in tested_values:
        tested = float(tested_values[key])
        reference = float(reference_values[key])
        if not math.isfinite(tested) or not math.isfinite(reference):
            raise ValueError(f"paired component contains a non-finite value at {key!r}")
        deltas[key] = tested - reference

    family_effects = _hierarchical_family_means(deltas, metadata)
    ordered_effects = np.asarray(
        [family_effects[family] for family in sorted(family_effects)],
        dtype=float,
    )
    ci_low, ci_high = _hierarchical_family_interval(
        deltas,
        metadata,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    p_value: float | None = None
    if ordered_effects.size >= 2:
        if np.allclose(ordered_effects, 0.0):
            p_value = 1.0
        else:
            from scipy.stats import wilcoxon

            p_value = float(
                wilcoxon(
                    ordered_effects,
                    alternative="two-sided",
                    zero_method="pratt",
                ).pvalue
            )
    return {
        "test_id": test_id,
        "component_id": component_id,
        "tested_method": tested_method,
        "reference_method": reference_method,
        "pair_count": len(deltas),
        "family_count": len(family_effects),
        "effect_family_macro": float(np.mean(ordered_effects)),
        "effect_ci95_low": ci_low,
        "effect_ci95_high": ci_high,
        "wilcoxon_p_value": p_value,
        "bootstrap_replicates": bootstrap_replicates if len(family_effects) >= 2 else 0,
        "bootstrap_seed": bootstrap_seed,
        "available": p_value is not None and ci_low is not None and ci_high is not None,
        "interval_supports_superiority": ci_high is not None and ci_high < 0.0,
    }


def build_global_holm_manifest(
    components: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply one Holm step-down correction to the six fixed joint claims."""

    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_component in components:
        component = dict(raw_component)
        test_id = str(component.get("test_id", ""))
        component_id = str(component.get("component_id", ""))
        if test_id not in GLOBAL_HOLM_TEST_IDS:
            raise ValueError(f"unknown global Holm test ID: {test_id}")
        if component_id not in GLOBAL_HOLM_COMPONENT_IDS[test_id]:
            raise ValueError(f"unexpected component {component_id!r} for {test_id}")
        identity = (test_id, component_id)
        if identity in indexed:
            raise ValueError(f"duplicate global Holm component: {test_id}/{component_id}")
        p_value = component.get("wilcoxon_p_value")
        if p_value is not None:
            numeric_p = float(p_value)
            if not math.isfinite(numeric_p) or not 0.0 <= numeric_p <= 1.0:
                raise ValueError(f"invalid component p-value for {test_id}/{component_id}")
            component["wilcoxon_p_value"] = numeric_p
        indexed[identity] = component

    test_rows: list[dict[str, Any]] = []
    for test_id in GLOBAL_HOLM_TEST_IDS:
        required = GLOBAL_HOLM_COMPONENT_IDS[test_id]
        present = [
            indexed[(test_id, component_id)]
            for component_id in required
            if (test_id, component_id) in indexed
        ]
        missing = [
            component_id for component_id in required if (test_id, component_id) not in indexed
        ]
        unavailable = [
            str(component["component_id"])
            for component in present
            if component.get("wilcoxon_p_value") is None
            or not bool(component.get("available", False))
        ]
        complete = not missing and not unavailable
        joint_p = (
            max(float(component["wilcoxon_p_value"]) for component in present) if complete else 1.0
        )
        interval_support = complete and all(
            bool(component.get("interval_supports_superiority", False)) for component in present
        )
        test_rows.append(
            {
                "test_id": test_id,
                "required_component_ids": list(required),
                "present_component_count": len(present),
                "missing_component_ids": missing,
                "unavailable_component_ids": unavailable,
                "component_complete": complete,
                "joint_raw_p_value": joint_p,
                "holm_rank": None,
                "holm_adjusted_p_value": None,
                "interval_supports_superiority": interval_support,
                "superiority_eligible": False,
            }
        )

    order_index = {test_id: index for index, test_id in enumerate(GLOBAL_HOLM_TEST_IDS)}
    ordered = sorted(
        range(len(test_rows)),
        key=lambda index: (
            float(test_rows[index]["joint_raw_p_value"]),
            order_index[str(test_rows[index]["test_id"])],
        ),
    )
    previous = 0.0
    family_size = len(GLOBAL_HOLM_TEST_IDS)
    for rank, index in enumerate(ordered, start=1):
        row = test_rows[index]
        adjusted = min(
            1.0,
            max(previous, (family_size - rank + 1) * float(row["joint_raw_p_value"])),
        )
        row["holm_rank"] = rank
        row["holm_adjusted_p_value"] = adjusted
        row["superiority_eligible"] = bool(
            row["component_complete"] and row["interval_supports_superiority"] and adjusted < 0.05
        )
        previous = adjusted

    normalized_components = [
        indexed[identity]
        for identity in sorted(
            indexed,
            key=lambda identity: (
                order_index[identity[0]],
                GLOBAL_HOLM_COMPONENT_IDS[identity[0]].index(identity[1]),
            ),
        )
    ]
    return {
        "schema_version": 1,
        "family_id": "bfais_r2_global_holm_six_v1",
        "family_size": family_size,
        "test_order": list(GLOBAL_HOLM_TEST_IDS),
        "joint_p_value_rule": "maximum required component p-value; 1 if any component unavailable",
        "adjustment": "Holm step-down across the six fixed joint claims",
        "alpha": 0.05,
        "components": normalized_components,
        "tests": test_rows,
    }


def write_global_holm_manifest(
    *,
    components: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write the complete component and six-test statistical manifest."""

    payload = build_global_holm_manifest(components)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "global_holm_manifest.json"
    csv_path = output / "global_holm_tests.csv"
    temporary_json = json_path.with_suffix(".json.tmp")
    temporary_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_json.replace(json_path)
    fieldnames = tuple(payload["tests"][0])
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload["tests"])
    temporary_csv.replace(csv_path)
    return {
        "global_holm_manifest_json": str(json_path),
        "global_holm_tests_csv": str(csv_path),
    }


__all__ = [
    "GLOBAL_HOLM_COMPONENT_IDS",
    "GLOBAL_HOLM_TEST_IDS",
    "build_global_holm_manifest",
    "paired_family_component",
    "write_global_holm_manifest",
]
