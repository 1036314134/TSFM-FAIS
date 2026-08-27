from __future__ import annotations

import json
from pathlib import Path

import pytest

from tsfm_fais.r2_statistics import (
    GLOBAL_HOLM_COMPONENT_IDS,
    GLOBAL_HOLM_TEST_IDS,
    build_global_holm_manifest,
    paired_family_component,
    write_global_holm_manifest,
)


def _components(joint_p_values: list[float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for test_id, joint_p in zip(GLOBAL_HOLM_TEST_IDS, joint_p_values, strict=True):
        first, second = GLOBAL_HOLM_COMPONENT_IDS[test_id]
        rows.extend(
            (
                {
                    "test_id": test_id,
                    "component_id": first,
                    "wilcoxon_p_value": joint_p / 2.0,
                    "available": True,
                    "interval_supports_superiority": True,
                },
                {
                    "test_id": test_id,
                    "component_id": second,
                    "wilcoxon_p_value": joint_p,
                    "available": True,
                    "interval_supports_superiority": True,
                },
            )
        )
    return rows


def test_global_holm_uses_one_fixed_six_test_family() -> None:
    payload = build_global_holm_manifest(_components([0.01, 0.04, 0.03, 0.20, 0.50, 1.0]))
    by_id = {row["test_id"]: row for row in payload["tests"]}

    assert payload["family_size"] == 6
    assert payload["test_order"] == list(GLOBAL_HOLM_TEST_IDS)
    assert by_id["H1_forecast_vs_recon"]["holm_adjusted_p_value"] == pytest.approx(0.06)
    assert by_id["H3_structured_vs_independent"]["holm_adjusted_p_value"] == pytest.approx(0.15)
    assert by_id["H2_block_vs_sequence"]["holm_adjusted_p_value"] == pytest.approx(0.16)
    assert by_id["H4_core_vs_external"]["holm_adjusted_p_value"] == pytest.approx(0.60)
    assert by_id["H5_lofo_core_vs_sequence"]["holm_adjusted_p_value"] == pytest.approx(1.0)
    assert by_id["H6_sundial_transfer"]["holm_adjusted_p_value"] == pytest.approx(1.0)


def test_missing_component_forces_joint_p_to_one_without_shrinking_family() -> None:
    components = _components([0.01] * 6)
    components = [
        component
        for component in components
        if not (
            component["test_id"] == "H2_block_vs_sequence"
            and component["component_id"] == "timesfm2p5"
        )
    ]

    payload = build_global_holm_manifest(components)
    row = next(row for row in payload["tests"] if row["test_id"] == "H2_block_vs_sequence")

    assert payload["family_size"] == 6
    assert row["component_complete"] is False
    assert row["missing_component_ids"] == ["timesfm2p5"]
    assert row["joint_raw_p_value"] == 1.0
    assert row["superiority_eligible"] is False


def test_paired_family_component_rejects_missing_pairs() -> None:
    with pytest.raises(ValueError, match="paired component keys differ"):
        paired_family_component(
            test_id="H1_forecast_vs_recon",
            component_id="chronos2",
            tested_method="seq_forecast",
            reference_method="seq_recon",
            tested_values={("forecast", "dataset", "episode-a"): 1.0},
            reference_values={("forecast", "dataset", "episode-b"): 2.0},
            metadata={},
            bootstrap_replicates=10,
        )


def test_paired_family_component_reports_five_level_interval() -> None:
    tested: dict[tuple[str, str, str], float] = {}
    reference: dict[tuple[str, str, str], float] = {}
    metadata: dict[tuple[str, str, str], dict[str, object]] = {}
    for family in ("family-a", "family-b"):
        for origin in (10, 20):
            key = ("chronos2", f"dataset-{family}", f"{family}-{origin}")
            tested[key] = 1.0
            reference[key] = 2.0
            metadata[key] = {
                "family_id": family,
                "dataset_id": f"dataset-{family}",
                "item_id": f"item-{family}",
                "forecast_origin": origin,
                "mask_realization_id": f"mask-{origin}",
            }

    component = paired_family_component(
        test_id="H1_forecast_vs_recon",
        component_id="chronos2",
        tested_method="seq_forecast",
        reference_method="seq_recon",
        tested_values=tested,
        reference_values=reference,
        metadata=metadata,
        bootstrap_replicates=50,
        bootstrap_seed=11,
    )

    assert component["pair_count"] == 4
    assert component["family_count"] == 2
    assert component["effect_family_macro"] == pytest.approx(-1.0)
    assert component["effect_ci95_low"] == pytest.approx(-1.0)
    assert component["effect_ci95_high"] == pytest.approx(-1.0)
    assert component["interval_supports_superiority"] is True


def test_global_holm_manifest_writer_preserves_components(tmp_path: Path) -> None:
    components = _components([0.01] * 6)
    outputs = write_global_holm_manifest(components=components, output_dir=tmp_path)

    payload = json.loads(Path(outputs["global_holm_manifest_json"]).read_text(encoding="utf-8"))
    assert len(payload["components"]) == 12
    assert len(payload["tests"]) == 6
    assert Path(outputs["global_holm_tests_csv"]).is_file()
