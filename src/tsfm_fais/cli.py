"""Command-line interface for configuration, auditing, and local verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from tsfm_fais.config import load_config
from tsfm_fais.contracts import BudgetSpec, ForecastSpec, TimeSeriesItem
from tsfm_fais.data import audit_dataset, load_dataset, load_manifest
from tsfm_fais.forecasting import (
    ForecastAdapterSpec,
    ForecastRegistry,
    ForecastRunner,
    NativeForecast,
    default_forecast_registry,
)
from tsfm_fais.imputers import DEFAULT_REGISTRY
from tsfm_fais.pipeline import BlockwiseFAIS
from tsfm_fais.registry_configs import validate_project_configuration
from tsfm_fais.stages import StageInputs, finish_preparation, prepare_stage


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value).__name__)


def _config_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    validate_project_configuration(config)
    print(json.dumps(config.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0


def _data_audit(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    reports = []
    rejected = 0
    for spec in manifest.datasets:
        if not spec.enabled:
            continue
        try:
            report_payload = audit_dataset(spec, load_dataset(spec)).to_dict()
        except Exception as error:
            report_payload = {
                "dataset_id": spec.dataset_id,
                "accepted": False,
                "issues": [{"code": "load_error", "message": f"{type(error).__name__}: {error}"}],
            }
        rejected += int(not report_payload["accepted"])
        reports.append(report_payload)
        print(
            f"{spec.dataset_id}: "
            f"{'PASS' if report_payload['accepted'] else 'REJECT'}"
        )
    payload = {"schema_version": 1, "datasets": reports}
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
    return 1 if rejected else 0


def _imputers_list(args: argparse.Namespace) -> int:
    for spec in DEFAULT_REGISTRY.specs():
        availability = DEFAULT_REGISTRY.availability(spec.imputer_id)
        suffix = (
            "available"
            if availability.available
            else f"missing={','.join(availability.missing)}"
        )
        print(
            f"{spec.imputer_id:20} mode={spec.mode:20} cost={spec.cost_tier} {suffix}"
        )
    return 0


def _forecasters_list(args: argparse.Namespace) -> int:
    for spec in default_forecast_registry().specs():
        print(
            f"{spec.model_id:16} mode={spec.mode:24} "
            f"max_context={spec.max_context} extra={spec.optional_extra}"
        )
    return 0


class _MockUnivariate:
    def predict_native(self, contexts, horizon, quantile_levels, num_samples):
        values = np.asarray(contexts)
        point = np.repeat(values[:, -1:], horizon, axis=1)
        quantiles = np.repeat(point[..., None], len(quantile_levels), axis=-1)
        return NativeForecast(point=point, quantiles=quantiles)


class _MockJoint:
    def predict_native(self, contexts, horizon, quantile_levels, num_samples):
        values = np.asarray(contexts)
        point = np.repeat(values[:, -1:, :], horizon, axis=1)
        quantiles = np.repeat(point[..., None], len(quantile_levels), axis=-1)
        return NativeForecast(point=point, quantiles=quantiles)


def _smoke(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    time = np.arange(48, dtype=float)
    values = np.stack(
        [
            np.sin(2 * np.pi * time / 12),
            np.cos(2 * np.pi * time / 12),
            0.05 * time + np.sin(2 * np.pi * time / 6),
        ],
        axis=1,
    )
    item = TimeSeriesItem(
        item_id="smoke",
        values=values,
        variate_names=("a", "b", "c"),
        start=pd.Timestamp("2026-01-01"),
        freq="H",
        timestamps=pd.date_range("2026-01-01", periods=48, freq="h"),
        metadata={"period": 12},
    )
    mask = np.ones_like(values, dtype=bool)
    mask[12:18, 0] = False
    mask[-5:, 1] = False
    spec = ForecastSpec(
        model_id="mock_univariate",
        mode="independent_univariate",
        horizon=config.experiment.horizon,
        target_indices=(0, 1, 2),
    )
    pipeline = BlockwiseFAIS(config=config)
    first = pipeline.impute(item, mask, spec, BudgetSpec(max_candidates=3), seed=config.seed)
    second = pipeline.impute(item, mask, spec, BudgetSpec(max_candidates=3), seed=config.seed)
    if not np.array_equal(first.values[mask], values[mask]):
        raise AssertionError("smoke imputation changed observed values")
    if not np.isfinite(first.values).all():
        raise AssertionError("smoke imputation produced non-finite values")
    if first.routing.assignments != second.routing.assignments or not np.allclose(
        first.values, second.values
    ):
        raise AssertionError("smoke imputation is not deterministic")
    registry = ForecastRegistry()
    registry.register(
        ForecastAdapterSpec(
            model_id="mock_univariate",
            mode="independent_univariate",
            factory="builtins:object",
            model_name="mock",
            optional_extra="none",
            max_context=48,
            output_type="quantile",
        )
    )
    registry.register(
        ForecastAdapterSpec(
            model_id="mock_joint",
            mode="joint_multivariate",
            factory="builtins:object",
            model_name="mock",
            optional_extra="none",
            max_context=48,
            output_type="quantile",
        )
    )
    runner = ForecastRunner(
        registry,
        adapters={"mock_univariate": _MockUnivariate(), "mock_joint": _MockJoint()},
    )
    uni = runner.predict(first.values[None, ...], spec)
    joint = runner.predict(
        first.values[None, ...],
        ForecastSpec(
            model_id="mock_joint",
            mode="joint_multivariate",
            horizon=config.experiment.horizon,
            target_indices=(0, 1, 2),
        ),
    )
    expected = (1, config.experiment.horizon, 3)
    if uni.point.shape != expected or joint.point.shape != expected:
        raise AssertionError("forecast smoke shape mismatch")
    print("SMOKE PASS")
    return 0


def _run_stage(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    validate_project_configuration(config)
    preparation_inputs = StageInputs(
        audit_artifact=_optional_path(args.audit_artifact),
        imputer_artifacts=_optional_path(args.imputer_artifacts),
        labels_artifact=_optional_path(args.labels_artifact),
        router_artifact=_optional_path(args.router_artifact),
        forecaster_artifact=_optional_path(args.forecaster_artifact),
        forecaster_id=args.forecaster_id,
    )
    preparation = prepare_stage(
        config,
        args.config,
        args.stage,
        preparation_inputs,
        run_id=args.run_id,
    )
    if args.execute:
        from tsfm_fais.stage_execution import execute_prepared_stage

        outputs = execute_prepared_stage(preparation, config, preparation_inputs)
        print(json.dumps(outputs, indent=2, ensure_ascii=False))
        return 0
    manifest = finish_preparation(preparation)
    print(f"STAGE PREPARED: {manifest}")
    return 0


def _optional_path(value: str | None) -> Path | None:
    return None if value is None else Path(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fais")
    commands = parser.add_subparsers(dest="command", required=True)

    config = commands.add_parser("config")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    validate = config_commands.add_parser("validate")
    validate.add_argument("--config", required=True)
    validate.set_defaults(handler=_config_validate)

    data = commands.add_parser("data")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    audit = data_commands.add_parser("audit")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--output")
    audit.set_defaults(handler=_data_audit)

    imputers = commands.add_parser("imputers")
    imputers_commands = imputers.add_subparsers(dest="imputers_command", required=True)
    imputer_list = imputers_commands.add_parser("list")
    imputer_list.set_defaults(handler=_imputers_list)

    forecasters = commands.add_parser("forecasters")
    forecaster_commands = forecasters.add_subparsers(dest="forecasters_command", required=True)
    forecaster_list = forecaster_commands.add_parser("list")
    forecaster_list.set_defaults(handler=_forecasters_list)

    run = commands.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument(
        "--stage",
        required=True,
        choices=("fit-imputers", "labels", "train-router", "impute"),
    )
    run.add_argument("--run-id")
    run.add_argument("--audit-artifact")
    run.add_argument("--imputer-artifacts")
    run.add_argument("--labels-artifact")
    run.add_argument("--router-artifact")
    run.add_argument(
        "--forecaster-artifact",
        help=(
            "local checkpoint path; for multi-model labels, use a directory with "
            "model-ID children or a JSON ID-to-path mapping"
        ),
    )
    run.add_argument(
        "--forecaster-id",
        help="registered model ID, or comma-separated IDs for the labels stage",
    )
    run.add_argument(
        "--execute",
        action="store_true",
        help="explicitly execute the prepared stage; may train models or run inference",
    )
    run.set_defaults(handler=_run_stage)

    smoke = commands.add_parser("smoke")
    smoke.add_argument(
        "--config",
        required=True,
        help="explicit project configuration; no source-checkout path is assumed",
    )
    smoke.set_defaults(handler=_smoke)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
