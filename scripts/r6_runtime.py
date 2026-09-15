"""Shared frozen runtime for the R6 confirmation; no fitting or outcome access."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate_followup_motm import matching_candidate  # noqa: E402
from evaluate_timesfm_vendor_missing import TimesFMVendorMissingAdapter  # noqa: E402
from probe_differentiable_imputation import parameter_digest  # noqa: E402
from replay_preforecast_student import query_candidate_points  # noqa: E402

from tsfm_fais.contracts import ForecastSpec  # noqa: E402
from tsfm_fais.forecasting import ForecastRunner, default_forecast_registry  # noqa: E402
from tsfm_fais.utility_experiment import file_sha256  # noqa: E402


def make_forecaster(model_id, legacy_bundle, previous_forecasts):
    legacy = json.loads((legacy_bundle / "manifest.json").read_text(encoding="utf-8"))
    previous = json.loads(
        (previous_forecasts / model_id / "manifest.json").read_text(encoding="utf-8")
    )
    for name, digest in legacy["identity"]["runtime_code_sha256"].items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError("the previously verified forecasting runtime changed")
    registry = default_forecast_registry()
    joint = registry.get(model_id).mode == "joint_multivariate"
    path = legacy["identity"]["forecaster_artifacts"][model_id]
    adapter = (
        registry.build(model_id, model_name=path, device="cuda", batch_size=8)
        if joint
        else TimesFMVendorMissingAdapter(model_name=path, device="cuda", batch_size=8)
    )
    runner = ForecastRunner(registry, {model_id: adapter})
    backbone = adapter._ensure_backend().model.eval().requires_grad_(False)
    digest = parameter_digest(backbone)
    if digest != previous["parameter_sha256"]:
        raise ValueError("the forecasting parameters differ from the prior audited model")
    return runner, adapter, backbone, digest, joint


def forecast_spec(model_id, horizon, joint):
    if horizon not in (96, 192):
        raise ValueError("only the two registered R6 horizons are allowed")
    return ForecastSpec(
        model_id,
        "joint_multivariate" if joint else "independent_univariate",
        horizon,
        context_length=96,
        target_indices=[0, 1],
    )


def query_r6_candidates(runner, spec, context, candidates, actions, motm, mean, scale, *, joint):
    points, distinct = query_candidate_points(
        runner, spec, context, candidates, actions, [0, 1], mean, scale, joint=joint
    )
    reused = matching_candidate(context, candidates, actions, motm, joint=joint)
    if reused is None:
        extra = (runner.predict(motm[None], spec).point[0] - mean[:2]) / scale[:2]
    else:
        extra = points[reused]
    result = np.concatenate([points, extra[None]], axis=0)
    if result.shape != (8, spec.horizon, 2) or not np.isfinite(result).all():
        raise ValueError("the registered candidate forecasts have invalid shape or support")
    return result, {
        "primary_distinct_contexts": distinct,
        "motm_reused_candidate_index": reused,
        "total_distinct_contexts": distinct + int(reused is None),
    }
