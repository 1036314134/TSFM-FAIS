"""Registered sensor-group interventions with unchanged R31 learning operations."""

import hashlib

import numpy as np
from forecast_calibration_core import FEATURE_NAMES as FEATURE_NAMES
from forecast_calibration_core import LEARNED as LEARNED
from forecast_calibration_core import NEW_METHODS as NEW_METHODS
from forecast_calibration_core import PARENT as PARENT
from forecast_calibration_core import ROOT as ROOT
from forecast_calibration_core import SEED as SEED
from forecast_calibration_core import calibration_context as target_context
from forecast_calibration_core import calibration_population as base_population
from forecast_calibration_core import calibration_sources as calibration_sources
from forecast_calibration_core import evaluation_rows as evaluation_rows
from forecast_calibration_core import evaluation_sources as evaluation_sources
from forecast_calibration_core import fixed_mae_fit as fixed_mae_fit
from forecast_calibration_core import load_npz as load_npz
from forecast_calibration_core import mixed_context as mixed_context
from forecast_calibration_core import new_gate as new_gate
from forecast_calibration_core import normalized_features as normalized_features
from forecast_calibration_core import observable_features as observable_features
from forecast_calibration_core import read_json as read_json
from forecast_calibration_core import smooth_mae as smooth_mae
from forecast_calibration_core import tensor_inputs as tensor_inputs

BASE = ROOT / "artifacts/iclr27-r32"
PATTERNS = ("targets_only", "local_pollutants", "regional_pollutants")


def calibration_population(records):
    original, eligibility = base_population(records)
    result = []
    for row in original:
        for pattern in PATTERNS:
            case_id = (
                row["case_id"]
                if pattern == "targets_only"
                else hashlib.sha256(f"r32|{row['case_id']}|{pattern}".encode()).hexdigest()[:20]
            )
            result.append(
                {
                    **row,
                    "case_id": case_id,
                    "outage_pattern": pattern,
                    "parent_case_id": row["case_id"],
                }
            )
    if len({r["case_id"] for r in result}) != len(result):
        raise ValueError("co-outage calibration cases must be unique")
    return result, eligibility


def calibration_context(full, row):
    x = target_context(full, row)
    pattern = row["outage_pattern"]
    if pattern not in PATTERNS:
        raise ValueError("unknown registered outage pattern")
    if pattern != "targets_only":
        x[-row["outage_age"] :, :6] = np.nan
    if pattern == "regional_pollutants":
        x[-row["outage_age"] :, 11:17] = np.nan
    return x
