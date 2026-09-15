"""An explicit input boundary for selection after querying candidate forecasts."""

from .preforecast import METADATA, STATIC_FEATURES

RESPONSE_FEATURES = tuple(
    "response." + name
    for name in (
        "mean_change",
        "signed_change",
        "max_change",
        "early_change",
        "late_change",
        "pool_distance",
        "departure",
        "signed_departure",
        "slope_change",
        "variation",
        "has_quantiles",
        "interval_width",
    )
)
FORECAST_FEATURES = STATIC_FEATURES + RESPONSE_FEATURES


def forecast_response_inputs(frame):
    observed = {name for name in frame if name.startswith(("static.", "response."))}
    if observed != set(FORECAST_FEATURES):
        raise ValueError("candidate-forecast inputs require the complete audited feature list")
    return frame[[name for name in METADATA if name in frame] + list(FORECAST_FEATURES)].copy()
