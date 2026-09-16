"""Time-preserving origin adjustment for a fixed forecast information scope."""

import numpy as np


def trailing_gap(context, target=None):
    values = np.asarray(context)
    observed = np.isfinite(values).any(1) if target is None else np.isfinite(values[:, target])
    indices = np.flatnonzero(observed)
    return len(values) if not len(indices) else len(values) - 1 - int(indices[-1])


def supported_gaps(context, joint):
    gaps = (
        [trailing_gap(context)] * 2 if joint else [trailing_gap(context, slot) for slot in (0, 1)]
    )
    return [gap if 1 <= gap <= 48 else 0 for gap in gaps]


def bridge_slice(forecast, gap):
    forecast = np.asarray(forecast)
    if not 0 <= gap <= 48 or forecast.shape[0] != 96 + gap:
        raise ValueError("bridge output must cover the original missing tail and H96")
    return forecast[gap : gap + 96]


def mask_tail(context, gap):
    if context.shape[0] != 96 or not 1 <= gap <= 48 or not np.isfinite(context).all():
        raise ValueError("synthetic tail probes require a complete L96 history")
    result = np.array(context, copy=True)
    result[-gap:] = np.nan
    return result
