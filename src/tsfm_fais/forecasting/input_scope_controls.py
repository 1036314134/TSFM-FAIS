"""Controlled target/covariate replacements between completed histories."""

import numpy as np

VARIANTS = ("base", "targets_only", "covariates_only", "both")


def crossed_inputs(context, base, changed, targets):
    context, base, changed = map(lambda value: np.asarray(value, float), (context, base, changed))
    targets = tuple(targets)
    if (
        context.ndim != 2
        or base.shape != context.shape
        or changed.shape != context.shape
        or not np.isfinite(base).all()
        or not np.isfinite(changed).all()
        or not targets
        or len(set(targets)) != len(targets)
        or min(targets) < 0
        or max(targets) >= context.shape[1]
    ):
        raise ValueError("aligned finite completions and valid distinct targets are required")
    observed = np.isfinite(context)
    np.testing.assert_array_equal(base[observed], context[observed])
    np.testing.assert_array_equal(changed[observed], context[observed])
    target_update, covariate_update = base.copy(), changed.copy()
    target_update[:, targets] = changed[:, targets]
    covariate_update[:, targets] = base[:, targets]
    return np.stack([base, target_update, covariate_update, changed])
