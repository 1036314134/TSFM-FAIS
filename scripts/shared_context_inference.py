"""Compute a shared context decision once before assigning it to both targets."""

import numpy as np
from train_calibrated_source_gates import probability_from_state


def broadcast_context_weights(state, features):
    if len(features) % 2:
        raise ValueError("shared-context inference requires paired target rows")
    np.testing.assert_array_equal(features[::2], features[1::2])
    probability = probability_from_state(state, np.ascontiguousarray(features[::2]))
    return np.repeat(probability, 2, axis=0)
