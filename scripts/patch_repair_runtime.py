"""Differentiable H96 forecasts with the installed TimesFM batch layout preserved."""

import torch

from tsfm_fais.forecasting.chronos_differentiable import chronos_median
from tsfm_fais.forecasting.timesfm_differentiable import running_stats


def padded_timesfm_median(core, context, targets=(0, 1)):
    from timesfm.torch.util import revin

    selected = context[:, list(targets)].T.float()
    count, length = selected.shape
    if length % core.p or count > 8 or core.o != 128 or core.aridx != 5:
        raise ValueError("the H96 canonical TimesFM repair layout changed")
    values = selected.new_zeros((8, length))
    values[:count] = selected
    masks = torch.ones_like(values, dtype=torch.bool)
    masks[:count] = False
    positive = (values >= 0).all(-1, keepdim=True)
    mean, scale = values.mean(-1, keepdim=True), values.std(-1, keepdim=True)
    normalized = revin(values, mean, scale)

    def prefill(inputs):
        patches = inputs.reshape(8, -1, core.p)
        patch_masks = masks.reshape(8, -1, core.p)
        n = inputs.new_zeros(8)
        mu, sigma = n.clone(), n.clone()
        means, scales = [], []
        for index in range(patches.shape[1]):
            n, mu, sigma = running_stats(n, mu, sigma, patches[:, index], patch_masks[:, index])
            means.append(mu)
            scales.append(sigma)
        means, scales = torch.stack(means, 1), torch.stack(scales, 1)
        encoded = torch.where(patch_masks, 0.0, revin(patches, means, scales))
        (_, _, output, _), _ = core(encoded, patch_masks, decode_caches=None)
        restored = revin(output, means, scales, reverse=True).reshape(8, -1, core.o, core.q)
        return restored[:, -1, :96, 5]

    point = (prefill(normalized) - prefill(-normalized)) / 2
    point = revin(point, mean, scale, reverse=True)
    point = torch.where(positive, point.clamp_min(0), point)
    return point[:count].T


def differentiable_point(model_id, adapter, backbone, values, targets=(0, 1)):
    if model_id == "chronos2":
        return chronos_median(adapter._ensure_backend(), values, 96, list(targets))
    return padded_timesfm_median(backbone, values, targets)


def repair_loss(point, truth, mean, scale, penalty, weight=1.0):
    error = (point.double() - truth.double()) / scale
    prediction_loss = 0.5 * (error.square() + torch.sqrt(error.square() + 1e-6)).mean()
    return weight * prediction_loss + 1e-3 * penalty
