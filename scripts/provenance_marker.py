"""Temporarily modify tokenizer marker features without changing value or attention inputs."""

import numpy as np
import torch


class ProvenanceMarker:
    def __init__(self, backbone, model_id, observed, enabled):
        self.backbone = backbone
        self.joint = model_id == "chronos2"
        self.observed = np.asarray(observed, bool)
        self.enabled = bool(enabled)
        self.calls = []
        self.hook = None

    def __enter__(self):
        if self.observed.ndim != 2 or len(self.observed) != 96:
            raise ValueError("the marker probe supports [96,D] inputs only")
        module = self.backbone.input_patch_embedding if self.joint else self.backbone.tokenizer
        self.hook = module.register_forward_pre_hook(self.before)
        return self

    def before(self, module, args):
        if len(args) != 1 or not isinstance(args[0], torch.Tensor):
            raise ValueError("the tokenizer input contract changed")
        before = args[0]
        expected = before.detach().clone()
        affected = not self.joint or len(self.calls) == 0
        patch = before.shape[-1] // (3 if self.joint else 2)
        if affected:
            if before.shape[1] * patch != 96:
                raise ValueError("the original context grid differs from the token grid")
            if self.joint:
                if before.shape[0] != self.observed.shape[1]:
                    raise ValueError("Chronos marker channels are not aligned")
                marker = torch.as_tensor(self.observed.T.copy(), device=before.device).reshape(
                    before.shape[0], -1, patch
                )
                expected[..., -patch:] = marker.to(before.dtype)
            else:
                if self.observed.shape[1] != 1:
                    raise ValueError("TimesFM marker requests must contain exactly one target")
                marker = torch.as_tensor(~self.observed[:, 0], device=before.device).reshape(
                    -1, patch
                )
                expected[0, :, -patch:] = marker.to(before.dtype)
        after = expected if self.enabled else before
        np.testing.assert_array_equal(
            before[..., :-patch].detach().cpu(), after[..., :-patch].detach().cpu()
        )
        if not self.joint:
            np.testing.assert_array_equal(before[1:].detach().cpu(), after[1:].detach().cpu())
        self.calls.append(
            {
                "before": before.detach().cpu().numpy().copy(),
                "after": after.detach().cpu().numpy().copy(),
                "expected": expected.cpu().numpy().copy(),
                "affected": affected,
            }
        )
        return (after,)

    def __exit__(self, exc_type, exc_value, traceback):
        self.hook.remove()
        self.hook = None
        if exc_type is None and len(self.calls) != 2:
            raise ValueError(
                "expected Chronos history/future or TimesFM positive/negative tokenizer calls"
            )
