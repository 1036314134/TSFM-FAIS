"""Verify a fixed binary-marker intervention without reading or scoring future targets."""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.dataset  # noqa: F401
import torch
from latent_source_inputs import ROOT, read_json
from probe_differentiable_imputation import parameter_digest
from provenance_marker import ProvenanceMarker
from r6_runtime import make_forecaster

from tsfm_fais.contracts import ForecastSpec
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed interface diagnostics")
    output.mkdir(parents=True, exist_ok=True)
    prepared_root = ROOT / "artifacts/iclr27-r20/tail-inputs-v001"
    source = read_json(prepared_root / "smoke_preparation.json")
    cases = source["cases"][:3]
    records = []
    torch.set_num_threads(1)
    started = perf_counter()
    for model_id in ("chronos2", "timesfm2p5"):
        runner, adapter, backbone, digest, joint = make_forecaster(
            model_id,
            ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
            ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
        )
        for entry in cases:
            path = prepared_root / entry["path"]
            if file_sha256(path) != entry["sha256"]:
                raise ValueError("a fixed diagnostic input changed")
            with np.load(path, allow_pickle=False) as data:
                ids = data["candidate_ids"].tolist()
                raw = data["candidate_values"][ids.index("seasonal_lag")]
                observed = np.isfinite(data["context"])
            effective = raw if joint else raw[:, :1]
            marker = observed if joint else observed[:, :1]
            spec = ForecastSpec(
                model_id,
                "joint_multivariate" if joint else "independent_univariate",
                96,
                context_length=96,
                target_indices=[0, 1] if joint else [0],
            )

            def predict(runner=runner, effective=effective, spec=spec):
                return runner.predict_missing(effective[None], spec).point[0]

            plain = predict()
            values = {"plain": plain, "effective": effective, "observed": marker}
            logs = {}
            for name, given, enabled in (
                ("passthrough", marker, False),
                ("complete_identity", np.ones_like(marker), True),
                ("provenance", marker, True),
            ):
                with ProvenanceMarker(backbone, model_id, given, enabled) as intervention:
                    values[name] = predict()
                logs[name] = intervention.calls
                for call_index, call in enumerate(intervention.calls):
                    for field in ("before", "after", "expected"):
                        values[f"{name}_{call_index}_{field}"] = call[field]
                if name != "provenance":
                    np.testing.assert_array_equal(plain, values[name])
            np.testing.assert_array_equal(predict(), plain)
            with ProvenanceMarker(backbone, model_id, marker, True):
                np.testing.assert_array_equal(predict(), values["provenance"])
            for plain_call, altered_call in zip(
                logs["passthrough"], logs["provenance"], strict=True
            ):
                np.testing.assert_array_equal(plain_call["before"], altered_call["before"])
            target = output / model_id / f"{entry['case_id']}.npz"
            _save_npz(target, **values)
            records.append(
                {
                    "model_id": model_id,
                    "case_id": entry["case_id"],
                    "input_sha256": entry["sha256"],
                    "path": str(target.relative_to(output)),
                    "sha256": file_sha256(target),
                    "tokenizer_calls_per_request": len(logs["provenance"]),
                    "changed_marker_features": sum(
                        int(np.count_nonzero(call["before"] != call["after"]))
                        for call in logs["provenance"]
                    ),
                    "maximum_forecast_change_not_an_error_metric": float(
                        abs(plain - values["provenance"]).max()
                    ),
                    "parameter_sha256": digest,
                }
            )
            print(f"verified {model_id} {entry['case_id']}", flush=True)
        if parameter_digest(backbone) != digest:
            raise ValueError("a frozen forecasting parameter changed")
        del predict, runner, adapter, backbone
        gc.collect()
        torch.cuda.empty_cache()
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "marker_module_sha256": file_sha256(ROOT / "scripts/provenance_marker.py"),
            "plan_sha256": file_sha256(ROOT / "docs/iclr2027/R21_PROVENANCE_INTERFACE_PLAN.md"),
            "records": records,
            "wall_seconds": perf_counter() - started,
            "future_targets_read": False,
            "accuracy_improvement_established": False,
            "limits": "internal marker-feature intervention; native mask semantics and model training distribution are not confidence semantics",
        },
    )


if __name__ == "__main__":
    main()
