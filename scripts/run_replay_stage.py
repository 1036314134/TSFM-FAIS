"""Measure stage wall time and preparation work without changing numerical operations."""

import argparse
import hashlib
import json
import runpy
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    record = {
        "script": str(args.script.resolve()),
        "script_sha256": hashlib.sha256(args.script.read_bytes()).hexdigest(),
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "stage": args.stage,
        "arguments": argv,
        "started_unix": time.time(),
    }
    started = time.perf_counter()
    sys.path.insert(0, str(args.script.resolve().parent))
    if args.stage == "prepare":
        import pyarrow.dataset  # noqa: F401
        from matched_replay_pool import NativeReplayPool

        from tsfm_fais.imputers.motm import MOTMReference
        from tsfm_fais.imputers.runner import CandidateRunner

        record.update(
            pool_setup_seconds=0.0,
            pool_setups=0,
            classical_prefix_fit_seconds=0.0,
            classical_prefix_fit_calls=0,
            motm_seconds=0.0,
            motm_context_calls=0,
            motm_fitted_variables=0,
            detail_scope="this invocation; cached smoke preparation retains its original total case timings",
        )
        old_init, old_fit, old_motm = (
            NativeReplayPool.__init__,
            CandidateRunner.fit,
            MOTMReference.impute,
        )

        def pool_init(self, *positional, **keywords):
            then = time.perf_counter()
            try:
                return old_init(self, *positional, **keywords)
            finally:
                record["pool_setup_seconds"] += time.perf_counter() - then
                record["pool_setups"] += 1

        def fit(self, *positional, **keywords):
            then = time.perf_counter()
            try:
                return old_fit(self, *positional, **keywords)
            finally:
                record["classical_prefix_fit_seconds"] += time.perf_counter() - then
                record["classical_prefix_fit_calls"] += 1

        def impute(self, *positional, **keywords):
            then = time.perf_counter()
            try:
                result = old_motm(self, *positional, **keywords)
                record["motm_fitted_variables"] += result[1]["fitted_variables"]
                return result
            finally:
                record["motm_seconds"] += time.perf_counter() - then
                record["motm_context_calls"] += 1

        NativeReplayPool.__init__ = pool_init
        CandidateRunner.fit = fit
        MOTMReference.impute = impute
    sys.argv = [str(args.script), *argv]
    try:
        runpy.run_path(str(args.script), run_name="__main__")
        record["status"] = "completed"
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["wall_seconds"] = time.perf_counter() - started
        record["ended_unix"] = time.time()
        args.record_root.mkdir(parents=True, exist_ok=True)
        path = args.record_root / f"{args.stage}-{time.time_ns()}.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
