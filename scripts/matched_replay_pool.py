"""Reuse verified native-prefix imputers before all historical replay anchors."""

from pathlib import Path

import numpy as np
from latent_source_inputs import read_json
from matched_replay_sources import timestamp
from prepare_native_confirmation import ACTIONS, DEEP_PARAMS, complete_candidates

from tsfm_fais.contracts import SeriesBatch
from tsfm_fais.imputers import DEFAULT_REGISTRY, CandidateRunner
from tsfm_fais.utility_experiment import file_sha256


class NativeReplayPool:
    def __init__(self, source, all_sources):
        self.source = source
        self.period = source["period"]
        self.runner = CandidateRunner()
        self.artifacts = {}
        prefix = source["values"][: source["prefix_end"]]
        self.defaults = np.nanmedian(prefix, axis=0)
        self.params = {"seasonal_lag": {"period": max(2, self.period)}}
        self.fit_records = []
        self.classical_fit_calls = 0
        source_manifest = read_json(source["root"] / "manifest.json")
        for entry in source["dataset"]["imputers"]:
            name = entry["candidate_id"]
            path = Path(entry["path"])
            if not path.is_absolute():
                path = source["root"] / path
            if file_sha256(path) != entry["sha256"]:
                raise ValueError("a frozen native imputer record changed")
            record = read_json(path)
            if (
                record["status"] != "fitted"
                or record["identity_sha256"] != source_manifest["identity_sha256"]
            ):
                raise ValueError("a native prefix imputer is not the registered fit")
            for file in record["files"]:
                if file_sha256(path.parent / name / file["path"]) != file["sha256"]:
                    raise ValueError("a frozen native imputer file changed")
            self.artifacts[name] = DEFAULT_REGISTRY.create(name, **DEEP_PARAMS).load_artifact(
                path.parent / name
            )
            self.fit_records.append(
                {"candidate_id": name, "path": str(path), "sha256": entry["sha256"]}
            )
        training_path = source["root"] / "imputers" / source["dataset_id"] / "training_batch.npz"
        if file_sha256(training_path) != source["dataset"]["training_batch_sha256"]:
            raise ValueError("the original imputer training batch changed")
        item_map = {
            row["item_id"]: row
            for row in all_sources
            if row["dataset_id"] == source["dataset_id"] and row["cohort"] == source["cohort"]
        }
        cutoffs = []
        with np.load(training_path, allow_pickle=False) as saved:
            for index, identifier in enumerate(saved["window_ids"].tolist()):
                item, rest = identifier.rsplit("@", 1)
                start = int(rest.split("|", 1)[0])
                fitted_source = item_map[item]
                if start + 96 > fitted_source["prefix_end"]:
                    raise ValueError("a frozen imputer used a post-prefix training window")
                mask = saved["observed"][index]
                np.testing.assert_array_equal(
                    saved["values"][index][mask], fitted_source["values"][start : start + 96][mask]
                )
                cutoffs.append(timestamp(fitted_source, start + 96))
        self.latest_training_boundary = max(cutoffs)
        batch = SeriesBatch(
            prefix[None], np.isfinite(prefix[None]), metadata={"period": self.period}
        )
        for action in ACTIONS:
            if action not in self.artifacts and (
                action == "seasonal_lag" or DEFAULT_REGISTRY.get_spec(action).fit_scope == "dataset"
            ):
                self.artifacts[action] = self.runner.fit(
                    action, batch, {"period": self.period}, params=self.params.get(action)
                )
                self.classical_fit_calls += 1

    def complete(self, context, origin, seed):
        if timestamp(self.source, origin - 96) < self.latest_training_boundary:
            raise ValueError("a historical context precedes the frozen imputer training boundary")
        artifacts = dict(self.artifacts)
        seasonal = dict(artifacts["seasonal_lag"])
        seasonal["profiles"] = np.roll(
            seasonal["profiles"], -(origin - 96) % seasonal["period"], axis=1
        )
        artifacts["seasonal_lag"] = seasonal
        batch = SeriesBatch(
            context[None], np.isfinite(context[None]), metadata={"period": self.period}
        )
        result = self.runner.run_many(ACTIONS, batch, artifacts, seed=seed, params=self.params)
        return complete_candidates(context, result, self.defaults)
