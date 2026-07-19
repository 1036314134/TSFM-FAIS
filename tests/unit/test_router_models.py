from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from tsfm_fais.contracts import ForecastSpec, TimeSeriesItem
from tsfm_fais.pipeline import BlockwiseFAIS
from tsfm_fais.routing import RouterBundle as PublicRouterBundle
from tsfm_fais.routing.models import (
    PairwiseRiskModel,
    RankerModel,
    RouterBundle,
    losses_to_relevance,
)


def test_losses_to_relevance_preserves_ties_and_inverts_loss_order() -> None:
    losses = np.asarray([3.0, 1.0, 1.0, 8.0, 4.0])
    relevance = losses_to_relevance(losses, (3, 2))
    np.testing.assert_array_equal(relevance, [0, 1, 1, 0, 1])


def test_losses_to_relevance_suppresses_numerical_near_ties() -> None:
    losses = np.asarray([1.0, 1.0 + 1e-8, 2.0])

    relevance = losses_to_relevance(losses, (3,))

    np.testing.assert_array_equal(relevance, [1, 1, 0])


def test_router_bundle_writes_schema_and_dependency_manifest(tmp_path) -> None:
    assert PublicRouterBundle is RouterBundle
    bundle = RouterBundle(
        prior=RankerModel(),
        unary=RankerModel(),
        pairwise=PairwiseRiskModel(),
        feature_names=("block_length", "candidate_cost"),
        pair_feature_names=("pair_gap",),
        candidate_ids=("locf", "linear_interp"),
        categorical_maps={"candidate": {"locf": 0, "linear_interp": 1}},
        metadata={"training_split": "synthetic"},
    )
    directory = bundle.save(tmp_path / "router")
    restored = RouterBundle.load(directory)
    assert restored.feature_names == bundle.feature_names
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["feature_names"] == list(bundle.feature_names)
    assert manifest["pair_feature_names"] == ["pair_gap"]
    assert manifest["candidate_ids"] == ["locf", "linear_interp"]
    assert "numpy" in manifest["dependency_versions"]
    pipeline = BlockwiseFAIS.load("configs/smoke.yaml", directory)
    assert pipeline.router.candidate_ids == bundle.candidate_ids


def test_pipeline_load_resolves_router_linked_imputer_artifacts(tmp_path) -> None:
    artifacts = tmp_path / "imputers"
    dataset_dir = artifacts / "toy"
    dataset_dir.mkdir(parents=True)
    np.savez(
        dataset_dir / "training_statistics.npz",
        medians=np.asarray([1.0, 2.0]),
        correlation=np.eye(2),
    )
    joblib.dump({"fitted": True}, dataset_dir / "knn_multivariate.joblib")
    (artifacts / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": {
                    "toy": {
                        "statistics": "training_statistics.npz",
                        "candidates": {
                            "knn_multivariate": {
                                "status": "fitted",
                                "serializer": "joblib",
                                "path": "knn_multivariate.joblib",
                            },
                            "mice": {
                                "status": "failed",
                                "reason": "synthetic fit failure",
                            },
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    router = RouterBundle(
        prior=RankerModel(),
        unary=RankerModel(),
        pairwise=PairwiseRiskModel(),
        feature_names=(),
        candidate_ids=("knn_multivariate", "mice"),
        metadata={"imputer_artifacts": str(artifacts)},
    )
    router_path = router.save(tmp_path / "router-with-lineage")
    pipeline = BlockwiseFAIS.load("configs/smoke.yaml", router_path)
    complete_item = TimeSeriesItem(
        item_id="complete",
        values=np.arange(8, dtype=float).reshape(4, 2),
        variate_names=("a", "b"),
        start=pd.Timestamp("2026-01-01"),
        freq="h",
    )
    no_op = pipeline.impute(
        complete_item,
        np.ones_like(complete_item.values, dtype=bool),
        ForecastSpec("mock", "independent_univariate", 1, target_indices=(0,)),
    )
    np.testing.assert_array_equal(no_op.values, complete_item.values)

    item = TimeSeriesItem(
        item_id="toy-item",
        values=np.arange(12, dtype=float).reshape(6, 2),
        variate_names=("a", "b"),
        start=pd.Timestamp("2026-01-01"),
        freq="h",
        metadata={"dataset_id": "toy"},
    )

    pipeline._ensure_item_artifacts(item)

    assert pipeline.imputer_artifacts["knn_multivariate"] == {"fitted": True}
    assert "mice" in pipeline.artifact_load_failures
    np.testing.assert_array_equal(pipeline.training_medians, [1.0, 2.0])
