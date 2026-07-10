"""Block-wise imputer routing."""

from .blocks import (
    assemble_routed_values,
    block_mask,
)
from .blocks import build_block_graph as build_legacy_block_graph
from .blocks import (
    detect_missing_blocks,
    extract_missing_blocks,
    validate_blocks,
)
from .bundle import RouterBundle as InferenceRouterBundle
from .features import (
    RoutingFeatureExtractor,
    RoutingFeatureTable,
    pair_features,
    proxy_pairwise_scores,
    proxy_unary_scores,
)
from .graph import BlockEdge, BlockGraph, build_block_graph
from .lightgbm_model import LazyLightGBMRegressor, LightGBMUnaryModel
from .metrics import (
    block_candidate_losses,
    forecast_degradation,
    masked_mae,
    masked_mse,
    masked_rmse,
    routing_regret,
    top_k_hit,
)
from .models import (
    PairwiseRiskModel,
    RankerModel,
    RouterBundle,
    RouterTrainer,
)
from .shortlist import CandidateShortlister, ShortlistResult, shortlist_candidates
from .solver import (
    BeamSearchSolver,
    ExhaustiveSolver,
    RoutingProblem,
    SolverResult,
    assignment_energy,
    beam_search,
    exhaustive_search,
    greedy_shortlist,
    solve_routing,
)
from .teacher import (
    RoutingTeacher,
    TeacherBuilder,
    TeacherLabel,
    TeacherTargets,
    TeacherWeights,
    replace_block,
)

__all__ = [
    "BeamSearchSolver",
    "BlockEdge",
    "BlockGraph",
    "CandidateShortlister",
    "ExhaustiveSolver",
    "InferenceRouterBundle",
    "LazyLightGBMRegressor",
    "LightGBMUnaryModel",
    "PairwiseRiskModel",
    "RankerModel",
    "RouterBundle",
    "RoutingFeatureExtractor",
    "RoutingFeatureTable",
    "RoutingProblem",
    "RoutingTeacher",
    "RouterTrainer",
    "ShortlistResult",
    "SolverResult",
    "TeacherTargets",
    "TeacherWeights",
    "TeacherBuilder",
    "TeacherLabel",
    "assemble_routed_values",
    "assignment_energy",
    "beam_search",
    "block_candidate_losses",
    "block_mask",
    "build_block_graph",
    "build_legacy_block_graph",
    "detect_missing_blocks",
    "exhaustive_search",
    "extract_missing_blocks",
    "forecast_degradation",
    "masked_mae",
    "masked_mse",
    "masked_rmse",
    "pair_features",
    "proxy_pairwise_scores",
    "proxy_unary_scores",
    "routing_regret",
    "greedy_shortlist",
    "replace_block",
    "shortlist_candidates",
    "solve_routing",
    "top_k_hit",
    "validate_blocks",
]
