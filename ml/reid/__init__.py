# Re-ID module — flank extraction, preprocessing, embedding, identity decisions.
from ml.reid.identity_decision_engine import (
    CandidateMatch,
    IdentityDecision,
    IdentityDecisionEngine,
    IdentityDecisionResult,
)
from ml.reid.flank_extractor import FlankExtractor, FlankSide
from ml.reid.preprocessing import PREPROCESSING_VERSION, PreprocessConfig
from ml.reid.stripe_processor import ProcessedStripe, StripeProcessor
from ml.reid.tiger_reid_encoder import (
    DEMO_MODEL_VERSION,
    EmbeddingResult,
    ReIDModelUnavailable,
    TigerReIDEncoder,
)

__all__ = [
    "TigerReIDEncoder",
    "EmbeddingResult",
    "ReIDModelUnavailable",
    "DEMO_MODEL_VERSION",
    "IdentityDecisionEngine",
    "IdentityDecision",
    "CandidateMatch",
    "IdentityDecisionResult",
    "FlankExtractor",
    "FlankSide",
    "StripeProcessor",
    "ProcessedStripe",
    "PreprocessConfig",
    "PREPROCESSING_VERSION",
]