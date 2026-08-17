"""
IdentityDecisionEngine — Tiger Intelligence System
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

class IdentityDecision(str, Enum):
    AUTO_MATCH = "auto_match"
    HUMAN_REVIEW = "human_review"
    NEW_INDIVIDUAL = "new_individual"

@dataclass
class CandidateMatch:
    tiger_id: str
    similarity_score: float
    observation_count: int
    rank: int

@dataclass
class IdentityDecisionResult:
    decision: IdentityDecision
    matched_tiger_id: Optional[str]
    candidates: List[CandidateMatch]
    top_score: float
    confidence: float
    reason: str
    auto_match_threshold: float
    review_threshold: float
    new_individual_threshold: float

class IdentityDecisionEngine:
    def __init__(
        self,
        auto_match_threshold: float = 0.90,
        review_threshold: float = 0.75,
        new_individual_threshold: float = 0.60,
    ):
        self.auto_match_threshold = auto_match_threshold
        self.review_threshold = review_threshold
        self.new_individual_threshold = new_individual_threshold
    
    def decide(self, candidates: List[CandidateMatch]) -> IdentityDecisionResult:
        """
        Apply threshold logic to candidate matches.
        """
        if not candidates:
            return IdentityDecisionResult(
                decision=IdentityDecision.NEW_INDIVIDUAL,
                matched_tiger_id=None,
                candidates=[],
                top_score=0.0,
                confidence=1.0,
                reason="No candidates provided.",
                auto_match_threshold=self.auto_match_threshold,
                review_threshold=self.review_threshold,
                new_individual_threshold=self.new_individual_threshold
            )
            
        sorted_candidates = sorted(candidates, key=lambda c: c.similarity_score, reverse=True)
        top_cand = sorted_candidates[0]
        top_score = top_cand.similarity_score
        
        confidence = self._compute_confidence(sorted_candidates)
        
        decision = IdentityDecision.NEW_INDIVIDUAL
        matched_id = None
        
        if top_score >= self.auto_match_threshold:
            # Check gap
            if len(sorted_candidates) == 1 or (top_score - sorted_candidates[1].similarity_score >= 0.05):
                decision = IdentityDecision.AUTO_MATCH
                matched_id = top_cand.tiger_id
            else:
                decision = IdentityDecision.HUMAN_REVIEW
        elif top_score >= self.review_threshold:
            decision = IdentityDecision.HUMAN_REVIEW
        else:
            decision = IdentityDecision.NEW_INDIVIDUAL
            
        reason = self._build_reason(decision, sorted_candidates)
        
        return IdentityDecisionResult(
            decision=decision,
            matched_tiger_id=matched_id,
            candidates=sorted_candidates,
            top_score=top_score,
            confidence=confidence,
            reason=reason,
            auto_match_threshold=self.auto_match_threshold,
            review_threshold=self.review_threshold,
            new_individual_threshold=self.new_individual_threshold
        )
    
    def _compute_confidence(self, candidates: List[CandidateMatch]) -> float:
        """Compute overall confidence based on top score and score gap."""
        if not candidates:
            return 0.0
        top = candidates[0].similarity_score
        if len(candidates) > 1:
            gap = top - candidates[1].similarity_score
            # Higher gap = higher confidence
            gap_factor = min(1.0, gap / 0.1)
            return float(min(1.0, top * (0.8 + 0.2 * gap_factor)))
        return float(min(1.0, top))
    
    def _build_reason(self, decision: IdentityDecision, candidates: List[CandidateMatch]) -> str:
        """Build human-readable explanation of the decision."""
        if not candidates:
            return "No candidates found."
            
        top = candidates[0]
        if decision == IdentityDecision.AUTO_MATCH:
            return f"Auto matched with {top.tiger_id} (score {top.similarity_score:.3f}, sufficiently above threshold)."
        elif decision == IdentityDecision.HUMAN_REVIEW:
            if top.similarity_score >= self.auto_match_threshold:
                return f"High score ({top.similarity_score:.3f}) but gap to 2nd candidate is too small. Needs review."
            return f"Score ({top.similarity_score:.3f}) is in the review zone."
        else:
            return f"Top score ({top.similarity_score:.3f}) is below all thresholds. Likely a new individual."
