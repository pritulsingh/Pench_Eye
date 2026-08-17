from typing import List, Dict, Any

try:
    from ml.reid.identity_decision_engine import IdentityDecisionEngine, CandidateMatch
except ImportError:
    class CandidateMatch:
        def __init__(self, tiger_id: str, similarity_score: float, observation_count: int = 1, rank: int = 1):
            self.tiger_id = tiger_id
            self.similarity_score = similarity_score
            self.observation_count = observation_count
            self.rank = rank

    class IdentityDecisionEngine:
        def __init__(self, auto_match_threshold: float = 0.90, review_threshold: float = 0.75, new_individual_threshold: float = 0.60):
            self.auto_match_threshold = auto_match_threshold
            self.review_threshold = review_threshold
            self.new_individual_threshold = new_individual_threshold

        def decide(self, candidates):
            if not candidates:
                return type('Result', (), {'decision': 'new_individual', 'matched_tiger_id': None, 'confidence': 0.0, 'candidates': [], 'reason': 'No candidates.'})()
            best = candidates[0]
            score = best.similarity_score if hasattr(best, 'similarity_score') else best.get('score', 0.0)
            tid = best.tiger_id if hasattr(best, 'tiger_id') else best.get('tiger_id')
            if score >= self.auto_match_threshold:
                return type('Result', (), {'decision': 'auto_match', 'matched_tiger_id': tid, 'confidence': score, 'candidates': candidates, 'reason': f'Auto matched with {tid}'})()
            elif score >= self.review_threshold:
                return type('Result', (), {'decision': 'human_review', 'matched_tiger_id': None, 'confidence': score, 'candidates': candidates, 'reason': 'Review required.'})()
            else:
                return type('Result', (), {'decision': 'new_individual', 'matched_tiger_id': None, 'confidence': score, 'candidates': candidates, 'reason': 'New individual.'})()


class IdentityDecisionService:
    def __init__(self):
        from app.core.config import settings
        self.engine = IdentityDecisionEngine(
            auto_match_threshold=settings.AUTO_MATCH_THRESHOLD,
            review_threshold=settings.REVIEW_THRESHOLD,
            new_individual_threshold=settings.NEW_INDIVIDUAL_THRESHOLD,
        )
        
    def decide(self, similarity_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        candidates = []
        for i, item in enumerate(similarity_results):
            score = item.get('score', item.get('similarity_score', 0.0))
            candidates.append(CandidateMatch(
                tiger_id=item.get('tiger_id', ''),
                similarity_score=score,
                observation_count=item.get('observation_count', 1),
                rank=item.get('rank', i + 1)
            ))
            
        result = self.engine.decide(candidates)
        dec_val = result.decision.value if hasattr(result.decision, 'value') else str(result.decision)
        return {
            "decision": dec_val,
            "tiger_id": getattr(result, 'matched_tiger_id', None),
            "confidence": getattr(result, 'confidence', 0.0),
            "candidates": similarity_results,
            "reason": getattr(result, 'reason', '')
        }


identity_decision_service = IdentityDecisionService()
