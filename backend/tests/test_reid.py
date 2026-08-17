"""Unit and smoke tests for Tiger Re-ID and Decision Services."""
import unittest
import numpy as np
from app.services.identity_decision_service import identity_decision_service


class TestReIDAndDecision(unittest.TestCase):
    def test_identity_decision_auto_match(self):
        candidates = [{'tiger_id': 'TIGER-001', 'score': 0.95, 'rank': 1}]
        decision = identity_decision_service.decide(candidates)
        self.assertEqual(decision['decision'], 'auto_match')
        self.assertEqual(decision['tiger_id'], 'TIGER-001')

    def test_identity_decision_human_review(self):
        candidates = [{'tiger_id': 'TIGER-002', 'score': 0.82, 'rank': 1}]
        decision = identity_decision_service.decide(candidates)
        self.assertEqual(decision['decision'], 'human_review')
        self.assertIsNone(decision['tiger_id'])

    def test_identity_decision_new_individual(self):
        candidates = [{'tiger_id': 'TIGER-003', 'score': 0.45, 'rank': 1}]
        decision = identity_decision_service.decide(candidates)
        self.assertEqual(decision['decision'], 'new_individual')
        self.assertIsNone(decision['tiger_id'])

    def test_identity_decision_empty_candidates(self):
        decision = identity_decision_service.decide([])
        self.assertEqual(decision['decision'], 'new_individual')


if __name__ == '__main__':
    unittest.main()
