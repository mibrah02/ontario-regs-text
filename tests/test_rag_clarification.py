from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag import answer_question_result


class RagClarificationTests(unittest.TestCase):
    def test_deer_season_without_wmu_or_method_requests_both(self) -> None:
        result = answer_question_result("when is deer season")
        self.assertEqual(result.kind, "clarify")
        self.assertEqual(result.expected_detail, "wmu_and_method")
        self.assertIn("WMU number and method", result.text)

    def test_deer_season_with_wmu_requests_method(self) -> None:
        result = answer_question_result("when is deer season in WMU 65")
        self.assertEqual(result.kind, "clarify")
        self.assertEqual(result.expected_detail, "method")
        self.assertIn("bows only, or guns", result.text)

    def test_bows_only_question_returns_exact_structured_quote(self) -> None:
        result = answer_question_result("when is deer season in WMU 65 bows only")
        self.assertEqual(result.kind, "answer")
        self.assertIn("p.44", result.text)
        self.assertIn("65 October 1 to October 4", result.text)

    def test_broad_rabbit_rules_question_requests_specific_topic(self) -> None:
        result = answer_question_result("what rabbits hunting rules")
        self.assertEqual(result.kind, "clarify")
        self.assertEqual(result.expected_detail, "topic")
        self.assertIn("rabbit daily limit", result.text)

    def test_broad_deer_rules_question_requests_specific_topic(self) -> None:
        result = answer_question_result("what deer hunting rules")
        self.assertEqual(result.kind, "clarify")
        self.assertEqual(result.expected_detail, "topic")
        self.assertIn("deer daily limit", result.text)

    def test_broad_general_hunting_rules_question_requests_specific_topic(self) -> None:
        result = answer_question_result("what hunting rules")
        self.assertEqual(result.kind, "clarify")
        self.assertEqual(result.expected_detail, "topic")
        self.assertIn("deer season in WMU 65", result.text)


if __name__ == "__main__":
    unittest.main()
