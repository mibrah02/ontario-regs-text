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

    def test_can_i_hunt_rabbits_returns_natural_guidance(self) -> None:
        result = answer_question_result("can I hunt rabbits in Ontario")
        self.assertEqual(result.kind, "clarify")
        self.assertEqual(result.expected_detail, "topic")
        self.assertIn("Rabbit hunting is covered here", result.text)

    def test_duck_daily_limit_without_district_requests_district(self) -> None:
        result = answer_question_result("duck daily limit")
        self.assertEqual(result.kind, "clarify")
        self.assertEqual(result.expected_detail, "district")
        self.assertIn("Southern or Central District", result.text)

    def test_duck_daily_limit_with_district_requests_wmu(self) -> None:
        result = answer_question_result("duck daily limit in Southern District")
        self.assertEqual(result.kind, "clarify")
        self.assertEqual(result.expected_detail, "wmu")
        self.assertIn("reply with your WMU", result.text)

    def test_duck_daily_limit_with_short_district_requests_wmu(self) -> None:
        result = answer_question_result("duck daily limit Southern")
        self.assertEqual(result.kind, "clarify")
        self.assertEqual(result.expected_detail, "wmu")
        self.assertIn("reply with your WMU", result.text)

    def test_duck_daily_limit_with_district_and_wmu_returns_short_quote(self) -> None:
        result = answer_question_result("duck daily limit in Southern District WMU 65")
        self.assertEqual(result.kind, "answer")
        self.assertIn("Ontario Summary of Migratory Birds Hunting Regulations", result.text)
        self.assertIn("p.4", result.text)
        self.assertIn("6 (in WMUs 60 to 87E", result.text)


    def test_duck_daily_limit_with_wmu_infers_district_and_returns_short_quote(self) -> None:
        result = answer_question_result("duck daily limit WMU 65")
        self.assertEqual(result.kind, "answer")
        self.assertIn("Ontario Summary of Migratory Birds Hunting Regulations", result.text)
        self.assertIn("6 (in WMUs 60 to 87E", result.text)

    def test_duck_daily_limit_with_toronto_infers_district_and_requests_wmu(self) -> None:
        result = answer_question_result("duck daily limit Toronto")
        self.assertEqual(result.kind, "clarify")
        self.assertEqual(result.expected_detail, "wmu")
        self.assertIn("reply with your WMU", result.text)

    def test_duck_daily_limit_with_ontario_stays_ambiguous(self) -> None:
        result = answer_question_result("duck daily limit in Ontario")
        self.assertEqual(result.kind, "clarify")
        self.assertEqual(result.expected_detail, "district")
        self.assertIn("reply with the district", result.text)

if __name__ == "__main__":
    unittest.main()
