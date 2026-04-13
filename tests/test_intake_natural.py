from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.main as main_module
import app.rag as rag


class IntakeNaturalTests(unittest.TestCase):
    def test_orphan_fragment_detection(self) -> None:
        self.assertTrue(rag._is_orphan_fragment("WMU 65"))
        self.assertTrue(rag._is_orphan_fragment("guns"))
        self.assertFalse(rag._is_orphan_fragment("deer 65 when"))
        self.assertFalse(rag._is_orphan_fragment("orange?"))

    def test_merge_explicit_details_preserves_wmu_and_district(self) -> None:
        merged = rag._merge_explicit_details("duck limit south 65", "duck daily limit Southern District")
        self.assertEqual(merged, "duck daily limit Southern District WMU 65")

    def test_merge_explicit_details_preserves_bow_method(self) -> None:
        merged = rag._merge_explicit_details("season deer 65 bow", "when is deer season in WMU 65")
        self.assertEqual(merged, "when is deer season in WMU 65 bows only")

    def test_orange_shortening_keeps_meaningful_phrase(self) -> None:
        shortened = main_module._shorten_answer_quote(
            "orange?",
            "All licensed hunters, including bow hunters, must wear hunter orange."
        )
        self.assertEqual(shortened, "hunter orange.")

    def test_interpret_keeps_wmu_for_rough_deer_question(self) -> None:
        original_invoke = rag._invoke_model_call
        try:
            rag._invoke_model_call = lambda callable_obj, timeout_seconds: SimpleNamespace(
                content=json.dumps({
                    "action": "search",
                    "normalized_question": "when is deer season",
                    "reply_text": "",
                    "pending_question": "",
                    "expected_detail": "",
                })
            )
            outcome = rag.interpret_incoming_message("deer 65 when")
        finally:
            rag._invoke_model_call = original_invoke

        self.assertEqual(outcome.action, "search")
        self.assertEqual(outcome.normalized_question, "when is deer season WMU 65")

    def test_interpret_keeps_bow_and_wmu_for_rough_deer_question(self) -> None:
        original_invoke = rag._invoke_model_call
        try:
            rag._invoke_model_call = lambda callable_obj, timeout_seconds: SimpleNamespace(
                content=json.dumps({
                    "action": "search",
                    "normalized_question": "when is deer season in WMU 65",
                    "reply_text": "",
                    "pending_question": "",
                    "expected_detail": "",
                })
            )
            outcome = rag.interpret_incoming_message("season deer 65 bow")
        finally:
            rag._invoke_model_call = original_invoke

        self.assertEqual(outcome.action, "search")
        self.assertEqual(outcome.normalized_question, "when is deer season in WMU 65 bows only")

    def test_interpret_keeps_district_and_wmu_for_rough_duck_question(self) -> None:
        original_invoke = rag._invoke_model_call
        try:
            rag._invoke_model_call = lambda callable_obj, timeout_seconds: SimpleNamespace(
                content=json.dumps({
                    "action": "search",
                    "normalized_question": "duck daily limit Southern District",
                    "reply_text": "",
                    "pending_question": "",
                    "expected_detail": "",
                })
            )
            outcome = rag.interpret_incoming_message("duck limit south 65")
        finally:
            rag._invoke_model_call = original_invoke

        self.assertEqual(outcome.action, "search")
        self.assertEqual(outcome.normalized_question, "duck daily limit Southern District WMU 65")


    def test_extract_district_terms_infers_from_wmu(self) -> None:
        self.assertEqual(rag._extract_district_terms("duck daily limit WMU 65"), {"Southern District"})

    def test_extract_district_terms_infers_from_toronto(self) -> None:
        self.assertEqual(rag._extract_district_terms("duck daily limit Toronto"), {"Southern District"})

    def test_extract_district_terms_does_not_infer_from_ontario(self) -> None:
        self.assertEqual(rag._extract_district_terms("duck daily limit in Ontario"), set())

    def test_interpret_promotes_toronto_to_search_when_llm_over_clarifies_district(self) -> None:
        original_invoke = rag._invoke_model_call
        try:
            rag._invoke_model_call = lambda callable_obj, timeout_seconds: SimpleNamespace(
                content=json.dumps({
                    "action": "clarify",
                    "normalized_question": "",
                    "reply_text": "Need one more detail: reply with the district, for example Southern or Central District.",
                    "pending_question": "duck daily limit",
                    "expected_detail": "district",
                })
            )
            outcome = rag.interpret_incoming_message("duck daily limit Toronto")
        finally:
            rag._invoke_model_call = original_invoke

        self.assertEqual(outcome.action, "search")
        self.assertEqual(outcome.normalized_question, "duck daily limit Toronto")


if __name__ == "__main__":
    unittest.main()
