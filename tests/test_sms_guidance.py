from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.main as main_module
from app.db import clear_pending_clarification, get_pending_clarification
from app.rag import IntakeOutcome


def fake_interpret(question: str, pending_state: dict[str, str] | None = None) -> IntakeOutcome:
    text = question.strip().lower()
    pending_state = pending_state or {}
    pending_question = pending_state.get("question", "")
    expected_detail = pending_state.get("expected_detail", "")

    if text in {"hello", "help", "hi", "hey"}:
        return IntakeOutcome(action="guide", reply_text=main_module.GUIDANCE_RESPONSE)

    if pending_question:
        if text in {"thanks", "what do you mean"}:
            detail = expected_detail or "method"
            if detail == "wmu_and_method":
                reply = "Still need two details: reply with the WMU number and method, for example WMU 65 bows only. Informational only. Not legal advice. Verify current regs."
            elif detail == "method":
                reply = "Still need one detail: reply with bows only, or guns. Informational only. Not legal advice. Verify current regs."
            else:
                reply = "Please ask again with one specific topic, for example: daily limit, possession limit, season, tag, or hunter orange. Informational only. Not legal advice. Verify current regs."
            return IntakeOutcome(
                action="clarify",
                reply_text=reply,
                pending_question=pending_question,
                expected_detail=detail,
            )
        if expected_detail == "district" and text in {"southern", "southern district", "central", "central district", "northern", "northern district", "hudson-james", "hudson-james bay", "hudson-james bay district"}:
            normalized = {
                "southern": "Southern District",
                "southern district": "Southern District",
                "central": "Central District",
                "central district": "Central District",
                "northern": "Northern District",
                "northern district": "Northern District",
                "hudson-james": "Hudson-James Bay District",
                "hudson-james bay": "Hudson-James Bay District",
                "hudson-james bay district": "Hudson-James Bay District",
            }[text]
            return IntakeOutcome(action="search", normalized_question=f"{pending_question} {normalized}")
        if expected_detail == "wmu" and text.startswith("wmu"):
            return IntakeOutcome(action="search", normalized_question=f"{pending_question} {question.strip()}")
        if text == "bows only":
            return IntakeOutcome(action="search", normalized_question=f"{pending_question} bows only")

    if text == "guns":
        return IntakeOutcome(action="guide", reply_text=main_module.GUIDANCE_RESPONSE)

    return IntakeOutcome(action="search", normalized_question=question)


class SmsGuidanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.phone = main_module.TEST_BYPASS_PHONE
        self.key = main_module._normalize_phone(self.phone)
        clear_pending_clarification(self.key)
        self.original_interpret = main_module.interpret_incoming_message
        main_module.interpret_incoming_message = fake_interpret

    def tearDown(self) -> None:
        main_module.interpret_incoming_message = self.original_interpret
        clear_pending_clarification(self.key)

    def test_hello_returns_guidance_without_counting_usage(self) -> None:
        calls = {"count": 0}

        def fake_increment(phone: str) -> int:
            calls["count"] += 1
            return calls["count"]

        original_increment = main_module.increment_free_question_count
        try:
            main_module.increment_free_question_count = fake_increment
            reply = main_module.build_sms_reply("+16475550123", "hello")
        finally:
            main_module.increment_free_question_count = original_increment

        self.assertIn("Ask an Ontario hunting question", reply)
        self.assertEqual(calls["count"], 0)

    def test_help_returns_guidance(self) -> None:
        reply = main_module.build_sms_reply("+16475550123", "help")
        self.assertIn("exact quote from the official summary", reply)

    def test_standalone_method_fragment_returns_guidance(self) -> None:
        reply = main_module.build_sms_reply("+16475550123", "guns")
        self.assertIn("Ask an Ontario hunting question", reply)

    def test_pending_thanks_repeats_clarification(self) -> None:
        first = main_module.build_sms_reply(self.phone, "when is deer season")
        self.assertIn("Need two details", first)
        reply = main_module.build_sms_reply(self.phone, "thanks")
        self.assertIn("Still need two details", reply)
        self.assertIsNotNone(get_pending_clarification(self.key))

    def test_pending_meta_message_repeats_clarification(self) -> None:
        first = main_module.build_sms_reply(self.phone, "when is deer season in WMU 65")
        self.assertIn("bows only, or guns", first)
        reply = main_module.build_sms_reply(self.phone, "what do you mean")
        self.assertIn("Still need one detail", reply)
        self.assertIsNotNone(get_pending_clarification(self.key))

    def test_state_resets_after_answer(self) -> None:
        main_module.build_sms_reply(self.phone, "when is deer season in WMU 65")
        answer = main_module.build_sms_reply(self.phone, "bows only")
        self.assertIn("p.44", answer)
        self.assertIsNone(get_pending_clarification(self.key))
        fresh = main_module.build_sms_reply(self.phone, "guns")
        self.assertIn("Ask an Ontario hunting question", fresh)

    def test_broad_species_interest_returns_natural_species_guidance(self) -> None:
        reply = main_module.build_sms_reply(self.phone, "how do I hunt deer")
        self.assertIn("Deer hunting is covered here", reply)
        self.assertIn("deer season", reply)

    def test_broad_general_interest_returns_specific_question_prompt(self) -> None:
        reply = main_module.build_sms_reply(self.phone, "what can I hunt in Ontario")
        self.assertIn("Many species are covered", reply)
        self.assertIn("duck daily bag limit", reply)


    def test_short_district_follow_up_requests_wmu_when_answer_is_too_long(self) -> None:
        first = main_module.build_sms_reply(self.phone, "duck daily limit")
        self.assertIn("reply with the district", first)
        answer = main_module.build_sms_reply(self.phone, "Southern")
        self.assertIn("reply with your WMU", answer)

    def test_wmu_follow_up_after_district_returns_shorter_answer(self) -> None:
        main_module.build_sms_reply(self.phone, "duck daily limit")
        second = main_module.build_sms_reply(self.phone, "Southern")
        self.assertIn("reply with your WMU", second)
        third = main_module.build_sms_reply(self.phone, "WMU 65")
        self.assertIn("Ontario Summary of Migratory Birds Hunting Regulations", third)
        self.assertIn("WMUs 60 to 87E", third)


if __name__ == "__main__":
    unittest.main()
