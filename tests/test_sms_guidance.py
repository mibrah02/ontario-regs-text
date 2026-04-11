from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.main as main_module
from app.db import clear_pending_clarification, get_pending_clarification


class SmsGuidanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.phone = main_module.TEST_BYPASS_PHONE
        self.key = main_module._normalize_phone(self.phone)
        clear_pending_clarification(self.key)

    def tearDown(self) -> None:
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

        self.assertIn("Ask an Ontario hunting regs question", reply)
        self.assertEqual(calls["count"], 0)

    def test_help_returns_guidance(self) -> None:
        reply = main_module.build_sms_reply("+16475550123", "help")
        self.assertIn("exact quote from the 2026 Summary", reply)

    def test_standalone_method_fragment_returns_guidance(self) -> None:
        reply = main_module.build_sms_reply("+16475550123", "guns")
        self.assertIn("Ask an Ontario hunting regs question", reply)

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
        self.assertIn("Ask an Ontario hunting regs question", fresh)


if __name__ == "__main__":
    unittest.main()
