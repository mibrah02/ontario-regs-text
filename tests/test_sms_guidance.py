from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.main as main_module


class SmsGuidanceTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
