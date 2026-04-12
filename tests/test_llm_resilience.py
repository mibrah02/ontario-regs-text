from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from langchain.schema import Document

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.main as main_module
import app.rag as rag_module
from app.rag import AnswerOutcome


class _SlowResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _SlowModel:
    def __init__(self, content: str, sleep_seconds: float) -> None:
        self._content = content
        self._sleep_seconds = sleep_seconds

    def invoke(self, prompt: str):
        time.sleep(self._sleep_seconds)
        return _SlowResponse(self._content)


class _FakeVectorStore:
    def similarity_search(self, query: str, k: int = 8):
        return [
            Document(
                page_content="Deer may be hunted with equipment listed in this section.",
                metadata={"page_num": 12, "chunk_type": "paragraph", "url": rag_module.PDF_URL},
            )
        ]


class LlmResilienceTests(unittest.TestCase):
    def test_intake_timeout_falls_back_quickly(self) -> None:
        original_get_intake_model = rag_module._get_intake_model
        original_timeout = rag_module.INTAKE_TIMEOUT_SECONDS
        try:
            rag_module._get_intake_model = lambda: _SlowModel('{"action":"guide","reply_text":"late"}', 0.2)
            rag_module.INTAKE_TIMEOUT_SECONDS = 0.02
            start = time.monotonic()
            result = rag_module.interpret_incoming_message("what are daily limits for rabbits")
            elapsed = time.monotonic() - start
        finally:
            rag_module._get_intake_model = original_get_intake_model
            rag_module.INTAKE_TIMEOUT_SECONDS = original_timeout

        self.assertLess(elapsed, 0.15)
        self.assertEqual(result.action, "search")
        self.assertEqual(result.normalized_question, "what are daily limits for rabbits")

    def test_answer_timeout_returns_not_found_quickly(self) -> None:
        original_get_chat_model = rag_module._get_chat_model
        original_load_vectorstore = rag_module._load_vectorstore
        original_direct_match = rag_module._direct_structured_match
        original_rewrite = rag_module._rewrite_search_query
        original_timeout = rag_module.ANSWER_TIMEOUT_SECONDS
        try:
            rag_module._get_chat_model = lambda: _SlowModel("slow", 0.2)
            rag_module._load_vectorstore = lambda: _FakeVectorStore()
            rag_module._direct_structured_match = lambda vectorstore, question: []
            rag_module._rewrite_search_query = lambda question: question
            rag_module.ANSWER_TIMEOUT_SECONDS = 0.02
            start = time.monotonic()
            result = rag_module.answer_question_result("what calibre is allowed for deer")
            elapsed = time.monotonic() - start
        finally:
            rag_module._get_chat_model = original_get_chat_model
            rag_module._load_vectorstore = original_load_vectorstore
            rag_module._direct_structured_match = original_direct_match
            rag_module._rewrite_search_query = original_rewrite
            rag_module.ANSWER_TIMEOUT_SECONDS = original_timeout

        self.assertLess(elapsed, 0.15)
        self.assertEqual(result.kind, "not_found")
        self.assertTrue(result.text.startswith("Not found in the official Ontario hunting or Ontario migratory bird summaries."))

    def test_specific_topic_question_overrides_topic_clarification(self) -> None:
        original_get_intake_model = rag_module._get_intake_model
        try:
            rag_module._get_intake_model = lambda: _SlowModel(
                '{"action":"clarify","normalized_question":"","reply_text":"Ask one specific rabbit question","pending_question":"rabbit","expected_detail":"topic"}',
                0.0,
            )
            result = rag_module.interpret_incoming_message("what are daily limits for rabbits")
        finally:
            rag_module._get_intake_model = original_get_intake_model

        self.assertEqual(result.action, "search")
        self.assertEqual(result.normalized_question, "what are daily limits for rabbits")

    def test_answer_cache_skips_repeat_model_call(self) -> None:
        original_cache = dict(rag_module.ANSWER_CACHE)
        original_get_chat_model = rag_module._get_chat_model
        original_load_vectorstore = rag_module._load_vectorstore
        original_direct_match = rag_module._direct_structured_match
        original_rewrite = rag_module._rewrite_search_query
        try:
            calls = {"count": 0}

            class _CountingModel:
                def invoke(self, prompt: str):
                    calls["count"] += 1
                    return _SlowResponse(
                        '2026 Ontario Hunting Regulations Summary, p.12: "Cached sample." ontario.ca/hunting\nInformational only. Not legal advice. Verify current regs.'
                    )

            rag_module.ANSWER_CACHE.clear()
            rag_module._get_chat_model = lambda: _CountingModel()
            rag_module._load_vectorstore = lambda: _FakeVectorStore()
            rag_module._direct_structured_match = lambda vectorstore, question: []
            rag_module._rewrite_search_query = lambda question: question

            first = rag_module.answer_question_result("what calibre is allowed for deer")
            second = rag_module.answer_question_result("what calibre is allowed for deer")
        finally:
            rag_module.ANSWER_CACHE.clear()
            rag_module.ANSWER_CACHE.update(original_cache)
            rag_module._get_chat_model = original_get_chat_model
            rag_module._load_vectorstore = original_load_vectorstore
            rag_module._direct_structured_match = original_direct_match
            rag_module._rewrite_search_query = original_rewrite

        self.assertEqual(calls["count"], 1)
        self.assertEqual(first.text, second.text)

    def test_sms_reply_is_normalized_for_trial_segment_budget(self) -> None:
        original_interpret = main_module.interpret_incoming_message
        original_answer = main_module.answer_question_result
        try:
            main_module.interpret_incoming_message = lambda question, pending_state=None: rag_module.IntakeOutcome(
                action="search",
                normalized_question=question,
            )
            main_module.answer_question_result = lambda question: AnswerOutcome(
                text=(
                    '2026 Ontario Hunting Regulations Summary, p.96: "Cottontail and European hare 36, 37, 42–50, 53–95 Daily limit of 5 Cottontail and 3 European Hare. Possession limit of 15 of each species." '
                    'ontario.ca/hunting\nInformational only. Not legal advice. Verify current regs.'
                ),
                kind="answer",
            )
            reply = main_module.build_sms_reply(main_module.TEST_BYPASS_PHONE, "what are daily limits for rabbits")
        finally:
            main_module.interpret_incoming_message = original_interpret
            main_module.answer_question_result = original_answer

        self.assertNotIn("–", reply)
        self.assertIn("42-50", reply)
        self.assertLessEqual(main_module._estimate_sms_segments(reply), main_module.SMS_SEGMENT_LIMIT)

    def test_build_sms_reply_survives_intake_exception(self) -> None:
        original_interpret = main_module.interpret_incoming_message
        original_answer = main_module.answer_question_result
        try:
            def _raise(question: str, pending_state=None):
                raise RuntimeError("boom")

            main_module.interpret_incoming_message = _raise
            main_module.answer_question_result = lambda question: AnswerOutcome(
                text=(
                    '2026 Ontario Hunting Regulations Summary, p.12: "Sample exact sentence from PDF." '
                    'ontario.ca/hunting\nInformational only. Not legal advice. Verify current regs.'
                ),
                kind="answer",
            )
            reply = main_module.build_sms_reply(main_module.TEST_BYPASS_PHONE, "what are daily limits for rabbits")
        finally:
            main_module.interpret_incoming_message = original_interpret
            main_module.answer_question_result = original_answer

        self.assertIn("Sample exact sentence", reply)

    def test_sms_webhook_falls_back_instead_of_500(self) -> None:
        original_build = main_module.build_sms_reply
        try:
            def _raise(from_number: str, question: str) -> str:
                raise RuntimeError("boom")

            main_module.build_sms_reply = _raise
            client = TestClient(main_module.app)
            response = client.post(
                "/sms",
                data={"From": "+16475550123", "Body": "what are daily limits for rabbits", "MessageSid": "SM-resilience"},
            )
        finally:
            main_module.build_sms_reply = original_build

        self.assertEqual(response.status_code, 200)
        self.assertIn("Not found in the official Ontario hunting or Ontario migratory bird summaries.", response.text)


if __name__ == "__main__":
    unittest.main()
