from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.main as main_module
from app.rag import AnswerOutcome, IntakeOutcome


def main() -> None:
    # Keep the test local and deterministic by replacing networked services.
    main_module.answer_question_result = lambda question: AnswerOutcome(
        text=(
            'The summary says: "Sample exact sentence from PDF." '
            '(2026 Ontario Hunting Regulations Summary, p.12, ontario.ca/hunting). '
            'Verify current regs.'
        ),
        kind="answer",
    )
    main_module.interpret_incoming_message = lambda question, pending_state=None: IntakeOutcome(
        action="search",
        normalized_question=question,
    )
    main_module.get_checkout_url = lambda phone: (
        f"https://checkout.stripe.com/pay/demo?client_reference_id={phone}"
    )
    main_module.is_paid_user = lambda phone: False
    main_module.get_pending_clarification = lambda phone: None
    main_module.set_pending_clarification = lambda phone, question: None
    main_module.clear_pending_clarification = lambda phone: None
    main_module.is_event_processed = lambda provider, event_id: False
    main_module.mark_event_processed = lambda provider, event_id, event_type: None

    cached_replies: dict[str, str] = {}

    def fake_get_cached_sms_reply(message_sid: str) -> str | None:
        return cached_replies.get(message_sid)

    def fake_cache_sms_reply(message_sid: str, phone: str, body: str, reply_text: str) -> None:
        cached_replies[message_sid] = reply_text

    main_module.get_cached_sms_reply = fake_get_cached_sms_reply
    main_module.cache_sms_reply = fake_cache_sms_reply

    counter = {"count": 0}

    def fake_increment_free_question_count(phone: str) -> int:
        counter["count"] += 1
        return counter["count"]

    main_module.increment_free_question_count = fake_increment_free_question_count

    client = TestClient(main_module.app)

    for attempt in range(1, 5):
        response = client.post(
            "/sms",
            data={
                "From": "+12895550123",
                "Body": "Can I hunt deer on Sundays in Ontario?",
                "MessageSid": f"SM{attempt}",
            },
        )
        print(f"Attempt {attempt} status: {response.status_code}")
        print(response.text)
        print("-" * 40)

    duplicate = client.post(
        "/sms",
        data={
            "From": "+12895550123",
            "Body": "Can I hunt deer on Sundays in Ontario?",
            "MessageSid": "SM4",
        },
    )
    print(f"Duplicate status: {duplicate.status_code}")
    print(duplicate.text)
    print("Counter after duplicate:", counter["count"])


if __name__ == "__main__":
    main()
