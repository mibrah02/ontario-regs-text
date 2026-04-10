from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.main as main_module


def main() -> None:
    # Keep the test local and deterministic by replacing networked services.
    main_module.answer_question = lambda question: (
        '2026 Ontario Hunting Regulations Summary, p.12: '
        '"Sample exact sentence from PDF." ontario.ca/hunting\n'
        "Informational only. Not legal advice. Verify current regs."
    )
    main_module.get_checkout_url = lambda phone: (
        f"https://checkout.stripe.com/pay/demo?client_reference_id={phone}"
    )
    main_module.is_paid_user = lambda phone: False
    main_module.get_pending_clarification = lambda phone: None
    main_module.set_pending_clarification = lambda phone, question: None
    main_module.clear_pending_clarification = lambda phone: None

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
            },
        )
        print(f"Attempt {attempt} status: {response.status_code}")
        print(response.text)
        print("-" * 40)


if __name__ == "__main__":
    main()
