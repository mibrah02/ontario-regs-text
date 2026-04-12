from __future__ import annotations
import json
import os
from urllib.parse import quote
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from dotenv import load_dotenv
from twilio.twiml.messaging_response import MessagingResponse

from app.db import (
    cache_sms_reply,
    clear_pending_clarification,
    get_cached_sms_reply,
    get_pending_clarification,
    increment_free_question_count,
    init_db,
    is_event_processed,
    is_paid_user,
    mark_event_processed,
    set_pending_clarification,
)
from app.rag import AnswerOutcome, IntakeOutcome, answer_question_result, ensure_index, interpret_incoming_message
from app.stripe import construct_event, get_checkout_url, handle_checkout_completed


load_dotenv()

FREE_QUESTION_LIMIT = 3
NOT_FOUND_RESPONSE = "Not found in 2026 Summary. Check ontario.ca or call MNRF 1-800-667-1940. Info only. Not legal advice. Verify current regs."
GUIDANCE_RESPONSE = (
    "Ask an Ontario hunting regs question, for example: hunter orange requirement, deer season WMU 65 bows only, or moose calf tag rules. "
    "I reply with the exact quote from the 2026 Summary. Info only. Not legal advice. Verify current regs."
)
TEST_BYPASS_PHONE = os.getenv("TEST_BYPASS_PHONE", "+16472626664")
PAYWALL_TEST_PREFIX = "PAYWALL "
SEEDED_PAID_NUMBERS = {item.strip() for item in os.getenv("SEEDED_PAID_NUMBERS", "").split(",") if item.strip()}


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    ensure_index()
    yield


app = FastAPI(title="Ontario Regs Text", lifespan=lifespan)


def _normalize_phone(phone: str) -> str:
    digits = "".join(ch for ch in phone if ch.isdigit())
    if len(digits) == 10:
        digits = "1" + digits
    return f"+{digits}" if digits else phone


def _is_test_bypass_number(phone: str) -> bool:
    return _normalize_phone(phone) == _normalize_phone(TEST_BYPASS_PHONE)


def _is_seeded_paid_number(phone: str) -> bool:
    normalized = _normalize_phone(phone)
    return any(normalized == _normalize_phone(item) for item in SEEDED_PAID_NUMBERS)


def _payment_entry_url(phone: str) -> str:
    base_url = os.environ["BASE_URL"].rstrip("/")
    return f"{base_url}/buy/{quote(_normalize_phone(phone), safe='')}"


def _serialize_pending_state(question: str, expected_detail: str | None) -> str:
    return json.dumps({"question": question, "expected_detail": expected_detail}, ensure_ascii=True)


def _deserialize_pending_state(raw: str | None) -> dict[str, str] | None:
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"question": raw, "expected_detail": "method"}
    if not isinstance(payload, dict):
        return None
    question = str(payload.get("question", "")).strip()
    expected_detail = str(payload.get("expected_detail", "")).strip() or "method"
    if not question:
        return None
    return {"question": question, "expected_detail": expected_detail}


def _pending_state(from_number: str) -> dict[str, str] | None:
    raw_pending = get_pending_clarification(_normalize_phone(from_number))
    return _deserialize_pending_state(raw_pending)


def _track_clarification(from_number: str, outcome: AnswerOutcome) -> None:
    key = _normalize_phone(from_number)
    if outcome.kind == "clarify" and outcome.pending_question:
        set_pending_clarification(key, _serialize_pending_state(outcome.pending_question, outcome.expected_detail))
    else:
        clear_pending_clarification(key)


def _track_intake(from_number: str, outcome: IntakeOutcome) -> None:
    key = _normalize_phone(from_number)
    if outcome.action == "clarify" and outcome.pending_question:
        set_pending_clarification(key, _serialize_pending_state(outcome.pending_question, outcome.expected_detail))
    else:
        clear_pending_clarification(key)


def _safe_intake(question: str, pending_state: dict[str, str] | None) -> IntakeOutcome:
    try:
        return interpret_incoming_message(question, pending_state)
    except Exception:
        return IntakeOutcome(action="search", normalized_question=question)


def _safe_answer(from_number: str, question: str) -> str:
    try:
        outcome = answer_question_result(question)
    except Exception:
        outcome = AnswerOutcome(text=NOT_FOUND_RESPONSE, kind="not_found")
    _track_clarification(from_number, outcome)
    return outcome.text


def build_sms_reply(from_number: str, question: str) -> str:
    if question.startswith(PAYWALL_TEST_PREFIX):
        try:
            get_checkout_url(from_number)
            return f"First 3 free. Unlimited: {_payment_entry_url(from_number)}"
        except Exception:
            return "Free limit reached. Payment link unavailable. Try again later."

    pending_state = _pending_state(from_number)
    intake = _safe_intake(question, pending_state)

    if intake.action in {"guide", "clarify"}:
        _track_intake(from_number, intake)
        return intake.reply_text or GUIDANCE_RESPONSE

    search_question = intake.normalized_question or question
    clear_pending_clarification(_normalize_phone(from_number))

    if _is_test_bypass_number(from_number):
        return _safe_answer(from_number, search_question)

    if _is_seeded_paid_number(from_number) or is_paid_user(from_number):
        return _safe_answer(from_number, search_question)

    free_count = increment_free_question_count(_normalize_phone(from_number))
    if free_count <= FREE_QUESTION_LIMIT:
        return _safe_answer(from_number, search_question)

    try:
        get_checkout_url(from_number)
        return f"First 3 free. Unlimited: {_payment_entry_url(from_number)}"
    except Exception:
        return "Free limit reached. Payment link unavailable. Try again later."


@app.get("/health")
async def healthcheck() -> PlainTextResponse:
    return PlainTextResponse("ok")


@app.get("/success")
async def checkout_success() -> HTMLResponse:
    return HTMLResponse(
        """
        <html>
          <body>
            <h1>Ontario Regs Text activated</h1>
            <p>Your phone number is now approved for paid replies after Stripe confirms payment.</p>
            <p>You can go back to texting your Twilio number.</p>
          </body>
        </html>
        """
    )


@app.get("/buy/{phone}")
async def buy_redirect(phone: str) -> RedirectResponse:
    return RedirectResponse(get_checkout_url(phone), status_code=303)


@app.get("/cancel")
async def checkout_cancel() -> HTMLResponse:
    return HTMLResponse(
        """
        <html>
          <body>
            <h1>Checkout canceled</h1>
            <p>No problem. Your first 3 questions are still free.</p>
          </body>
        </html>
        """
    )


@app.post("/sms")
async def sms_webhook(request: Request) -> PlainTextResponse:
    form = await request.form()
    from_number = str(form.get("From", "")).strip()
    question = str(form.get("Body", "")).strip()
    message_sid = str(form.get("MessageSid", "")).strip()

    if not from_number:
        raise HTTPException(status_code=400, detail="Missing From number.")
    if not question:
        raise HTTPException(status_code=400, detail="Missing SMS body.")

    if message_sid:
        try:
            cached_reply = get_cached_sms_reply(message_sid)
        except Exception:
            cached_reply = None
        if cached_reply:
            twiml = MessagingResponse()
            twiml.message(cached_reply)
            return PlainTextResponse(str(twiml), media_type="application/xml")

    try:
        reply_text = build_sms_reply(from_number, question)
    except Exception:
        reply_text = NOT_FOUND_RESPONSE

    if message_sid:
        try:
            cache_sms_reply(message_sid, _normalize_phone(from_number), question, reply_text)
        except Exception:
            pass

    twiml = MessagingResponse()
    twiml.message(reply_text)
    return PlainTextResponse(str(twiml), media_type="application/xml")


@app.post("/stripe")
async def stripe_webhook(request: Request) -> JSONResponse:
    payload = await request.body()
    signature = request.headers.get("stripe-signature", "")

    try:
        event = construct_event(payload, signature)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    event_id = str(event.get("id", "")).strip()
    event_type = str(event.get("type", "")).strip()
    if event_id and is_event_processed("stripe", event_id):
        return JSONResponse({"received": True, "duplicate": True})

    if event_type == "checkout.session.completed":
        handle_checkout_completed(event["data"]["object"])

    if event_id:
        mark_event_processed("stripe", event_id, event_type)

    return JSONResponse({"received": True})
