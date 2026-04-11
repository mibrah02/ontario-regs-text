from __future__ import annotations
import json
import os
import re
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
from app.rag import AnswerOutcome, answer_question_result, ensure_index
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
METHOD_REPLY_TERMS = ("bow", "bows only", "rifle", "rifles", "shotgun", "shotguns", "muzzle", "muzzle-loading", "gun", "guns")
WMU_REPLY_RE = re.compile(r"\b(?:wmu\s*)?\d{1,3}[a-z]?\b", re.IGNORECASE)
GUIDANCE_TERMS = {
    "hi",
    "hello",
    "hey",
    "help",
    "start",
    "menu",
    "info",
    "usage",
    "how do i use this",
    "how to use this",
    "what can you do",
    "what do you do",
}
ACK_TERMS = {
    "thanks",
    "thank you",
    "thx",
    "ok",
    "okay",
    "cool",
    "got it",
    "lol",
    "haha",
}
META_TERMS = {
    "what do you mean",
    "which one",
    "can you clarify",
    "clarify",
    "not sure",
    "i dont know",
    "i do not know",
}


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


def _looks_like_method_reply(question: str) -> bool:
    lowered = question.lower().strip()
    return any(term in lowered for term in METHOD_REPLY_TERMS)


def _normalized_message_key(question: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", question.lower()).strip()


def _is_guidance_message(question: str) -> bool:
    normalized = _normalized_message_key(question)
    return normalized in GUIDANCE_TERMS


def _is_ack_message(question: str) -> bool:
    normalized = _normalized_message_key(question)
    return normalized in ACK_TERMS


def _is_meta_message(question: str) -> bool:
    normalized = _normalized_message_key(question)
    return normalized in META_TERMS


def _is_context_free_fragment(question: str) -> bool:
    normalized = _normalized_message_key(question)
    if not normalized:
        return True
    if normalized in METHOD_REPLY_TERMS:
        return True
    if normalized.startswith("wmu ") or bool(re.fullmatch(r"(?:wmu\s*)?\d{1,3}[a-z]?", normalized)):
        return True
    return False


def _clarification_reminder(pending_state: dict[str, str]) -> str:
    expected_detail = pending_state.get("expected_detail", "method")
    if expected_detail == "wmu":
        return "Still need one detail: reply with the WMU number, for example WMU 65. Informational only. Not legal advice. Verify current regs."
    if expected_detail == "wmu_and_method":
        return "Still need two details: reply with the WMU number and method, for example WMU 65 bows only. Informational only. Not legal advice. Verify current regs."
    return "Still need one detail: reply with bows only, or guns. Informational only. Not legal advice. Verify current regs."


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


def _looks_like_pending_follow_up(question: str, pending_state: dict[str, str]) -> bool:
    expected_detail = pending_state.get("expected_detail", "method")
    stripped = question.strip()

    if expected_detail == "method":
        return _looks_like_method_reply(stripped)
    if expected_detail == "wmu":
        return bool(WMU_REPLY_RE.search(stripped))
    if expected_detail == "wmu_and_method":
        return bool(WMU_REPLY_RE.search(stripped)) or _looks_like_method_reply(stripped)
    return False


def _question_with_clarification(from_number: str, question: str) -> str:
    raw_pending = get_pending_clarification(_normalize_phone(from_number))
    pending_state = _deserialize_pending_state(raw_pending)
    if pending_state and _looks_like_pending_follow_up(question, pending_state):
        clear_pending_clarification(_normalize_phone(from_number))
        return f"{pending_state['question']} {question}".strip()
    return question


def _pending_state(from_number: str) -> dict[str, str] | None:
    raw_pending = get_pending_clarification(_normalize_phone(from_number))
    return _deserialize_pending_state(raw_pending)


def _should_repeat_clarification(question: str, pending_state: dict[str, str] | None) -> bool:
    if not pending_state:
        return False
    if _looks_like_pending_follow_up(question, pending_state):
        return False
    return _is_ack_message(question) or _is_guidance_message(question) or _is_meta_message(question) or _is_context_free_fragment(question)


def _track_clarification(from_number: str, outcome: AnswerOutcome) -> None:
    key = _normalize_phone(from_number)
    if outcome.kind == "clarify" and outcome.pending_question:
        set_pending_clarification(key, _serialize_pending_state(outcome.pending_question, outcome.expected_detail))
    else:
        clear_pending_clarification(key)


def _safe_answer(from_number: str, question: str) -> str:
    pending_state = _pending_state(from_number)
    if _should_repeat_clarification(question, pending_state):
        return _clarification_reminder(pending_state)

    effective_question = _question_with_clarification(from_number, question)
    try:
        outcome = answer_question_result(effective_question)
    except Exception:
        outcome = AnswerOutcome(text=NOT_FOUND_RESPONSE, kind="not_found")
    _track_clarification(from_number, outcome)
    return outcome.text


def build_sms_reply(from_number: str, question: str) -> str:
    pending_state = _pending_state(from_number)
    if (_is_guidance_message(question) or _is_ack_message(question) or _is_meta_message(question)) and not pending_state:
        return GUIDANCE_RESPONSE

    if _is_context_free_fragment(question) and not pending_state:
        return GUIDANCE_RESPONSE

    if question.startswith(PAYWALL_TEST_PREFIX):
        try:
            get_checkout_url(from_number)
            return f"First 3 free. Unlimited: {_payment_entry_url(from_number)}"
        except Exception:
            return "Free limit reached. Payment link unavailable. Try again later."

    if _is_test_bypass_number(from_number):
        return _safe_answer(from_number, question)

    if _is_seeded_paid_number(from_number) or is_paid_user(from_number):
        return _safe_answer(from_number, question)

    free_count = increment_free_question_count(_normalize_phone(from_number))
    if free_count <= FREE_QUESTION_LIMIT:
        return _safe_answer(from_number, question)

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
        cached_reply = get_cached_sms_reply(message_sid)
        if cached_reply:
            twiml = MessagingResponse()
            twiml.message(cached_reply)
            return PlainTextResponse(str(twiml), media_type="application/xml")

    reply_text = build_sms_reply(from_number, question)

    if message_sid:
        cache_sms_reply(message_sid, _normalize_phone(from_number), question, reply_text)

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
