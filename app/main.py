from __future__ import annotations
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
from app.rag import answer_question, ensure_index
from app.stripe import construct_event, get_checkout_url, handle_checkout_completed


load_dotenv()

FREE_QUESTION_LIMIT = 3
DISCLAIMER = "Info only. Not legal advice. Verify current regs."
NOT_FOUND_RESPONSE = "Not found in 2026 Summary. Check ontario.ca or call MNRF 1-800-667-1940. Info only. Not legal advice. Verify current regs."
TEST_BYPASS_PHONE = os.getenv("TEST_BYPASS_PHONE", "+16472626664")
PAYWALL_TEST_PREFIX = "PAYWALL "
SEEDED_PAID_NUMBERS = {item.strip() for item in os.getenv("SEEDED_PAID_NUMBERS", "").split(",") if item.strip()}
CLARIFICATION_PREFIX = "Your question matches multiple deer season entries in the 2026 Summary."
METHOD_REPLY_TERMS = ("bow", "bows only", "rifle", "rifles", "shotgun", "shotguns", "muzzle", "muzzle-loading", "gun", "guns")


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


def _question_with_clarification(from_number: str, question: str) -> str:
    pending = get_pending_clarification(_normalize_phone(from_number))
    if pending and _looks_like_method_reply(question):
        clear_pending_clarification(_normalize_phone(from_number))
        return f"{pending} {question}"
    return question


def _track_clarification(from_number: str, original_question: str, reply_text: str) -> None:
    key = _normalize_phone(from_number)
    if reply_text.startswith(CLARIFICATION_PREFIX):
        set_pending_clarification(key, original_question)
    else:
        clear_pending_clarification(key)


def _safe_answer(from_number: str, question: str) -> str:
    effective_question = _question_with_clarification(from_number, question)
    try:
        reply_text = answer_question(effective_question)
    except Exception:
        reply_text = NOT_FOUND_RESPONSE
    _track_clarification(from_number, effective_question, reply_text)
    return reply_text


def build_sms_reply(from_number: str, question: str) -> str:
    if question.startswith(PAYWALL_TEST_PREFIX):
        try:
            checkout_url = get_checkout_url(from_number)
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
        checkout_url = get_checkout_url(from_number)
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
