from __future__ import annotations

import json
import os
import re
import unicodedata
from contextlib import asynccontextmanager
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
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
NOT_FOUND_RESPONSE = "Not found in the official Ontario hunting or Ontario migratory bird summaries. Check ontario.ca or canada.ca. Info only. Not legal advice. Verify current regs."
GUIDANCE_RESPONSE = (
    "Ask an Ontario hunting question, for example: hunter orange requirement, deer season WMU 65 bows only, rabbit daily limit, or duck daily bag limit in the Southern District. "
    "I reply with the exact quote from the official summary. Info only. Not legal advice. Verify current regs."
)
TEST_BYPASS_PHONE = os.getenv("TEST_BYPASS_PHONE", "+16472626664")
PAYWALL_TEST_PREFIX = "PAYWALL "
SEEDED_PAID_NUMBERS = {item.strip() for item in os.getenv("SEEDED_PAID_NUMBERS", "").split(",") if item.strip()}
SMS_SEGMENT_LIMIT = max(1, int(os.getenv("SMS_SEGMENT_LIMIT", "2")))
SMS_TRANSLATION_TABLE = str.maketrans(
    {
        "\u00a0": " ",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
    }
)
STOPWORDS = {
    "what",
    "when",
    "where",
    "which",
    "who",
    "can",
    "i",
    "are",
    "is",
    "the",
    "for",
    "in",
    "of",
    "to",
    "my",
    "on",
    "and",
    "a",
    "an",
    "do",
    "does",
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


def _normalize_sms_transport(text: str) -> str:
    text = text.translate(SMS_TRANSLATION_TABLE)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return text.strip()


def _estimate_sms_segments(text: str) -> int:
    normalized = _normalize_sms_transport(text)
    length = len(normalized)
    if length == 0:
        return 1
    if length <= 160:
        return 1
    return (length + 152) // 153


def _answer_components(reply_text: str) -> tuple[str, str, str] | None:
    normalized = reply_text.strip()
    first_quote = normalized.find('"')
    last_quote = normalized.rfind('"')
    if first_quote == -1 or last_quote <= first_quote:
        return None
    prefix = normalized[:first_quote]
    quote = normalized[first_quote + 1:last_quote]
    suffix = normalized[last_quote + 2:].strip()
    if ', p.' not in prefix or not quote or not suffix:
        return None
    return prefix, quote, suffix


def _question_keywords(question: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9]+", question.lower())
        if len(word) >= 4 and word not in STOPWORDS
    }


def _trim_fragment_to_relevant_start(question: str, fragment: str) -> str:
    lowered_question = question.lower()
    lowered_fragment = fragment.lower()
    priority_phrases: list[str] = []
    if "limit" in lowered_question:
        priority_phrases.extend(["daily limit", "daily limits", "possession limit", "possession limits"])
    if "season" in lowered_question:
        priority_phrases.extend(["open season", "season"])
    if "orange" in lowered_question:
        priority_phrases.append("orange")
    if "tag" in lowered_question:
        priority_phrases.append("tag")
    if "wmu" in lowered_question:
        priority_phrases.append("wmu")

    for phrase in priority_phrases:
        index = lowered_fragment.find(phrase)
        if index != -1:
            return fragment[index:].strip()
    return fragment.strip()


def _shorten_answer_quote(question: str, quote: str) -> str:
    keywords = _question_keywords(question)
    if not keywords:
        return quote.strip()

    fragments = [fragment.strip() for fragment in re.split(r"(?<=\.)\s+", quote) if fragment.strip()]
    if len(fragments) <= 1:
        return _trim_fragment_to_relevant_start(question, quote)

    selected: list[str] = []
    for fragment in fragments:
        lowered = fragment.lower()
        overlap = sum(1 for keyword in keywords if keyword in lowered)
        if overlap > 0:
            trimmed = _trim_fragment_to_relevant_start(question, fragment)
            if trimmed and trimmed not in selected:
                selected.append(trimmed)

    if not selected:
        return _trim_fragment_to_relevant_start(question, quote)
    return " ".join(selected).strip()


def _fit_reply_to_sms_limit(question: str, reply_text: str) -> str:
    normalized = _normalize_sms_transport(reply_text)
    if _estimate_sms_segments(normalized) <= SMS_SEGMENT_LIMIT:
        return normalized

    parts = _answer_components(normalized)
    if not parts:
        return normalized

    prefix, quote, suffix = parts
    shortened_quote = _shorten_answer_quote(question, quote)
    candidate = _normalize_sms_transport(f'{prefix}"{shortened_quote}" {suffix}')
    if _estimate_sms_segments(candidate) <= SMS_SEGMENT_LIMIT:
        return candidate

    if "." in shortened_quote:
        first_sentence = shortened_quote.split(".", 1)[0].strip()
        if first_sentence:
            fallback = _normalize_sms_transport(f'{prefix}"{first_sentence}." {suffix}')
            if _estimate_sms_segments(fallback) <= SMS_SEGMENT_LIMIT:
                return fallback

    return candidate


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
            return _fit_reply_to_sms_limit(question, f"First 3 free. Unlimited: {_payment_entry_url(from_number)}")
        except Exception:
            return _fit_reply_to_sms_limit(question, "Free limit reached. Payment link unavailable. Try again later.")

    pending_state = _pending_state(from_number)
    intake = _safe_intake(question, pending_state)

    if intake.action in {"guide", "clarify"}:
        _track_intake(from_number, intake)
        return _fit_reply_to_sms_limit(question, intake.reply_text or GUIDANCE_RESPONSE)

    search_question = intake.normalized_question or question
    clear_pending_clarification(_normalize_phone(from_number))

    if _is_test_bypass_number(from_number):
        return _fit_reply_to_sms_limit(question, _safe_answer(from_number, search_question))

    if _is_seeded_paid_number(from_number) or is_paid_user(from_number):
        return _fit_reply_to_sms_limit(question, _safe_answer(from_number, search_question))

    free_count = increment_free_question_count(_normalize_phone(from_number))
    if free_count <= FREE_QUESTION_LIMIT:
        return _fit_reply_to_sms_limit(question, _safe_answer(from_number, search_question))

    try:
        get_checkout_url(from_number)
        return _fit_reply_to_sms_limit(question, f"First 3 free. Unlimited: {_payment_entry_url(from_number)}")
    except Exception:
        return _fit_reply_to_sms_limit(question, "Free limit reached. Payment link unavailable. Try again later.")


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
        reply_text = _fit_reply_to_sms_limit(question, NOT_FOUND_RESPONSE)

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
