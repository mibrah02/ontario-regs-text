from __future__ import annotations
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from dotenv import load_dotenv
from twilio.twiml.messaging_response import MessagingResponse

from app.db import init_db, is_paid_user
from app.rag import answer_question, ensure_index
from app.stripe import construct_event, get_checkout_url, handle_checkout_completed


load_dotenv()

FREE_QUESTION_LIMIT = 3
free_question_counts: defaultdict[str, int] = defaultdict(int)
DISCLAIMER = "Informational only. Not legal advice. Verify current regs."


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    ensure_index()
    yield


app = FastAPI(title="Ontario Regs Text", lifespan=lifespan)


def build_sms_reply(from_number: str, question: str) -> str:
    if is_paid_user(from_number):
        return answer_question(question)

    free_question_counts[from_number] += 1
    if free_question_counts[from_number] <= FREE_QUESTION_LIMIT:
        return answer_question(question)

    checkout_url = get_checkout_url(from_number)
    return f"First 3 questions free. Unlimited: {checkout_url} {DISCLAIMER}"


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

    if not from_number:
        raise HTTPException(status_code=400, detail="Missing From number.")
    if not question:
        raise HTTPException(status_code=400, detail="Missing SMS body.")

    reply_text = build_sms_reply(from_number, question)
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

    if event["type"] == "checkout.session.completed":
        handle_checkout_completed(event["data"]["object"])

    return JSONResponse({"received": True})
