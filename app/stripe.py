from __future__ import annotations

import os

from dotenv import load_dotenv
import stripe as stripe_sdk

from app.db import mark_paid


load_dotenv()

stripe_sdk.api_key = os.getenv("STRIPE_SECRET_KEY", "")


def get_checkout_url(phone: str) -> str:
    base_url = os.environ["BASE_URL"].rstrip("/")
    session = stripe_sdk.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": os.environ["STRIPE_PRICE_ID"], "quantity": 1}],
        success_url=f"{base_url}/success?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{base_url}/cancel",
        client_reference_id=phone,
        metadata={"phone": phone},
        allow_promotion_codes=False,
    )
    return session.url


def handle_checkout_completed(session: dict) -> None:
    phone = session.get("client_reference_id") or session.get("metadata", {}).get("phone")
    if not phone:
        raise ValueError("Stripe session missing client_reference_id/phone metadata.")
    mark_paid(phone)


def construct_event(payload: bytes, signature: str) -> stripe_sdk.Event:
    webhook_secret = os.environ["STRIPE_WEBHOOK_SECRET"]
    return stripe_sdk.Webhook.construct_event(payload, signature, webhook_secret)
