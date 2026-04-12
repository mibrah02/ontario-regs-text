# Ontario Regs Text

Production-ready SMS bot for Ontario hunting regulation questions. The bot only returns exact quotes from the official Ontario hunting summary and the official Ontario migratory birds summary, with page references and disclaimers.

## What is included

- FastAPI app with `/sms`, `/stripe`, and `/health`
- Multi-source PDF-to-FAISS indexing pipeline using LangChain + OpenAI embeddings
- LLM-based SMS intake layer that interprets greetings, vague questions, and follow-ups before retrieval
- SQLAlchemy-backed app state with SQLite by default and Railway/Postgres-ready configuration
- Idempotent Twilio SMS caching and Stripe webhook processing
- Stripe Checkout subscription flow for `$49 CAD / year`
- Local SMS webhook simulation script
- Railway-friendly startup config via `Procfile`

## App structure

```text
/app
  main.py
  rag.py
  db.py
  stripe.py
/.env.example
/requirements.txt
/README.md
/tests/simulate_sms.py
/data
```

## Liability guardrail

The answer path is locked to a source-aware quote-only prompt in [`app/rag.py`](/Users/mohamedi/Documents/ontario-regs-text/app/rag.py). It only allows exact quotes from the official indexed sources and formats answers like:

```text
{source_title}, p.{page}: "{exact sentence from PDF}" {citation_url}
Informational only. Not legal advice. Verify current regs.
```

If no matching quote is found in the indexed official sources, it returns the safe fallback.

## Local setup

1. Install Python 3.11.
2. Create and activate a virtualenv.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Copy `.env.example` to `.env` and fill in the values.
5. The repo expects these PDFs:
   - `data/2026-ontario-hunting-regulations-summary.pdf`
   - `data/ontario-migratory-birds-2025-2026.pdf`
6. If you replace either file, keep the same filenames or update `SOURCE_PDF_PATH` and `FEDERAL_WATERFOWL_PDF_PATH`.
7. Build the FAISS index:

```bash
python -c "from app.rag import build_index; build_index()"
```

This writes `data/index.faiss` and `data/index.pkl` using both official sources when both PDFs are present.

8. Start the app:

```bash
uvicorn app.main:app --reload
```

9. Test locally:

```bash
python tests/simulate_sms.py
python3 -m unittest discover -s tests -p "test_*.py"
```

## Twilio setup you must do

1. Create a Twilio account.
2. Buy a local Ontario SMS-capable number, ideally `289` or `905`.
3. In Twilio Console, open the phone number configuration.
4. Set the incoming message webhook to:

```text
https://<your-domain>/sms
```

5. Use HTTP `POST`.

## Stripe setup you must do

1. Create a Stripe account.
2. Create a product named `Ontario Regs Text`.
3. Create a recurring yearly price for `49 CAD`.
4. Copy the `Price ID` into `STRIPE_PRICE_ID`.
5. Create a webhook endpoint pointing to:

```text
https://<your-domain>/stripe
```

6. Subscribe to the `checkout.session.completed` event.
7. Copy the webhook signing secret into `STRIPE_WEBHOOK_SECRET`.
8. Copy your secret API key into `STRIPE_SECRET_KEY`.

## Railway deploy

1. Push this repo to GitHub.
2. Create a new Railway project from that GitHub repo.
3. Railway will detect the `Procfile`.
4. Add these environment variables in Railway:
   - `OPENAI_API_KEY`
   - `STRIPE_SECRET_KEY`
   - `STRIPE_PRICE_ID`
   - `STRIPE_WEBHOOK_SECRET`
   - `TWILIO_ACCOUNT_SID`
   - `TWILIO_AUTH_TOKEN`
   - `BASE_URL`
   - `SOURCE_PDF_PATH=data/2026-ontario-hunting-regulations-summary.pdf`
   - `FEDERAL_WATERFOWL_PDF_PATH=data/ontario-migratory-birds-2025-2026.pdf`
   - `FAISS_INDEX_DIR=data`
   - `DATABASE_URL=sqlite+pysqlite:///data/app.db`
   - `SQLITE_DB_PATH=data/app.db`
   - `CLARIFICATION_TTL_MINUTES=30`
   - `INTAKE_MODEL=gpt-4o-mini`
   - `REWRITE_MODEL=gpt-4o-mini`
   - `ANSWER_MODEL=gpt-4o`
   - `INTAKE_TIMEOUT_SECONDS=3.5`
   - `REWRITE_TIMEOUT_SECONDS=2.5`
   - `ANSWER_TIMEOUT_SECONDS=5.5`
   - `OPENAI_MAX_RETRIES=1`
   - `MODEL_CALL_WORKERS=4`
   - `SMS_SEGMENT_LIMIT=2`
5. In Railway shell or predeploy command, build the index once:

```bash
python -c "from app.rag import build_index; build_index()"
```

6. Deploy.
7. Copy the generated public URL and paste it into Twilio and Stripe.
8. Update `BASE_URL` in Railway to that exact public URL and redeploy once.

## Migrating to Postgres later

When you are ready to replace the default SQLite state with Railway Postgres:

1. Create a Railway Postgres service and copy its connection URL.
2. In Railway Variables, set `DATABASE_URL` to that Postgres URL.
3. Keep `SQLITE_DB_PATH` unchanged for local fallback.
4. To migrate existing local/SQLite state into Postgres, run:

```bash
cd "/Users/mohamedi/Documents/ontario-regs-text"
TARGET_DATABASE_URL="postgresql://..." python3 scripts/migrate_state.py
```

The migration copies:
- paid users
- free usage counters
- pending clarification state
- cached inbound SMS replies
- processed Stripe events

After migration, redeploy Railway so the app starts using Postgres for live state.

## Local curl test for `/sms`

```bash
curl -X POST http://127.0.0.1:8000/sms \
  -d "From=+12895550123" \
  -d "Body=What calibre is allowed for deer?"
```

## Notes

- Free usage, clarification state, inbound SMS caching, and processed Stripe events are stored in the database.
- Incoming SMS first go through a small intent-model call so the bot can interpret natural messages before quote retrieval.
- Intake, rewrite, and final answer model calls all have bounded deadlines and fail closed so the SMS webhook returns a reply instead of hanging.
- Outbound SMS replies are normalized to carrier-friendly ASCII and fit to a segment budget so Twilio Trial accounts do not drop long Unicode replies.
- The app accepts Railway-style `postgres://` or `postgresql://` values and normalizes them to SQLAlchemy `postgresql+psycopg://`.
- Duplicate Twilio deliveries reuse the cached reply instead of recomputing retrieval or incrementing usage.
- Duplicate Stripe webhook events are ignored safely after the first successful processing.
- The bot never uses outside knowledge by design.
- Waterfowl / migratory bird answers are sourced from the official federal Ontario migratory bird summary, not the Ontario provincial hunting summary.
- If retrieval is unclear, it must return the safe fallback for the official indexed summaries.
- Stripe webhook setup can wait until after deployment, when you have the real public `/stripe` URL.
