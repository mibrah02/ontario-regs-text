# Ontario Regs Text

Production-ready SMS bot for Ontario hunting regulation questions. The bot only returns exact quotes from the 2026 Ontario Hunting Regulations Summary PDF with a page reference and disclaimer.

## What is included

- FastAPI app with `/sms`, `/stripe`, and `/health`
- PDF-to-FAISS indexing pipeline using LangChain + OpenAI embeddings
- SQLAlchemy-backed app state with SQLite by default and Postgres-ready configuration
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

The answer path is locked to this exact system prompt:

```text
You are a search tool for the 2026 Ontario Hunting Regulations Summary only.
Never summarize, never interpret, never use outside knowledge.
User question: {question}
Relevant pages: {pages}
If the answer is clearly on these pages, reply with EXACTLY this format:
'2026 Ontario Hunting Regulations Summary, p.{page}: "{exact sentence from PDF}" ontario.ca/hunting
Informational only. Not legal advice. Verify current regs.'
If not found, reply: 'Not found in 2026 Summary. Check ontario.ca or call MNRF 1-800-667-1940. Informational only. Not legal advice. Verify current regs.'
Do not add anything else.
```

## Local setup

1. Install Python 3.11.
2. Create and activate a virtualenv.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Copy `.env.example` to `.env` and fill in the values.
5. The repo already expects the PDF at `data/2026-ontario-hunting-regulations-summary.pdf`.
6. If you replace that file, keep the same filename or update `SOURCE_PDF_PATH`.
7. Build the FAISS index:

```bash
python -c "from app.rag import build_index; build_index()"
```

This writes `data/index.faiss` and `data/index.pkl`.

8. Start the app:

```bash
uvicorn app.main:app --reload
```

9. Test locally:

```bash
python tests/simulate_sms.py
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
   - `FAISS_INDEX_DIR=data`
   - `DATABASE_URL=sqlite+pysqlite:///data/app.db`
   - `SQLITE_DB_PATH=data/app.db`
   - `CLARIFICATION_TTL_MINUTES=30`
5. In Railway shell or predeploy command, build the index once:

```bash
python -c "from app.rag import build_index; build_index()"
```

6. Deploy.
7. Copy the generated public URL and paste it into Twilio and Stripe.
8. Update `BASE_URL` in Railway to that exact public URL and redeploy once.

## Local curl test for `/sms`

```bash
curl -X POST http://127.0.0.1:8000/sms \
  -d "From=+12895550123" \
  -d "Body=What calibre is allowed for deer?"
```

## Notes

- Free usage and clarification state are stored in the database.
- The default database is SQLite, but the app is wired to accept a Postgres `DATABASE_URL`.
- The bot never uses outside knowledge by design.
- If retrieval is unclear, it must return the `Not found in 2026 Summary...` fallback.
- Stripe webhook setup can wait until after deployment, when you have the real public `/stripe` URL.
