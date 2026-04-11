# Ontario Regs Text V2 Architecture Plan

## Goal

Build a production SMS product that feels conversational, stays liability-safe, and preserves strong margins.

Product goals:

- natural user experience over SMS
- exact quoted output from the Ontario source
- short replies that minimize SMS segments
- low OpenAI cost per resolved question
- robust state handling across deploys and scale
- easy monitoring of fallback rate, answer rate, and conversion rate

## What V1 got right

- Fast path from idea to working SMS prototype
- strict quote-first safety posture
- Twilio, Stripe, Railway, and OpenAI are already connected
- PDF ingestion and vector search are in place
- ambiguity is starting to be handled instead of guessed

## What V1 gets wrong architecturally

### 1. Too much intelligence happens inside each live SMS request

Current request path does all of this in real time:

- question interpretation
- vector search
- reranking
- ambiguity handling
- final answer generation

That increases:

- latency
- OpenAI cost
- failure surface
- variation in output quality

### 2. Conversation state is in memory

Current state such as:

- free question counts
- pending clarification context

is stored in Python memory.

That breaks if:

- Railway restarts
- a second instance is added
- a deploy happens in the middle of a conversation

### 3. The retrieval layer is still too document-shaped

Even after improvements, the core source still starts from page text extraction.

For production, the real object model should be:

- rules
- season rows
- definitions
- requirements
- prohibitions
- unit-specific records

not just:

- PDF page chunks

### 4. SQLite is too small for production state

SQLite is acceptable for prototype billing flags, but weak for:

- multi-instance app servers
- durable clarification state
- analytics
- auditability
- retry-safe webhook processing

### 5. The system is still too LLM-dependent in the hot path

The LLM should help with:

- query rewrite
- ambiguous intent classification
- final exact-quote packaging

The LLM should not be the primary reasoning engine for every live request.

## V2 design principles

### 1. Structured retrieval over raw semantic retrieval

The system should search structured records first, not pages first.

### 2. Bounded convergent clarification

If the answer is ambiguous:

- ask for the exact missing detail
- store the pending context
- continue only while the conversation is clearly converging toward a specific quote

Do not drag the user into open-ended back-and-forth.

Use these guardrails:

- continue clarification only when each reply reduces uncertainty
- prefer slot-like prompts such as species, WMU, method, or resident status
- stop and guide the user to restate the question if ambiguity is not shrinking
- use `Not found` only when the user has already provided enough detail and no matching quote exists

### 3. Deterministic before generative

Use deterministic routing wherever possible:

- identify phone
- identify payment state
- identify whether the question maps to a known table/rule type
- identify whether clarification is needed and what exact detail is missing

Only then use the LLM for narrow tasks.

### 4. Minimize SMS segments

Every part of the system should optimize for:

- short clarification prompts that request only the missing detail
- one short exact answer
- one short paywall link

### 5. Fail closed safely

If confidence is low:

- ask for the missing detail when the question is under-specified, or
- return the safe fallback only when the question is specific enough and still has no matching quote

Never guess.

## Recommended V2 system

### Core components

- `API service`
  FastAPI app for SMS, Stripe, health, and admin endpoints.
- `State store`
  Postgres for users, subscriptions, conversations, clarification state, and analytics.
- `Document ingestion pipeline`
  Offline parser that converts the PDF into structured records and embeddings.
- `Search layer`
  Hybrid retrieval across structured metadata and semantic vectors.
- `Decision layer`
  Cheap classifier/router for answer, clarify, paywall, or fallback.
- `Answer layer`
  Exact-quote composer that returns the shortest compliant quote.
- `Metrics layer`
  Tracks answer quality, ambiguity rate, fallback rate, and cost.

## Proposed runtime flow

### Step 1. SMS arrives

Twilio sends:

- `From`
- `Body`

to `/sms`.

### Step 2. Load conversation and subscription state

Lookup in Postgres:

- user by phone number
- paid status
- existing pending clarification
- usage counters

### Step 3. Normalize and classify the message

Classify whether the message is:

- a fresh regulation question
- a clarification reply
- a payment/paywall interaction
- unsupported/off-topic

This classifier can be:

- rule-based first
- LLM fallback only when needed

### Step 4. If clarification is pending

If the user previously received a clarification prompt:

- merge the new reply into the pending question
- re-run retrieval with the completed question
- clear pending state if resolved

### Step 5. Retrieve from structured sources

Search in this order:

1. structured metadata filters
2. exact token hits
3. semantic vector search

If one high-confidence result exists:

- return the exact quote

If several plausible results exist and the missing detail is identifiable:

- ask a short clarification that tells the user exactly what to provide

If the question is already specific enough and nothing useful exists:

- return the fallback

### Step 6. Return the SMS reply

Possible outputs:

- exact quote
- one or more bounded clarification prompts when the conversation is converging
- payment link
- fallback

## Target conversation behavior

### Ideal direct answer

User:

`What is the hunter orange requirement in Ontario?`

Bot:

`2026 Ontario Hunting Regulations Summary, p.30: "A hunter orange garment and head cover must be worn." ontario.ca/hunting
Informational only. Not legal advice. Verify current regs.`

### Ideal clarification

User:

`When is deer season in WMU 65`

Bot:

`WMU 65 has multiple deer season entries. Reply with one: bows only, or guns. Info only. Not legal advice. Verify current regs.`

User:

`bows only`

Bot:

`2026 Ontario Hunting Regulations Summary, p.44: "65 October 1 to October 4, October 15 to November 1, November 16 to November 29, December 7 to December 31" ontario.ca/hunting
Informational only. Not legal advice. Verify current regs.`

That is the right definition of “natural chat” for this product:

- human-friendly
- minimal turns when possible
- guided follow-ups only when needed
- exact source at the end

## Clarification policy

The bot should distinguish between two very different cases.

### 1. Missing detail

The answer may exist, but the question is under-specified.

Examples of missing detail:

- species
- WMU
- method
- resident vs non-resident
- season type

In this case, do not say `Not found`.

Instead, guide the user with the exact missing detail.

Example:

- user: `when is deer season in wmu 65`
- bot: `Need one more detail: reply with bows only, or guns. Info only. Not legal advice. Verify current regs.`

### 2. Not found

The user has already provided enough detail, but the system still cannot find a matching quote in the 2026 Summary.

Only then should the bot return:

`Not found in 2026 Summary. Check ontario.ca or call MNRF 1-800-667-1940. Informational only. Not legal advice. Verify current regs.`

### Clarification stopping rules

Allow multiple follow-ups only while the conversation is converging.

Recommended guardrails:

- continue only if the candidate set is shrinking
- stop if the ambiguity does not shrink after 2 consecutive turns
- stop after 3 clarification turns max
- stop if the user reply is off-topic or too vague
- when stopping, guide the user to ask again with the specific missing fields

Preferred fallback guidance:

`Please ask again with species, WMU, and method. Info only. Not legal advice. Verify current regs.`

## Recommended data model

### `users`

- `id`
- `phone`
- `created_at`
- `last_seen_at`

### `subscriptions`

- `user_id`
- `status`
- `stripe_customer_id`
- `stripe_subscription_id`
- `paid_at`
- `expires_at`

### `usage_counters`

- `user_id`
- `free_questions_used`
- `updated_at`

### `conversation_state`

- `user_id`
- `pending_type`
- `pending_question`
- `pending_options_json`
- `expires_at`
- `updated_at`

### `messages`

- `id`
- `user_id`
- `direction`
- `body`
- `message_type`
- `created_at`

### `source_records`

- `id`
- `year`
- `page_num`
- `record_type`
- `species`
- `wmu`
- `method`
- `resident_status`
- `section`
- `source_text`
- `source_url`

### `source_embeddings`

- `source_record_id`
- `embedding`

### `webhook_events`

- `provider`
- `event_id`
- `processed_at`
- `status`

This makes webhook processing idempotent and auditable.

## Recommended ingestion model

Do not store only “page chunks.”

Store several record types:

- `paragraph_rule`
- `definition`
- `season_row`
- `exception_note`
- `licence_requirement`
- `equipment_requirement`

Each record should include metadata when possible:

- species
- WMU
- method
- resident vs non-resident
- season category
- page number

Example `season_row`:

- `species = white-tailed deer`
- `wmu = 65`
- `method = bows only`
- `resident_status = resident/non-resident`
- `source_text = exact row text`
- `page_num = 44`

This is the key shift that makes the bot scalable and robust.

## Search architecture

### Recommended retrieval strategy

Use hybrid retrieval:

- metadata filtering first
- keyword/token retrieval second
- semantic search third

Example:

Question:

`when is deer season in wmu 65`

Search pipeline:

1. detect `deer`
2. detect `WMU 65`
3. pull candidate `season_row` records for deer + WMU 65
4. if multiple methods exist, ask for clarification
5. if one method exists after clarification, quote exact row

This is better than hoping vector search will infer everything correctly.

## LLM usage in V2

Use the LLM in narrow, cheap roles:

### 1. Query rewrite

Turn user language into a better search query.

### 2. Intent classification

Decide whether the message is:

- a new question
- a clarification reply
- billing/paywall
- unsupported

### 3. Final packaging

Return the exact quote in the required output format.

Do not use the LLM as the primary source of truth.

## Cost and margin strategy

### Reduce Twilio costs

- keep replies short
- prefer one-turn answers
- clarification messages should be one segment
- payment links should be short branded links

### Reduce OpenAI costs

- use a cheap model for query rewrite and classification
- reserve stronger models only when needed
- avoid calling the final answer model if deterministic retrieval already found one exact record

### Reduce operational costs

- durable Postgres state
- idempotent Stripe/Twilio handling
- background ingestion, not runtime ingestion

## Reliability and robustness improvements

### 1. Postgres instead of SQLite

This is the biggest infrastructure improvement needed.

### 2. Idempotent webhooks

Store Stripe and Twilio event IDs to avoid duplicate processing.

### 3. Background ingestion job

Ingestion should run when the PDF changes, not during SMS requests.

### 4. Metrics and tracing

Track:

- `% direct answer`
- `% one clarification`
- `% fallback`
- `% paywall shown`
- `% paid conversion`
- average SMS segments per answer
- average OpenAI cost per question

### 5. Alerting

Alert if:

- fallback rate spikes
- Stripe webhook failures increase
- Twilio delivery failures increase
- app latency rises above threshold

## Suggested technology changes

### Keep

- FastAPI
- Twilio
- Stripe
- Railway for now
- OpenAI

### Change

- SQLite -> Postgres
- page-first indexing -> structured record ingestion
- in-memory conversation state -> Postgres conversation state

### Optional next-stage upgrades

- Railway Postgres cutover once early launch usage justifies the extra monthly cost
- pgvector instead of local FAISS
- background worker for ingestion
- short branded domain for paywall links
- admin dashboard for analytics

## Migration plan

### Phase A. Stabilize conversation UX

- move pending clarification state to Postgres
- move free-question counts to Postgres
- keep current app structure

### Phase B. Replace raw page retrieval with structured retrieval

- build ingestion script that emits structured records
- embed structured records
- add metadata filters before vector search

### Phase C. Reduce LLM usage on the hot path

- deterministic answer path for exact structured hits
- LLM only for rewrite/classification/formatting

### Phase D. Add observability

- message logs
- answer quality metrics
- billing funnel metrics

## Recommended near-term priority order

1. move conversation state and counters to Postgres
2. implement structured source records
3. introduce hybrid retrieval with metadata filters
4. shorten and standardize clarification prompts
5. add analytics and error monitoring

## Bottom line

V2 should not be “a smarter chatbot.”

It should be:

- a structured retrieval system
- with minimal conversational state
- that asks one smart follow-up when needed
- and then returns the exact quote

That is what will make it:

- more natural for users
- safer legally
- cheaper per conversation
- more scalable operationally
