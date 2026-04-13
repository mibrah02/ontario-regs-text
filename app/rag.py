from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pypdf import PdfReader


load_dotenv()

PROVINCIAL_SOURCE_URL = "https://www.ontario.ca/document/ontario-hunting-regulations-summary"
PROVINCIAL_CITATION_URL = "ontario.ca/hunting"
FEDERAL_MIGRATORY_SOURCE_URL = "https://www.canada.ca/en/environment-climate-change/services/migratory-game-bird-hunting/regulations-provincial-territorial-summaries/ontario.html"
FEDERAL_MIGRATORY_PDF_URL = "https://www.canada.ca/content/dam/eccc/documents/pdf/oiseaux-migrateurs-migratory-bird/arcom-mbhrs/2025-2026/25038.01-CWS-MBHS-2025-2026-WEB-ON-V03-EN.pdf"
FEDERAL_MIGRATORY_CITATION_URL = "canada.ca/migratory-game-bird-hunting"
PDF_URL = PROVINCIAL_SOURCE_URL
DEFAULT_PDF_PATH = Path(os.getenv("SOURCE_PDF_PATH", "data/2026-ontario-hunting-regulations-summary.pdf"))
FEDERAL_WATERFOWL_PDF_PATH = Path(os.getenv("FEDERAL_WATERFOWL_PDF_PATH", "data/ontario-migratory-birds-2025-2026.pdf"))
INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", "data"))
NOT_FOUND_RESPONSE = (
    "Not found in the official Ontario hunting or Ontario migratory bird summaries. Check ontario.ca or canada.ca. "
    "Informational only. Not legal advice. Verify current regs."
)
INTAKE_MODEL = os.getenv("INTAKE_MODEL", "gpt-4o-mini")
REWRITE_MODEL = os.getenv("REWRITE_MODEL", "gpt-4o-mini")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4o")
INTAKE_TIMEOUT_SECONDS = float(os.getenv("INTAKE_TIMEOUT_SECONDS", "3.5"))
REWRITE_TIMEOUT_SECONDS = float(os.getenv("REWRITE_TIMEOUT_SECONDS", "2.5"))
ANSWER_TIMEOUT_SECONDS = float(os.getenv("ANSWER_TIMEOUT_SECONDS", "5.5"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "1"))
MODEL_CALL_WORKERS = max(2, int(os.getenv("MODEL_CALL_WORKERS", "4")))
MODEL_CALL_POOL = ThreadPoolExecutor(max_workers=MODEL_CALL_WORKERS)
ANSWER_CACHE_TTL_SECONDS = float(os.getenv("ANSWER_CACHE_TTL_SECONDS", "21600"))
ANSWER_CACHE_MAX_ITEMS = max(32, int(os.getenv("ANSWER_CACHE_MAX_ITEMS", "512")))
ANSWER_CACHE: dict[str, tuple[float, "AnswerOutcome"]] = {}
SYSTEM_PROMPT = """You are a search tool for the provided official Ontario hunting regulation sources only.
Never summarize, never interpret, never use outside knowledge.
User question: {question}
Relevant source pages: {pages}
Each source page includes source_title, citation_url, and page.
If the answer is clearly on these pages, return the shortest exact quote that answers the question.
If the source is a table, return only the exact matching row or cell text needed, not the whole table.
Use the provided table context, district, WMU, and method context to distinguish between different entries.
Preserve source wording, but collapse repeated spaces and line breaks into normal readable spaces.
Reply with EXACTLY this format:
'{{source_title}}, p.{{page}}: "{{exact sentence from PDF}}" {{citation_url}}
Informational only. Not legal advice. Verify current regs.'
If not found, reply: '{not_found_response}'
Do not add anything else."""
SEARCH_REWRITE_PROMPT = """Rewrite the user question into a short search query for the official Ontario hunting and Ontario migratory bird summaries.
Keep species names, WMU numbers, districts, season concepts, licence concepts, and gear terms when relevant.
Do not answer the question. Do not add commentary. Return one short search query only.
User question: {question}
"""


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    source_title: str
    source_url: str
    citation_url: str
    pdf_path: Path
    year: str
    source_kind: str


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        source_id="ontario_hunting_summary_2026",
        source_title="2026 Ontario Hunting Regulations Summary",
        source_url=PROVINCIAL_SOURCE_URL,
        citation_url=PROVINCIAL_CITATION_URL,
        pdf_path=DEFAULT_PDF_PATH,
        year="2026",
        source_kind="provincial",
    ),
    SourceSpec(
        source_id="ontario_migratory_birds_2025_2026",
        source_title="Ontario Summary of Migratory Birds Hunting Regulations, August 2025 to July 2026",
        source_url=FEDERAL_MIGRATORY_PDF_URL,
        citation_url=FEDERAL_MIGRATORY_CITATION_URL,
        pdf_path=FEDERAL_WATERFOWL_PDF_PATH,
        year="2025-2026",
        source_kind="migratory",
    ),
)

WMU_ROW_START_RE = re.compile(r"^(?:\d{1,3}[A-Z]?)(?:,\s*\d{1,3}[A-Z]?)*(?:\s*,\s*\d{1,3}[A-Z]?)*\b")
TABLE_PAGE_MARKERS = ("Wildlife Management Unit", "Resident — open season", "Resident - open season")
SKIP_LINE_RE = re.compile(
    r"^(?:Hunting Regulations Summary|White-tailed Deer|Wildlife Management Unit|Resident|Non-resident|General Regulations|\d+)$",
    re.IGNORECASE,
)
MIGRATORY_TABLE_MARKERS = (
    "Open Seasons and Daily Bag and Possession Limits for Migratory Game Birds in Ontario",
    "Special Measures for Overabundant Species in Ontario",
    "Area Species Open Season Daily Bag Limit Possession Limit",
)
MIGRATORY_HEADER_SKIP_RE = re.compile(
    r"^(?:Summary of Migratory Birds|Hunting Regulations|August 2025|to July 2026|Ontario|Open Seasons and Daily Bag and Possession Limits for Migratory Game Birds in Ontario|Area Species Open Season Daily Bag Limit Possession Limit|Special Measures for Overabundant Species in Ontario|The information presented here is a summary of the law\.|For more information, consult|Environment and Climate Change Canada|Canadian Wildlife Service|Regional Office|Tel\.:|enviroinfo@ec\.gc\.ca|Report your Migratory Bird Bands:|www\.reportband\.gov|ISSN )",
    re.IGNORECASE,
)
MIGRATORY_SPECIES_START_RE = re.compile(
    r"^(?:Ducks\b|Canada Geese\b|Snow Geese\b|Geese \(other than\b|Rails \(other than\b|Woodcock\b|Snipe\b|Mourning Doves\b)",
    re.IGNORECASE,
)
MIGRATORY_DATA_LINE_RE = re.compile(
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December|No open season|N/A|\bNo limit\b|\bin WMU|\bin WMUs|\bmunicipalities\b|\bSunday gun hunting\b)",
    re.IGNORECASE,
)
DISTRICT_PATTERNS = {
    "hudson-james bay district": "Hudson-James Bay District",
    "hudson james bay district": "Hudson-James Bay District",
    "hudson-james bay": "Hudson-James Bay District",
    "hudson james bay": "Hudson-James Bay District",
    "hudson-james": "Hudson-James Bay District",
    "northern district": "Northern District",
    "northern": "Northern District",
    "central district": "Central District",
    "central": "Central District",
    "southern district": "Southern District",
    "southern": "Southern District",
}

SPECIES_PATTERNS = {
    "white-tailed deer": "deer",
    "deer": "deer",
    "moose": "moose",
    "black bear": "bear",
    "bear": "bear",
    "wild turkey": "turkey",
    "turkey": "turkey",
    "elk": "elk",
    "wolf": "wolf",
    "coyote": "coyote",
    "rabbit": "rabbit",
    "hare": "rabbit",
    "cottontail": "rabbit",
    "snowshoe": "rabbit",
    "duck": "duck",
    "ducks": "duck",
    "waterfowl": "waterfowl",
    "migratory bird": "waterfowl",
    "migratory birds": "waterfowl",
    "canada goose": "canada goose",
    "canada geese": "canada goose",
    "cackling goose": "canada goose",
    "cackling geese": "canada goose",
    "snow goose": "snow goose",
    "snow geese": "snow goose",
    "ross's goose": "snow goose",
    "ross's geese": "snow goose",
    "goose": "goose",
    "geese": "goose",
    "mourning dove": "mourning dove",
    "woodcock": "woodcock",
    "snipe": "snipe",
    "rail": "rail",
    "rails": "rail",
    "coot": "coot",
    "coots": "coot",
    "gallinule": "gallinule",
    "gallinules": "gallinule",
}
WATERFOWL_SPECIES = {"duck", "waterfowl", "canada goose", "snow goose", "goose", "mourning dove", "woodcock", "snipe", "rail", "coot", "gallinule"}
WATERFOWL_DETAIL_TERMS = ("season", "bag", "daily", "limit", "possession", "open season")


def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


def _get_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=ANSWER_MODEL,
        temperature=0,
        timeout=ANSWER_TIMEOUT_SECONDS,
        max_retries=OPENAI_MAX_RETRIES,
    )


def _get_intake_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=INTAKE_MODEL,
        temperature=0,
        timeout=INTAKE_TIMEOUT_SECONDS,
        max_retries=OPENAI_MAX_RETRIES,
    )


def _get_rewrite_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=REWRITE_MODEL,
        temperature=0,
        timeout=REWRITE_TIMEOUT_SECONDS,
        max_retries=OPENAI_MAX_RETRIES,
    )


def _invoke_model_call(callable_obj, timeout_seconds: float):
    future = MODEL_CALL_POOL.submit(callable_obj)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError:
        future.cancel()
        return None


def _normalize_page_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def _normalize_inline(text: str) -> str:
    text = _normalize_page_text(text)
    return re.sub(r" ?\n ?", " ", text).strip()


def _normalize_model_output(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = text.strip()
    if len(text) >= 2 and text[0] == "'" and text[-1] == "'":
        text = text[1:-1].strip()
    return text


def _normalize_question_cache_key(question: str) -> str:
    lowered = question.lower().strip()
    lowered = re.sub(r"[^a-z0-9 ]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _prune_answer_cache(now: float) -> None:
    expired = [key for key, (expires_at, _) in ANSWER_CACHE.items() if expires_at <= now]
    for key in expired:
        ANSWER_CACHE.pop(key, None)
    if len(ANSWER_CACHE) <= ANSWER_CACHE_MAX_ITEMS:
        return
    overflow = len(ANSWER_CACHE) - ANSWER_CACHE_MAX_ITEMS
    for key in sorted(ANSWER_CACHE, key=lambda item: ANSWER_CACHE[item][0])[:overflow]:
        ANSWER_CACHE.pop(key, None)


@dataclass
class AnswerOutcome:
    text: str
    kind: str
    pending_question: str | None = None
    expected_detail: str | None = None


@dataclass
class IntakeOutcome:
    action: str
    normalized_question: str = ""
    reply_text: str = ""
    pending_question: str | None = None
    expected_detail: str | None = None


def _get_cached_answer(question: str):
    key = _normalize_question_cache_key(question)
    if not key:
        return None
    now = monotonic()
    cached = ANSWER_CACHE.get(key)
    if not cached:
        return None
    expires_at, outcome = cached
    if expires_at <= now:
        ANSWER_CACHE.pop(key, None)
        return None
    return AnswerOutcome(
        text=outcome.text,
        kind=outcome.kind,
        pending_question=outcome.pending_question,
        expected_detail=outcome.expected_detail,
    )


def _set_cached_answer(question: str, outcome: AnswerOutcome) -> None:
    key = _normalize_question_cache_key(question)
    if not key:
        return
    now = monotonic()
    ANSWER_CACHE[key] = (
        now + ANSWER_CACHE_TTL_SECONDS,
        AnswerOutcome(
            text=outcome.text,
            kind=outcome.kind,
            pending_question=outcome.pending_question,
            expected_detail=outcome.expected_detail,
        ),
    )
    _prune_answer_cache(now)


def _compress_repeated_table_row(text: str) -> str:
    match = WMU_ROW_START_RE.match(text)
    if not match:
        return text

    prefix = match.group(0).strip()
    remainder = text[match.end():].strip()
    tokens = remainder.split()
    if not tokens or len(tokens) % 2 != 0:
        return text

    midpoint = len(tokens) // 2
    if tokens[:midpoint] != tokens[midpoint:]:
        return text

    return f"{prefix} {' '.join(tokens[:midpoint])}".strip()


def _format_exact_quote(document: Document) -> str:
    page = document.metadata.get("page_num", "?")
    quote = _normalize_inline(document.page_content)
    if document.metadata.get("chunk_type") == "table_row":
        quote = _compress_repeated_table_row(quote)
    quote = quote.replace('"', '\\"')
    source_title = document.metadata.get("source_title", "Official Hunting Summary")
    citation_url = document.metadata.get("citation_url", PROVINCIAL_CITATION_URL)
    return (
        f'{source_title}, p.{page}: "{quote}" {citation_url}\n'
        "Informational only. Not legal advice. Verify current regs."
    )


def _extract_species_terms(question: str) -> set[str]:
    lowered = question.lower()
    return {label for pattern, label in SPECIES_PATTERNS.items() if pattern in lowered}


def _infer_species(text: str) -> str | None:
    matches = _extract_species_terms(text)
    if len(matches) == 1:
        return next(iter(matches))
    lowered = text.lower()
    for pattern, label in SPECIES_PATTERNS.items():
        if pattern in lowered:
            return label
    return None


def _extract_wmu_terms(question: str) -> set[str]:
    return {match.upper() for match in re.findall(r"\b(?:WMU\s*)?(\d{1,3}[A-Z]?)\b", question, re.IGNORECASE)}


def _extract_district_terms(question: str) -> set[str]:
    lowered = question.lower()
    return {label for pattern, label in DISTRICT_PATTERNS.items() if pattern in lowered}


def _extract_method_terms(question: str) -> set[str]:
    lowered = question.lower()
    terms = set()
    if "bows only" in lowered or "bow" in lowered:
        terms.add("bows only")
    if any(term in lowered for term in ("rifle", "rifles", "shotgun", "shotguns", "muzzle", "muzzle-loading", "gun", "guns")):
        terms.add("guns")
    return terms


def _doc_matches_wmu(document: Document, wmu_terms: set[str]) -> bool:
    content = document.page_content.upper()
    return any(re.search(rf"^\s*{re.escape(term)}(?:\b|,)", content) for term in wmu_terms)


def _doc_matches_method(document: Document, method_terms: set[str]) -> bool:
    context = document.metadata.get("table_context", "").lower()
    if not method_terms:
        return True
    if "bows only" in method_terms and "bows only" in context:
        return True
    if "guns" in method_terms and "rifles, shotguns, muzzle-loading guns and bows" in context:
        return True
    return False


def _doc_matches_district(document: Document, district_terms: set[str]) -> bool:
    if not district_terms:
        return True
    return document.metadata.get("district") in district_terms


def _doc_matches_species(document: Document, species_terms: set[str]) -> bool:
    doc_species = str(document.metadata.get("species", "")).strip().lower()
    if not species_terms:
        return True
    if doc_species in species_terms:
        return True
    if "goose" in species_terms and doc_species in {"goose", "canada goose", "snow goose"}:
        return True
    if "waterfowl" in species_terms and doc_species in WATERFOWL_SPECIES:
        return True
    return False


def _direct_structured_match(vectorstore: FAISS, question: str) -> list[Document]:
    species_terms = _extract_species_terms(question)
    wmu_terms = _extract_wmu_terms(question)
    method_terms = _extract_method_terms(question)
    district_terms = _extract_district_terms(question)

    if species_terms and wmu_terms and method_terms:
        candidates: list[Document] = []
        for document in vectorstore.docstore._dict.values():
            if document.metadata.get("chunk_type") != "table_row":
                continue
            if not _doc_matches_species(document, species_terms):
                continue
            if not _doc_matches_wmu(document, wmu_terms):
                continue
            if not _doc_matches_method(document, method_terms):
                continue
            candidates.append(document)
        return candidates[:3]

    if species_terms and district_terms:
        candidates = []
        for document in vectorstore.docstore._dict.values():
            if document.metadata.get("chunk_type") != "table_row":
                continue
            if not _doc_matches_district(document, district_terms):
                continue
            if not _doc_matches_species(document, species_terms):
                continue
            candidates.append(document)
        return candidates[:3]

    return []


def _extract_paragraph_chunks(page_text: str) -> list[str]:
    paragraphs: list[str] = []
    for block in page_text.split("\n\n"):
        cleaned = _normalize_inline(block)
        if len(cleaned) >= 80:
            paragraphs.append(cleaned)
    return paragraphs


def _looks_like_wmu_row_start(line: str) -> bool:
    if not WMU_ROW_START_RE.match(line):
        return False
    return not line.lower().startswith(("resident", "non-resident"))


def _extract_table_context(page_text: str) -> str:
    if not any(marker in page_text for marker in TABLE_PAGE_MARKERS):
        return ""

    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    context_lines: list[str] = []
    for line in lines:
        if SKIP_LINE_RE.match(line):
            continue
        if _looks_like_wmu_row_start(line):
            break
        context_lines.append(line)

    return _normalize_inline("\n".join(context_lines[:6]))


def _extract_table_row_chunks(page_text: str) -> list[str]:
    if not any(marker in page_text for marker in TABLE_PAGE_MARKERS):
        return []

    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    rows: list[str] = []
    current: list[str] = []

    for line in lines:
        if SKIP_LINE_RE.match(line):
            continue
        if line.startswith("*"):
            if current:
                rows.append(_normalize_inline("\n".join(current)))
                current = []
            continue

        if _looks_like_wmu_row_start(line):
            if current:
                rows.append(_normalize_inline("\n".join(current)))
            current = [line]
            continue

        if current:
            current.append(line)

    if current:
        rows.append(_normalize_inline("\n".join(current)))

    deduped: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row not in seen and len(row) >= 30:
            deduped.append(row)
            seen.add(row)
    return deduped


def _looks_like_migratory_area_start(line: str) -> bool:
    lowered = line.lower()
    return lowered in {
        "hudson-james bay district",
        "northern district",
        "central district",
        "southern district",
        "hudson-james",
        "hudson-james bay",
        "northern",
        "central",
        "southern",
    } or lowered.startswith("southern district (")


def _normalize_migratory_lines(page_text: str) -> list[str]:
    raw_lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    lines: list[str] = []
    index = 0
    while index < len(raw_lines):
        line = raw_lines[index]
        next_line = raw_lines[index + 1] if index + 1 < len(raw_lines) else ""
        if line in {"Hudson-James", "Hudson-James Bay", "Northern", "Central", "Southern"} and next_line in {"Bay District", "District"}:
            lines.append(f"{line} {next_line}")
            index += 2
            continue
        if line == "Hudson-James Bay" and next_line == "District":
            lines.append("Hudson-James Bay District")
            index += 2
            continue
        lines.append(line)
        index += 1
    return lines


def _extract_migratory_table_row_chunks(page_text: str) -> list[dict[str, str]]:
    if not any(marker in page_text for marker in MIGRATORY_TABLE_MARKERS):
        return []

    lines = _normalize_migratory_lines(page_text)
    rows: list[dict[str, str]] = []
    current_area_parts: list[str] = []
    species_lines: list[str] = []
    data_lines: list[str] = []

    def current_area() -> str:
        return _normalize_inline(" ".join(current_area_parts))

    def flush() -> None:
        nonlocal species_lines, data_lines
        area = current_area()
        if not area or not species_lines or not data_lines:
            species_lines = []
            data_lines = []
            return
        species_text = _normalize_inline(" ".join(species_lines))
        text = _normalize_inline("\n".join([area, species_text, *data_lines]))
        if len(text) < 30:
            species_lines = []
            data_lines = []
            return
        rows.append(
            {
                "text": text,
                "district": area,
                "species": _infer_species(species_text) or _infer_species(text) or "",
                "table_context": area,
            }
        )
        species_lines = []
        data_lines = []

    for line in lines:
        if MIGRATORY_HEADER_SKIP_RE.match(line):
            continue
        if line.startswith("Notice:"):
            continue
        if line.startswith("For more information") or line.startswith("You may also direct your questions"):
            flush()
            break

        if _looks_like_migratory_area_start(line):
            flush()
            current_area_parts = [line]
            continue

        if current_area_parts and not species_lines and not MIGRATORY_SPECIES_START_RE.match(line) and not MIGRATORY_DATA_LINE_RE.search(line):
            current_area_parts.append(line)
            continue

        if MIGRATORY_SPECIES_START_RE.match(line):
            flush()
            species_lines = [line]
            data_lines = []
            continue

        if species_lines and not data_lines and not MIGRATORY_DATA_LINE_RE.search(line):
            species_lines.append(line)
            continue

        if species_lines:
            data_lines.append(line)

    flush()

    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if row["text"] not in seen:
            deduped.append(row)
            seen.add(row["text"])
    return deduped


def _base_metadata(source: SourceSpec, page_num: int) -> dict[str, Any]:
    return {
        "page_num": page_num,
        "year": source.year,
        "url": source.source_url,
        "source_id": source.source_id,
        "source_title": source.source_title,
        "citation_url": source.citation_url,
        "source_kind": source.source_kind,
    }


def _page_documents_from_source(source: SourceSpec) -> list[Document]:
    reader = PdfReader(str(source.pdf_path))
    documents: list[Document] = []

    for index, page in enumerate(reader.pages, start=1):
        page_text = _normalize_page_text(page.extract_text() or "")
        if not page_text:
            continue

        metadata = _base_metadata(source, index)
        page_species = _infer_species(page_text)
        page_metadata = dict(metadata)
        if page_species and len(_extract_species_terms(page_text)) == 1:
            page_metadata["species"] = page_species

        documents.append(Document(page_content=page_text, metadata={**page_metadata, "chunk_type": "page"}))

        for paragraph in _extract_paragraph_chunks(page_text):
            paragraph_metadata = dict(metadata)
            paragraph_species = _infer_species(paragraph)
            if paragraph_species and len(_extract_species_terms(paragraph)) == 1:
                paragraph_metadata["species"] = paragraph_species
            elif page_species and len(_extract_species_terms(page_text)) == 1:
                paragraph_metadata["species"] = page_species
            documents.append(
                Document(
                    page_content=paragraph,
                    metadata={**paragraph_metadata, "chunk_type": "paragraph"},
                )
            )

        if source.source_kind == "provincial":
            table_context = _extract_table_context(page_text)
            for row in _extract_table_row_chunks(page_text):
                row_metadata = {**metadata, "chunk_type": "table_row"}
                if table_context:
                    row_metadata["table_context"] = table_context
                row_species = _infer_species(row) or page_species
                if row_species:
                    row_metadata["species"] = row_species
                documents.append(Document(page_content=row, metadata=row_metadata))
            continue

        for row in _extract_migratory_table_row_chunks(page_text):
            row_metadata = {
                **metadata,
                "chunk_type": "table_row",
                "table_context": row.get("table_context", ""),
                "district": row.get("district", ""),
            }
            if row.get("species"):
                row_metadata["species"] = row["species"]
            documents.append(Document(page_content=row["text"], metadata=row_metadata))

    if not documents:
        raise ValueError(f"No text could be extracted from the PDF: {source.pdf_path}")

    return documents


def _configured_sources() -> list[SourceSpec]:
    return [source for source in SOURCES if source.pdf_path.exists()]


def build_index(pdf_path: str | Path = DEFAULT_PDF_PATH) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    primary_override = Path(pdf_path)
    documents: list[Document] = []
    for source in SOURCES:
        if source.source_id == "ontario_hunting_summary_2026":
            actual_source = SourceSpec(
                source_id=source.source_id,
                source_title=source.source_title,
                source_url=source.source_url,
                citation_url=source.citation_url,
                pdf_path=primary_override,
                year=source.year,
                source_kind=source.source_kind,
            )
        else:
            actual_source = source
        if not actual_source.pdf_path.exists():
            continue
        documents.extend(_page_documents_from_source(actual_source))
    if not documents:
        raise FileNotFoundError("No source PDFs were found for indexing.")
    vectorstore = FAISS.from_documents(documents, _get_embeddings())
    vectorstore.save_local(str(INDEX_DIR))


def ensure_index() -> None:
    if (INDEX_DIR / "index.faiss").exists():
        return
    build_index()


def _load_vectorstore() -> FAISS:
    if not (INDEX_DIR / "index.faiss").exists():
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_DIR / 'index.faiss'}. Run build_index() first."
        )

    return FAISS.load_local(
        str(INDEX_DIR),
        _get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def _rewrite_search_query(question: str) -> str:
    try:
        response = _invoke_model_call(
            lambda: _get_rewrite_model().invoke(SEARCH_REWRITE_PROMPT.format(question=question)),
            REWRITE_TIMEOUT_SECONDS,
        )
        if response is None:
            return question
        rewritten = _normalize_inline(response.content)
        return rewritten or question
    except Exception:
        return question


def _extract_query_terms(question: str) -> set[str]:
    return {match.upper() for match in re.findall(r"\b[0-9]{1,3}[A-Z]?\b", question)}


def _rerank_results(question: str, documents: list[Document]) -> list[Document]:
    query_terms = _extract_query_terms(question)
    species_terms = _extract_species_terms(question)
    district_terms = _extract_district_terms(question)

    def sort_key(document: Document) -> tuple[int, int, int, int]:
        content = document.page_content.upper()
        overlap = sum(1 for term in query_terms if term in content)
        table_bonus = 1 if document.metadata.get("chunk_type") == "table_row" else 0
        species_bonus = 1 if species_terms and _doc_matches_species(document, species_terms) else 0
        district_bonus = 1 if district_terms and _doc_matches_district(document, district_terms) else 0
        return (district_bonus, species_bonus, overlap, table_bonus)

    return sorted(documents, key=sort_key, reverse=True)


INTAKE_GUIDANCE_RESPONSE = (
    "Ask an Ontario hunting question, for example: hunter orange requirement, deer season WMU 65 bows only, rabbit daily limit, or duck daily bag limit in the Southern District. "
    "I reply with the exact quote from the official summary. Info only. Not legal advice. Verify current regs."
)

INTAKE_PROMPT = """You are the intake layer for Ontario Regs Text, an SMS bot that answers Ontario hunting regulation questions.
Your job is only to understand the user's message and choose the next step.
You do not answer legal or hunting questions yourself.

Return strict JSON only with these keys:
action: one of "guide", "clarify", or "search"
normalized_question: short plain-language query for downstream retrieval, or ""
reply_text: SMS text to send if action is guide or clarify, or ""
pending_question: canonical question to store if action is clarify, or ""
expected_detail: short label like "topic", "method", "wmu", "wmu_and_method", "district", or "detail", or ""

Rules:
- Use action=search when the user asked a specific hunting-regs question, even if phrasing is messy.
- Use action=guide for greetings, chit-chat, or messages with no usable hunting-regs question.
- Use action=clarify when the message is hunting-related but still missing an important detail.
- If pending context exists and the new message clearly starts a new question, ignore the pending context.
- If pending context exists and the new message fills the missing detail, combine it and use action=search.
- If pending context exists and the new message is meta, confusion, or acknowledgement, use action=clarify and repeat the missing detail briefly.
- For broad supported-species questions like "can I hunt rabbits in Ontario", keep it natural: you may briefly confirm the species is covered in the official summaries, then ask for one specific question.
- For waterfowl season or limit questions without a district, ask for the district: Hudson-James Bay, Northern, Central, or Southern.
- Keep reply_text short and natural for SMS.
- Never answer from memory and never quote regulations yourself.

Examples:
1. pending: none | message: "hello"
{"action":"guide","normalized_question":"","reply_text":"Ask an Ontario hunting question, for example: hunter orange requirement, deer season WMU 65 bows only, rabbit daily limit, or duck daily bag limit in the Southern District. I reply with the exact quote from the official summary. Info only. Not legal advice. Verify current regs.","pending_question":"","expected_detail":""}
2. pending: {"question":"when is deer season","expected_detail":"wmu_and_method"} | message: "WMU 65"
{"action":"clarify","normalized_question":"","reply_text":"Need one more detail: reply with bows only, or guns. Informational only. Not legal advice. Verify current regs.","pending_question":"when is deer season in WMU 65","expected_detail":"method"}
3. pending: {"question":"when is deer season in WMU 65","expected_detail":"method"} | message: "bows only"
{"action":"search","normalized_question":"when is deer season in WMU 65 bows only","reply_text":"","pending_question":"","expected_detail":""}
4. pending: none | message: "what rabbits hunting rules"
{"action":"clarify","normalized_question":"","reply_text":"Rabbit hunting is covered here. Ask one specific rabbit question, for example: rabbit daily limit, rabbit possession limit, or rabbit season in your WMU. Informational only. Not legal advice. Verify current regs.","pending_question":"rabbit","expected_detail":"topic"}
5. pending: none | message: "can I hunt rabbits in Ontario"
{"action":"clarify","normalized_question":"","reply_text":"Rabbit hunting is covered here. Ask one specific rabbit question, for example: rabbit daily limit, rabbit possession limit, or rabbit season in your WMU. Informational only. Not legal advice. Verify current regs.","pending_question":"rabbit","expected_detail":"topic"}
6. pending: none | message: "duck daily limit"
{"action":"clarify","normalized_question":"","reply_text":"Need one more detail: reply with the district, for example Southern or Central District. Informational only. Not legal advice. Verify current regs.","pending_question":"duck daily limit","expected_detail":"district"}
7. pending: none | message: "duck daily limit in Southern District"
{"action":"search","normalized_question":"duck daily limit Southern District","reply_text":"","pending_question":"","expected_detail":""}
8. pending: none | message: "guns"
{"action":"guide","normalized_question":"","reply_text":"Ask an Ontario hunting question, for example: hunter orange requirement, deer season WMU 65 bows only, rabbit daily limit, or duck daily bag limit in the Southern District. I reply with the exact quote from the official summary. Info only. Not legal advice. Verify current regs.","pending_question":"","expected_detail":""}

Pending question: {pending_question}
Expected detail: {expected_detail}
Incoming message: {message}
"""


def _normalize_intake_key(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


INTAKE_SPECIFIC_TOPIC_TERMS = (
    "season",
    "limit",
    "daily",
    "bag",
    "possession",
    "wmu",
    "unit",
    "district",
    "tag",
    "licence",
    "license",
    "orange",
    "calibre",
    "caliber",
    "gun",
    "guns",
    "bow",
    "bows",
    "rifle",
    "shotgun",
    "muzzle",
    "sunday",
    "date",
    "dates",
    "resident",
    "nonresident",
    "non-resident",
    "calf",
    "antlerless",
    "shot",
)


def _message_has_specific_topic_hint(message: str) -> bool:
    lowered = message.lower()
    return any(term in lowered for term in INTAKE_SPECIFIC_TOPIC_TERMS)


def _clarification_reminder_text(expected_detail: str | None) -> str:
    if expected_detail == "wmu":
        return "Still need one detail: reply with the WMU number, for example WMU 65. Informational only. Not legal advice. Verify current regs."
    if expected_detail == "wmu_and_method":
        return "Still need two details: reply with the WMU number and method, for example WMU 65 bows only. Informational only. Not legal advice. Verify current regs."
    if expected_detail == "district":
        return "Still need one detail: reply with the district, for example Southern or Central District. Informational only. Not legal advice. Verify current regs."
    if expected_detail == "topic":
        return "Please ask again with one specific topic, for example: daily limit, possession limit, season, tag, or hunter orange. Informational only. Not legal advice. Verify current regs."
    return "Still need one detail: reply with bows only, or guns. Informational only. Not legal advice. Verify current regs."


def _looks_like_follow_up_fragment(message: str, expected_detail: str | None) -> bool:
    lowered = message.lower().strip()
    if not lowered:
        return False
    if expected_detail in {"method", "wmu_and_method"} and _question_specifies_method(lowered):
        return True
    if expected_detail in {"wmu", "wmu_and_method"} and _extract_wmu_terms(lowered):
        return True
    if expected_detail == "district" and _extract_district_terms(lowered):
        return True
    if expected_detail == "topic":
        return len(re.findall(r"\w+", lowered)) <= 6
    return False


def _looks_like_new_question(message: str, pending_question: str, expected_detail: str | None) -> bool:
    if not pending_question:
        return False
    if _looks_like_follow_up_fragment(message, expected_detail):
        return False

    lowered = message.lower().strip()
    word_count = len(re.findall(r"\w+", lowered))
    if word_count < 3:
        return False

    pending_species = _infer_species(pending_question)
    message_species = _infer_species(message)
    if message_species and message_species != pending_species:
        return True

    return lowered.startswith(("what", "when", "where", "which", "who", "can", "do", "does", "is", "are", "how", "tell me"))


def _message_is_broad_interest(question: str) -> bool:
    lowered = question.lower().strip()
    if any(term in lowered for term in TOPIC_HINT_TERMS):
        return False
    return any(term in lowered for term in BROAD_QUERY_TERMS)


def _natural_species_interest_outcome(question: str) -> AnswerOutcome | None:
    species_terms = _extract_species_terms(question)
    if len(species_terms) != 1:
        return None
    if not _message_is_broad_interest(question):
        return None
    species = next(iter(species_terms))
    return _clarification_outcome(_supported_species_prompt(species), species, "topic")


def _natural_general_interest_outcome(question: str) -> AnswerOutcome | None:
    lowered = question.lower().strip()
    if _extract_species_terms(question):
        return None
    if not _message_is_broad_interest(question):
        return None
    return _clarification_outcome(
        "Many species are covered in the official summaries. Ask one specific question, for example: deer season, rabbit daily limit, or duck daily bag limit in the Southern District. Informational only. Not legal advice. Verify current regs.",
        "hunting species",
        "topic",
    )


def _supported_species_prompt(species: str) -> str:
    examples = {
        "rabbit": "rabbit daily limit, rabbit possession limit, or rabbit season in your WMU",
        "duck": "duck daily bag limit in the Southern District, duck possession limit in the Northern District, or non-toxic shot for migratory birds",
        "waterfowl": "duck daily bag limit in the Southern District, Canada goose season in the Central District, or non-toxic shot for migratory birds",
        "canada goose": "Canada goose daily bag limit in the Southern District, Canada goose season in the Central District, or Sunday gun hunting rules",
        "snow goose": "Snow goose season in the Southern District, Snow goose bag limit, or special conservation harvest dates",
    }
    labels = {
        "rabbit": "Rabbit hunting is covered here.",
        "duck": "Duck hunting is covered here.",
        "waterfowl": "Waterfowl hunting is covered here.",
        "canada goose": "Canada goose hunting is covered here.",
        "snow goose": "Snow goose hunting is covered here.",
        "deer": "Deer hunting is covered here.",
        "moose": "Moose hunting is covered here.",
        "bear": "Bear hunting is covered here.",
        "turkey": "Turkey hunting is covered here.",
        "elk": "Elk hunting is covered here.",
        "wolf": "Wolf hunting is covered here.",
        "coyote": "Coyote hunting is covered here.",
        "mourning dove": "Mourning dove hunting is covered here.",
        "woodcock": "Woodcock hunting is covered here.",
        "snipe": "Snipe hunting is covered here.",
    }
    fallback_species = species.replace(" ", " ")
    example_text = examples.get(species, f"{fallback_species} season, {fallback_species} daily limit, or {fallback_species} tag rules")
    intro = labels.get(species, f"{fallback_species.capitalize()} hunting is covered here.")
    return f"{intro} Ask one specific {fallback_species} question, for example: {example_text}. Informational only. Not legal advice. Verify current regs."


def _fallback_interpret_incoming_message(message: str, pending_state: dict[str, str] | None = None) -> IntakeOutcome:
    normalized = _normalize_intake_key(message)
    pending_question = (pending_state or {}).get("question", "").strip()
    expected_detail = (pending_state or {}).get("expected_detail", "").strip() or None

    if pending_question:
        if normalized in {"thanks", "thank you", "thx", "ok", "okay", "cool", "got it", "lol", "haha", "what do you mean", "which one", "can you clarify", "clarify", "not sure", "i dont know", "i do not know"}:
            return IntakeOutcome(
                action="clarify",
                reply_text=_clarification_reminder_text(expected_detail),
                pending_question=pending_question,
                expected_detail=expected_detail,
            )
        if _looks_like_follow_up_fragment(message, expected_detail):
            combined = f"{pending_question} {message.strip()}".strip()
            return IntakeOutcome(action="search", normalized_question=combined)

    if normalized in {"hi", "hello", "hey", "help", "start", "menu", "info", "usage", "how do i use this", "how to use this", "what can you do", "what do you do"}:
        return IntakeOutcome(action="guide", reply_text=INTAKE_GUIDANCE_RESPONSE)

    if normalized in {"thanks", "thank you", "thx", "ok", "okay", "cool", "got it", "lol", "haha"}:
        return IntakeOutcome(action="guide", reply_text=INTAKE_GUIDANCE_RESPONSE)

    if not normalized:
        return IntakeOutcome(action="guide", reply_text=INTAKE_GUIDANCE_RESPONSE)

    if len(re.findall(r"\w+", normalized)) <= 3 and (_question_specifies_method(normalized) or bool(_extract_wmu_terms(normalized)) or bool(_extract_district_terms(normalized))):
        return IntakeOutcome(action="guide", reply_text=INTAKE_GUIDANCE_RESPONSE)

    species_terms = _extract_species_terms(message)
    lowered = message.lower()
    if len(species_terms) == 1 and any(term in lowered for term in ("can i hunt", "what", "rules", "regulations", "hunt", "hunting")) and not _message_has_specific_topic_hint(message):
        species = next(iter(species_terms))
        return IntakeOutcome(
            action="clarify",
            reply_text=_supported_species_prompt(species),
            pending_question=species,
            expected_detail="topic",
        )

    if any(species in species_terms for species in WATERFOWL_SPECIES) and any(term in lowered for term in WATERFOWL_DETAIL_TERMS) and not _extract_district_terms(lowered):
        return IntakeOutcome(
            action="clarify",
            reply_text="Need one more detail: reply with the district, for example Southern or Central District. Informational only. Not legal advice. Verify current regs.",
            pending_question=message.strip(),
            expected_detail="district",
        )

    return IntakeOutcome(action="search", normalized_question=message.strip())


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def interpret_incoming_message(message: str, pending_state: dict[str, str] | None = None) -> IntakeOutcome:
    pending_question = (pending_state or {}).get("question", "").strip()
    expected_detail = (pending_state or {}).get("expected_detail", "").strip() or ""
    prompt = (
        INTAKE_PROMPT.replace("{pending_question}", pending_question or "none")
        .replace("{expected_detail}", expected_detail or "none")
        .replace("{message}", message.strip())
    )
    try:
        response = _invoke_model_call(
            lambda: _get_intake_model().invoke(prompt),
            INTAKE_TIMEOUT_SECONDS,
        )
        if response is None:
            return _fallback_interpret_incoming_message(message, pending_state)
        payload = json.loads(_strip_json_fence(str(response.content)))
        action = str(payload.get("action", "")).strip().lower()
        if action not in {"guide", "clarify", "search"}:
            raise ValueError(f"Unexpected intake action: {action!r}")
        normalized_question = _normalize_inline(str(payload.get("normalized_question", "")))
        reply_text = _normalize_inline(str(payload.get("reply_text", "")))
        stored_question = _normalize_inline(str(payload.get("pending_question", ""))) or None
        stored_detail = _normalize_inline(str(payload.get("expected_detail", ""))) or None

        follow_up_like = bool(pending_question) and _looks_like_follow_up_fragment(message, expected_detail or None)
        new_question_like = bool(pending_question) and _looks_like_new_question(message, pending_question, expected_detail or None)

        if action == "search" and not normalized_question:
            normalized_question = message.strip()
        if action in {"guide", "clarify"} and not reply_text:
            raise ValueError("Missing reply_text for non-search action")

        natural_species = _natural_species_interest_outcome(message)
        if action != "search" and natural_species is not None:
            return IntakeOutcome(
                action="clarify",
                reply_text=natural_species.text,
                pending_question=natural_species.pending_question,
                expected_detail=natural_species.expected_detail,
            )

        natural_general = _natural_general_interest_outcome(message)
        if action != "search" and natural_general is not None:
            return IntakeOutcome(
                action="clarify",
                reply_text=natural_general.text,
                pending_question=natural_general.pending_question,
                expected_detail=natural_general.expected_detail,
            )

        if action == "clarify" and stored_detail == "topic" and _message_has_specific_topic_hint(message):
            action = "search"
            normalized_question = message.strip()
            reply_text = ""
            stored_question = None
            stored_detail = None

        if action == "clarify" and new_question_like and not follow_up_like:
            if not stored_question or stored_question == pending_question:
                species = _infer_species(message)
                stored_question = species or message.strip()
        if action == "search" and new_question_like and not follow_up_like and pending_question and normalized_question == pending_question:
            normalized_question = message.strip()

        return IntakeOutcome(
            action=action,
            normalized_question=normalized_question,
            reply_text=reply_text,
            pending_question=stored_question,
            expected_detail=stored_detail,
        )
    except Exception:
        return _fallback_interpret_incoming_message(message, pending_state)


METHOD_TERMS = ("bow", "bows", "rifle", "rifles", "shotgun", "shotguns", "muzzle", "muzzle-loading", "gun", "guns")


def _question_specifies_method(question: str) -> bool:
    lowered = question.lower()
    return any(term in lowered for term in METHOD_TERMS)


def _is_deer_season_question(question: str) -> bool:
    lowered = question.lower()
    return "deer" in lowered and "season" in lowered


BROAD_QUERY_TERMS = (
    "rule",
    "rules",
    "regulation",
    "regulations",
    "hunt",
    "hunting",
    "allowed",
    "what about",
    "what are",
    "what is",
    "can i",
    "am i allowed",
    "how do i hunt",
    "i want to hunt",
    "tell me about",
    "what can i hunt",
)
TOPIC_HINT_TERMS = (
    "season",
    "limit",
    "daily",
    "bag",
    "possession",
    "wmu",
    "unit",
    "district",
    "tag",
    "licence",
    "license",
    "orange",
    "calibre",
    "caliber",
    "gun",
    "guns",
    "bow",
    "bows",
    "rifle",
    "shotgun",
    "muzzle",
    "sunday",
    "date",
    "dates",
    "resident",
    "nonresident",
    "non-resident",
    "calf",
    "antlerless",
    "shot",
)


def _pretty_species_name(label: str) -> str:
    names = {
        "deer": "deer",
        "moose": "moose",
        "bear": "bear",
        "turkey": "turkey",
        "elk": "elk",
        "wolf": "wolf",
        "coyote": "coyote",
        "rabbit": "rabbit",
        "duck": "duck",
        "waterfowl": "waterfowl",
        "canada goose": "Canada goose",
        "snow goose": "Snow goose",
        "goose": "goose",
        "mourning dove": "mourning dove",
        "woodcock": "woodcock",
        "snipe": "snipe",
        "rail": "rail",
        "coot": "coot",
        "gallinule": "gallinule",
    }
    return names.get(label, label)


def _clarification_outcome(text: str, question: str, expected_detail: str) -> AnswerOutcome:
    return AnswerOutcome(
        text=text,
        kind="clarify",
        pending_question=question,
        expected_detail=expected_detail,
    )


def _broad_species_question_clarification(question: str) -> AnswerOutcome | None:
    lowered = question.lower()
    if any(term in lowered for term in TOPIC_HINT_TERMS):
        return None

    species_terms = _extract_species_terms(question)
    if len(species_terms) != 1:
        return None

    word_count = len(re.findall(r"\w+", lowered))
    has_broad_cue = any(term in lowered for term in BROAD_QUERY_TERMS)
    if not has_broad_cue and word_count > 4:
        return None

    species = next(iter(species_terms))
    return _clarification_outcome(_supported_species_prompt(species), species, "topic")


def _broad_general_question_clarification(question: str) -> AnswerOutcome | None:
    lowered = question.lower()
    if any(term in lowered for term in TOPIC_HINT_TERMS):
        return None

    word_count = len(re.findall(r"\w+", lowered))
    if word_count > 6:
        return None
    if not any(term in lowered for term in BROAD_QUERY_TERMS):
        return None

    return _clarification_outcome(
        "Many species are covered in the official summaries. Ask one specific question, for example: deer season in WMU 65, rabbit daily limit, or duck daily bag limit in the Southern District. Informational only. Not legal advice. Verify current regs.",
        question,
        "topic",
    )


def _missing_deer_season_details(question: str) -> list[str]:
    if not _is_deer_season_question(question):
        return []

    missing: list[str] = []
    if not _extract_wmu_terms(question):
        missing.append("wmu")
    if not _question_specifies_method(question):
        missing.append("method")
    return missing


def _waterfowl_district_missing(question: str) -> bool:
    species_terms = _extract_species_terms(question)
    if not species_terms.intersection(WATERFOWL_SPECIES):
        return False
    lowered = question.lower()
    if not any(term in lowered for term in WATERFOWL_DETAIL_TERMS):
        return False
    if _extract_district_terms(question):
        return False
    return True


def _build_deer_season_clarification(question: str, missing_details: list[str]) -> AnswerOutcome:
    if missing_details == ["wmu", "method"] or set(missing_details) == {"wmu", "method"}:
        return _clarification_outcome(
            "Need two details: reply with the WMU number and method, for example: WMU 65 bows only. Informational only. Not legal advice. Verify current regs.",
            question,
            "wmu_and_method",
        )

    if missing_details == ["wmu"]:
        return _clarification_outcome(
            "Need one more detail: reply with the WMU number, for example WMU 65. Informational only. Not legal advice. Verify current regs.",
            question,
            "wmu",
        )

    return _clarification_outcome(
        "WMU deer season has multiple entries. Reply with one: bows only, or guns. Informational only. Not legal advice. Verify current regs.",
        question,
        "method",
    )


def _build_waterfowl_district_clarification(question: str) -> AnswerOutcome:
    species_terms = _extract_species_terms(question)
    species_name = _pretty_species_name(next(iter(species_terms))) if species_terms else "waterfowl"
    return _clarification_outcome(
        f"Need one more detail for {species_name}: reply with the district, for example Southern or Central District. Informational only. Not legal advice. Verify current regs.",
        question,
        "district",
    )


def _results_are_ambiguous(question: str, documents: list[Document]) -> bool:
    if _question_specifies_method(question):
        return False

    contexts = {
        document.metadata.get("table_context", "")
        for document in documents
        if document.metadata.get("chunk_type") == "table_row" and document.metadata.get("table_context")
    }
    if len(contexts) > 1:
        return True

    if not _extract_district_terms(question):
        districts = {document.metadata.get("district", "") for document in documents if document.metadata.get("district")}
        if len(districts) > 1:
            return True

    return False


def _ambiguity_clarification(question: str, documents: list[Document]) -> AnswerOutcome:
    districts = {document.metadata.get("district", "") for document in documents if document.metadata.get("district")}
    if len(districts) > 1 and not _extract_district_terms(question):
        return _clarification_outcome(
            "Your question matches multiple districts. Please ask again with one district: Hudson-James Bay District, Northern District, Central District, or Southern District. Informational only. Not legal advice. Verify current regs.",
            question,
            "district",
        )

    contexts = [
        document.metadata.get("table_context", "")
        for document in documents
        if document.metadata.get("chunk_type") == "table_row" and document.metadata.get("table_context")
    ]
    joined = " ".join(contexts).lower()

    if "bows only" in joined and "rifles, shotguns, muzzle-loading guns and bows" in joined:
        return _clarification_outcome(
            "Your question matches multiple entries in the official summary. Reply with one: bows only, or guns. Informational only. Not legal advice. Verify current regs.",
            question,
            "method",
        )

    return _clarification_outcome(
        "Your question matches multiple entries. Please ask again with the missing district, hunting method, or season type. Informational only. Not legal advice. Verify current regs.",
        question,
        "detail",
    )


def _format_pages(documents: list[Document]) -> str:
    formatted_pages: list[dict[str, Any]] = []
    for document in documents:
        formatted_pages.append(
            {
                "page": document.metadata.get("page_num"),
                "chunk_type": document.metadata.get("chunk_type"),
                "source_title": document.metadata.get("source_title"),
                "citation_url": document.metadata.get("citation_url"),
                "url": document.metadata.get("url", PDF_URL),
                "context": document.metadata.get("table_context", ""),
                "district": document.metadata.get("district", ""),
                "species": document.metadata.get("species", ""),
                "text": _normalize_page_text(document.page_content),
            }
        )
    return json.dumps(formatted_pages, ensure_ascii=True)


def answer_question_result(question: str) -> AnswerOutcome:
    cached = _get_cached_answer(question)
    if cached is not None:
        return cached

    broad_species = _broad_species_question_clarification(question)
    if broad_species:
        _set_cached_answer(question, broad_species)
        return broad_species

    broad_general = _broad_general_question_clarification(question)
    if broad_general:
        _set_cached_answer(question, broad_general)
        return broad_general

    missing_details = _missing_deer_season_details(question)
    if missing_details:
        outcome = _build_deer_season_clarification(question, missing_details)
        _set_cached_answer(question, outcome)
        return outcome

    if _waterfowl_district_missing(question):
        outcome = _build_waterfowl_district_clarification(question)
        _set_cached_answer(question, outcome)
        return outcome

    vectorstore = _load_vectorstore()
    direct_matches = _direct_structured_match(vectorstore, question)
    if direct_matches:
        if len(direct_matches) == 1:
            outcome = AnswerOutcome(text=_format_exact_quote(direct_matches[0]), kind="answer")
            _set_cached_answer(question, outcome)
            return outcome
        results = direct_matches
    else:
        retrieval_query = _rewrite_search_query(question)
        results = vectorstore.similarity_search(retrieval_query, k=8)
        results = _rerank_results(f"{question} {retrieval_query}", results)[:3]

    if _results_are_ambiguous(question, results):
        outcome = _ambiguity_clarification(question, results)
        _set_cached_answer(question, outcome)
        return outcome

    prompt = SYSTEM_PROMPT.format(
        question=question,
        pages=_format_pages(results),
        not_found_response=NOT_FOUND_RESPONSE,
    )
    response = _invoke_model_call(
        lambda: _get_chat_model().invoke(prompt),
        ANSWER_TIMEOUT_SECONDS,
    )
    if response is None:
        outcome = AnswerOutcome(text=NOT_FOUND_RESPONSE, kind="not_found")
        _set_cached_answer(question, outcome)
        return outcome
    normalized = _normalize_model_output(response.content)
    kind = "not_found" if normalized == NOT_FOUND_RESPONSE else "answer"
    outcome = AnswerOutcome(text=normalized, kind=kind)
    _set_cached_answer(question, outcome)
    return outcome


def answer_question(question: str) -> str:
    return answer_question_result(question).text
