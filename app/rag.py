from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pypdf import PdfReader


load_dotenv()

PDF_URL = "https://www.ontario.ca/document/ontario-hunting-regulations-summary"
DEFAULT_PDF_PATH = Path(os.getenv("SOURCE_PDF_PATH", "data/2026-ontario-hunting-regulations-summary.pdf"))
INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", "data"))
SYSTEM_PROMPT = """You are a search tool for the 2026 Ontario Hunting Regulations Summary only.
Never summarize, never interpret, never use outside knowledge.
User question: {question}
Relevant pages: {pages}
If the answer is clearly on these pages, return the shortest exact quote that answers the question.
If the source is a table, return only the exact matching row or cell text needed, not the whole table.
Use the provided table context to distinguish between different season tables or weapon types.
If multiple relevant entries differ by method, season type, or weapon and the user did not specify enough detail, reply with the not-found fallback.
Preserve source wording, but collapse repeated spaces and line breaks into normal readable spaces.
Reply with EXACTLY this format:
'2026 Ontario Hunting Regulations Summary, p.{{page}}: "{{exact sentence from PDF}}" ontario.ca/hunting
Informational only. Not legal advice. Verify current regs.'
If not found, reply: 'Not found in 2026 Summary. Check ontario.ca or call MNRF 1-800-667-1940. Informational only. Not legal advice. Verify current regs.'
Do not add anything else."""
SEARCH_REWRITE_PROMPT = """Rewrite the user question into a short search query for the 2026 Ontario Hunting Regulations Summary.
Keep species names, WMU numbers, season concepts, licence concepts, and gear terms when relevant.
Do not answer the question. Do not add commentary. Return one short search query only.
User question: {question}
"""

WMU_ROW_START_RE = re.compile(r"^(?:\d{1,3}[A-Z]?)(?:,\s*\d{1,3}[A-Z]?)*(?:\s*,\s*\d{1,3}[A-Z]?)*\b")
TABLE_PAGE_MARKERS = ("Wildlife Management Unit", "Resident — open season", "Resident - open season")
SKIP_LINE_RE = re.compile(
    r"^(?:Hunting Regulations Summary|White-tailed Deer|Wildlife Management Unit|Resident|Non-resident|General Regulations|\d+)$",
    re.IGNORECASE,
)


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
}

def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


def _get_chat_model() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o", temperature=0)


def _get_rewrite_model() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o", temperature=0)


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
    return (
        f'2026 Ontario Hunting Regulations Summary, p.{page}: "{quote}" ontario.ca/hunting\n'
        "Informational only. Not legal advice. Verify current regs."
    )


def _infer_species(text: str) -> str | None:
    lowered = text.lower()
    for pattern, label in SPECIES_PATTERNS.items():
        if pattern in lowered:
            return label
    return None


def _extract_species_terms(question: str) -> set[str]:
    lowered = question.lower()
    return {label for pattern, label in SPECIES_PATTERNS.items() if pattern in lowered}


def _extract_wmu_terms(question: str) -> set[str]:
    return {match.upper() for match in re.findall(r"\b(?:WMU\s*)?(\d{1,3}[A-Z]?)\b", question, re.IGNORECASE)}


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


def _direct_structured_match(vectorstore: FAISS, question: str) -> list[Document]:
    species_terms = _extract_species_terms(question)
    wmu_terms = _extract_wmu_terms(question)
    method_terms = _extract_method_terms(question)
    if not species_terms or not wmu_terms or not method_terms:
        return []

    candidates: list[Document] = []
    for document in vectorstore.docstore._dict.values():
        if document.metadata.get("chunk_type") != "table_row":
            continue
        if species_terms and document.metadata.get("species") not in species_terms:
            continue
        if not _doc_matches_wmu(document, wmu_terms):
            continue
        if not _doc_matches_method(document, method_terms):
            continue
        candidates.append(document)

    return candidates[:3]


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


def _page_documents_from_pdf(pdf_path: str | Path) -> list[Document]:
    reader = PdfReader(str(pdf_path))
    documents: list[Document] = []

    for index, page in enumerate(reader.pages, start=1):
        page_text = _normalize_page_text(page.extract_text() or "")
        if not page_text:
            continue

        metadata = {
            "page_num": index,
            "year": 2026,
            "url": PDF_URL,
        }
        species = _infer_species(page_text)
        if species:
            metadata["species"] = species

        documents.append(
            Document(
                page_content=page_text,
                metadata={**metadata, "chunk_type": "page"},
            )
        )

        for paragraph in _extract_paragraph_chunks(page_text):
            documents.append(
                Document(
                    page_content=paragraph,
                    metadata={**metadata, "chunk_type": "paragraph"},
                )
            )

        table_context = _extract_table_context(page_text)
        for row in _extract_table_row_chunks(page_text):
            row_metadata = {**metadata, "chunk_type": "table_row"}
            if table_context:
                row_metadata["table_context"] = table_context
            documents.append(
                Document(
                    page_content=row,
                    metadata=row_metadata,
                )
            )

    if not documents:
        raise ValueError("No text could be extracted from the PDF.")

    return documents


def build_index(pdf_path: str | Path = DEFAULT_PDF_PATH) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    documents = _page_documents_from_pdf(pdf_path)
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
        response = _get_rewrite_model().invoke(SEARCH_REWRITE_PROMPT.format(question=question))
        rewritten = _normalize_inline(response.content)
        return rewritten or question
    except Exception:
        return question


def _extract_query_terms(question: str) -> set[str]:
    return {match.upper() for match in re.findall(r"\b[0-9]{1,3}[A-Z]?\b", question)}


def _rerank_results(question: str, documents: list[Document]) -> list[Document]:
    query_terms = _extract_query_terms(question)
    species_terms = _extract_species_terms(question)

    def sort_key(document: Document) -> tuple[int, int, int]:
        content = document.page_content.upper()
        overlap = sum(1 for term in query_terms if term in content)
        table_bonus = 1 if document.metadata.get("chunk_type") == "table_row" else 0
        species_bonus = 1 if species_terms and document.metadata.get("species") in species_terms else 0
        return (species_bonus, overlap, table_bonus)

    return sorted(documents, key=sort_key, reverse=True)

@dataclass
class AnswerOutcome:
    text: str
    kind: str
    pending_question: str | None = None
    expected_detail: str | None = None


METHOD_TERMS = ("bow", "bows", "rifle", "rifles", "shotgun", "shotguns", "muzzle", "muzzle-loading", "gun", "guns")


def _question_specifies_method(question: str) -> bool:
    lowered = question.lower()
    return any(term in lowered for term in METHOD_TERMS)


def _is_deer_season_question(question: str) -> bool:
    lowered = question.lower()
    return "deer" in lowered and "season" in lowered


def _missing_deer_season_details(question: str) -> list[str]:
    if not _is_deer_season_question(question):
        return []

    missing: list[str] = []
    if not _extract_wmu_terms(question):
        missing.append("wmu")
    if not _question_specifies_method(question):
        missing.append("method")
    return missing


def _clarification_outcome(text: str, question: str, expected_detail: str) -> AnswerOutcome:
    return AnswerOutcome(
        text=text,
        kind="clarify",
        pending_question=question,
        expected_detail=expected_detail,
    )


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


def _results_are_ambiguous(question: str, documents: list[Document]) -> bool:
    if _question_specifies_method(question):
        return False

    contexts = {
        document.metadata.get("table_context", "")
        for document in documents
        if document.metadata.get("chunk_type") == "table_row" and document.metadata.get("table_context")
    }
    return len(contexts) > 1



def _ambiguity_clarification(question: str, documents: list[Document]) -> AnswerOutcome:
    contexts = [
        document.metadata.get("table_context", "")
        for document in documents
        if document.metadata.get("chunk_type") == "table_row" and document.metadata.get("table_context")
    ]
    joined = " ".join(contexts).lower()

    if "bows only" in joined and "rifles, shotguns, muzzle-loading guns and bows" in joined:
        return _clarification_outcome(
            "Your question matches multiple entries in the 2026 Summary. Reply with one: bows only, or guns. Informational only. Not legal advice. Verify current regs.",
            question,
            "method",
        )

    return _clarification_outcome(
        "Your question matches multiple entries in the 2026 Summary. Please ask again with the hunting method or season type. Informational only. Not legal advice. Verify current regs.",
        question,
        "method",
    )

def _format_pages(documents: list[Document]) -> str:
    formatted_pages: list[dict[str, Any]] = []
    for document in documents:
        formatted_pages.append(
            {
                "page": document.metadata.get("page_num"),
                "chunk_type": document.metadata.get("chunk_type"),
                "url": document.metadata.get("url", PDF_URL),
                "context": document.metadata.get("table_context", ""),
                "text": _normalize_page_text(document.page_content),
            }
        )
    return json.dumps(formatted_pages, ensure_ascii=True)


def answer_question_result(question: str) -> AnswerOutcome:
    missing_details = _missing_deer_season_details(question)
    if missing_details:
        return _build_deer_season_clarification(question, missing_details)

    vectorstore = _load_vectorstore()
    direct_matches = _direct_structured_match(vectorstore, question)
    if direct_matches:
        if len(direct_matches) == 1:
            return AnswerOutcome(text=_format_exact_quote(direct_matches[0]), kind="answer")
        results = direct_matches
    else:
        retrieval_query = _rewrite_search_query(question)
        results = vectorstore.similarity_search(retrieval_query, k=8)
        results = _rerank_results(f"{question} {retrieval_query}", results)[:3]
    if _results_are_ambiguous(question, results):
        return _ambiguity_clarification(question, results)

    prompt = SYSTEM_PROMPT.format(
        question=question,
        pages=_format_pages(results),
    )
    response = _get_chat_model().invoke(prompt)
    normalized = _normalize_model_output(response.content)
    kind = "not_found" if normalized.startswith("Not found in 2026 Summary.") else "answer"
    return AnswerOutcome(text=normalized, kind=kind)


def answer_question(question: str) -> str:
    return answer_question_result(question).text
