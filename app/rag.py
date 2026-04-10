from __future__ import annotations

import json
import os
import re
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
Preserve source wording, but collapse repeated spaces and line breaks into normal readable spaces.
Reply with EXACTLY this format:
'2026 Ontario Hunting Regulations Summary, p.{{page}}: "{{exact sentence from PDF}}" ontario.ca/hunting
Informational only. Not legal advice. Verify current regs.'
If not found, reply: 'Not found in 2026 Summary. Check ontario.ca or call MNRF 1-800-667-1940. Informational only. Not legal advice. Verify current regs.'
Do not add anything else."""

WMU_ROW_START_RE = re.compile(r"^(?:\d{1,3}[A-Z]?)(?:,\s*\d{1,3}[A-Z]?)*(?:\s*,\s*\d{1,3}[A-Z]?)*\b")
TABLE_PAGE_MARKERS = ("Wildlife Management Unit", "Resident — open season", "Resident - open season")
SKIP_LINE_RE = re.compile(
    r"^(?:Hunting Regulations Summary|White-tailed Deer|Wildlife Management Unit|Resident|Non-resident|General Regulations|\d+)$",
    re.IGNORECASE,
)


def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


def _get_chat_model() -> ChatOpenAI:
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
    return text.strip()


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

        for row in _extract_table_row_chunks(page_text):
            documents.append(
                Document(
                    page_content=row,
                    metadata={**metadata, "chunk_type": "table_row"},
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



def _extract_query_terms(question: str) -> set[str]:
    return {match.upper() for match in re.findall(r"\b[0-9]{1,3}[A-Z]?\b", question)}


def _rerank_results(question: str, documents: list[Document]) -> list[Document]:
    query_terms = _extract_query_terms(question)
    if not query_terms:
        return documents

    def sort_key(document: Document) -> tuple[int, int]:
        content = document.page_content.upper()
        overlap = sum(1 for term in query_terms if term in content)
        table_bonus = 1 if document.metadata.get("chunk_type") == "table_row" else 0
        return (overlap, table_bonus)

    return sorted(documents, key=sort_key, reverse=True)

def _format_pages(documents: list[Document]) -> str:
    formatted_pages: list[dict[str, Any]] = []
    for document in documents:
        formatted_pages.append(
            {
                "page": document.metadata.get("page_num"),
                "chunk_type": document.metadata.get("chunk_type"),
                "url": document.metadata.get("url", PDF_URL),
                "text": _normalize_page_text(document.page_content),
            }
        )
    return json.dumps(formatted_pages, ensure_ascii=True)


def answer_question(question: str) -> str:
    vectorstore = _load_vectorstore()
    results = vectorstore.similarity_search(question, k=8)
    results = _rerank_results(question, results)[:3]
    prompt = SYSTEM_PROMPT.format(
        question=question,
        pages=_format_pages(results),
    )
    response = _get_chat_model().invoke(prompt)
    return _normalize_model_output(response.content)
