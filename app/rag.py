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


def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


def _get_chat_model() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o", temperature=0)


def _normalize_page_text(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"[ 	]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def _normalize_model_output(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"[ 	]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return text.strip()


def _page_documents_from_pdf(pdf_path: str | Path) -> list[Document]:
    reader = PdfReader(str(pdf_path))
    documents: list[Document] = []

    for index, page in enumerate(reader.pages, start=1):
        page_text = _normalize_page_text(page.extract_text() or "")
        if not page_text:
            continue

        documents.append(
            Document(
                page_content=page_text,
                metadata={
                    "page_num": index,
                    "year": 2026,
                    "url": PDF_URL,
                },
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


def _format_pages(documents: list[Document]) -> str:
    formatted_pages: list[dict[str, Any]] = []
    for document in documents:
        formatted_pages.append(
            {
                "page": document.metadata.get("page_num"),
                "url": document.metadata.get("url", PDF_URL),
                "text": _normalize_page_text(document.page_content),
            }
        )
    return json.dumps(formatted_pages, ensure_ascii=True)


def answer_question(question: str) -> str:
    vectorstore = _load_vectorstore()
    results = vectorstore.similarity_search(question, k=3)
    prompt = SYSTEM_PROMPT.format(
        question=question,
        pages=_format_pages(results),
    )
    response = _get_chat_model().invoke(prompt)
    return _normalize_model_output(response.content)
