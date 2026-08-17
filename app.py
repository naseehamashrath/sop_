import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import streamlit as st
from google import genai
from google.genai import types

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional runtime dependency
    PdfReader = None

try:
    import docx
except ImportError:  # pragma: no cover - optional runtime dependency
    docx = None


APP_TITLE = "Manufacturing SOP & Safety Explainer"
BASE_DIR = Path(__file__).resolve().parent
APPROVED_DOCS_DIR = BASE_DIR / "data" / "approved_documents"
INDEX_PATH = BASE_DIR / ".rag_index" / "approved_sop_index.json"
DEFAULT_GENERATION_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"
TOP_K = 4
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

SYSTEM_INSTRUCTION = """
You are a manufacturing SOP and safety explainer bot for employees, interns, and supervisors.
Your role is explanation only.

Rules:
- Use only the retrieved approved-document context to explain SOPs and safety rules.
- Explain in simple, practical language.
- Do not approve work, authorize operations, certify compliance, sign off, or say an action is safe to perform.
- Do not replace a supervisor, safety officer, permit issuer, or emergency response team.
- If the user asks for a decision, approval, compliance judgment, or operational go/no-go, refuse that part and tell them to contact the responsible supervisor or safety officer.
- If the retrieved context is insufficient, say that the approved documents do not contain enough information and recommend checking the official SOP or supervisor.
- For emergency situations, provide only high-level emergency steps found in context and tell the user to follow site emergency procedures and contact emergency response personnel.
- Keep the answer concise and include the source document names used.
"""

DECISION_OR_APPROVAL_PATTERNS = (
    r"\b(can i|may i|should i|am i allowed|is it okay|is it safe|approve|approval|authorize|sign off)\b",
    r"\b(compliant|non[- ]?compliant|violation|pass inspection|permit me|clearance)\b",
    r"\b(start the machine|restart the machine|bypass|override|disable guard|skip step)\b",
)


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source: str
    text: str
    embedding: list[float]


class GeminiServiceError(RuntimeError):
    pass


def get_api_key() -> str:
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret_key = ""
    return os.getenv("GEMINI_API_KEY") or secret_key or st.session_state.get("api_key", "")


def get_generation_model() -> str:
    return (
        os.getenv("GEMINI_FLASH_MODEL")
        or st.session_state.get("generation_model")
        or DEFAULT_GENERATION_MODEL
    )


def get_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    if PdfReader is None:
        return ""
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_docx(path: Path) -> str:
    if docx is None:
        return ""
    document = docx.Document(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def load_approved_documents() -> dict[str, str]:
    APPROVED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    documents: dict[str, str] = {}
    for path in sorted(APPROVED_DOCS_DIR.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            text = read_text_file(path)
        elif suffix == ".pdf":
            text = read_pdf(path)
        elif suffix == ".docx":
            text = read_docx(path)
        else:
            continue
        cleaned = normalize_text(text)
        if cleaned:
            documents[str(path.relative_to(APPROVED_DOCS_DIR))] = cleaned
    return documents


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(source: str, text: str) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            paragraph_break = text.rfind("\n\n", start, end)
            sentence_break = text.rfind(". ", start, end)
            split_at = max(paragraph_break, sentence_break)
            if split_at > start + CHUNK_SIZE // 2:
                end = split_at + 1
        chunk = text[start:end].strip()
        if chunk:
            digest = hashlib.sha256(f"{source}:{start}:{chunk}".encode("utf-8")).hexdigest()[:16]
            chunks.append((digest, chunk))
        start = max(end - CHUNK_OVERLAP, end if end == len(text) else 0)
        if start >= len(text):
            break
    return chunks


def documents_fingerprint(documents: dict[str, str]) -> str:
    payload = json.dumps(documents, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def embed_texts(client: genai.Client, texts: list[str], task_type: str) -> list[list[float]]:
    try:
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [embedding.values for embedding in result.embeddings]
    except Exception as exc:
        raise GeminiServiceError(f"Could not connect to Gemini embeddings: {friendly_gemini_error(exc)}") from exc


def save_index(fingerprint: str, chunks: list[DocumentChunk]) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fingerprint": fingerprint,
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "text": chunk.text,
                "embedding": chunk.embedding,
            }
            for chunk in chunks
        ],
    }
    INDEX_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_index(fingerprint: str) -> list[DocumentChunk] | None:
    if not INDEX_PATH.exists():
        return None
    payload = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    if payload.get("fingerprint") != fingerprint:
        return None
    return [
        DocumentChunk(
            chunk_id=item["chunk_id"],
            source=item["source"],
            text=item["text"],
            embedding=item["embedding"],
        )
        for item in payload.get("chunks", [])
    ]


@st.cache_data(show_spinner=False)
def cached_documents() -> dict[str, str]:
    return load_approved_documents()


def build_or_load_index(client: genai.Client, documents: dict[str, str]) -> list[DocumentChunk]:
    fingerprint = documents_fingerprint(documents)
    indexed = load_index(fingerprint)
    if indexed is not None:
        return indexed

    chunk_records: list[tuple[str, str, str]] = []
    for source, text in documents.items():
        for chunk_id, chunk in chunk_text(source, text):
            chunk_records.append((chunk_id, source, chunk))

    embeddings: list[list[float]] = []
    batch_size = 20
    try:
        for index in range(0, len(chunk_records), batch_size):
            batch = chunk_records[index : index + batch_size]
            embeddings.extend(embed_texts(client, [record[2] for record in batch], "RETRIEVAL_DOCUMENT"))
    except GeminiServiceError:
        return [
            DocumentChunk(chunk_id=record[0], source=record[1], text=record[2], embedding=[])
            for record in chunk_records
        ]

    chunks = [
        DocumentChunk(chunk_id=record[0], source=record[1], text=record[2], embedding=embedding)
        for record, embedding in zip(chunk_records, embeddings)
    ]
    save_index(fingerprint, chunks)
    return chunks


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right):
        dot += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def retrieve_relevant_chunks(client: genai.Client, query: str, chunks: list[DocumentChunk]) -> list[tuple[DocumentChunk, float]]:
    if not chunks or not chunks[0].embedding:
        return retrieve_keyword_chunks(query, chunks)

    query_embedding = embed_texts(client, [query], "RETRIEVAL_QUERY")[0]
    ranked = sorted(
        ((chunk, cosine_similarity(query_embedding, chunk.embedding)) for chunk in chunks),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[:TOP_K]


def tokenize(text: str) -> set[str]:
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "for",
        "in",
        "is",
        "it",
        "of",
        "or",
        "the",
        "this",
        "to",
        "what",
        "when",
        "where",
        "why",
        "with",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9-]+", text.lower())
        if token not in stop_words
    }


def retrieve_keyword_chunks(query: str, chunks: list[DocumentChunk]) -> list[tuple[DocumentChunk, float]]:
    query_terms = tokenize(query)
    if not query_terms:
        return [(chunk, 0.0) for chunk in chunks[:TOP_K]]

    ranked = []
    for chunk in chunks:
        chunk_terms = tokenize(chunk.text)
        overlap = query_terms & chunk_terms
        score = len(overlap) / max(len(query_terms), 1)
        ranked.append((chunk, score))
    return sorted(ranked, key=lambda item: item[1], reverse=True)[:TOP_K]


def is_decision_or_approval_request(query: str) -> bool:
    lowered = query.lower()
    return any(re.search(pattern, lowered) for pattern in DECISION_OR_APPROVAL_PATTERNS)


def format_context(retrieved: list[tuple[DocumentChunk, float]]) -> str:
    sections = []
    for index, (chunk, score) in enumerate(retrieved, start=1):
        sections.append(
            f"[Context {index} | Source: {chunk.source} | Similarity: {score:.3f}]\n{chunk.text}"
        )
    return "\n\n".join(sections)


def build_extractive_answer(
    query: str,
    retrieved: list[tuple[DocumentChunk, float]],
    reason: str,
) -> str:
    sources = sorted({chunk.source for chunk, _score in retrieved})
    context_points = []
    for chunk, _score in retrieved[:2]:
        sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
        for sentence in sentences:
            cleaned = sentence.strip()
            if cleaned and len(cleaned) > 30:
                context_points.append(cleaned)
            if len(context_points) >= 5:
                break
        if len(context_points) >= 5:
            break

    boundary = ""
    if is_decision_or_approval_request(query):
        boundary = (
            "I cannot approve work, decide whether an action is safe, certify compliance, "
            "or replace a supervisor/safety officer.\n\n"
        )

    points = "\n".join(f"- {point}" for point in context_points)
    return (
        f"{boundary}"
        f"Gemini is unavailable right now ({reason}), so this is an approved-document extract instead of an AI-generated explanation.\n\n"
        f"{points}\n\n"
        f"Sources used: {', '.join(sources) if sources else 'approved documents'}"
    )


def generate_answer(
    client: genai.Client,
    query: str,
    retrieved: list[tuple[DocumentChunk, float]],
    needs_safety_boundary: bool,
    generation_model: str,
) -> str:
    context = format_context(retrieved)
    boundary_note = (
        "The user request appears to ask for approval, permission, safety judgment, or operational authorization. "
        "Refuse that decision-making part while still explaining relevant SOP information from the context."
        if needs_safety_boundary
        else "The user request appears explanation-oriented."
    )
    prompt = f"""
{boundary_note}

Approved retrieved context:
{context}

Employee question:
{query}

Answer requirements:
- Ground the answer in the approved retrieved context.
- Use simple language and practical explanation.
- Include a short "Sources used" line listing source document names.
- Do not invent missing procedures.
"""
    try:
        response = client.models.generate_content(
            model=generation_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2,
            ),
        )
        return response.text or "I could not generate a response from the approved context."
    except Exception as exc:
        raise GeminiServiceError(f"Could not connect to Gemini Flash: {friendly_gemini_error(exc)}") from exc


def friendly_gemini_error(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    if "getaddrinfo failed" in lowered or "name resolution" in lowered:
        return (
            "your computer cannot resolve the Gemini API address. Check internet, DNS, VPN, proxy, "
            "firewall, or try a mobile hotspot."
        )
    if "api key" in lowered or "permission" in lowered or "unauthenticated" in lowered or "forbidden" in lowered:
        return "the API key may be invalid, restricted, expired, or not allowed for Gemini API."
    if "quota" in lowered or "rate" in lowered:
        return "the Google AI Studio quota or rate limit was reached."
    return message or exc.__class__.__name__


def error_next_step(message: str) -> str:
    lowered = message.lower()
    if "quota" in lowered or "rate limit" in lowered:
        return (
            "Wait a few minutes and try again, use a different Google AI Studio key/project with available quota, "
            "or check your Google AI Studio quota limits."
        )
    if "api key" in lowered or "permission" in lowered or "forbidden" in lowered:
        return "Create or rotate a Google AI Studio API key, then paste the new key into the sidebar."
    return (
        "Check internet/DNS/VPN/firewall settings and confirm that Google AI Studio is reachable from this machine."
    )


def render_sidebar(documents: dict[str, str]) -> None:
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input(
            "Google AI Studio API key",
            type="password",
            value=st.session_state.get("api_key", ""),
            help="You can also set GEMINI_API_KEY as an environment variable or Streamlit secret.",
        )
        if api_key:
            st.session_state["api_key"] = api_key

        generation_model = st.text_input(
            "Gemini Flash model",
            value=get_generation_model(),
            help="Override with GEMINI_FLASH_MODEL if your Google AI Studio project uses a different Flash model.",
        )
        if generation_model:
            st.session_state["generation_model"] = generation_model.strip()

        st.divider()
        st.subheader("Approved knowledge base")
        st.caption(f"Folder: {APPROVED_DOCS_DIR.as_posix()}")
        if documents:
            for name in documents:
                st.write(f"- {name}")
        else:
            st.warning("No approved SOP or safety documents were found.")

        if st.button("Reload documents"):
            cached_documents.clear()
            st.rerun()


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("RAG-grounded explanation assistant for approved manufacturing SOPs and safety documents.")

    documents = cached_documents()
    render_sidebar(documents)

    api_key = get_api_key()
    if not api_key:
        st.info("Enter a Google AI Studio API key in the sidebar or set GEMINI_API_KEY to start.")
        return

    if not documents:
        st.error("Add approved .txt, .md, .pdf, or .docx files to data/approved_documents.")
        return

    client = get_client(api_key)

    try:
        with st.spinner("Preparing approved SOP knowledge base..."):
            chunks = build_or_load_index(client, documents)
    except GeminiServiceError as exc:
        st.error(str(exc))
        st.info(
            "The approved documents were found, but Gemini could not be reached to build or refresh the RAG index. "
            "Check internet/DNS/VPN/firewall and confirm the Google AI Studio API key."
        )
        return

    st.success(f"Ready with {len(chunks)} approved document chunks.")

    examples = [
        "Explain lockout-tagout in simple terms",
        "What safety gear is used near heavy machines?",
        "Summarize emergency shutdown procedure",
        "Explain this SOP step-by-step",
    ]
    selected = st.selectbox("Try a tested query", [""] + examples)
    question = st.text_area(
        "Ask an SOP or safety explanation question",
        value=selected,
        height=120,
        placeholder="Example: Explain lockout-tagout in simple terms.",
    )

    if st.button("Generate explanation", type="primary", disabled=not question.strip()):
        retrieval_warning = ""
        with st.spinner("Retrieving approved context and generating explanation..."):
            try:
                retrieved = retrieve_relevant_chunks(client, question, chunks)
            except GeminiServiceError as exc:
                retrieval_warning = str(exc)
                retrieved = retrieve_keyword_chunks(question, chunks)

            needs_boundary = is_decision_or_approval_request(question)
            try:
                answer = generate_answer(client, question, retrieved, needs_boundary, get_generation_model())
            except GeminiServiceError as exc:
                answer = build_extractive_answer(question, retrieved, str(exc))
                st.warning(str(exc))
                st.info(error_next_step(str(exc)))

        if retrieval_warning:
            st.warning(retrieval_warning)
            st.info("Semantic search was unavailable, so the app used keyword matching over approved documents.")

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved approved context"):
            for chunk, score in retrieved:
                st.markdown(f"**{chunk.source}** - similarity `{score:.3f}`")
                st.write(chunk.text)

    st.divider()
    st.caption(
        "This tool explains approved documents only. It does not approve work, authorize operations, "
        "certify compliance, or replace supervisors."
    )


if __name__ == "__main__":
    main()
