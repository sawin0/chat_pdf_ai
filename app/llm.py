import os
import re
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None

MAX_CONTEXT_LENGTH = 6000

SYSTEM_PROMPT = (
    "You are a grounded RAG assistant. "
    "Use only the supplied context to answer. "
    "Always answer in Nepali language. "
    "If the context is insufficient, reply exactly: माफ गर्नुहोस्, यसबारे मलाई कुनै जानकारी छैन, त्यसैले मद्दत गर्न सकिनँ।"
)


def _build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Question:\n{question}\n\n"
                "Context:\n{context}\n\n"
                "Answer in concise plain text.",
            ),
        ]
    )


def _is_complex_question(question: str) -> bool:
    lowered = question.lower()
    complexity_terms = [
        "compare",
        "difference",
        "analyze",
        "explain why",
        "impact",
        "summary",
        "summarize",
        "pros and cons",
    ]
    return len(question) > 80 or any(term in lowered for term in complexity_terms)


def _get_candidate_providers(question: str) -> List[str]:
    # Route simple prompts to low-cost/free-tier-friendly providers first.
    if _is_complex_question(question):
        return ["gemini", "deepseek", "groq", "grok", "openai"]
    return ["gemini", "groq", "deepseek", "grok", "openai"]


def _provider_llm(provider: str):
    if provider == "gemini" and os.getenv("GEMINI_API_KEY"):
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"), temperature=0
        )

    if provider == "deepseek" and os.getenv("DEEPSEEK_API_KEY"):
        return ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            temperature=0,
        )

    if provider == "groq" and os.getenv("GROQ_API_KEY"):
        if ChatGroq is None:
            return None
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )

    grok_key = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
    if provider == "grok" and grok_key:
        return ChatOpenAI(
            model=os.getenv("GROK_MODEL", "grok-2-latest"),
            api_key=grok_key,
            base_url="https://api.x.ai/v1",
            temperature=0,
        )

    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
        )

    return None


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[\w\u0900-\u097F]+", text.lower())


def _is_song_request(question: str) -> bool:
    lowered = question.lower()
    song_terms = ["गीत", "गित", "भजन", "lyrics", "song", "कीर्तन", "संकीर्तन"]
    return any(term in lowered for term in song_terms)


def _extract_song_lines(question: str, context: str, max_lines: int = 8) -> str:
    segments = [
        part.strip()
        for part in re.split(r"(?<=[।॥!?])\s+|\n+", context)
        if part.strip()
    ]
    if not segments:
        return ""

    q_tokens = set(_tokenize(question))
    bad_markers = ["*यो तस्बिरमा", "सम्पादक", "copyright", "http://", "https://"]
    devotional_markers = ["राधा", "राधे", "गोविन्द", "गोविंद"]
    anchor_terms = [t for t in devotional_markers if t in question.lower()]

    scored = []
    for i, line in enumerate(segments):
        lowered = line.lower()
        if any(marker in lowered for marker in bad_markers):
            continue
        if len(line) > 180:
            continue
        if anchor_terms and not any(anchor in lowered for anchor in anchor_terms):
            continue

        line_tokens = set(_tokenize(line))
        overlap = len(q_tokens.intersection(line_tokens))
        marker_boost = (
            4 if any(marker in lowered for marker in devotional_markers) else 0
        )
        verse_boost = 1 if "॥" in line or "।" in line else 0
        length_penalty = 1 if len(line) > 120 else 0
        score = overlap + marker_boost + verse_boost - length_penalty

        if score > 0:
            scored.append((score, i, line))

    if not scored:
        return ""

    scored.sort(key=lambda x: (-x[0], x[1]))
    selected = scored[:max_lines]
    # Preserve original document order in final output.
    selected.sort(key=lambda x: x[1])

    dedup = []
    seen = set()
    for _, _, line in selected:
        if line in seen:
            continue
        seen.add(line)
        dedup.append(line)

    return "\n".join(dedup)


def _extractive_fallback(question: str, context: str, max_sentences: int = 4) -> str:
    if _is_song_request(question):
        song_lines = _extract_song_lines(question, context)
        if song_lines:
            return song_lines

    # Deterministic fallback when all external providers fail.
    normalized_context = " ".join(context.split())
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?।])\s+|\n+", normalized_context)
        if s.strip()
    ]

    if not sentences:
        return "माफ गर्नुहोस्, यसबारे मलाई कुनै जानकारी छैन, त्यसैले मद्दत गर्न सकिनँ।"

    q_tokens = set(_tokenize(question))
    stop_words = {
        "the",
        "is",
        "are",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "what",
        "which",
        "how",
        "why",
        "when",
        "where",
        "को",
        "का",
        "कि",
        "र",
        "मा",
        "के",
        "किन",
        "कसरी",
        "कुन",
    }
    q_tokens = {t for t in q_tokens if t not in stop_words and len(t) > 1}

    if not q_tokens:
        top = sentences[:max_sentences]
        return "\n".join(top)

    ranked = []
    for s in sentences:
        s_tokens = set(_tokenize(s))
        overlap = len(q_tokens.intersection(s_tokens))
        if overlap > 0:
            ranked.append((overlap, s))

    if not ranked:
        top = sentences[:max_sentences]
        return "\n".join(top)

    ranked.sort(key=lambda x: x[0], reverse=True)
    selected = []
    seen = set()
    for _, sentence in ranked:
        if sentence in seen:
            continue
        seen.add(sentence)
        selected.append(sentence)
        if len(selected) >= max_sentences:
            break
    return "\n".join(selected)


def _extract_content(response) -> str:
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for chunk in content:
            if isinstance(chunk, dict):
                text = chunk.get("text")
                if text:
                    parts.append(str(text))
            elif isinstance(chunk, str):
                parts.append(chunk)
        return "\n".join(parts).strip()
    return str(content).strip()


def ask_rag_llm(question: str, context: str) -> str:
    question = str(question or "").strip()
    context = str(context or "").strip()

    if not context:
        return "Error: No relevant content found in the indexed PDFs. Please process a PDF first using the /process-pdf endpoint."

    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH].rstrip() + "\n\n...[truncated]"

    prompt = _build_prompt()
    providers = _get_candidate_providers(question)
    errors = []
    any_provider_configured = False

    for provider in providers:
        llm = _provider_llm(provider)
        if llm is None:
            continue
        any_provider_configured = True

        try:
            chain = prompt | llm
            response = chain.invoke({"question": question, "context": context})
            answer = _extract_content(response)
            if answer:
                return answer
        except Exception as exc:
            errors.append(f"{provider}: {exc}")
            continue

    if errors:
        return _extractive_fallback(question, context)

    if not any_provider_configured:
        return _extractive_fallback(question, context)

    return "माफ गर्नुहोस्, यसबारे मलाई कुनै जानकारी छैन, त्यसैले मद्दत गर्न सकिनँ।"
