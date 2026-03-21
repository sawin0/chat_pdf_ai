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
FALLBACK = "माफ गर्नुहोस्, यसबारे मलाई कुनै जानकारी छैन, त्यसैले मद्दत गर्न सकिनँ।"


# ----------------- SYSTEM PROMPT -----------------
SYSTEM_PROMPT = (
    "You are a grounded Retrieval-Augmented Generation (RAG) assistant.\n"
    "Follow these rules strictly:\n"
    "- Use ONLY the provided context.\n"
    "- Extract ONLY the part that directly answers the question.\n"
    "- Ignore irrelevant text like descriptions, metadata, or noise unless explicitly asked.\n"
    "- If the question asks for a song/poem, return ONLY the poem lines.\n"
    "- If the question asks to describe an image, then ONLY describe the image-related content from the context.\n"
    "- If the question is NOT about an image, ignore any image descriptions in the context.\n"
    "- Do NOT include explanations unless explicitly asked.\n"
    "- Context may contain OCR errors; interpret carefully.\n"
    f"- If context is insufficient, reply EXACTLY with:\n  {FALLBACK}\n"
    "- Always answer in Nepali.\n"
    "- Keep answer concise and clean.\n"
)


def _build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Question:\n{question}\n\n"
                "Task:\n- Identify if this is: [poem | image | general]\n"
                "- Answer accordingly.\n\n"
                "Context:\n{context}\n\nAnswer:",
            ),
        ]
    )


# ----------------- RESPONSE PARSER -----------------
def _extract_content(response) -> str:
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            for chunk in content
        ).strip()
    return str(content).strip()


# ----------------- CONTEXT CLEANING -----------------
def clean_context(text: str) -> str:
    lines = text.split("\n")
    filtered = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Remove OCR/image noise
        if any(word in line for word in ["तस्बिर", "देखिन्छ", "पृष्ठभूमि"]):
            continue

        # Remove overly long garbage lines
        if len(line) > 2000:
            continue

        filtered.append(line)

    return "\n".join(filtered)


# ----------------- INTENT DETECTION -----------------
def detect_intent(question: str) -> str:
    q = question.lower()

    if any(word in q for word in ["गीत", "song", "भजन"]):
        return "poem"

    if any(word in q for word in ["तस्बिर", "image", "चित्र", "describe"]):
        return "image"

    return "general"


# ----------------- ANSWER VALIDATION -----------------
def is_valid_answer(answer: str, question: str, intent: str) -> bool:
    if not answer:
        return False

    # Reject obvious OCR/image noise
    if any(word in answer for word in ["तस्बिर", "देखिन्छ", "पृष्ठभूमि"]):
        return False

    # Poem validation
    if intent == "poem":
        if not any(word in answer for word in ["राधे", "गोविन्द", "कृष्ण"]):
            return False

    return True


# ----------------- LLM PROVIDERS -----------------
def _provider_llm(provider: str):
    if provider == "gemini" and os.getenv("GEMINI_API_KEY"):
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            temperature=0,
        )

    if provider == "deepseek" and os.getenv("DEEPSEEK_API_KEY"):
        return ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            temperature=0,
        )

    if provider == "groq" and os.getenv("GROQ_API_KEY") and ChatGroq:
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )

    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
        )

    return None


def _get_candidate_providers() -> List[str]:
    return ["gemini", "groq", "deepseek", "openai"]


# ----------------- MAIN FUNCTION -----------------
def ask_rag_llm(question: str, context: str) -> str:
    question = str(question or "").strip()
    context = str(context or "").strip()

    if not context:
        return FALLBACK

    # ✅ Clean context
    context = clean_context(context)

    # ✅ Detect intent
    intent = detect_intent(question)

    prompt = _build_prompt()

    providers = _get_candidate_providers()

    for provider in providers:
        llm = _provider_llm(provider)
        if llm is None:
            continue

        try:
            chain = prompt | llm
            response = chain.invoke({"question": question, "context": context})
            answer = _extract_content(response)

            print("\n--- RAW ANSWER ---\n", answer)

            # ✅ Validate answer
            if is_valid_answer(answer, question, intent):
                return answer

        except Exception as e:
            print(f"\n❌ ERROR from {provider}: {e}\n")
            continue
    # ✅ Safe fallback (NO leakage)
    return FALLBACK
