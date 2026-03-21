import os
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


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


# ----------------- LLM PROVIDER -----------------
def _get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        api_key=api_key,
        temperature=0,
    )


# ----------------- MAIN FUNCTION -----------------
def ask_rag_llm(question: str, context: str) -> str:
    question = str(question or "").strip()
    raw_context = str(context or "").strip()

    print(
        "\n--- RAW INPUT ---\n"
        f"Question length: {len(question)}\n"
        f"Context length: {len(raw_context)}\n"
        f"Context preview:\n{raw_context[:800]}\n"
    )

    if not raw_context:
        return FALLBACK

    context = clean_context(raw_context)

    if not context and raw_context:
        print("\n⚠️ Cleaned context became empty. Falling back to raw context.\n")
        context = raw_context

    print(
        "\n--- CLEANED CONTEXT ---\n"
        f"Context length: {len(context)}\n"
        f"Context preview:\n{context[:800]}\n"
    )

    intent = detect_intent(question)
    prompt = _build_prompt()

    print("\n--- PROMPT ---\n" f"Question:\n{question}\n\n" f"Context:\n{context}\n")

    llm = _get_llm()
    if llm is None:
        print("\n❌ No GROQ_API_KEY set.\n")
        return FALLBACK

    try:
        chain = prompt | llm
        response = chain.invoke({"question": question, "context": context})
        answer = _extract_content(response)

        print("\n--- RAW ANSWER ---\n", answer)

        if is_valid_answer(answer, question, intent):
            return answer
    except Exception as e:
        print(f"\n❌ ERROR from groq: {e}\n")

    return FALLBACK
