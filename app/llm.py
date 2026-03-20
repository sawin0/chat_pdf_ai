MAX_CONTEXT_LENGTH = 3000


def ask_local_llm(question: str, context: str) -> str:
    question = str(question or "").strip()
    context = str(context or "").strip()

    if not context:
        return "I don't know"

    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH].rstrip() + "\n\n...[truncated]"

    return (
        "Retrieved context relevant to the question:\n\n"
        f"Question: {question}\n\n"
        f"{context}"
    )
