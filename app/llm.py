import os
from ollama import ResponseError, chat

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

MAX_PROMPT_LENGTH = 3000  # max characters for context + question


def ask_local_llm(
    question: str, context: str, model: str = "phi3:mini", max_tokens: int = 256
) -> str:
    # Ensure inputs are strings
    question = str(question or "")
    context = str(context or "")

    # Prompt guardrail: force LLM to answer only from context.
    system_prompt = (
        "You are a helpful assistant that answers questions ONLY using the provided context. "
        "If the answer is not contained in the context, respond with 'I don't know'."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    # Truncate user prompt if too long.
    if len(user_prompt) > MAX_PROMPT_LENGTH:
        user_prompt = user_prompt[:MAX_PROMPT_LENGTH] + "\n\n...[truncated]"

    try:
        # Use the official Ollama Python SDK chat API.
        os.environ["OLLAMA_HOST"] = OLLAMA_HOST
        response = chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"num_predict": max_tokens},
        )

        # SDK returns message content as response.message.content.
        answer = ""
        if getattr(response, "message", None) is not None:
            answer = str(getattr(response.message, "content", "") or "")
        elif isinstance(response, dict):
            answer = str(response.get("message", {}).get("content", "") or "")

        if answer:
            return answer.strip()
        return "Error: LLM returned an empty response."

    except ResponseError as e:
        err_detail = str(getattr(e, "error", str(e))).strip()
        status_code = getattr(e, "status_code", None)

        if status_code == 404 and "model" in err_detail.lower():
            return (
                f"Error: Model '{model}' is not available in Ollama. "
                f"Run: docker compose exec ollama ollama pull {model}"
            )

        if status_code is not None:
            return f"Error: Ollama request failed with status {status_code}" + (
                f" ({err_detail})" if err_detail else ""
            )

        return f"Error: Ollama request failed ({err_detail})"
    except Exception as e:
        return f"Error: Failed to connect to LLM server ({e})"
