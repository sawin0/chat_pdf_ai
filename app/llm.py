import requests
import os

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

MAX_PROMPT_LENGTH = 3000  # max characters for context + question


def ask_local_llm(
    question: str, context: str, model: str = "phi3:mini", max_tokens: int = 256
) -> str:
    # Ensure inputs are strings
    question = str(question or "")
    context = str(context or "")

    # Prompt guardrail: force LLM to answer only from context
    prompt = (
        "You are a helpful assistant that answers questions ONLY using the provided context. "
        "If the answer is not contained in the context, respond with 'I don't know'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    # Truncate prompt if too long
    if len(prompt) > MAX_PROMPT_LENGTH:
        prompt = prompt[:MAX_PROMPT_LENGTH] + "\n\n...[truncated]"

    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }

        # Use Ollama's native generate endpoint.
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate", json=payload, timeout=150
        )
        if response.status_code >= 400:
            err_detail = ""
            try:
                err_detail = str(response.json().get("error", "")).strip()
            except ValueError:
                err_detail = response.text.strip()

            if response.status_code == 404 and "model" in err_detail.lower():
                return (
                    f"Error: Model '{model}' is not available in Ollama. "
                    f"Run: docker compose exec ollama ollama pull {model}"
                )

            return (
                f"Error: Ollama request failed with status {response.status_code}"
                + (f" ({err_detail})" if err_detail else "")
            )

        result = response.json()

        # /api/generate returns text in `response` when stream=false.
        answer = result.get("response", "")
        if answer:
            return answer.strip()
        return "Error: LLM returned an empty response."

    except requests.exceptions.RequestException as e:
        return f"Error: Failed to connect to LLM server ({e})"
    except ValueError:
        return "Error: Invalid response from LLM server."
