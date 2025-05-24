import requests
import json

MODEL = "gemma3:1b"
OLLAMA_URL = "http://localhost:11434/"


def generate(prompt: str, context: list, top_k: float, top_p: float, temp: float):
    """
    Generate a response from the model.

    Args:
        prompt (str): The prompt to generate a response from
        context (list): Ollama's internal context for maintaining conversation state
        top_k (float): The top_k parameter
        top_p (float): The top_p parameter
        temp (float): The temperature parameter

    Raises:
        Exception: If the request fails

    Returns:
        tuple: The response and the updated context
    """
    formatted_prompt = f"User: {prompt}\nAssistant:"

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": MODEL,
            "prompt": formatted_prompt,
            "context": context,
            "options": {"top_k": top_k, "temperature": temp, "top_p": top_p},
        },
        stream=False,
    )
    response.raise_for_status()
    final_response = ""
    for line in response.iter_lines():
        body = json.loads(line)
        response_part = body.get("response", "")
        if "error" in body:
            raise Exception(body["error"])
        final_response += response_part
        if body.get("done", False):
            context = body.get("context", [])
            break
    return final_response, context
