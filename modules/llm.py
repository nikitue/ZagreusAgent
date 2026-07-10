import requests

def generate_response(user_text):
    """
    Sends the user text to the local Ollama server.
    The keep_alive=0 parameter acts as our VRAM Orchestrator,
    instantly freeing the GPU for the F5-TTS model.
    """
    print("[System] Sending prompt to LLM...")
    
    url = "http://localhost:11434/api/generate"
    
    # We pass the user's text inside a system prompt instructing 
    # the LLM to act like Zagreus from Hades.
    payload = {
        "model": "qwen2.5:3b",
        "prompt": f"You are Zagreus from the game Hades. Respond directly, concisely, and with his characteristic rebellious but polite tone. User: {user_text}",
        "stream": False,
        "keep_alive": 0
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        return f"Blood and darkness, something went wrong with my thoughts: {e}"