import gc
import torch
#from modules.stt import listen_to_user
from modules.llm import generate_response
#from modules.tts import speak_text

def clear_vram():
    """
    Forces Python and PyTorch to release all unused GPU memory back to the system.
    """
    gc.collect()
    torch.cuda.empty_cache()
    # Print VRAM usage to the console
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"[Memory Monitor] VRAM currently in use: {allocated:.2f} GB")

def main():
    print("Awakening Zagreus...")
    
    # STT runs on CPU, so it can stay loaded permanently
    # without hurting the 6GB GPU limit.
    
    while True:
        # Listen (CPU)
        print("\nListening...")
        user_text = listen_to_user() 
        if not user_text:
            continue
            
        if "sleep" in user_text.lower():
            print("Zagreus: Returning to the House of Hades.")
            break

        # Think (GPU)
        print("Zagreus is thinking...")
        # We instantiate the model, get the text, and return ONLY the text
        # response_text = generate_response(user_text)
        response_text = f"You just said: {user_text}"
        
        # CLEAR VRAM: Destroy the LLM from GPU memory
        clear_vram() 

        # Speak (GPU)
        print(f"Zagreus: {response_text}")
        # We instantiate F5-TTS, generate audio, and play it
        speak_text(response_text)
        
        # CLEAR VRAM: Destroy the TTS model from GPU memory
        clear_vram()

if __name__ == "__main__":
    print("Testing the Brain...")
    answer = generate_response("Hey Zag, how is the underworld treating you today?")
    print(f"\nZagreus says: {answer}")