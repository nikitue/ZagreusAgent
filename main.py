import gc
import torch
from modules.stt import record_audio, transcribe_audio
from modules.llm import generate_response
from modules.tts import speak_text

def clear_vram():
    """
    Forces Python and PyTorch to release all unused GPU memory back to the system.
    """
    gc.collect()
    torch.cuda.empty_cache()
    # Print VRAM usage to the console
    #allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    #print(f"[Memory Monitor] VRAM currently in use: {allocated:.2f} GB")

def main():
    print("Awakening Zagreus...")

    #Start clean
    clear_vram()

    while True:
        try:
            # Listen
            print("\nListening...")
            raw_audio = record_audio(duration=10)
            user_text = transcribe_audio(raw_audio)
                
            # Whisper finishes, immediately sweep it out of VRAM
            clear_vram() 
            if not user_text:
                continue

            # If the microphone just heard background noise, skip the loop
            if len(user_text) < 2:
                continue

            print(f"\nYou: \"{user_text}\"")
                
            # Secret kill-switch to exit the loop gracefully
            if "sleep" in user_text.lower() or "goodbye" in user_text.lower():
                print("Zagreus: Returning to the House of Hades.")
                break
                
            # Think (GPU)
            print("Zagreus is thinking...")
            response_text = generate_response(user_text)
                
            print(f"Zagreus: {response_text}") 
            clear_vram() # Just to be safe (keep_alive=0 is enough)

            # Speak (GPU)
            # We instantiate F5-TTS, generate audio, and play it
            speak_text(response_text)
            clear_vram()
        except KeyboardInterrupt:
            # Allows you to press Ctrl+C in the terminal to safely exit
            print("\n[System] Shutting down Zagreus...")
            break
        except Exception as e:
            print(f"\n[System Error]: {e}")
            clear_vram()

if __name__ == "__main__":
    main()