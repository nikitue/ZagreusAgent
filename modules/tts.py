import sounddevice as sd
from f5_tts.api import F5TTS

def speak_text(text):
    """
    Loads F5-TTS, clones Zagreus's voice, streams the audio, 
    and then finishes so VRAM can be cleared.
    """
    print("[Voice] Loading F5-TTS onto GPU...")
    # This will automatically download the F5-TTS model weights on the first run
    tts = F5TTS(model="F5TTS_Base", device="cuda")

    ref_audio = "assets/zagreusVO.wav"
    ref_text = "Say, Father... back when you drew lots with your brothers, and wound up with the Underworld, rather than the heavens or the seas... which would you have claimed, if the choice was yours?"
    print(f"Zagreus is saying: {text}")
    
    print("[Voice] Synthesizing speech...")
    
    # We use .infer() to generate the audio array
    output = tts.infer(
        ref_file=ref_audio,
        ref_text=ref_text,
        gen_text=text
    )
    
    # Safely extract the audio array and the sample rate 
    audio_wav = output[0]
    sample_rate = output[1]
    
    print("[Voice] Playing audio...")
    sd.play(audio_wav, samplerate=sample_rate, blocking=True)
        
    print("[Voice] Finished speaking.")

# Test main
if __name__ == "__main__":
    print("Testing the Voice...")
    
    # The text we want the AI to generate and speak in Zagreus's voice
    test_sentence = "Hey there, Niki. The Underworld is looking a bit brighter today, wouldn't you say? How can I assist you..."
    
    speak_text(test_sentence)
    