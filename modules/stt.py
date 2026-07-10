import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import os
import sys

# Explicitly tell Windows where to find the pip-installed NVIDIA CUDA 12 files
site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
cublas_bin = os.path.join(site_packages, "nvidia", "cublas", "bin")
cudnn_bin = os.path.join(site_packages, "nvidia", "cudnn", "bin")

# Prepend to PATH for underlying C/C++ libraries
os.environ["PATH"] = cublas_bin + os.pathsep + cudnn_bin + os.pathsep + os.environ["PATH"]

# Add directory directly for Python 3.8+ DLL loading
if hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(cublas_bin)
        os.add_dll_directory(cudnn_bin)
    except FileNotFoundError:
        pass

def record_audio(duration=4, sample_rate=16000):
    """Records audio from the microphone into a numpy array."""
    print(f"\n[Ears] Listening for {duration} seconds...")
    
    # Record mono audio at 16kHz (Whisper's native standard)
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    
    print("[Ears] Processing audio...")
    # Flatten the array because sounddevice returns a 2D array [[sample], [sample]]
    return audio.flatten()

def transcribe_audio(audio_data):
    """Transcribes the raw numpy audio array using faster-whisper on the GPU."""
    # Use 'base.en' because it's fast, accurate, and tiny.
    # Use float16 to make sure it runs optimally on RTX 3060.
    model = WhisperModel("base.en", device="cuda", compute_type="float16")
    
    # Transcribe the numpy array directly without saving to a file
    segments, info = model.transcribe(
                    audio_data,
                    beam_size=5,
                    initial_prompt="Zagreus, Hades, Underworld, Olympus, Agent, Assistant." # Understand easier the name Zagreus
                    )
    
    # Combine the transcribed segments into a single string
    text = "".join([segment.text for segment in segments])
    return text.strip()

#Test main
if __name__ == "__main__":
    print("Testing the Ears...")
    
    # Record voice
    raw_audio = record_audio(duration=4)
    
    # Convert to text
    text_output = transcribe_audio(raw_audio)
    
    print(f"\n[Result] Whisper heard: \"{text_output}\"")