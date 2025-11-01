import torch
import torch.serialization
from TTS.api import TTS
from pydub import AudioSegment
import glob

# Add safe globals to bypass the security check
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    torch.serialization.add_safe_globals([XttsConfig])
except ImportError:
    pass

# Fallback: disable weights_only check
_original_load = torch.load
def _patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

# Step 1: Combine all MP3 samples with proper formatting
print("Combining voice samples...")
samples = glob.glob("enrollment_samples/*.mp3")
print(f"Found {len(samples)} samples")

combined = AudioSegment.empty()
for sample in samples:
    print(f"  Adding: {sample}")
    audio = AudioSegment.from_file(sample, format="mp3")
    
    # Convert to mono if stereo
    if audio.channels > 1:
        audio = audio.set_channels(1)
    
    # Resample to 22050 Hz (optimal for TTS)
    if audio.frame_rate != 22050:
        audio = audio.set_frame_rate(22050)
    
    combined += audio

# Ensure final audio is properly formatted
combined = combined.set_channels(1)  # Mono
combined = combined.set_frame_rate(22050)  # 22050 Hz for best TTS quality

combined.export("combined_voice.wav", format="wav")
print(f"Combined voice saved: combined_voice.wav")
print(f"  - Sample rate: {combined.frame_rate} Hz")
print(f"  - Channels: {combined.channels} (mono)")
print(f"  - Duration: {len(combined)/1000:.2f} seconds")

# Step 2: Initialize TTS
print("\nLoading TTS model...")
tts = TTS("tts_models/multilingual/maasdulti-dataset/xtts_v2")

# Step 3: Generate speech
print("Generating speech...")
tts.tts_to_file(
    text="The quantum entanglement experiments yielded promising results for our neural network architecture.",
    speaker_wav="combined_voice.wav",
    file_path="output.mp3",
    language="en"
)

print("\nDone! Check output.mp3")