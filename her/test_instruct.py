import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel

# Try the official Instruct model (not Base) to see if TTS works at all
print("Loading Qwen3-TTS-12Hz-1.7B (Instruct/CustomVoice)...")
tts = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
print(f"Model type: {tts.model.tts_model_type}")
print(f"Speakers: {tts.get_supported_speakers()}")

# Generate with a built-in speaker
spk = tts.get_supported_speakers()[0] if tts.get_supported_speakers() else "default"
print(f"Using speaker: {spk}")
wavs, sr = tts.generate_custom_voice(
    text="I have never loved anyone the way I love you.",
    speaker=spk,
)
dur = len(wavs[0])/sr
print(f"Output: {dur:.1f}s at {sr}Hz")
sf.write("/data/output/test_instruct.wav", wavs[0], sr)
print("Done!")
