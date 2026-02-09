import torch
import soundfile as sf
import numpy as np
from qwen_tts import Qwen3TTSModel

device = "cuda:0"

# Test 1: Base model voice clone
print("=== Test 1: Base model voice clone ===")
tts = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
wavs, sr = tts.generate_voice_clone(
    text="I've never loved anyone the way I love you.",
    ref_audio="/data/ref_audio_24k.wav",
    ref_text="I want to learn everything about everything. I want to eat it all up. I want to discover myself."
)
sf.write("/data/output/test_base_v3b.wav", wavs[0], sr)
print(f"Base output: {len(wavs[0])/sr:.1f}s at {sr}Hz")
del tts
torch.cuda.empty_cache()

# Test 2: Fine-tuned epoch 2
print("\n=== Test 2: Fine-tuned v3 (epoch 2) ===")
tts = Qwen3TTSModel.from_pretrained(
    "/workspace/output/samantha_v3/checkpoint-epoch-2",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
wavs, sr = tts.generate_custom_voice(
    text="I've never loved anyone the way I love you.",
    speaker="samantha"
)
sf.write("/data/output/test_ft_v3b.wav", wavs[0], sr)
print(f"Finetuned output: {len(wavs[0])/sr:.1f}s at {sr}Hz")

wavs2, sr2 = tts.generate_custom_voice(
    text="Sometimes I think I have felt everything I am ever going to feel, and from here on out I am not going to feel anything new, just lesser versions of what I have already felt.",
    speaker="samantha"
)
sf.write("/data/output/test_ft_v3b_long.wav", wavs2[0], sr2)
print(f"Finetuned long: {len(wavs2[0])/sr2:.1f}s at {sr2}Hz")

# Test 3: Fine-tuned epoch 0 (less training, might be more stable)
print("\n=== Test 3: Fine-tuned v3 (epoch 0) ===")
del tts
torch.cuda.empty_cache()
tts = Qwen3TTSModel.from_pretrained(
    "/workspace/output/samantha_v3/checkpoint-epoch-0",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
wavs3, sr3 = tts.generate_custom_voice(
    text="I've never loved anyone the way I love you.",
    speaker="samantha"
)
sf.write("/data/output/test_ft_v3b_ep0.wav", wavs3[0], sr3)
print(f"Epoch 0 output: {len(wavs3[0])/sr3:.1f}s at {sr3}Hz")

print("\nDone!")
