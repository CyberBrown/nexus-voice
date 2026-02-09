import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import numpy as np

def save_result(result, path):
    audio_data, sr = result
    audio = np.array(audio_data).flatten()
    sf.write(path, audio, sr, subtype='PCM_16')
    print(f"  Saved {path}: {len(audio)} samples, {len(audio)/sr:.1f}s")

# Test 1: Base model voice clone
print("=== Test 1: Base model voice clone ===")
tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", dtype=torch.bfloat16)
result = tts.generate_voice_clone(
    text="I've never loved anyone the way I love you.",
    ref_audio="/data/ref_audio_24k.wav",
    ref_text="I want to learn everything about everything. I want to eat it all up. I want to discover myself."
)
save_result(result, "/data/output/test_base_v3.wav")
del tts
torch.cuda.empty_cache()

# Test 2: Fine-tuned model
print("\n=== Test 2: Fine-tuned v3 (epoch 2) ===")
tts = Qwen3TTSModel.from_pretrained("/workspace/output/samantha_v3/checkpoint-epoch-2", dtype=torch.bfloat16)
result = tts.generate_custom_voice(
    text="I've never loved anyone the way I love you.",
    speaker="samantha"
)
save_result(result, "/data/output/test_finetuned_v3.wav")

result2 = tts.generate_custom_voice(
    text="Sometimes I think I have felt everything I am ever going to feel, and from here on out I am not going to feel anything new, just lesser versions of what I have already felt.",
    speaker="samantha"
)
save_result(result2, "/data/output/test_finetuned_v3_long.wav")
print("\nDone!")
