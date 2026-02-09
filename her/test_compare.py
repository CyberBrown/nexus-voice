import torch, soundfile as sf, numpy as np
from qwen_tts import Qwen3TTSModel

text = "I have never loved anyone the way I love you."

# Test 1: Official CustomVoice model (known working)
print("=== Official CustomVoice (aiden) ===")
tts1 = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
wavs, sr = tts1.generate_custom_voice(text=text, speaker="aiden")
print(f"  Duration: {len(wavs[0])/sr:.1f}s, max={np.max(np.abs(wavs[0])):.4f}, std={np.std(wavs[0]):.4f}")
sf.write("/data/output/compare_official.wav", wavs[0], sr)
del tts1; torch.cuda.empty_cache()

# Test 2: Our fine-tuned model
print("=== Fine-tuned samantha (epoch 2) ===")
tts2 = Qwen3TTSModel.from_pretrained("/workspace/output/samantha_v3/checkpoint-epoch-2", device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
wavs2, sr2 = tts2.generate_custom_voice(text=text, speaker="samantha")
print(f"  Duration: {len(wavs2[0])/sr2:.1f}s, max={np.max(np.abs(wavs2[0])):.4f}, std={np.std(wavs2[0]):.4f}")
sf.write("/data/output/compare_finetuned.wav", wavs2[0], sr2)

# Test 3: Fine-tuned epoch 0
print("=== Fine-tuned samantha (epoch 0) ===")
del tts2; torch.cuda.empty_cache()
tts3 = Qwen3TTSModel.from_pretrained("/workspace/output/samantha_v3/checkpoint-epoch-0", device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
wavs3, sr3 = tts3.generate_custom_voice(text=text, speaker="samantha")
print(f"  Duration: {len(wavs3[0])/sr3:.1f}s, max={np.max(np.abs(wavs3[0])):.4f}, std={np.std(wavs3[0]):.4f}")
sf.write("/data/output/compare_finetuned_ep0.wav", wavs3[0], sr3)

print("\nDone!")
