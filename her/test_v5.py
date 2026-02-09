import torch, soundfile as sf, numpy as np
from qwen_tts import Qwen3TTSModel

print("=== v5 embedding check ===")
tts = Qwen3TTSModel.from_pretrained("/workspace/output/samantha_v5/checkpoint-epoch-2", device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
emb = tts.model.talker.model.codec_embedding.weight.data[3000].float()
print(f"  samantha: mean={emb.mean():.6f}, std={emb.std():.6f}, norm={emb.norm():.4f}")

print("\n=== v5 inference ===")
wavs, sr = tts.generate_custom_voice(text="I have never loved anyone the way I love you.", speaker="samantha")
dur = len(wavs[0])/sr
print(f"  Short: {dur:.1f}s, max={np.max(np.abs(wavs[0])):.4f}, std={np.std(wavs[0]):.4f}")
sf.write("/data/output/test_v5_short.wav", wavs[0], sr)

wavs2, sr2 = tts.generate_custom_voice(
    text="Sometimes I think I have felt everything I am ever going to feel, and from here on out I am not going to feel anything new, just lesser versions of what I have already felt.",
    speaker="samantha"
)
dur2 = len(wavs2[0])/sr2
print(f"  Long: {dur2:.1f}s, max={np.max(np.abs(wavs2[0])):.4f}, std={np.std(wavs2[0]):.4f}")
sf.write("/data/output/test_v5_long.wav", wavs2[0], sr2)
print("Done!")
