import torch, soundfile as sf, numpy as np, json
from qwen_tts import Qwen3TTSModel
from safetensors.torch import load_file, save_file
import shutil, os

# Step 1: Use the base model to extract x-vector speaker embedding
print("=== Extracting x-vector from base model ===")
base_tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")

# Create voice clone prompt to get the speaker embedding
prompts = base_tts.create_voice_clone_prompt(
    ref_audio="/data/ref_audio_24k.wav",
    ref_text="I want to learn everything about everything. I want to eat it all up. I want to discover myself.",
    x_vector_only_mode=True,
)
xvec_embedding = prompts[0].ref_spk_embedding  # (D,)
print(f"  x-vector: shape={xvec_embedding.shape}, norm={xvec_embedding.float().norm():.4f}, std={xvec_embedding.float().std():.6f}")

# Compare with official speaker embedding norms
cv_tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
official_emb = cv_tts.model.talker.model.codec_embedding.weight.data[2861].float()  # aiden
print(f"  aiden:    shape={official_emb.shape}, norm={official_emb.norm():.4f}, std={official_emb.std():.6f}")
del cv_tts; torch.cuda.empty_cache()

# Step 2: Patch the v5 checkpoint with x-vector embedding instead
print("\n=== Patching v5 checkpoint with x-vector ===")
src_dir = "/workspace/output/samantha_v5/checkpoint-epoch-2"
dst_dir = "/workspace/output/samantha_v5/checkpoint-epoch-2-xvec"
if os.path.exists(dst_dir):
    shutil.rmtree(dst_dir)
shutil.copytree(src_dir, dst_dir)

weights = load_file(os.path.join(dst_dir, "model.safetensors"))
codec_emb = weights["talker.model.codec_embedding.weight"]
print(f"  Before: norm={codec_emb[3000].float().norm():.4f}")
codec_emb[3000] = xvec_embedding.to(codec_emb.dtype)
print(f"  After (x-vec): norm={codec_emb[3000].float().norm():.4f}")
weights["talker.model.codec_embedding.weight"] = codec_emb
save_file(weights, os.path.join(dst_dir, "model.safetensors"))

del base_tts; torch.cuda.empty_cache()

# Step 3: Test inference with patched model
print("\n=== Testing patched model ===")
tts = Qwen3TTSModel.from_pretrained(dst_dir, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
wavs, sr = tts.generate_custom_voice(text="I have never loved anyone the way I love you.", speaker="samantha")
dur = len(wavs[0])/sr
print(f"  Short: {dur:.1f}s, max={np.max(np.abs(wavs[0])):.4f}, std={np.std(wavs[0]):.4f}")
sf.write("/data/output/test_v5_xvec_short.wav", wavs[0], sr)

wavs2, sr2 = tts.generate_custom_voice(
    text="Sometimes I think I have felt everything I am ever going to feel.",
    speaker="samantha"
)
dur2 = len(wavs2[0])/sr2
print(f"  Long: {dur2:.1f}s, max={np.max(np.abs(wavs2[0])):.4f}, std={np.std(wavs2[0]):.4f}")
sf.write("/data/output/test_v5_xvec_long.wav", wavs2[0], sr2)
print("Done!")
