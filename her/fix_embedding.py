import torch, json, os, shutil
from safetensors.torch import save_file, load_file
import numpy as np

# Target stats from official speakers (average)
TARGET_STD = 0.333  # average of official speakers
TARGET_NORM = 15.08  # average of official speakers

checkpoint_dir = "/workspace/output/samantha_v3/checkpoint-epoch-2"
fixed_dir = "/workspace/output/samantha_v3/checkpoint-epoch-2-fixed"

# Copy checkpoint
if os.path.exists(fixed_dir):
    shutil.rmtree(fixed_dir)
shutil.copytree(checkpoint_dir, fixed_dir)

# Load weights
weights = load_file(os.path.join(fixed_dir, "model.safetensors"))
codec_emb = weights["talker.model.codec_embedding.weight"]

# Get samantha embedding at idx 3000
samantha_emb = codec_emb[3000].float()
print(f"Before: mean={samantha_emb.mean():.6f}, std={samantha_emb.std():.6f}, norm={samantha_emb.norm():.4f}")

# Normalize: scale to match target norm
current_norm = samantha_emb.norm()
scale = TARGET_NORM / current_norm
samantha_normalized = samantha_emb * scale

print(f"After:  mean={samantha_normalized.mean():.6f}, std={samantha_normalized.std():.6f}, norm={samantha_normalized.norm():.4f}")

# Write back
codec_emb[3000] = samantha_normalized.to(codec_emb.dtype)
weights["talker.model.codec_embedding.weight"] = codec_emb
save_file(weights, os.path.join(fixed_dir, "model.safetensors"))
print(f"Saved to {fixed_dir}")
