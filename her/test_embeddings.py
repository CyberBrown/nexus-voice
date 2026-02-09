import torch, json
from qwen_tts import Qwen3TTSModel
from safetensors import safe_open

# Load official CustomVoice model
print("=== Official CustomVoice speaker embeddings ===")
tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")

# Get codec embedding weight
codec_emb = tts.model.talker.model.codec_embedding.weight.data
print(f"Codec embedding shape: {codec_emb.shape}")

# Load config to find speaker IDs  
import json
config = json.load(open("/root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/" + 
    __import__('os').listdir("/root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/")[0] + "/config.json"))
spk_ids = config.get("talker_config", {}).get("spk_id", {})
print(f"Speaker IDs: {spk_ids}")

for name, idx in spk_ids.items():
    emb = codec_emb[idx].float()
    print(f"  {name} (idx={idx}): mean={emb.mean():.6f}, std={emb.std():.6f}, norm={emb.norm():.4f}, min={emb.min():.4f}, max={emb.max():.4f}")

del tts; torch.cuda.empty_cache()

# Load our fine-tuned model
print("\n=== Fine-tuned samantha embedding ===")
tts2 = Qwen3TTSModel.from_pretrained("/workspace/output/samantha_v3/checkpoint-epoch-2", device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
codec_emb2 = tts2.model.talker.model.codec_embedding.weight.data
emb_sam = codec_emb2[3000].float()
print(f"  samantha (idx=3000): mean={emb_sam.mean():.6f}, std={emb_sam.std():.6f}, norm={emb_sam.norm():.4f}, min={emb_sam.min():.4f}, max={emb_sam.max():.4f}")

print("\nDone!")
