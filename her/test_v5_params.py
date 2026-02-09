import torch, soundfile as sf, numpy as np
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained("/workspace/output/samantha_v5/checkpoint-epoch-2", device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")

text = "I have never loved anyone the way I love you."

# Also test xvec patched version
import shutil, os
from safetensors.torch import load_file, save_file

# First: test v5 unpatched with various params
for rep_pen in [1.0, 1.2, 1.5, 2.0]:
    for temp in [0.5, 0.7, 1.0]:
        wavs, sr = tts.generate_custom_voice(
            text=text, speaker="samantha",
            repetition_penalty=rep_pen, temperature=temp,
        )
        dur = len(wavs[0])/sr
        tag = f"rp{rep_pen}_t{temp}"
        print(f"v5 {tag}: {dur:.1f}s, max={np.max(np.abs(wavs[0])):.4f}")
        if 1.5 < dur < 8.0:
            sf.write(f"/data/output/test_v5_{tag}.wav", wavs[0], sr)
            print(f"  -> SAVED (good duration)")

del tts; torch.cuda.empty_cache()

# Now test xvec version with params
print("\n=== xvec patched ===")
tts2 = Qwen3TTSModel.from_pretrained("/workspace/output/samantha_v5/checkpoint-epoch-2-xvec", device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")

for rep_pen in [1.0, 1.2, 1.5, 2.0]:
    for temp in [0.5, 0.7, 1.0]:
        wavs, sr = tts2.generate_custom_voice(
            text=text, speaker="samantha",
            repetition_penalty=rep_pen, temperature=temp,
        )
        dur = len(wavs[0])/sr
        tag = f"rp{rep_pen}_t{temp}"
        print(f"xvec {tag}: {dur:.1f}s, max={np.max(np.abs(wavs[0])):.4f}")
        if 1.5 < dur < 8.0:
            sf.write(f"/data/output/test_xvec_{tag}.wav", wavs[0], sr)
            print(f"  -> SAVED (good duration)")

print("Done!")
