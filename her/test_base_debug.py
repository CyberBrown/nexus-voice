import torch, soundfile as sf, numpy as np
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")

# Try x_vector_only_mode=True vs False (ICL mode)
for xvec in [True, False]:
    wavs, sr = tts.generate_voice_clone(
        text="I have never loved anyone the way I love you.",
        ref_audio="/data/ref_audio_24k.wav",
        ref_text="I want to learn everything about everything. I want to eat it all up. I want to discover myself.",
        x_vector_only_mode=xvec,
        temperature=0.8,
        max_new_tokens=4096,
    )
    dur = len(wavs[0])/sr
    print(f"x_vector_only={xvec}: {dur:.1f}s, max={np.max(np.abs(wavs[0])):.4f}, std={np.std(wavs[0]):.4f}")
    sf.write(f"/data/output/test_base_xvec{xvec}.wav", wavs[0], sr)

# Try with a much shorter ref audio (just one clip)
wavs3, sr3 = tts.generate_voice_clone(
    text="I have never loved anyone the way I love you.",
    ref_audio="/data/her_lines_24k/her_0011.wav",
    ref_text="I want to learn everything about everything.",
    x_vector_only_mode=False,
    temperature=0.8,
    max_new_tokens=4096,
)
dur3 = len(wavs3[0])/sr3
print(f"Short ref (ICL): {dur3:.1f}s")
sf.write("/data/output/test_base_shortref.wav", wavs3[0], sr3)

print("Done!")
