import torch, numpy as np, soundfile as sf
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")

for temp in [0.5, 0.7, 1.0]:
    for max_tokens in [2048, 4096]:
        wavs, sr = tts.generate_voice_clone(
            text="I have never loved anyone the way I love you.",
            ref_audio="/data/ref_audio_24k.wav",
            ref_text="I want to learn everything about everything. I want to eat it all up. I want to discover myself.",
            temperature=temp,
            max_new_tokens=max_tokens,
        )
        dur = len(wavs[0])/sr
        print(f"temp={temp} max_tokens={max_tokens}: {dur:.1f}s")
        if dur > 2.0:
            sf.write(f"/data/output/test_base_t{temp}_m{max_tokens}.wav", wavs[0], sr)
