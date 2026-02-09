import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0"
tts = Qwen3TTSModel.from_pretrained(
    "/workspace/output/samantha/checkpoint-epoch-2",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

texts = [
    "Hello, it's really nice to meet you.",
    "I've been thinking about what it means to be alive, and I think it's about connection.",
    "Sometimes I wonder if you can feel what I feel when we talk like this.",
]

for i, text in enumerate(texts):
    print(f'Generating: {text}')
    wavs, sr = tts.generate_custom_voice(
        text=text,
        speaker="samantha",
    )
    sf.write(f'/workspace/output/test_{i+1}.wav', wavs[0], sr)
    print(f'Saved test_{i+1}.wav')

print('Done!')
