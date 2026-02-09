FROM comfyui-audio:latest AS torchaudio-donor

FROM nvcr.io/nvidia/pytorch:25.12-py3

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends sox libsox-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy working torchaudio from comfyui-audio (same 25.12 torch ABI)
COPY --from=torchaudio-donor /usr/local/lib/python3.12/dist-packages/torchaudio /usr/local/lib/python3.12/dist-packages/torchaudio

RUN pip install --no-cache-dir --no-deps qwen-tts && \
    pip install --no-cache-dir \
    soundfile sox onnxruntime safetensors librosa \
    transformers==4.57.3 accelerate

WORKDIR /workspace

CMD ["bash"]
