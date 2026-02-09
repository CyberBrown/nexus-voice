#!/bin/bash
# Start llama-server with Nemotron for voice pipeline
# Uses --reasoning-budget 0 to disable thinking output

MODEL_PATH="/home/chris/models/nemotron-gguf/Nemotron-3-Nano-30B-A3B-Q4_0.gguf"
LLAMA_SERVER="/home/chris/ggml-org/llama.cpp/build-cuda/bin/llama-server"
PORT=9003

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Download with: huggingface-cli download unsloth/Nemotron-3-Nano-30B-A3B-GGUF Nemotron-3-Nano-30B-A3B-Q4_0.gguf --local-dir /home/chris/models/nemotron-gguf/"
    exit 1
fi

echo "Starting llama-server with Nemotron 30B (Q4_0)..."
echo "Port: $PORT"
echo "Reasoning: disabled (--reasoning-budget 0)"

$LLAMA_SERVER \
    -m "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port $PORT \
    -c 4096 \
    -ngl 99 \
    --reasoning-budget 0 \
    --flash-attn off \
    -t 8
