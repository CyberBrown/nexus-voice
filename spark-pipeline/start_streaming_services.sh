#!/bin/bash
# Start Streaming Voice Chat Services on DGX Spark
# ================================================
#
# Usage:
#   # Default (fast, no cloning)
#   ./start_streaming_services.sh
#
#   # With Parakeet STT (10-15x faster than Whisper)
#   ./start_streaming_services.sh --stt parakeet
#
#   # With 1.5B TTS for voice cloning
#   ./start_streaming_services.sh --tts 1.5b --voice-sample ~/voices/my-voice.wav
#
#   # Both upgrades
#   ./start_streaming_services.sh --stt parakeet --tts 1.5b --voice-sample ~/voices/my-voice.wav
#
# Service ports:
#   8025 - Whisper STT (default)
#   8029 - Parakeet STT (when --stt parakeet)
#   8027 - VibeVoice TTS (0.5B streaming or 1.5B cloning)
#   8028 - Voice Chat Orchestrator
#   11434 - Ollama LLM

set -e

# Parse arguments
STT_ENGINE="whisper"
TTS_MODEL="0.5b"
VOICE_SAMPLE=""
TTS_VOICE="en-Emma_woman"

while [[ $# -gt 0 ]]; do
    case $1 in
        --stt)
            STT_ENGINE="$2"
            shift 2
            ;;
        --tts)
            TTS_MODEL="$2"
            shift 2
            ;;
        --voice-sample)
            VOICE_SAMPLE="$2"
            shift 2
            ;;
        --voice)
            TTS_VOICE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --stt <whisper|parakeet>  STT engine (default: whisper)"
            echo "  --tts <0.5b|1.5b>         TTS model size (default: 0.5b)"
            echo "  --voice-sample <path>     Voice sample WAV for 1.5B cloning"
            echo "  --voice <name>            Voice preset for 0.5B (default: en-Emma_woman)"
            echo "  -h, --help                Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Default (Whisper + 0.5B)"
            echo "  $0 --stt parakeet                     # Fast Parakeet STT"
            echo "  $0 --tts 1.5b --voice-sample ~/v.wav  # Voice cloning"
            echo "  $0 --stt parakeet --tts 1.5b --voice-sample ~/v.wav"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate arguments
if [[ "$TTS_MODEL" == "1.5b" && -z "$VOICE_SAMPLE" ]]; then
    echo "Error: --voice-sample is required when using --tts 1.5b"
    exit 1
fi

if [[ -n "$VOICE_SAMPLE" && ! -f "$VOICE_SAMPLE" ]]; then
    echo "Error: Voice sample file not found: $VOICE_SAMPLE"
    exit 1
fi

# Determine STT port
if [[ "$STT_ENGINE" == "parakeet" ]]; then
    STT_PORT=8029
else
    STT_PORT=8025
fi

echo "=========================================="
echo "  Starting Streaming Voice Chat Services"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  STT Engine:  $STT_ENGINE (port $STT_PORT)"
echo "  TTS Model:   $TTS_MODEL"
if [[ "$TTS_MODEL" == "0.5b" ]]; then
    echo "  TTS Voice:   $TTS_VOICE"
else
    echo "  Voice Sample: $VOICE_SAMPLE"
fi
echo ""

mkdir -p ~/ggml-org/logs

# Kill any existing instances
echo "Stopping existing services..."
pkill -f whisper-server 2>/dev/null || true
pkill -f parakeet_stt_server 2>/dev/null || true
pkill -f vibevoice_streaming_server 2>/dev/null || true
pkill -f voice_chat_streaming 2>/dev/null || true
sleep 2

# Start STT Server
if [[ "$STT_ENGINE" == "parakeet" ]]; then
    echo ""
    echo "[1/3] Starting Parakeet STT server (port 8029)..."
    echo "      This may take a moment to load NeMo models..."
    cd ~/spark-voice-pipeline
    nohup python3 parakeet_stt_server.py --port 8029 \
        > ~/ggml-org/logs/parakeet-stt.log 2>&1 &
    echo "      PID: $!"

    # Wait for Parakeet
    echo "  [..] Waiting for Parakeet STT..."
    for i in {1..120}; do
        if curl -s http://localhost:8029/health 2>/dev/null | grep -q '"status":"ok"'; then
            echo "  [OK] Parakeet STT (252x RTF, 3.5% WER)"
            break
        fi
        if [ $i -eq 120 ]; then
            echo "  [!] Parakeet not ready - check logs/parakeet-stt.log"
        fi
        sleep 1
    done
else
    echo ""
    echo "[1/3] Starting Whisper STT server (port 8025)..."
    cd ~/ggml-org/whisper.cpp/build-cuda/bin
    LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH nohup ./whisper-server \
        -m ~/ggml-org/whisper.cpp/models/ggml-large-v3-turbo-q8_0.bin \
        --host 0.0.0.0 \
        --port 8025 \
        > ~/ggml-org/logs/whisper-server.log 2>&1 &
    echo "      PID: $!"

    # Wait for Whisper
    for i in {1..30}; do
        if curl -s http://localhost:8025/health 2>/dev/null | grep -q '"status":"ok"'; then
            echo "  [OK] Whisper STT (20x RTF, 2.9% WER)"
            break
        fi
        sleep 1
    done
fi

# Check Ollama
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  [OK] Ollama LLM"
else
    echo "  [!] Ollama not running - start with: ollama serve"
fi

# Start VibeVoice TTS
echo ""
echo "[2/3] Starting VibeVoice TTS (port 8027)..."
if [[ "$TTS_MODEL" == "1.5b" ]]; then
    echo "      Loading 1.5B model with voice cloning..."
    cd ~/spark-voice-pipeline
    nohup python3 vibevoice_streaming_server.py \
        --port 8027 \
        --model 1.5b \
        --voice-sample "$VOICE_SAMPLE" \
        > ~/ggml-org/logs/vibevoice-streaming.log 2>&1 &
else
    echo "      Loading 0.5B streaming model..."
    cd ~/spark-voice-pipeline
    nohup python3 vibevoice_streaming_server.py \
        --port 8027 \
        --model 0.5b \
        --voice "$TTS_VOICE" \
        > ~/ggml-org/logs/vibevoice-streaming.log 2>&1 &
fi
echo "      PID: $!"

# Wait for VibeVoice
echo "  [..] Waiting for VibeVoice (~30-60s)..."
for i in {1..90}; do
    if curl -s http://localhost:8027/health 2>/dev/null | grep -q '"status":"ok"'; then
        echo "  [OK] VibeVoice TTS ($TTS_MODEL)"
        break
    fi
    if [ $i -eq 90 ]; then
        echo "  [!] VibeVoice not ready - check logs/vibevoice-streaming.log"
    fi
    sleep 1
done

# Start Orchestrator
echo ""
echo "[3/3] Starting Voice Chat Orchestrator (port 8028)..."
cd ~/spark-voice-pipeline
nohup python3 voice_chat_streaming.py \
    --port 8028 \
    --stt-port $STT_PORT \
    > ~/ggml-org/logs/orchestrator.log 2>&1 &
echo "      PID: $!"

sleep 2
if curl -s http://localhost:8028/health 2>/dev/null | grep -q '"status":"ok"'; then
    echo "  [OK] Orchestrator"
else
    echo "  [!] Orchestrator may still be starting..."
fi

echo ""
echo "=========================================="
echo "  Streaming Voice Chat Services Ready"
echo "=========================================="
echo ""
echo "Services:"
if [[ "$STT_ENGINE" == "parakeet" ]]; then
    echo "  Parakeet STT:    http://localhost:8029 (252x RTF)"
else
    echo "  Whisper STT:     http://localhost:8025 (20x RTF)"
fi
echo "  Ollama LLM:      http://localhost:11434"
if [[ "$TTS_MODEL" == "1.5b" ]]; then
    echo "  VibeVoice TTS:   http://localhost:8027/tts (1.5B, cloning)"
else
    echo "  VibeVoice TTS:   ws://localhost:8027/stream (0.5B, streaming)"
fi
echo "  Orchestrator:    ws://localhost:8028/voice"
echo ""
echo "Client usage (run on your laptop):"
echo "  python voice_chat_client_streaming.py --spark-host 10.0.0.104"
echo ""
if [[ "$STT_ENGINE" == "parakeet" ]]; then
    echo "Expected latency: ~300-400ms to first audio (Parakeet + 0.5B)"
else
    echo "Expected latency: ~800ms to first audio (Whisper + 0.5B)"
fi
echo ""
echo "Logs: ~/ggml-org/logs/"
echo "=========================================="
