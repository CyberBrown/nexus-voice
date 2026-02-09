#!/usr/bin/env python3
"""
Chatterbox TTS Server - Zero-shot voice cloning with Her voice.

Uses Chatterbox-Turbo (350M params) for fast, high-quality TTS.

Usage:
  python chatterbox_tts_server.py --port 8027
"""

import argparse
import asyncio
import io
import json
import os
import sys
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio as ta
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import Response
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect, WebSocketState

# Add chatterbox to path
sys.path.insert(0, '/home/chris/chatterbox/src')
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Configuration
DEFAULT_PORT = 8027
MODEL_PATH = '/home/chris/.cache/huggingface/hub/models--ResembleAI--chatterbox-turbo/snapshots/749d1c1a46eb10492095d68fbcf55691ccf137cd'
VOICE_SAMPLE_PATH = Path("/home/chris/ggml-org/voice samples/her audio samples.wav")

app = FastAPI(title="Chatterbox TTS Server")


class TTSRequest(BaseModel):
    text: str


# Global model state
tts_model: Optional[ChatterboxTurboTTS] = None


def load_models(device: str = "cuda"):
    """Load Chatterbox model and prepare Her voice."""
    global tts_model

    print(f"[startup] Loading Chatterbox-Turbo from {MODEL_PATH}")
    tts_model = ChatterboxTurboTTS.from_local(MODEL_PATH, device=device)
    print(f"[startup] Model loaded: sample_rate={tts_model.sr}")

    # Prepare Her voice conditioning
    if VOICE_SAMPLE_PATH.exists():
        print(f"[startup] Preparing voice from {VOICE_SAMPLE_PATH}")

        # Load audio and ensure float32
        import librosa
        wav, sr = librosa.load(str(VOICE_SAMPLE_PATH), sr=24000)
        wav = wav.astype(np.float32)

        # Take first 10 seconds
        max_samples = 10 * sr
        if len(wav) > max_samples:
            wav = wav[:max_samples]

        # Save temp file with correct format
        temp_path = Path("/tmp/her_voice_chatterbox.wav")
        import soundfile as sf
        sf.write(str(temp_path), wav, sr)

        # Prepare conditionals
        tts_model.prepare_conditionals(str(temp_path), exaggeration=0.5)
        print(f"[startup] Voice conditioning prepared!")
    else:
        print(f"[startup] WARNING: Voice sample not found: {VOICE_SAMPLE_PATH}")

    return tts_model


@torch.no_grad()
def generate_audio(text: str) -> np.ndarray:
    """Generate audio using Chatterbox with Her voice."""
    global tts_model

    if tts_model is None:
        raise RuntimeError("TTS model not loaded")

    t0 = time.perf_counter()

    # Generate audio
    wav = tts_model.generate(
        text,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.2,
    )

    # Convert to numpy float32
    audio = wav.squeeze().cpu().numpy().astype(np.float32)

    t_total = (time.perf_counter() - t0) * 1000
    duration = len(audio) / tts_model.sr if len(audio) > 0 else 0
    print(f"[tts] '{text[:40]}...' | {t_total:.0f}ms | {duration:.2f}s")

    return audio


@app.on_event("startup")
async def startup():
    """Load models on startup."""
    device = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    load_models(device=device)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": "chatterbox-turbo",
        "model_size": "350m",
        "streaming": True,
        "sample_rate": tts_model.sr if tts_model else 24000,
        "voice": "her",
    }


@app.post("/tts")
async def tts_http(request: TTSRequest):
    """HTTP endpoint for TTS - returns WAV audio."""
    audio = generate_audio(request.text)

    if len(audio) == 0:
        return Response(content=b"", media_type="audio/wav")

    # Convert float32 to int16 for WAV
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)

    # Create WAV in memory
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(24000)
        wav.writeframes(audio_int16.tobytes())

    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")


@app.websocket("/ws")
async def websocket_tts(ws: WebSocket):
    """WebSocket endpoint for streaming TTS."""
    await ws.accept()

    try:
        while True:
            data = await ws.receive()

            if "text" in data:
                text = data["text"]
                if not text.strip():
                    continue

                # Generate audio
                audio = generate_audio(text)

                if len(audio) > 0:
                    # Stream in chunks (2400 samples = 100ms at 24kHz)
                    chunk_size = 2400
                    for i in range(0, len(audio), chunk_size):
                        chunk = audio[i:i + chunk_size]
                        await ws.send_bytes(chunk.tobytes())

                await ws.send_json({"done": True, "samples": len(audio)})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"error": str(e)})
        except:
            pass
    finally:
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.close()


def main():
    parser = argparse.ArgumentParser(description="Chatterbox TTS Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.environ["DEVICE"] = args.device

    print(f"Starting Chatterbox TTS Server on {args.host}:{args.port}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Voice: {VOICE_SAMPLE_PATH}")
    print(f"  Device: {args.device}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
