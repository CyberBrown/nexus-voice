#!/usr/bin/env python3
"""
Kyutai 0.75B Streaming TTS Server - Uses audio prefix conditioning.

This model uses a short audio prefix for voice cloning.
Optimized for speed with minimal prefix length.

Usage:
  python kyutai_streaming_075b.py --port 8030
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import Response
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect, WebSocketState
import sphn
import wave
import io

from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import TTSModel

class TTSRequest(BaseModel):
    text: str


# Configuration
DEFAULT_PORT = 8027  # Match orchestrator expectations
WEIGHTS_DIR = Path.home() / "moshi-tts-weights"
VOICE_PREFIX_PATH = Path("/tmp/her_voice_3s.wav")

app = FastAPI(title="Kyutai 0.75B Streaming TTS Server")

# Global model state
tts_model: Optional[TTSModel] = None
voice_prefix = None  # Encoded prefix frames


def load_models(device: str = "cuda", dtype=torch.bfloat16):
    """Load the 0.75B TTS model and voice prefix."""
    global tts_model, voice_prefix

    print(f"[startup] Loading Kyutai 0.75B TTS from {WEIGHTS_DIR}")

    with open(WEIGHTS_DIR / "config.json") as f:
        config = json.load(f)

    valid_lm_keys = {
        "dim", "text_card", "existing_text_padding_id", "n_q", "dep_q", "card",
        "num_heads", "num_layers", "hidden_scale", "causal", "layer_scale",
        "context", "max_period", "gating", "norm", "positional_embedding",
        "depformer_dim", "depformer_dim_feedforward", "depformer_num_heads",
        "depformer_num_layers", "depformer_layer_scale", "depformer_multi_linear",
        "depformer_context", "depformer_max_period", "depformer_gating",
        "depformer_pos_emb", "depformer_weights_per_step", "delays",
        "conditioners", "fuser", "cross_attention", "text_card_out",
        "demux_second_stream", "depformer_low_rank_embeddings",
        "depformer_weights_per_step_schedule",
    }

    lm_config = {k: v for k, v in config.items() if k in valid_lm_keys}

    info = CheckpointInfo(
        moshi_weights=Path(WEIGHTS_DIR / config["moshi_name"]),
        mimi_weights=Path(WEIGHTS_DIR / config["mimi_name"]),
        tokenizer=Path(WEIGHTS_DIR / config["tokenizer_name"]),
        raw_config=config,
        lm_config=lm_config,
        tts_config=config.get('tts_config', {}),
        model_id=config.get('model_id', {}),
    )

    tts_model = TTSModel.from_checkpoint_info(
        info,
        device=device,
        dtype=dtype,
        n_q=16,
    )

    print(f"[startup] TTS 0.75B loaded: sample_rate={tts_model.mimi.sample_rate}, frame_rate={tts_model.mimi.frame_rate}")
    print(f"[startup] multi_speaker={tts_model.multi_speaker}")

    # Load voice prefix
    if VOICE_PREFIX_PATH.exists():
        print(f"[startup] Loading voice prefix from {VOICE_PREFIX_PATH}")
        voice_prefix = tts_model.get_prefix(VOICE_PREFIX_PATH)
        print(f"[startup] Voice prefix loaded: {voice_prefix.shape}")
    else:
        print(f"[startup] WARNING: Voice prefix not found: {VOICE_PREFIX_PATH}")
        voice_prefix = None

    return tts_model


@torch.no_grad()
def generate_audio(text: str) -> np.ndarray:
    """Generate audio using voice prefix conditioning."""
    global tts_model, voice_prefix

    if tts_model is None:
        raise RuntimeError("TTS model not loaded")

    t0 = time.perf_counter()

    # Prepare the script
    entries = tts_model.prepare_script([text], padding_between=1)
    
    # Empty condition attributes (0.75B doesn't use safetensors conditioning)
    empty_attrs = tts_model.make_condition_attributes([])

    # Generate with voice prefix
    prefixes = [voice_prefix] if voice_prefix is not None else None
    result = tts_model.generate(
        [entries],
        [empty_attrs],
        prefixes=prefixes,
    )

    # Decode frames using streaming mode
    pcm_chunks = []
    with tts_model.mimi.streaming(1):
        for frame in result.frames[tts_model.delay_steps:]:
            pcm = tts_model.mimi.decode(frame[:, 1:, :])
            pcm = torch.clip(pcm, -1, 1)
            pcm_chunks.append(pcm)

    # Skip first 2 frames (removes click) and trim to end_step
    pcm_chunks = pcm_chunks[2:]
    end_step = result.end_steps[0]

    if pcm_chunks and end_step is not None:
        audio = torch.cat([pcm[0, 0, :] for pcm in pcm_chunks[:end_step]], dim=0)
        audio = audio.detach().cpu().numpy().astype(np.float32)
    elif pcm_chunks:
        audio = torch.cat([pcm[0, 0, :] for pcm in pcm_chunks], dim=0)
        audio = audio.detach().cpu().numpy().astype(np.float32)
    else:
        audio = np.array([], dtype=np.float32)

    t_total = (time.perf_counter() - t0) * 1000
    duration = len(audio) / tts_model.mimi.sample_rate if len(audio) > 0 else 0
    print(f"[tts] '{text[:40]}' | total:{t_total:.0f}ms dur:{duration:.2f}s")

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
        "model": "moshi-tts",
        "model_size": "750m",
        "streaming": True,
        "sample_rate": tts_model.mimi.sample_rate if tts_model else 24000,
        "format": "float32",
        "multi_speaker": tts_model.multi_speaker if tts_model else False,
        "voice_loaded": voice_prefix is not None,
    }


@app.post("/tts")
async def tts_http(request: TTSRequest):
    """HTTP endpoint for TTS - returns WAV audio."""
    audio = generate_audio(request.text)

    if len(audio) == 0:
        return Response(content=b"", media_type="audio/wav")

    # Convert float32 to int16 for WAV
    audio_int16 = (audio * 32767).astype(np.int16)

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
    parser = argparse.ArgumentParser(description="Kyutai 0.75B Streaming TTS Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.environ["DEVICE"] = args.device

    print(f"Starting Kyutai 0.75B Streaming TTS Server on {args.host}:{args.port}")
    print(f"  Weights: {WEIGHTS_DIR}")
    print(f"  Voice Prefix: {VOICE_PREFIX_PATH}")
    print(f"  Device: {args.device}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
