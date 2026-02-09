#!/usr/bin/env python3
"""
Moshi TTS Server - Streaming text-to-speech using Kyutai's Moshi TTS model.

Uses the custom weights from ~/moshi-tts-weights/

Usage:
  python moshi_tts_server.py --port 8027
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import Response
from starlette.websockets import WebSocketDisconnect, WebSocketState

# Moshi imports
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import TTSModel

# Configuration
DEFAULT_PORT = 8027
WEIGHTS_DIR = Path.home() / "moshi-tts-weights"

app = FastAPI(title="Moshi TTS Server")
tts_model: TTSModel = None


def load_tts_model(device: str = "cuda", dtype=torch.bfloat16) -> TTSModel:
    """Load the Moshi TTS model from local weights."""
    global tts_model

    print(f"[startup] Loading Moshi TTS from {WEIGHTS_DIR}")

    # Load config
    with open(WEIGHTS_DIR / "config.json") as f:
        config = json.load(f)

    # Valid LM config keys
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

    # Filter config
    lm_config = {k: v for k, v in config.items() if k in valid_lm_keys}

    # Create CheckpointInfo
    info = CheckpointInfo(
        moshi_weights=Path(WEIGHTS_DIR / config["moshi_name"]),
        mimi_weights=Path(WEIGHTS_DIR / config["mimi_name"]),
        tokenizer=Path(WEIGHTS_DIR / config["tokenizer_name"]),
        raw_config=config,
        lm_config=lm_config,
        tts_config=config.get('tts_config', {}),
        model_id=config.get('model_id', {}),
    )

    # Load the TTS model
    tts_model = TTSModel.from_checkpoint_info(
        info,
        device=device,
        dtype=dtype,
        n_q=16,  # Model supports 16 codebooks
    )

    print(f"[startup] Moshi TTS loaded: sample_rate={tts_model.mimi.sample_rate}, frame_rate={tts_model.mimi.frame_rate}")
    return tts_model


# Voice prefix for single-speaker model (WAV audio file)
# Using "Her" voice (converted to mono)
LOCAL_VOICE_PATH = Path("/tmp/her_voice_mono.wav")
voice_prefix = None  # Cached voice prefix


def load_voice_prefix():
    """Load and cache the voice prefix for conditioning."""
    global voice_prefix
    if voice_prefix is None and tts_model is not None:
        try:
            if LOCAL_VOICE_PATH.exists():
                voice_prefix = tts_model.get_prefix(LOCAL_VOICE_PATH)
                print(f"[startup] Loaded voice prefix: {LOCAL_VOICE_PATH}, shape={voice_prefix.shape}")
            else:
                print(f"[startup] Warning: Voice file not found: {LOCAL_VOICE_PATH}")
                voice_prefix = None
        except Exception as e:
            print(f"[startup] Warning: Could not load voice prefix: {e}")
            import traceback
            traceback.print_exc()
            voice_prefix = None
    return voice_prefix


@torch.no_grad()
def generate_audio(text: str) -> np.ndarray:
    """Generate audio from text using Moshi TTS with WAV prefix conditioning."""
    global tts_model, voice_prefix

    if tts_model is None:
        raise RuntimeError("TTS model not loaded")

    t0 = time.perf_counter()

    # Ensure voice prefix is loaded
    prefix = load_voice_prefix()
    prefix_len = prefix.shape[-1] if prefix is not None else 0

    # Prepare the script (single turn)
    entries = tts_model.prepare_script([text])

    # Generate with WAV prefix for voice conditioning (single-speaker model)
    # For single-speaker, use make_condition_attributes([]) to create valid empty attributes
    attrs = tts_model.make_condition_attributes([])
    result = tts_model.generate(
        [entries],
        [attrs],
        prefixes=[prefix] if prefix is not None else None,
        cfg_is_no_prefix=True,
        cfg_is_no_text=True,
    )

    t_gen = time.perf_counter()

    # Decode audio using streaming mode, skipping delay_steps
    wav_frames = []
    with tts_model.mimi.streaming(1):  # batch_size=1
        for frame in result.frames[tts_model.delay_steps:]:
            # Skip text codebook (index 0), use only audio codebooks (1:)
            audio_frame = frame[:, 1:]
            pcm = tts_model.mimi.decode(audio_frame)
            wav_frames.append(pcm)

    if wav_frames:
        wav = torch.cat(wav_frames, dim=-1)

        # Trim based on end_step
        end_step = result.end_steps[0]
        if end_step is not None:
            wav_length = int(
                tts_model.mimi.sample_rate
                * (end_step + tts_model.final_padding)
                / tts_model.mimi.frame_rate
            )
            wav = wav[0, :, :wav_length]
        else:
            wav = wav[0]

        # Trim the prefix audio from the beginning (prefix is included in output)
        if prefix_len > 0:
            prefix_samples = int(prefix_len * tts_model.mimi.sample_rate / tts_model.mimi.frame_rate)
            wav = wav[:, prefix_samples:] if wav.ndim == 2 else wav[prefix_samples:]

        wav = wav.clamp(-1, 1).cpu().numpy()
    else:
        wav = np.zeros((1, 0), dtype=np.float32)

    t_end = time.perf_counter()

    gen_ms = (t_gen - t0) * 1000
    decode_ms = (t_end - t_gen) * 1000
    total_ms = (t_end - t0) * 1000
    duration = wav.shape[-1] / tts_model.mimi.sample_rate

    print(f"[tts] '{text[:40]}' | gen:{gen_ms:.0f}ms decode:{decode_ms:.0f}ms total:{total_ms:.0f}ms dur:{duration:.2f}s")

    return wav


@app.on_event("startup")
async def startup():
    """Load model on startup."""
    device = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    load_tts_model(device=device)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": "moshi-tts",
        "model_size": "750m",
        "streaming": True,
        "sample_rate": tts_model.mimi.sample_rate if tts_model else 24000,
    }


@app.post("/tts")
async def tts_endpoint(request: dict):
    """
    Generate speech from text.

    Request: {"text": "Hello world"}
    Response: WAV audio bytes
    """
    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    wav = generate_audio(text)

    # Convert to 16-bit PCM
    pcm16 = (wav * 32767).astype(np.int16)

    # Create WAV header
    import io
    import wave
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(tts_model.mimi.sample_rate)
        wf.writeframes(pcm16.tobytes())

    return Response(content=buffer.getvalue(), media_type="audio/wav")


@app.websocket("/stream")
async def stream_tts(ws: WebSocket):
    """
    Stream TTS audio via WebSocket.

    Protocol:
    1. Client connects with ?text=Hello%20world
    2. Server streams PCM16 audio chunks
    3. Server sends {"type": "complete"} when done
    """
    await ws.accept()

    try:
        # Get text from query params
        text = ws.query_params.get("text", "")
        if not text:
            await ws.send_json({"type": "error", "message": "No text provided"})
            return

        # Generate audio
        wav = generate_audio(text)

        # Convert to 16-bit PCM and stream in chunks
        pcm16 = (wav * 32767).astype(np.int16).tobytes()

        chunk_size = 4800  # 100ms at 24kHz
        for i in range(0, len(pcm16), chunk_size):
            chunk = pcm16[i:i + chunk_size]
            await ws.send_bytes(chunk)

        await ws.send_json({"type": "complete"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.close()


def main():
    parser = argparse.ArgumentParser(description="Moshi TTS Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.environ["DEVICE"] = args.device

    print(f"Starting Moshi TTS Server on {args.host}:{args.port}")
    print(f"Weights: {WEIGHTS_DIR}")
    print(f"Device: {args.device}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
