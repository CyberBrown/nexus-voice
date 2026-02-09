#!/usr/bin/env python3
"""
Kyutai Streaming TTS Server - WebSocket streaming text-to-speech.

Uses the Her voice from the audio file, cached at startup.
Streams raw PCM float32 @ 24kHz back via WebSocket.

Usage:
  python kyutai_streaming_server.py --port 8030
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import sphn
import torch
import uvicorn
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

# Moshi imports
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import TTSModel

# Configuration
DEFAULT_PORT = 8030
WEIGHTS_DIR = Path.home() / "moshi-tts-weights"
MIMI_DIR = Path.home() / "moshi-weights" / "mimi"
VOICE_WAV = Path("/tmp/her_voice_3s.wav")  # 3-second clip for faster generation

app = FastAPI(title="Kyutai Streaming TTS Server")

# Global model state (loaded once at startup)
tts_model: Optional[TTSModel] = None
voice_prefix: Optional[torch.Tensor] = None
prefix_samples: int = 0


def load_models(device: str = "cuda", dtype=torch.bfloat16):
    """Load the TTS model and voice prefix once at startup."""
    global tts_model, voice_prefix, prefix_samples

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
        n_q=16,
    )

    print(f"[startup] TTS model loaded: sample_rate={tts_model.mimi.sample_rate}, frame_rate={tts_model.mimi.frame_rate}")

    # Load voice prefix ONCE at startup
    if VOICE_WAV.exists():
        print(f"[startup] Loading voice from {VOICE_WAV}")
        # Load audio and convert to mono if needed
        wav, _ = sphn.read(VOICE_WAV, sample_rate=tts_model.mimi.sample_rate)
        if wav.ndim > 1 and wav.shape[0] > 1:
            wav = wav.mean(axis=0)  # Convert stereo to mono
        # Ensure shape is [C, T] for mono
        if wav.ndim == 1:
            wav = wav[None, :]  # Add channel dimension -> [1, T]
        # Now add batch dimension -> [B, C, T]
        wav_tensor = torch.from_numpy(wav).to(device=tts_model.lm.device, dtype=torch.float32)[None]
        print(f"[startup] Audio tensor shape: {wav_tensor.shape}")

        # Encode to get prefix codes
        with torch.no_grad():
            voice_prefix = tts_model.mimi.encode(wav_tensor)[0, :, :-2]
        # Add null text token at position 0
        null_text = torch.full_like(voice_prefix[:1], tts_model.machine.token_ids.zero)
        voice_prefix = torch.cat([null_text, voice_prefix], dim=0)

        prefix_samples = int(
            (tts_model.mimi.sample_rate * voice_prefix.shape[-1]) / tts_model.mimi.frame_rate
        )
        print(f"[startup] Voice prefix cached: shape={voice_prefix.shape}, prefix_samples={prefix_samples}")
    else:
        print(f"[startup] WARNING: Voice file not found: {VOICE_WAV}")
        voice_prefix = None
        prefix_samples = 0

    return tts_model


@torch.no_grad()
def generate_streaming(text: str):
    """
    Generator that yields PCM float32 audio chunks as they're decoded.

    Yields:
        numpy.ndarray: float32 PCM audio chunks at 24kHz
    """
    global tts_model, voice_prefix, prefix_samples

    if tts_model is None:
        raise RuntimeError("TTS model not loaded")

    t0 = time.perf_counter()

    # Prepare the script
    entries = tts_model.prepare_script([text], padding_between=1)
    attrs = tts_model.make_condition_attributes([])

    # Prepare prefixes
    prefixes = [voice_prefix] if voice_prefix is not None else None

    # Track frames for trimming
    frame_count = 0
    samples_yielded = 0
    end_step = None

    def on_frame(frame):
        """Callback for each generated frame - decode and yield audio."""
        nonlocal frame_count, samples_yielded, end_step
        frame_count += 1

        # Skip delay_steps at the beginning
        if frame_count <= tts_model.delay_steps:
            return None

        # Check if all audio codebooks have valid values
        if (frame[:, 1:] == -1).any():
            return None

        # Decode the audio codebooks (skip text codebook at index 0)
        pcm = tts_model.mimi.decode(frame[:, 1:, :])
        pcm = pcm.squeeze(0).clamp(-1, 1).cpu().numpy()

        return pcm

    # Generate with streaming decode
    pcm_chunks = []
    with tts_model.mimi.streaming(1):  # batch_size=1
        result = tts_model.generate(
            [entries],
            [attrs],
            prefixes=prefixes,
            cfg_is_no_prefix=True,
            cfg_is_no_text=True,
        )

        # Decode each frame
        for frame in result.frames[tts_model.delay_steps:]:
            if (frame[:, 1:] == -1).any():
                continue
            pcm = tts_model.mimi.decode(frame[:, 1:, :])
            pcm = pcm.squeeze(0).clamp(-1, 1).cpu().numpy()
            pcm_chunks.append(pcm)

        end_step = result.end_steps[0]

    # Concatenate and trim
    if pcm_chunks:
        audio = np.concatenate(pcm_chunks, axis=-1)

        # Trim based on end_step
        if end_step is not None:
            wav_length = int(
                tts_model.mimi.sample_rate
                * (end_step + tts_model.final_padding)
                / tts_model.mimi.frame_rate
            )
            audio = audio[:, :wav_length]

        # Skip the voice prefix audio
        if prefix_samples > 0 and audio.shape[-1] > prefix_samples:
            audio = audio[:, prefix_samples:]

        # Convert to 1D
        audio = audio.squeeze()

        t_total = (time.perf_counter() - t0) * 1000
        duration = len(audio) / tts_model.mimi.sample_rate
        print(f"[tts] '{text[:40]}' | total:{t_total:.0f}ms dur:{duration:.2f}s")

        return audio.astype(np.float32)

    return np.array([], dtype=np.float32)


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
        "model": "kyutai-tts-streaming",
        "sample_rate": tts_model.mimi.sample_rate if tts_model else 24000,
        "format": "float32",
        "voice_loaded": voice_prefix is not None,
    }


@app.websocket("/ws")
async def websocket_tts(ws: WebSocket):
    """
    WebSocket endpoint for streaming TTS.

    Protocol:
    1. Client sends text as string messages
    2. Server streams back raw PCM float32 @ 24kHz as binary
    3. Server sends {"done": true} JSON when complete

    For streaming chunks:
    - Client can send partial text, server generates as it receives
    - Server sends audio chunks as they're decoded
    """
    await ws.accept()

    try:
        while True:
            # Receive text from client
            data = await ws.receive()

            if "text" in data:
                text = data["text"]
                if not text.strip():
                    continue

                # Generate audio
                audio = generate_streaming(text)

                if len(audio) > 0:
                    # Stream in chunks (2400 samples = 100ms at 24kHz)
                    chunk_size = 2400
                    for i in range(0, len(audio), chunk_size):
                        chunk = audio[i:i + chunk_size]
                        await ws.send_bytes(chunk.tobytes())

                # Signal completion
                await ws.send_json({"done": True, "samples": len(audio)})

            elif "bytes" in data:
                # Binary message - ignore for now
                pass

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
    parser = argparse.ArgumentParser(description="Kyutai Streaming TTS Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.environ["DEVICE"] = args.device

    print(f"Starting Kyutai Streaming TTS Server on {args.host}:{args.port}")
    print(f"  Weights: {WEIGHTS_DIR}")
    print(f"  Voice: {VOICE_WAV}")
    print(f"  Device: {args.device}")
    print(f"  Output: raw PCM float32 @ 24kHz")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
