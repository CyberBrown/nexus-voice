#!/usr/bin/env python3
"""
Whisper STT Server - Fast speech-to-text using OpenAI Whisper on CUDA.

Provides a drop-in replacement for Parakeet with detailed timing breakdown.

Usage:
  python whisper_stt_server.py --port 8029 --model base
"""

import argparse
import io
import os
import tempfile
import time
import wave
from typing import Optional

import numpy as np
import torch
import uvicorn
import whisper
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

# Configuration
DEFAULT_MODEL = "base"  # tiny, base, small, medium, large
DEFAULT_PORT = 8029
SAMPLE_RATE = 16000

app = FastAPI(title="Whisper STT Server")
model = None
model_name = None
device = None


def load_model(name: str, dev: str = "cuda"):
    """Load Whisper model."""
    global model, model_name, device

    print(f"[startup] Loading Whisper model: {name} on {dev}")
    t0 = time.perf_counter()

    model = whisper.load_model(name, device=dev)
    model_name = name
    device = dev

    load_ms = (time.perf_counter() - t0) * 1000
    print(f"[startup] Model loaded in {load_ms:.0f}ms")

    # Warmup inference
    print("[startup] Warming up...")
    dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1 second of silence
    _ = model.transcribe(dummy, language="en", fp16=(dev == "cuda"))
    print("[startup] Warmup complete, ready for inference")


def audio_bytes_to_array(audio_bytes: bytes):
    """Convert WAV bytes to numpy array."""
    try:
        with io.BytesIO(audio_bytes) as buf:
            with wave.open(buf, 'rb') as wav:
                n_channels = wav.getnchannels()
                sampwidth = wav.getsampwidth()
                framerate = wav.getframerate()
                n_frames = wav.getnframes()
                audio_data = wav.readframes(n_frames)

                if sampwidth == 2:
                    dtype = np.int16
                elif sampwidth == 4:
                    dtype = np.int32
                else:
                    dtype = np.int16

                audio_array = np.frombuffer(audio_data, dtype=dtype)

                if n_channels == 2:
                    audio_array = audio_array.reshape(-1, 2).mean(axis=1)

                audio_float = audio_array.astype(np.float32) / np.iinfo(dtype).max
                return audio_float, framerate
    except Exception as e:
        # Fallback: assume raw PCM 16kHz 16-bit mono
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0
        return audio_float, SAMPLE_RATE


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio

    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        # Linear interpolation fallback
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio)


@app.on_event("startup")
async def startup():
    """Load model on startup."""
    name = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    dev = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    load_model(name, dev)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": f"whisper-{model_name}",
        "device": device,
        "sample_rate": SAMPLE_RATE,
    }


@app.post("/inference")
async def inference(
    file: UploadFile = File(...),
    response_format: str = Form("json"),
):
    """
    Whisper-compatible inference endpoint with detailed timing.
    """
    global model

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    audio_bytes = await file.read()
    t_read = time.perf_counter()

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio data provided")

    # Parse audio
    try:
        audio_array, orig_sr = audio_bytes_to_array(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse audio: {e}")
    t_parse = time.perf_counter()

    # Resample if needed
    if orig_sr != SAMPLE_RATE:
        audio_array = resample_audio(audio_array, orig_sr, SAMPLE_RATE)
    t_resample = time.perf_counter()

    audio_duration = len(audio_array) / SAMPLE_RATE

    # Transcribe directly from numpy array (no temp file needed!)
    result = model.transcribe(
        audio_array,
        language="en",
        fp16=(device == "cuda"),
        initial_prompt=None,
    )
    t_infer = time.perf_counter()

    text = result["text"].strip()
    t_end = time.perf_counter()

    # Timing breakdown
    read_ms = (t_read - t0) * 1000
    parse_ms = (t_parse - t_read) * 1000
    resample_ms = (t_resample - t_parse) * 1000
    infer_ms = (t_infer - t_resample) * 1000
    total_ms = (t_end - t0) * 1000
    rtf = total_ms / (audio_duration * 1000) if audio_duration > 0 else 0

    print(f"[inference] '{text[:40]}' | read:{read_ms:.0f}ms parse:{parse_ms:.0f}ms resample:{resample_ms:.0f}ms infer:{infer_ms:.0f}ms total:{total_ms:.0f}ms RTF:{rtf:.3f}")

    return {"text": text}


@app.post("/transcribe")
async def transcribe(
    request: Request,
    file: Optional[UploadFile] = File(None),
):
    """Alternative transcription endpoint."""
    if file is not None:
        audio_bytes = await file.read()
    else:
        audio_bytes = await request.body()

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio data")

    t0 = time.perf_counter()

    audio_array, orig_sr = audio_bytes_to_array(audio_bytes)
    if orig_sr != SAMPLE_RATE:
        audio_array = resample_audio(audio_array, orig_sr, SAMPLE_RATE)

    audio_duration = len(audio_array) / SAMPLE_RATE

    result = model.transcribe(
        audio_array,
        language="en",
        fp16=(device == "cuda"),
    )

    text = result["text"].strip()
    inference_time = time.perf_counter() - t0
    rtf = inference_time / audio_duration if audio_duration > 0 else 0

    return {
        "text": text,
        "duration": audio_duration,
        "inference_time": inference_time,
        "rtf": rtf,
    }


def main():
    parser = argparse.ArgumentParser(description="Whisper STT Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.environ["MODEL_NAME"] = args.model
    os.environ["DEVICE"] = args.device

    print(f"Starting Whisper STT Server on {args.host}:{args.port}")
    print(f"Model: whisper-{args.model}, Device: {args.device}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
