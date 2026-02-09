#!/usr/bin/env python3
"""
Parakeet STT Server
NVIDIA NeMo-based speech-to-text server using Parakeet models.

Models available:
  - nvidia/parakeet-tdt-0.6b-v3: Best balance (252x RTF, 3.5% WER)
  - nvidia/parakeet-rnnt-0.6b: Alternative RNNT model
  - nvidia/parakeet-ctc-0.6b: CTC-based model

Endpoints:
  POST /transcribe    - Transcribe audio (multipart form or raw bytes)
  POST /inference     - Whisper-compatible endpoint
  GET  /health        - Health check

Usage:
  python parakeet_stt_server.py --port 8029 --model nvidia/parakeet-tdt-0.6b-v3
"""

import argparse
import asyncio
import io
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

# NeMo imports (lazy loaded for faster startup feedback)
nemo_asr = None
model = None

# Configuration
DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_PORT = 8029
SAMPLE_RATE = 16000

app = FastAPI(title="Parakeet STT Server")


def load_nemo():
    """Lazy load NeMo ASR module."""
    global nemo_asr
    if nemo_asr is None:
        print("[startup] Loading NeMo ASR module...")
        import nemo.collections.asr as nemo_asr_module
        nemo_asr = nemo_asr_module
    return nemo_asr


def load_model(model_name: str, device: str = "cuda"):
    """Load Parakeet ASR model."""
    global model

    print(f"[startup] Loading model: {model_name}")
    asr = load_nemo()

    # Determine model class based on model name
    if "tdt" in model_name.lower():
        model_class = asr.models.ASRModel
    elif "rnnt" in model_name.lower():
        model_class = asr.models.ASRModel
    elif "ctc" in model_name.lower():
        model_class = asr.models.ASRModel
    else:
        model_class = asr.models.ASRModel

    # Load from HuggingFace or local path
    try:
        model = model_class.from_pretrained(model_name, map_location=device)
    except Exception as e:
        print(f"[startup] Error loading from pretrained: {e}")
        # Try as a local .nemo file
        if Path(model_name).exists():
            model = model_class.restore_from(model_name, map_location=device)
        else:
            raise

    model.eval()
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()

    print(f"[startup] Model loaded on {device}")
    return model


def audio_bytes_to_array(audio_bytes: bytes) -> np.ndarray:
    """Convert audio bytes (WAV format) to numpy array."""
    import wave

    # Try to parse as WAV
    try:
        with io.BytesIO(audio_bytes) as buf:
            with wave.open(buf, 'rb') as wav:
                n_channels = wav.getnchannels()
                sampwidth = wav.getsampwidth()
                framerate = wav.getframerate()
                n_frames = wav.getnframes()

                audio_data = wav.readframes(n_frames)

                # Convert to numpy array
                if sampwidth == 2:
                    dtype = np.int16
                elif sampwidth == 4:
                    dtype = np.int32
                else:
                    dtype = np.int16

                audio_array = np.frombuffer(audio_data, dtype=dtype)

                # Convert to mono if stereo
                if n_channels == 2:
                    audio_array = audio_array.reshape(-1, 2).mean(axis=1)

                # Normalize to float32 [-1, 1]
                audio_float = audio_array.astype(np.float32) / np.iinfo(dtype).max

                return audio_float, framerate
    except Exception as e:
        # Fall back to raw PCM assumption (16kHz, 16-bit, mono)
        print(f"[transcribe] WAV parse failed, assuming raw PCM: {e}")
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
        # Simple linear interpolation fallback
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio)


@app.on_event("startup")
async def startup():
    """Load model on startup."""
    model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    device = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    load_model(model_name, device)

    app.state.model_name = model_name
    app.state.device = device
    print(f"[startup] Parakeet STT ready: {model_name}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": app.state.model_name,
        "device": app.state.device,
        "sample_rate": SAMPLE_RATE,
    }


@app.post("/transcribe")
async def transcribe(
    request: Request,
    file: Optional[UploadFile] = File(None),
):
    """
    Transcribe audio to text.

    Accepts:
      - Multipart form with 'file' field
      - Raw audio bytes in request body

    Returns:
      {"text": "transcription", "duration": 1.23, "rtf": 0.05}
    """
    global model

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Get audio bytes
    if file is not None:
        audio_bytes = await file.read()
    else:
        audio_bytes = await request.body()

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio data provided")

    start_time = time.time()

    # Convert to numpy array
    try:
        audio_array, orig_sr = audio_bytes_to_array(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse audio: {e}")

    # Resample to 16kHz if needed
    if orig_sr != SAMPLE_RATE:
        audio_array = resample_audio(audio_array, orig_sr, SAMPLE_RATE)

    audio_duration = len(audio_array) / SAMPLE_RATE

    # Save to temp file (NeMo expects file paths)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        # Write as WAV
        import wave
        with wave.open(tmp_path, 'wb') as wav_out:
            wav_out.setnchannels(1)
            wav_out.setsampwidth(2)
            wav_out.setframerate(SAMPLE_RATE)
            audio_int16 = (audio_array * 32767).astype(np.int16)
            wav_out.writeframes(audio_int16.tobytes())

    try:
        # Transcribe
        with torch.no_grad():
            transcriptions = model.transcribe([tmp_path])

        if isinstance(transcriptions, list):
            text = transcriptions[0] if transcriptions else ""
        else:
            text = str(transcriptions)

        # Handle NeMo's Hypothesis objects
        if hasattr(text, 'text'):
            text = text.text

    finally:
        # Cleanup temp file
        try:
            os.unlink(tmp_path)
        except:
            pass

    inference_time = time.time() - start_time
    rtf = inference_time / audio_duration if audio_duration > 0 else 0

    print(f"[transcribe] '{text[:50]}...' ({audio_duration:.2f}s audio, {inference_time*1000:.0f}ms, RTF={rtf:.3f})")

    return {
        "text": text.strip(),
        "duration": audio_duration,
        "inference_time": inference_time,
        "rtf": rtf,
    }


@app.post("/inference")
async def inference(
    file: UploadFile = File(...),
    response_format: str = Form("json"),
):
    """
    Whisper-compatible inference endpoint.

    This provides compatibility with existing Whisper clients.
    """
    global model

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    audio_bytes = await file.read()
    t_read = time.perf_counter()

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio data provided")

    # Convert to numpy array
    try:
        audio_array, orig_sr = audio_bytes_to_array(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse audio: {e}")
    t_parse = time.perf_counter()

    # Resample to 16kHz if needed
    if orig_sr != SAMPLE_RATE:
        audio_array = resample_audio(audio_array, orig_sr, SAMPLE_RATE)
    t_resample = time.perf_counter()

    audio_duration = len(audio_array) / SAMPLE_RATE

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        import wave
        with wave.open(tmp_path, 'wb') as wav_out:
            wav_out.setnchannels(1)
            wav_out.setsampwidth(2)
            wav_out.setframerate(SAMPLE_RATE)
            audio_int16 = (audio_array * 32767).astype(np.int16)
            wav_out.writeframes(audio_int16.tobytes())
    t_write = time.perf_counter()

    try:
        with torch.no_grad():
            transcriptions = model.transcribe([tmp_path])
        t_infer = time.perf_counter()

        if isinstance(transcriptions, list):
            text = transcriptions[0] if transcriptions else ""
        else:
            text = str(transcriptions)

        if hasattr(text, 'text'):
            text = text.text

    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass

    t_end = time.perf_counter()

    # Detailed timing breakdown
    read_ms = (t_read - t0) * 1000
    parse_ms = (t_parse - t_read) * 1000
    resample_ms = (t_resample - t_parse) * 1000
    write_ms = (t_write - t_resample) * 1000
    infer_ms = (t_infer - t_write) * 1000
    total_ms = (t_end - t0) * 1000

    print(f"[inference] '{text[:40]}' | read:{read_ms:.0f}ms parse:{parse_ms:.0f}ms resample:{resample_ms:.0f}ms write:{write_ms:.0f}ms infer:{infer_ms:.0f}ms total:{total_ms:.0f}ms")

    # Return in Whisper-compatible format
    return {
        "text": text.strip(),
    }


def main():
    parser = argparse.ArgumentParser(description="Parakeet STT Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Model name (HuggingFace) or path to .nemo file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Set environment variables for startup hook
    os.environ["MODEL_NAME"] = args.model
    os.environ["DEVICE"] = args.device

    print(f"Starting Parakeet STT Server on {args.host}:{args.port}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
