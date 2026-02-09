#!/usr/bin/env python3
"""
VibeVoice Streaming TTS Server
WebSocket-based streaming for lowest latency voice synthesis.

Supports two models:
  - 0.5B (VibeVoice-Realtime-0.5B): Streaming, low latency, preset voices
  - 1.5B (VibeVoice-1.5B): Higher quality, voice cloning from audio sample

Endpoints:
  WS  /stream?text=...     - Stream audio for text (query param)
  WS  /stream              - Stream audio for text (send JSON messages)
  POST /tts                - Batch TTS (returns WAV, for 1.5B or compatibility)
  POST /synthesize         - Alias for /tts
  GET /health              - Health check
  GET /voices              - List available voices

Usage:
  # 0.5B (default, streaming)
  python vibevoice_streaming_server.py --port 8027 --voice en-Emma_woman

  # 1.5B with voice cloning
  python vibevoice_streaming_server.py --model 1.5b --voice-sample ~/voices/my-voice.wav --port 8027
"""

import asyncio
import argparse
import copy
import io
import json
import os
import threading
import traceback
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse, Response
from starlette.websockets import WebSocketDisconnect, WebSocketState

# Model paths
MODEL_PATH_05B = "microsoft/VibeVoice-Realtime-0.5B"
MODEL_PATH_15B = "microsoft/VibeVoice-1.5B"

# Voices directory for 0.5B presets
VOICES_DIR = Path(__file__).parent / "VibeVoice/demo/voices/streaming_model"
SAMPLE_RATE = 24000
DEFAULT_PORT = 8027


class StreamingTTSService:
    """Streaming TTS service using VibeVoice-Realtime-0.5B."""

    def __init__(self, model_path: str, device: str = "cuda", inference_steps: int = 5):
        self.model_path = model_path
        self.inference_steps = inference_steps
        self.sample_rate = SAMPLE_RATE
        self.device = device
        self._torch_device = torch.device(device)
        self.model_size = "0.5b"

        self.processor = None
        self.model = None
        self.voice_presets: Dict[str, Path] = {}
        self.default_voice_key: Optional[str] = None
        self._voice_cache: Dict[str, Any] = {}

    def load(self) -> None:
        """Load model and processor."""
        from vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_streaming_processor import (
            VibeVoiceStreamingProcessor,
        )

        print(f"[startup] Loading 0.5B processor from {self.model_path}")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

        # Device-specific settings
        if self.device == "cuda":
            load_dtype = torch.bfloat16
            device_map = "cuda"
            attn_impl = "flash_attention_2"
        else:
            load_dtype = torch.float32
            device_map = self.device
            attn_impl = "sdpa"

        print(f"[startup] Loading model with dtype={load_dtype}, attn={attn_impl}")

        try:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=load_dtype,
                device_map=device_map,
                attn_implementation=attn_impl,
            )
        except Exception as e:
            if attn_impl == "flash_attention_2":
                print(f"[startup] flash_attention_2 failed, using SDPA: {e}")
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=device_map,
                    attn_implementation="sdpa",
                )
            else:
                raise

        self.model.eval()
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)

        # Load voice presets
        self.voice_presets = self._load_voice_presets()
        print(f"[startup] Found {len(self.voice_presets)} voice presets")

    def _load_voice_presets(self) -> Dict[str, Path]:
        """Scan for available voice presets."""
        if not VOICES_DIR.exists():
            print(f"[startup] Warning: Voices directory not found: {VOICES_DIR}")
            return {}

        presets = {}
        for pt_path in VOICES_DIR.glob("*.pt"):
            presets[pt_path.stem] = pt_path
        return dict(sorted(presets.items()))

    def load_voice(self, voice_key: str) -> None:
        """Load a voice preset into cache."""
        if voice_key not in self.voice_presets:
            raise ValueError(f"Voice not found: {voice_key}")

        if voice_key not in self._voice_cache:
            path = self.voice_presets[voice_key]
            print(f"[voice] Loading {voice_key} from {path}")
            self._voice_cache[voice_key] = torch.load(
                path, map_location=self._torch_device, weights_only=False
            )
        self.default_voice_key = voice_key

    def _get_voice(self, voice_key: Optional[str] = None) -> Any:
        """Get voice preset data."""
        key = voice_key or self.default_voice_key
        if key not in self._voice_cache:
            self.load_voice(key)
        return self._voice_cache[key]

    def _prepare_inputs(self, text: str, voice_data: Any) -> Dict:
        """Prepare model inputs."""
        processed = self.processor.process_input_with_cached_prompt(
            text=text.strip().replace("'", "'"),
            cached_prompt=voice_data,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return {k: v.to(self._torch_device) if hasattr(v, "to") else v
                for k, v in processed.items()}

    def stream(
        self,
        text: str,
        voice_key: Optional[str] = None,
        cfg_scale: float = 1.5,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[np.ndarray]:
        """Stream audio chunks for given text."""
        from vibevoice.modular.streamer import AudioStreamer

        if not text.strip():
            return

        voice_data = self._get_voice(voice_key)
        inputs = self._prepare_inputs(text, voice_data)

        audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        errors = []
        stop_signal = stop_event or threading.Event()

        def run_generation():
            try:
                self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={"do_sample": False},
                    audio_streamer=audio_streamer,
                    stop_check_fn=stop_signal.is_set,
                    verbose=False,
                    all_prefilled_outputs=copy.deepcopy(voice_data),
                )
            except Exception as e:
                errors.append(e)
                traceback.print_exc()
                audio_streamer.end()

        thread = threading.Thread(target=run_generation, daemon=True)
        thread.start()

        try:
            stream = audio_streamer.get_stream(0)
            for chunk in stream:
                if torch.is_tensor(chunk):
                    chunk = chunk.detach().cpu().to(torch.float32).numpy()
                else:
                    chunk = np.asarray(chunk, dtype=np.float32)

                if chunk.ndim > 1:
                    chunk = chunk.reshape(-1)

                # Normalize
                peak = np.max(np.abs(chunk)) if chunk.size else 0.0
                if peak > 1.0:
                    chunk = chunk / peak

                yield chunk.astype(np.float32)
        finally:
            stop_signal.set()
            audio_streamer.end()
            thread.join(timeout=5.0)
            if errors:
                raise errors[0]

    def synthesize_batch(self, text: str, voice_key: Optional[str] = None) -> np.ndarray:
        """Batch synthesis (non-streaming) for compatibility."""
        chunks = []
        for chunk in self.stream(text, voice_key=voice_key):
            chunks.append(chunk)
        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks)

    def chunk_to_pcm16(self, chunk: np.ndarray) -> bytes:
        """Convert float32 audio chunk to PCM16 bytes."""
        chunk = np.clip(chunk, -1.0, 1.0)
        return (chunk * 32767.0).astype(np.int16).tobytes()


class VoiceCloningTTSService:
    """TTS service using VibeVoice-1.5B with voice cloning support."""

    def __init__(self, model_path: str, voice_sample_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.voice_sample_path = voice_sample_path
        self.device = device
        self._torch_device = torch.device(device)
        self.sample_rate = SAMPLE_RATE
        self.model_size = "1.5b"

        self.processor = None
        self.model = None
        self.voice_embedding = None

    def load(self) -> None:
        """Load model, processor, and voice sample."""
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

        print(f"[startup] Loading 1.5B processor from {self.model_path}")
        self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)

        # Device-specific settings
        if self.device == "cuda":
            load_dtype = torch.bfloat16
            attn_impl = "flash_attention_2"
        else:
            load_dtype = torch.float32
            attn_impl = "sdpa"

        print(f"[startup] Loading 1.5B model with dtype={load_dtype}, attn={attn_impl}")

        # Import the non-streaming model
        try:
            from vibevoice.modular.modeling_vibevoice import (
                VibeVoiceForConditionalGeneration,
            )
        except ImportError:
            # Fallback to streaming inference class if non-streaming not available
            from vibevoice.modular.modeling_vibevoice_streaming_inference import (
                VibeVoiceStreamingForConditionalGenerationInference as VibeVoiceForConditionalGeneration,
            )

        try:
            self.model = VibeVoiceForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=load_dtype,
                device_map=self.device if self.device == "cuda" else None,
                attn_implementation=attn_impl,
            )
        except Exception as e:
            if attn_impl == "flash_attention_2":
                print(f"[startup] flash_attention_2 failed, using SDPA: {e}")
                self.model = VibeVoiceForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=self.device if self.device == "cuda" else None,
                    attn_implementation="sdpa",
                )
            else:
                raise

        self.model.eval()

        # Load voice sample for cloning
        if self.voice_sample_path and Path(self.voice_sample_path).exists():
            print(f"[startup] Loading voice sample from {self.voice_sample_path}")
            self._load_voice_sample()
        else:
            print(f"[startup] Warning: Voice sample not found: {self.voice_sample_path}")

    def _load_voice_sample(self) -> None:
        """Load and prepare voice sample for cloning."""
        import scipy.io.wavfile as wav

        sample_rate, audio = wav.read(self.voice_sample_path)

        # Convert to float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0

        # Convert stereo to mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample to 24kHz if needed
        if sample_rate != SAMPLE_RATE:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
            except ImportError:
                # Simple resampling
                ratio = SAMPLE_RATE / sample_rate
                new_length = int(len(audio) * ratio)
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, new_length),
                    np.arange(len(audio)),
                    audio
                )

        self.voice_embedding = audio
        print(f"[startup] Voice sample loaded: {len(audio)/SAMPLE_RATE:.2f}s")

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to audio using voice cloning."""
        if self.voice_embedding is None:
            raise RuntimeError("No voice sample loaded for cloning")

        # Format as script for VibeVoice
        script = f"Speaker 1: {text}"

        # Process with voice sample
        inputs = self.processor(
            text=script,
            voice_samples=[self.voice_embedding],
            padding=True,
            return_tensors="pt",
        )

        # Move to device
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self._torch_device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=1.5,
                tokenizer=self.processor.tokenizer,
                generation_config={"do_sample": False},
            )

        # Extract audio
        if hasattr(outputs, 'speech_outputs') and outputs.speech_outputs:
            audio = outputs.speech_outputs[0]
            if torch.is_tensor(audio):
                audio = audio.cpu().numpy()
            return audio.squeeze().astype(np.float32)
        else:
            return np.array([], dtype=np.float32)

    def chunk_to_pcm16(self, chunk: np.ndarray) -> bytes:
        """Convert float32 audio chunk to PCM16 bytes."""
        chunk = np.clip(chunk, -1.0, 1.0)
        return (chunk * 32767.0).astype(np.int16).tobytes()


# FastAPI app
app = FastAPI(title="VibeVoice TTS Server")
service = None


@app.on_event("startup")
async def startup():
    global service

    model_size = os.environ.get("MODEL_SIZE", "0.5b")
    device = os.environ.get("DEVICE", "cuda")
    voice = os.environ.get("VOICE", "en-Emma_woman")
    voice_sample = os.environ.get("VOICE_SAMPLE", "")

    if model_size == "1.5b":
        if not voice_sample:
            raise RuntimeError("1.5B model requires --voice-sample argument")
        service = VoiceCloningTTSService(
            model_path=MODEL_PATH_15B,
            voice_sample_path=voice_sample,
            device=device,
        )
    else:
        service = StreamingTTSService(
            model_path=MODEL_PATH_05B,
            device=device,
        )

    service.load()

    if hasattr(service, 'load_voice') and voice:
        try:
            service.load_voice(voice)
        except ValueError as e:
            print(f"[startup] Warning: {e}")
            if service.voice_presets:
                first_voice = list(service.voice_presets.keys())[0]
                print(f"[startup] Using default voice: {first_voice}")
                service.load_voice(first_voice)

    app.state.service = service
    app.state.ws_lock = asyncio.Lock()
    app.state.model_size = model_size

    print(f"[startup] Ready: {model_size} model")
    if hasattr(service, 'default_voice_key'):
        print(f"[startup] Voice: {service.default_voice_key}")


@app.get("/health")
async def health():
    model_name = f"vibevoice-{app.state.model_size}"
    response = {
        "status": "ok",
        "model": model_name,
        "model_size": app.state.model_size,
        "streaming": app.state.model_size == "0.5b",
        "sample_rate": SAMPLE_RATE,
    }
    if hasattr(service, 'default_voice_key'):
        response["voice"] = service.default_voice_key
    if hasattr(service, 'voice_sample_path'):
        response["voice_sample"] = service.voice_sample_path
    return response


@app.get("/voices")
async def voices():
    if not service:
        return {"voices": []}

    if hasattr(service, 'voice_presets'):
        return {
            "voices": list(service.voice_presets.keys()),
            "default": service.default_voice_key,
            "model": "0.5b",
        }
    else:
        return {
            "voices": ["cloned"],
            "default": "cloned",
            "model": "1.5b",
            "voice_sample": service.voice_sample_path if hasattr(service, 'voice_sample_path') else None,
        }


@app.websocket("/stream")
async def websocket_stream(ws: WebSocket):
    """WebSocket endpoint for streaming TTS (0.5B only).

    Connect with text as query param: /stream?text=Hello
    Or send JSON messages: {"text": "Hello", "voice": "en-Emma_woman"}

    Receives: PCM16 audio bytes at 24kHz
    """
    await ws.accept()

    # Check if using streaming model
    if app.state.model_size != "0.5b":
        await ws.send_json({
            "type": "error",
            "message": "Streaming not supported with 1.5B model. Use /tts endpoint."
        })
        await ws.close(code=1003)
        return

    # Check if service is busy
    lock = app.state.ws_lock
    if lock.locked():
        await ws.send_json({"type": "error", "message": "Service busy"})
        await ws.close(code=1013)
        return

    async with lock:
        # Get text from query params or wait for message
        text = ws.query_params.get("text", "")
        voice = ws.query_params.get("voice")
        cfg = float(ws.query_params.get("cfg", "1.5"))

        if not text:
            # Wait for JSON message with text
            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=30.0)
                text = msg.get("text", "")
                voice = msg.get("voice", voice)
                cfg = msg.get("cfg", cfg)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "error", "message": "Timeout waiting for text"})
                await ws.close()
                return
            except Exception as e:
                await ws.send_json({"type": "error", "message": str(e)})
                await ws.close()
                return

        if not text:
            await ws.send_json({"type": "error", "message": "No text provided"})
            await ws.close()
            return

        print(f"[stream] Starting: {text[:50]}...")
        await ws.send_json({"type": "start", "text_length": len(text)})

        stop_event = threading.Event()
        chunk_count = 0
        total_samples = 0

        try:
            iterator = service.stream(text, voice_key=voice, cfg_scale=cfg, stop_event=stop_event)
            sentinel = object()

            while ws.client_state == WebSocketState.CONNECTED:
                chunk = await asyncio.to_thread(next, iterator, sentinel)
                if chunk is sentinel:
                    break

                pcm_bytes = service.chunk_to_pcm16(chunk)
                await ws.send_bytes(pcm_bytes)

                chunk_count += 1
                total_samples += len(chunk)

                if chunk_count == 1:
                    await ws.send_json({"type": "first_audio"})

        except WebSocketDisconnect:
            print("[stream] Client disconnected")
            stop_event.set()
        except Exception as e:
            print(f"[stream] Error: {e}")
            traceback.print_exc()
            try:
                await ws.send_json({"type": "error", "message": str(e)})
            except:
                pass
        finally:
            stop_event.set()
            duration = total_samples / SAMPLE_RATE if total_samples else 0
            print(f"[stream] Complete: {chunk_count} chunks, {duration:.2f}s audio")

            if ws.client_state == WebSocketState.CONNECTED:
                try:
                    await ws.send_json({
                        "type": "complete",
                        "chunks": chunk_count,
                        "duration": duration,
                    })
                    await ws.close()
                except:
                    pass


@app.post("/tts")
async def tts_batch(request: Request):
    """Batch TTS synthesis endpoint.

    Works with both 0.5B and 1.5B models.
    For 1.5B, uses voice cloning from the loaded sample.

    Input: JSON {"text": "...", "voice": "..." (optional for 0.5B)}
    Output: WAV audio file
    """
    try:
        data = await request.json()
    except:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    text = data.get("text", "")
    voice = data.get("voice")

    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    print(f"[tts] Synthesizing ({app.state.model_size}): {text[:50]}...")

    try:
        if app.state.model_size == "1.5b":
            audio = service.synthesize(text)
        else:
            audio = service.synthesize_batch(text, voice_key=voice)
    except Exception as e:
        print(f"[tts] Error: {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

    if len(audio) == 0:
        return JSONResponse({"error": "No audio generated"}, status_code=500)

    # Convert to WAV
    import scipy.io.wavfile as wav

    audio_int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    wav.write(buf, SAMPLE_RATE, audio_int16)
    buf.seek(0)

    return Response(content=buf.read(), media_type="audio/wav")


@app.post("/synthesize")
async def synthesize_batch(request: Request):
    """Alias for /tts endpoint (backwards compatibility)."""
    return await tts_batch(request)


def main():
    parser = argparse.ArgumentParser(description="VibeVoice TTS Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--model", choices=["0.5b", "1.5b"], default="0.5b",
                        help="Model size: 0.5b (streaming) or 1.5b (voice cloning)")
    parser.add_argument("--voice", default="en-Emma_woman",
                        help="Voice preset name (for 0.5B model)")
    parser.add_argument("--voice-sample", default="",
                        help="Path to voice sample WAV file (required for 1.5B model)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Validate 1.5B requirements
    if args.model == "1.5b" and not args.voice_sample:
        parser.error("--voice-sample is required when using --model 1.5b")

    if args.voice_sample and not Path(args.voice_sample).exists():
        parser.error(f"Voice sample file not found: {args.voice_sample}")

    # Set environment variables for startup hook
    os.environ["MODEL_SIZE"] = args.model
    os.environ["DEVICE"] = args.device
    os.environ["VOICE"] = args.voice
    os.environ["VOICE_SAMPLE"] = args.voice_sample

    print(f"Starting VibeVoice TTS Server on {args.host}:{args.port}")
    print(f"Model: {args.model}")
    if args.model == "0.5b":
        print(f"Voice: {args.voice}")
    else:
        print(f"Voice Sample: {args.voice_sample}")
    print(f"Device: {args.device}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
