#!/usr/bin/env python3
"""
Voice Chat Streaming Orchestrator
Runs on Spark, connects: Client -> STT -> LLM (streaming) -> Moshi TTS -> Client

Architecture:
  Client sends WAV audio via WebSocket
  -> STT transcribes (Whisper)
  -> LLM streams tokens, buffered into sentences
  -> Each sentence processed through Moshi TTS (~570ms for short phrases)
  -> PCM16 audio chunks (24kHz) streamed back to client

Usage:
  # Default (Whisper STT on 8025)
  python voice_chat_streaming.py --port 8028

Client connects to: ws://spark:8028/voice
"""

import asyncio
import argparse
import io
import json
import os
import re
import time
import wave
from typing import AsyncIterator, Optional

import aiohttp
import uvicorn
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

# Service endpoints (configurable via environment or args)
STT_PORT = int(os.environ.get("STT_PORT", "8025"))
STT_URL = f"http://localhost:{STT_PORT}/inference"
OLLAMA_URL = "http://localhost:11434/api/chat"
VLLM_URL = "http://localhost:8000/v1/chat/completions"
LLAMA_URL = "http://localhost:9003/v1/chat/completions"
TTS_WS_URL = "ws://localhost:8027/ws"
TTS_HTTP_URL = "http://localhost:8027/tts"
TTS_HEALTH_URL = "http://localhost:8027/health"

# TTS mode: "streaming" (WebSocket) or "batch" (HTTP) - auto-detected
TTS_MODE = "batch"  # Moshi TTS works best in batch mode
# LLM backend: "ollama", "vllm", or "llama" (llama.cpp)
LLM_BACKEND = "ollama"

# Configuration
SAMPLE_RATE_IN = 16000   # Input audio sample rate
SAMPLE_RATE_OUT = 24000  # Moshi TTS output sample rate
LLM_MODEL = "qwen3-4b-heretic:latest"
SYSTEM_PROMPT = "You are a helpful voice assistant. Keep responses concise - 1-2 sentences max. Do not use <think> tags or show your reasoning - just respond directly."

# Sentence boundary pattern
SENTENCE_END = re.compile(r'[.!?]\s*$')
# Pattern to strip thinking tags from Nemotron output
THINK_PATTERN = re.compile(r'<think>.*?</think>\s*', re.DOTALL)


app = FastAPI(title="Voice Chat Streaming Orchestrator")


async def transcribe_audio(audio_bytes: bytes) -> str:
    """Send audio to STT server and get transcription."""
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field('file', audio_bytes, filename='audio.wav', content_type='audio/wav')
        data.add_field('response_format', 'json')

        async with session.post(STT_URL, data=data, timeout=30) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"STT error: {resp.status} - {text}")
            result = await resp.json()
            return result.get("text", "").strip()


async def stream_llm_response(
    user_text: str,
    messages: list,
) -> AsyncIterator[str]:
    """Stream tokens from LLM (Ollama or vLLM)."""
    messages.append({"role": "user", "content": user_text})

    full_response = ""
    raw_buffer = ""  # Buffer to detect thinking pattern
    in_thinking = None  # None = unknown, True = in thinking, False = no thinking

    if LLM_BACKEND in ("vllm", "llama"):
        # OpenAI-compatible API (vLLM or llama.cpp)
        api_url = LLAMA_URL if LLM_BACKEND == "llama" else VLLM_URL
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "stream": True,
            "max_tokens": 250,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, json=payload, timeout=60) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"LLM error: {resp.status} - {text}")

                async for line in resp.content:
                    line_str = line.decode('utf-8').strip()
                    if not line_str or not line_str.startswith("data: "):
                        continue
                    if line_str == "data: [DONE]":
                        break
                    try:
                        data = json.loads(line_str[6:])
                        token = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if token:
                            raw_buffer += token

                            # Detect if response has thinking (look for </think> pattern)
                            if in_thinking is None:
                                # Check if this looks like thinking content
                                # Nemotron thinking usually starts lowercase or with analysis words
                                if "</think>" in raw_buffer:
                                    # Found end of thinking - extract content after
                                    in_thinking = False
                                    after_think = raw_buffer.split("</think>", 1)[-1].lstrip('\n')
                                    raw_buffer = ""
                                    if after_think:
                                        full_response += after_think
                                        yield after_think
                                    continue
                                elif len(raw_buffer) > 200:
                                    # No </think> after 200 chars - probably no thinking
                                    # Yield everything we've buffered
                                    in_thinking = False
                                    full_response += raw_buffer
                                    yield raw_buffer
                                    raw_buffer = ""
                                    continue
                                else:
                                    # Still detecting, buffer more
                                    continue

                            # Skip any stray think tags
                            if "<think>" in token or "</think>" in token:
                                continue

                            full_response += token
                            yield token
                    except json.JSONDecodeError:
                        continue

                # Flush any remaining buffer at end
                if raw_buffer and in_thinking is None:
                    full_response += raw_buffer
                    yield raw_buffer
    else:
        # Ollama API
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "stream": True,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(OLLAMA_URL, json=payload, timeout=60) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"LLM error: {resp.status} - {text}")

                async for line in resp.content:
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            full_response += token
                            yield token
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue

    # Add assistant response to history
    messages.append({"role": "assistant", "content": full_response})

    # Trim history if too long
    if len(messages) > 10:
        messages[:] = messages[:1] + messages[-8:]


async def stream_tts_audio(text: str, voice: str = None) -> AsyncIterator[bytes]:
    """Stream audio chunks from Moshi TTS via WebSocket."""
    if not text.strip():
        return

    async with aiohttp.ClientSession() as session:
        params = {"text": text}
        if voice:
            params["voice"] = voice

        url = f"{TTS_WS_URL}?{'&'.join(f'{k}={v}' for k, v in params.items())}"

        async with session.ws_connect(url) as ws:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    yield msg.data
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    # JSON status message
                    data = json.loads(msg.data)
                    if data.get("type") == "error":
                        raise RuntimeError(f"TTS error: {data.get('message')}")
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                    break


async def batch_tts_audio(text: str) -> AsyncIterator[bytes]:
    """Get audio from Moshi TTS via HTTP and stream PCM chunks."""
    if not text.strip():
        return

    async with aiohttp.ClientSession() as session:
        async with session.post(TTS_HTTP_URL, json={"text": text}, timeout=120) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(f"TTS error: {resp.status} - {error_text}")

            # Read WAV and convert to PCM16 chunks
            wav_bytes = await resp.read()

            # Skip WAV header (44 bytes) and yield PCM data in chunks
            pcm_data = wav_bytes[44:]
            chunk_size = 4800  # 100ms at 24kHz, 16-bit

            for i in range(0, len(pcm_data), chunk_size):
                yield pcm_data[i:i + chunk_size]


async def get_tts_audio(text: str, voice: str = None) -> AsyncIterator[bytes]:
    """Get TTS audio using appropriate mode (streaming or batch)."""
    if TTS_MODE == "batch":
        async for chunk in batch_tts_audio(text):
            yield chunk
    else:
        async for chunk in stream_tts_audio(text, voice):
            yield chunk


class SentenceBuffer:
    """Buffer LLM tokens and yield complete sentences."""

    def __init__(self):
        self.buffer = ""
        self.min_chars = 10  # Minimum chars before checking for sentence end

    def add(self, token: str) -> Optional[str]:
        """Add token, return complete sentence if available."""
        self.buffer += token

        # Check for sentence boundary
        if len(self.buffer) >= self.min_chars and SENTENCE_END.search(self.buffer):
            sentence = self.buffer.strip()
            self.buffer = ""
            return sentence
        return None

    def flush(self) -> Optional[str]:
        """Return any remaining text."""
        if self.buffer.strip():
            sentence = self.buffer.strip()
            self.buffer = ""
            return sentence
        return None


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "service": "voice-chat-orchestrator",
        "llm_backend": LLM_BACKEND,
        "llm_model": LLM_MODEL,
        "stt_url": STT_URL,
        "stt_port": STT_PORT,
    }


@app.websocket("/voice")
async def voice_chat(ws: WebSocket):
    """Main voice chat WebSocket endpoint.

    Protocol:
    1. Client sends: {"type": "audio", "data": "<base64 WAV>"}
    2. Server streams back:
       - {"type": "transcription", "text": "..."}
       - {"type": "llm_token", "token": "..."}
       - {"type": "audio", "data": "<base64 PCM16>"}  (or raw bytes)
       - {"type": "complete"}
    """
    await ws.accept()
    print("[orchestrator] Client connected")

    # Conversation history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        while ws.client_state == WebSocketState.CONNECTED:
            # Wait for audio from client
            try:
                msg = await ws.receive()
            except WebSocketDisconnect:
                break

            if msg["type"] == "websocket.disconnect":
                break

            # Handle different message types
            if "bytes" in msg:
                # Raw audio bytes
                audio_bytes = msg["bytes"]
            elif "text" in msg:
                try:
                    data = json.loads(msg["text"])
                    if data.get("type") == "ping":
                        await ws.send_json({"type": "pong"})
                        continue
                    elif data.get("type") == "audio":
                        import base64
                        audio_bytes = base64.b64decode(data["data"])
                    else:
                        continue
                except (json.JSONDecodeError, KeyError):
                    continue
            else:
                continue

            print(f"[orchestrator] Received {len(audio_bytes)} bytes of audio")
            request_start = time.perf_counter()

            # Step 1: Transcribe with STT (Whisper or Parakeet)
            stt_start = time.perf_counter()
            try:
                transcription = await transcribe_audio(audio_bytes)
                stt_ms = int((time.perf_counter() - stt_start) * 1000)
                if not transcription:
                    await ws.send_json({"type": "error", "message": "No speech detected"})
                    continue
                print(f"[STT: {stt_ms}ms] Transcription: {transcription}")
                await ws.send_json({"type": "transcription", "text": transcription})
            except Exception as e:
                print(f"[orchestrator] STT error: {e}")
                await ws.send_json({"type": "error", "message": f"Transcription failed: {e}"})
                continue

            # Step 2: Stream LLM response, buffer into sentences
            sentence_buffer = SentenceBuffer()
            full_response = ""
            sentences = []
            llm_start = time.perf_counter()
            llm_first_token_ms = None
            llm_done_ms = None
            tts_first_chunk_ms = None
            tts_total_ms = 0

            try:
                async for token in stream_llm_response(transcription, messages):
                    if llm_first_token_ms is None:
                        llm_first_token_ms = int((time.perf_counter() - llm_start) * 1000)
                        print(f"[LLM first token: {llm_first_token_ms}ms]")

                    full_response += token
                    await ws.send_json({"type": "llm_token", "token": token})

                    # Check for complete sentence
                    sentence = sentence_buffer.add(token)
                    if sentence:
                        sentences.append(sentence)
                        print(f"[orchestrator] Sentence ready: {sentence[:50]}...")

                        # Stream this sentence's audio immediately
                        tts_start = time.perf_counter()
                        try:
                            first_chunk = True
                            async for audio_chunk in get_tts_audio(sentence):
                                if first_chunk and tts_first_chunk_ms is None:
                                    tts_first_chunk_ms = int((time.perf_counter() - tts_start) * 1000)
                                    print(f"[TTS first chunk: {tts_first_chunk_ms}ms]")
                                    first_chunk = False
                                await ws.send_bytes(audio_chunk)
                            tts_total_ms += int((time.perf_counter() - tts_start) * 1000)
                        except Exception as e:
                            print(f"[orchestrator] TTS error for sentence: {e}")

                llm_done_ms = int((time.perf_counter() - llm_start) * 1000)

                # Flush remaining text
                remaining = sentence_buffer.flush()
                if remaining:
                    sentences.append(remaining)
                    print(f"[orchestrator] Final sentence: {remaining[:50]}...")
                    tts_start = time.perf_counter()
                    try:
                        first_chunk = True
                        async for audio_chunk in get_tts_audio(remaining):
                            if first_chunk and tts_first_chunk_ms is None:
                                tts_first_chunk_ms = int((time.perf_counter() - tts_start) * 1000)
                                print(f"[TTS first chunk: {tts_first_chunk_ms}ms]")
                                first_chunk = False
                            await ws.send_bytes(audio_chunk)
                        tts_total_ms += int((time.perf_counter() - tts_start) * 1000)
                    except Exception as e:
                        print(f"[orchestrator] TTS error for final: {e}")

            except Exception as e:
                print(f"[orchestrator] LLM error: {e}")
                await ws.send_json({"type": "error", "message": f"LLM failed: {e}"})
                continue

            # Calculate total time
            total_ms = int((time.perf_counter() - request_start) * 1000)
            timing_summary = f"[STT:{stt_ms}ms LLM:{llm_first_token_ms}/{llm_done_ms}ms TTS:{tts_first_chunk_ms}/{tts_total_ms}ms Total:{total_ms}ms]"
            print(f"[orchestrator] {timing_summary}")

            # Signal completion with timing
            await ws.send_json({
                "type": "complete",
                "full_response": full_response,
                "sentences": len(sentences),
                "timing": {
                    "stt_ms": stt_ms,
                    "llm_first_token_ms": llm_first_token_ms,
                    "llm_total_ms": llm_done_ms,
                    "tts_first_chunk_ms": tts_first_chunk_ms,
                    "tts_total_ms": tts_total_ms,
                    "total_ms": total_ms,
                },
            })
            print(f"[orchestrator] Complete: {len(sentences)} sentences")

    except WebSocketDisconnect:
        print("[orchestrator] Client disconnected")
    except Exception as e:
        print(f"[orchestrator] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if ws.client_state == WebSocketState.CONNECTED:
            try:
                await ws.close()
            except:
                pass
        print("[orchestrator] Session ended")


@app.websocket("/voice-simple")
async def voice_chat_simple(ws: WebSocket):
    """Simpler voice chat - sends full response then streams all audio.

    Less complex but slightly higher latency than sentence-streaming.
    """
    await ws.accept()
    print("[orchestrator-simple] Client connected")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        while ws.client_state == WebSocketState.CONNECTED:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                break

            if "bytes" not in msg:
                continue

            audio_bytes = msg["bytes"]
            print(f"[orchestrator-simple] Received {len(audio_bytes)} bytes")

            # Transcribe
            try:
                transcription = await transcribe_audio(audio_bytes)
                if not transcription:
                    continue
                await ws.send_json({"type": "transcription", "text": transcription})
            except Exception as e:
                await ws.send_json({"type": "error", "message": str(e)})
                continue

            # Get full LLM response
            full_response = ""
            async for token in stream_llm_response(transcription, messages):
                full_response += token
                await ws.send_json({"type": "llm_token", "token": token})

            await ws.send_json({"type": "llm_complete", "text": full_response})

            # Stream TTS audio
            try:
                async for audio_chunk in get_tts_audio(full_response):
                    await ws.send_bytes(audio_chunk)
            except Exception as e:
                await ws.send_json({"type": "error", "message": f"TTS: {e}"})

            await ws.send_json({"type": "complete"})

    except WebSocketDisconnect:
        pass
    finally:
        print("[orchestrator-simple] Session ended")


def detect_tts_mode():
    """Detect TTS mode by checking health endpoint."""
    global TTS_MODE
    import requests
    try:
        resp = requests.get(TTS_HEALTH_URL, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            model = data.get("model", "unknown")
            model_size = data.get("model_size", "unknown")

            # Use batch mode for all current TTS backends (more reliable)
            # Chatterbox and Moshi both work better with HTTP batch
            if model in ("moshi-tts", "chatterbox-turbo", "kyutai-tts-0.75b"):
                TTS_MODE = "batch"
                return f"{model} ({model_size})"
            elif data.get("streaming") == False or model_size == "1.5b":
                TTS_MODE = "batch"
                return f"batch ({model_size})"
            else:
                # Default to batch for reliability
                TTS_MODE = "batch"
                return f"batch ({model_size})"
    except:
        pass
    TTS_MODE = "batch"
    return "batch (default)"


def main():
    global LLM_MODEL, STT_PORT, STT_URL, TTS_MODE, LLM_BACKEND, TTS_HTTP_URL, TTS_WS_URL, TTS_HEALTH_URL

    parser = argparse.ArgumentParser(description="Voice Chat Streaming Orchestrator")
    parser.add_argument("--port", type=int, default=8028)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--llm-model", default="qwen3-4b-heretic:latest",
                        help="LLM model (qwen3-4b-heretic for Ollama, nemotron-3-nano for vLLM)")
    parser.add_argument("--llm-backend", default="ollama", choices=["ollama", "vllm", "llama"],
                        help="LLM backend: ollama, vllm, or llama (default: ollama)")
    parser.add_argument("--stt-port", type=int, default=8025,
                        help="STT server port (8025=Whisper, 8029=Parakeet)")
    parser.add_argument("--tts-port", type=int, default=8027,
                        help="TTS server port (8027=Chatterbox, 8031=Piper)")
    args = parser.parse_args()

    LLM_MODEL = args.llm_model
    LLM_BACKEND = args.llm_backend
    STT_PORT = args.stt_port
    STT_URL = f"http://localhost:{STT_PORT}/inference"
    TTS_HTTP_URL = f"http://localhost:{args.tts_port}/tts"
    TTS_WS_URL = f"ws://localhost:{args.tts_port}/ws"
    TTS_HEALTH_URL = f"http://localhost:{args.tts_port}/health"

    # Also set environment variable for consistency
    os.environ["STT_PORT"] = str(STT_PORT)

    # Detect TTS mode
    tts_mode_name = detect_tts_mode()

    llm_url = LLAMA_URL if LLM_BACKEND == "llama" else (VLLM_URL if LLM_BACKEND == "vllm" else OLLAMA_URL)
    print(f"Starting Voice Chat Orchestrator on {args.host}:{args.port}")
    print(f"  STT: Whisper ({STT_URL})")
    print(f"  LLM: {LLM_BACKEND}/{LLM_MODEL}")
    print(f"  TTS: {tts_mode_name}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
