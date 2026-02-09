#!/usr/bin/env python3
"""
Test script for the upgraded voice pipeline.

Tests:
  1. Parakeet STT server
  2. VibeVoice TTS server (0.5B and 1.5B)
  3. Full pipeline comparison

Usage:
  # Test Parakeet STT
  python test_pipeline.py --test stt --stt-port 8029

  # Test TTS (0.5B)
  python test_pipeline.py --test tts

  # Test TTS (1.5B with voice cloning)
  python test_pipeline.py --test tts --tts-port 8027

  # Full latency comparison
  python test_pipeline.py --test latency
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

import numpy as np
import requests


def generate_test_audio(duration_seconds: float = 3.0, sample_rate: int = 16000) -> bytes:
    """Generate a simple test audio file with silence (for testing endpoints)."""
    # Generate silence with a tiny bit of noise to simulate audio
    samples = int(duration_seconds * sample_rate)
    audio = np.random.randn(samples).astype(np.float32) * 0.001

    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)

    # Write to WAV bytes
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_int16.tobytes())

    buf.seek(0)
    return buf.read()


def test_stt_health(port: int) -> bool:
    """Test STT server health."""
    try:
        resp = requests.get(f"http://localhost:{port}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print(f"  [OK] STT server on port {port}")
            print(f"       Model: {data.get('model', 'unknown')}")
            print(f"       Device: {data.get('device', 'unknown')}")
            return True
        else:
            print(f"  [FAIL] STT server returned {resp.status_code}")
            return False
    except Exception as e:
        print(f"  [FAIL] STT server not responding: {e}")
        return False


def test_stt_transcribe(port: int, audio_path: str = None) -> bool:
    """Test STT transcription."""
    print(f"\nTesting STT transcription on port {port}...")

    if audio_path and Path(audio_path).exists():
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()
        print(f"  Using audio file: {audio_path}")
    else:
        audio_bytes = generate_test_audio(2.0)
        print("  Using generated test audio (silence)")

    try:
        start = time.time()
        resp = requests.post(
            f"http://localhost:{port}/transcribe",
            files={"file": ("test.wav", audio_bytes, "audio/wav")},
            timeout=30
        )
        elapsed = time.time() - start

        if resp.status_code == 200:
            data = resp.json()
            text = data.get("text", "")
            rtf = data.get("rtf", elapsed / 2.0)
            print(f"  [OK] Transcription: '{text[:50]}...' ({len(text)} chars)")
            print(f"       Latency: {elapsed*1000:.0f}ms, RTF: {rtf:.3f}")
            return True
        else:
            print(f"  [FAIL] Transcription failed: {resp.status_code} - {resp.text}")
            return False
    except Exception as e:
        print(f"  [FAIL] Transcription error: {e}")
        return False


def test_tts_health(port: int) -> bool:
    """Test TTS server health."""
    try:
        resp = requests.get(f"http://localhost:{port}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print(f"  [OK] TTS server on port {port}")
            print(f"       Model: {data.get('model', 'unknown')}")
            print(f"       Model Size: {data.get('model_size', 'unknown')}")
            print(f"       Streaming: {data.get('streaming', False)}")
            if 'voice' in data:
                print(f"       Voice: {data['voice']}")
            return True
        else:
            print(f"  [FAIL] TTS server returned {resp.status_code}")
            return False
    except Exception as e:
        print(f"  [FAIL] TTS server not responding: {e}")
        return False


def test_tts_synthesize(port: int, text: str = "Hello, this is a test of the voice pipeline.") -> bool:
    """Test TTS synthesis."""
    print(f"\nTesting TTS synthesis on port {port}...")

    try:
        start = time.time()
        resp = requests.post(
            f"http://localhost:{port}/tts",
            json={"text": text},
            timeout=60
        )
        elapsed = time.time() - start

        if resp.status_code == 200:
            audio_bytes = resp.content
            # Parse WAV to get duration
            with io.BytesIO(audio_bytes) as buf:
                with wave.open(buf, 'rb') as wav:
                    frames = wav.getnframes()
                    rate = wav.getframerate()
                    duration = frames / rate

            rtf = elapsed / duration if duration > 0 else 0
            print(f"  [OK] Synthesized {duration:.2f}s audio")
            print(f"       Latency: {elapsed*1000:.0f}ms, RTF: {rtf:.3f}")
            print(f"       Audio size: {len(audio_bytes)} bytes")

            # Save for playback
            output_path = "/tmp/tts_test_output.wav"
            with open(output_path, 'wb') as f:
                f.write(audio_bytes)
            print(f"       Saved to: {output_path}")
            return True
        else:
            print(f"  [FAIL] Synthesis failed: {resp.status_code} - {resp.text}")
            return False
    except Exception as e:
        print(f"  [FAIL] Synthesis error: {e}")
        return False


def test_tts_websocket(port: int, text: str = "Hello, this is a streaming test.") -> bool:
    """Test TTS WebSocket streaming."""
    print(f"\nTesting TTS WebSocket streaming on port {port}...")

    try:
        import websockets
    except ImportError:
        print("  [SKIP] websockets module not installed")
        return True

    async def stream_test():
        try:
            import urllib.parse
            url = f"ws://localhost:{port}/stream?text={urllib.parse.quote(text)}"

            start = time.time()
            first_chunk_time = None
            total_bytes = 0
            chunk_count = 0

            async with websockets.connect(url) as ws:
                async for message in ws:
                    if isinstance(message, bytes):
                        if first_chunk_time is None:
                            first_chunk_time = time.time() - start
                        total_bytes += len(message)
                        chunk_count += 1
                    else:
                        # JSON message
                        data = json.loads(message)
                        if data.get("type") == "complete":
                            break
                        elif data.get("type") == "error":
                            print(f"  [FAIL] Stream error: {data.get('message')}")
                            return False

            elapsed = time.time() - start
            # 16-bit audio at 24kHz
            audio_duration = total_bytes / (2 * 24000)

            print(f"  [OK] Streamed {chunk_count} chunks, {audio_duration:.2f}s audio")
            print(f"       First chunk: {first_chunk_time*1000:.0f}ms")
            print(f"       Total time: {elapsed*1000:.0f}ms")
            return True

        except Exception as e:
            print(f"  [FAIL] WebSocket error: {e}")
            return False

    return asyncio.run(stream_test())


def test_full_latency(stt_port: int = 8029, tts_port: int = 8027):
    """Test full pipeline latency estimation."""
    print("\n" + "=" * 50)
    print("Full Pipeline Latency Estimation")
    print("=" * 50)

    # Test STT
    print("\n1. STT Latency:")
    audio_bytes = generate_test_audio(2.0)

    try:
        start = time.time()
        resp = requests.post(
            f"http://localhost:{stt_port}/transcribe",
            files={"file": ("test.wav", audio_bytes, "audio/wav")},
            timeout=30
        )
        stt_time = (time.time() - start) * 1000
        if resp.status_code == 200:
            print(f"   STT: {stt_time:.0f}ms")
        else:
            print(f"   STT: FAILED")
            stt_time = 0
    except Exception as e:
        print(f"   STT: ERROR - {e}")
        stt_time = 0

    # Test TTS first chunk
    print("\n2. TTS First Chunk Latency:")
    try:
        import websockets
        import urllib.parse

        async def tts_first_chunk():
            url = f"ws://localhost:{tts_port}/stream?text={urllib.parse.quote('Hello')}"
            start = time.time()

            async with websockets.connect(url, close_timeout=5) as ws:
                async for message in ws:
                    if isinstance(message, bytes):
                        return (time.time() - start) * 1000
                    else:
                        data = json.loads(message)
                        if data.get("type") == "error":
                            return 0
            return 0

        tts_first = asyncio.run(tts_first_chunk())
        print(f"   TTS (first chunk): {tts_first:.0f}ms")
    except Exception as e:
        print(f"   TTS: ERROR - {e}")
        tts_first = 0

    # Estimate LLM time (assume ~100ms for first token with local model)
    llm_time = 100
    print(f"\n3. LLM First Token (estimated): ~{llm_time}ms")

    # Total
    total = stt_time + llm_time + tts_first
    print(f"\n{'=' * 50}")
    print(f"Estimated Total Latency to First Audio: ~{total:.0f}ms")
    print(f"{'=' * 50}")

    # Compare with baseline
    if stt_port == 8029:
        print("\nComparison (estimated):")
        print(f"  Current (Parakeet): ~{total:.0f}ms")
        # Estimate whisper at 20x RTF for 2s audio = 100ms
        whisper_estimate = 500 + llm_time + tts_first
        print(f"  Baseline (Whisper): ~{whisper_estimate:.0f}ms")
        print(f"  Improvement: ~{whisper_estimate - total:.0f}ms faster")


def main():
    parser = argparse.ArgumentParser(description="Test upgraded voice pipeline")
    parser.add_argument("--test", choices=["stt", "tts", "latency", "all"], default="all")
    parser.add_argument("--stt-port", type=int, default=8029, help="STT server port")
    parser.add_argument("--tts-port", type=int, default=8027, help="TTS server port")
    parser.add_argument("--audio", type=str, help="Audio file for STT test")
    parser.add_argument("--text", type=str, default="Hello, this is a test.",
                        help="Text for TTS test")
    args = parser.parse_args()

    print("=" * 50)
    print("Voice Pipeline Test Suite")
    print("=" * 50)

    if args.test in ["stt", "all"]:
        print(f"\n--- STT Tests (port {args.stt_port}) ---")
        test_stt_health(args.stt_port)
        test_stt_transcribe(args.stt_port, args.audio)

    if args.test in ["tts", "all"]:
        print(f"\n--- TTS Tests (port {args.tts_port}) ---")
        test_tts_health(args.tts_port)
        test_tts_synthesize(args.tts_port, args.text)
        test_tts_websocket(args.tts_port, args.text)

    if args.test in ["latency", "all"]:
        test_full_latency(args.stt_port, args.tts_port)

    print("\nDone!")


if __name__ == "__main__":
    main()
