#!/usr/bin/env python3
"""Quick test of the full voice pipeline: STT -> LLM -> TTS"""

import asyncio
import json
import wave
import io
import numpy as np

async def test_pipeline():
    import websockets
    
    # Record or use a test audio file
    # For now, let's use a pre-recorded test file or generate one
    
    # Create a simple test - send text directly to TTS first
    print("=== Testing TTS Server ===")
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8027/tts",
            json={"text": "Hello! Testing the Her voice pipeline."},
            timeout=60
        ) as resp:
            if resp.status == 200:
                audio = await resp.read()
                with open("/home/chris/spark-voice-pipeline/pipeline_test_tts.wav", "wb") as f:
                    f.write(audio)
                print(f"TTS test saved: pipeline_test_tts.wav ({len(audio)} bytes)")
            else:
                print(f"TTS failed: {resp.status}")
    
    print("\n=== Testing Full Pipeline ===")
    print("Connect to ws://localhost:8028/voice")
    print("Send audio bytes, receive transcription + LLM tokens + audio")
    
    # Create test audio with speech (you'd normally record this)
    # For demo, let's use a pre-existing audio file if available
    test_audio = "/home/chris/spark-voice-pipeline/test_input.wav"
    
    import os
    if not os.path.exists(test_audio):
        print(f"\nTo test full pipeline, create {test_audio} with your voice saying something.")
        print("Or use: arecord -d 3 -r 16000 -f S16_LE test_input.wav")
        return
    
    with open(test_audio, "rb") as f:
        audio_bytes = f.read()
    
    print(f"Using test audio: {test_audio} ({len(audio_bytes)} bytes)")
    
    async with websockets.connect("ws://localhost:8028/voice") as ws:
        # Send audio
        await ws.send(audio_bytes)
        print("Sent audio, waiting for response...")
        
        audio_chunks = []
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                audio_chunks.append(msg)
            else:
                data = json.loads(msg)
                if data.get("type") == "transcription":
                    print(f"Transcription: {data['text']}")
                elif data.get("type") == "llm_token":
                    print(data["token"], end="", flush=True)
                elif data.get("type") == "complete":
                    print(f"\n\nComplete! Timing: {data.get('timing', {})}")
                    break
        
        if audio_chunks:
            # Combine audio chunks (PCM16 at 24kHz)
            audio_data = b"".join(audio_chunks)
            
            # Save as WAV
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(audio_data)
            
            with open("/home/chris/spark-voice-pipeline/pipeline_test_response.wav", "wb") as f:
                f.write(buf.getvalue())
            
            print(f"Response audio saved: pipeline_test_response.wav")

if __name__ == "__main__":
    asyncio.run(test_pipeline())
