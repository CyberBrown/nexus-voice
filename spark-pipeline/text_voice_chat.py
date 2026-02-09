#!/usr/bin/env python3
"""
Simple Text-Based Voice Chat for Testing TTS
Type text, hear Samantha respond.

Usage: python text_voice_chat.py
"""

import io
import os
import sys
import requests
import subprocess

TTS_URL = "http://localhost:8032/synthesize"
SPEAKER = "her"

def play_audio(wav_bytes):
    """Play audio using aplay or ffplay."""
    try:
        proc = subprocess.Popen(
            ['aplay', '-q', '-'],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        proc.communicate(wav_bytes)
    except FileNotFoundError:
        # Try ffplay
        proc = subprocess.Popen(
            ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', '-'],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        proc.communicate(wav_bytes)

def generate_speech(text):
    """Generate speech from text."""
    params = {'text': text, 'speaker': SPEAKER}
    
    try:
        resp = requests.post(TTS_URL, params=params, timeout=60)
        if resp.status_code == 200:
            return resp.content
        else:
            print(f"TTS Error: {resp.status_code} - {resp.text}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    print("="*50)
    print("  Samantha Voice Chat (Text Mode)")
    print("  TTS: Qwen3-TTS @ localhost:8032")
    print("="*50)
    print("Type a message and press Enter to hear Samantha speak.")
    print("Type 'quit' or Ctrl+C to exit.\n")
    
    # Test connection
    try:
        resp = requests.get("http://localhost:8032/health", timeout=5)
        if resp.status_code == 200:
            print("✓ TTS server connected\n")
        else:
            print("⚠ TTS server not healthy\n")
    except:
        print("✗ Cannot connect to TTS server at localhost:8032\n")
        return
    
    while True:
        try:
            text = input("You: ").strip()
            if not text:
                continue
            if text.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break
            
            print("Samantha: [generating...]", end="", flush=True)
            audio = generate_speech(text)
            
            if audio:
                print(f"\rSamantha: {text[:50]}{'...' if len(text) > 50 else ''}  ")
                play_audio(audio)
            else:
                print("\rSamantha: [error generating speech]")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            break

if __name__ == "__main__":
    main()
