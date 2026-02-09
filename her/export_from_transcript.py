import json
import os
import subprocess
import sys

AUDIO_FILE = "/home/chris/projects/her/Her (2013) Bluray-2160p.wav"
TRANSCRIPT_FILE = "/home/chris/projects/her/labeled_transcript.json"
OUTPUT_DIR = "/home/chris/projects/her/her_lines"

if len(sys.argv) < 2:
    print("Usage: python3 export_from_transcript.py SPEAKER_XX")
    sys.exit(1)

target_speaker = sys.argv[1]
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(TRANSCRIPT_FILE) as f:
    segments = json.load(f)

target_segs = [s for s in segments if s["speaker"] == target_speaker]
print(f"Exporting {len(target_segs)} lines for {target_speaker}")

for idx, seg in enumerate(target_segs, 1):
    filename = os.path.join(OUTPUT_DIR, f"her_{idx:04d}.wav")
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(seg["start"]),
        "-i", AUDIO_FILE,
        "-t", str(seg["end"] - seg["start"]),
        "-acodec", "pcm_s16le",
        filename
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Exported her_{idx:04d}.wav [{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text'][:50]}")

print(f"\nDone! {len(target_segs)} files exported to {OUTPUT_DIR}")
