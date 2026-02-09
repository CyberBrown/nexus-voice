import re
import os
import subprocess

AUDIO_FILE = "/home/chris/projects/her/Her (2013) Bluray-2160p.wav"
TRANSCRIPT_FILE = "/home/chris/projects/her/Her (2013) Bluray-2160p.txt"
TARGET_SPEAKER = "her"
OUTPUT_DIR = "/home/chris/projects/her/her_lines"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def timecode_to_seconds(tc):
    """Convert HH:MM:SS:FF timecode to seconds (assuming 24fps)."""
    h, m, s, f = map(int, tc.split(":"))
    return h * 3600 + m * 60 + s + f / 24.0

# Parse transcript
entries = []
with open(TRANSCRIPT_FILE, "r") as f:
    lines = f.read().strip().split("\n")

i = 0
while i < len(lines):
    line = lines[i].strip()
    tc_match = re.match(r"\[(\d+:\d+:\d+:\d+)\s*-\s*(\d+:\d+:\d+:\d+)\]", line)
    if tc_match:
        start_tc = tc_match.group(1)
        end_tc = tc_match.group(2)
        if i + 1 < len(lines):
            speaker = lines[i + 1].strip()
            dialogue = ""
            if i + 2 < len(lines):
                dialogue = lines[i + 2].strip()
            entries.append((speaker, start_tc, end_tc, dialogue))
    i += 1

# Filter to target speaker
target_entries = [(s, st, et, d) for s, st, et, d in entries if s.lower() == TARGET_SPEAKER.lower()]
print(f"Found {len(target_entries)} lines for '{TARGET_SPEAKER}'")

for idx, (speaker, start_tc, end_tc, dialogue) in enumerate(target_entries, 1):
    start_s = timecode_to_seconds(start_tc)
    end_s = timecode_to_seconds(end_tc)
    duration = end_s - start_s
    filename = os.path.join(OUTPUT_DIR, f"her_{idx:04d}.wav")

    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(start_s),
        "-i", AUDIO_FILE,
        "-t", str(duration),
        "-acodec", "pcm_s16le",
        filename
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"Exported her_{idx:04d}.wav [{start_tc} - {end_tc}] {dialogue[:50]}")

print(f"\nDone! {len(target_entries)} files exported to {OUTPUT_DIR}")
