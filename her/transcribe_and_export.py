import whisper
import torch
import json
import os
import subprocess
from pyannote.audio import Pipeline

AUDIO_FILE = "/home/chris/projects/her/Her (2013) Bluray-2160p.wav"
OUTPUT_DIR = "/home/chris/projects/her/her_lines"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Transcribe with Whisper
whisper_path = "/home/chris/projects/her/whisper_transcript.json"
if os.path.exists(whisper_path):
    print("Loading existing Whisper transcript...")
    with open(whisper_path) as f:
        result = json.load(f)
else:
    print("Loading Whisper model...")
    model = whisper.load_model("medium", device="cuda")
    print("Transcribing audio (this may take a while)...")
    result = model.transcribe(AUDIO_FILE, language="en", word_timestamps=True)
    with open(whisper_path, "w") as f:
        json.dump(result, f, indent=2)
    del model
    torch.cuda.empty_cache()
print(f"Transcription: {len(result['segments'])} segments")

# Step 2: Speaker diarization with pyannote
print("Running speaker diarization...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN
)
pipeline.to(torch.device("cuda"))

diarization = pipeline(AUDIO_FILE)

# Build speaker timeline
speaker_segments = []
for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
    speaker_segments.append({
        "start": turn.start,
        "end": turn.end,
        "speaker": speaker
    })

with open("/home/chris/projects/her/diarization.json", "w") as f:
    json.dump(speaker_segments, f, indent=2)
print(f"Diarization done: {len(speaker_segments)} turns, speakers: {set(s['speaker'] for s in speaker_segments)}")

# Step 3: Assign speakers to whisper segments
def find_speaker(seg_start, seg_end, speaker_segs):
    """Find the speaker with most overlap for a given segment."""
    best_speaker = None
    best_overlap = 0
    for ss in speaker_segs:
        overlap_start = max(seg_start, ss["start"])
        overlap_end = min(seg_end, ss["end"])
        overlap = max(0, overlap_end - overlap_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = ss["speaker"]
    return best_speaker

labeled_segments = []
for seg in result["segments"]:
    speaker = find_speaker(seg["start"], seg["end"], speaker_segments)
    labeled_segments.append({
        "start": seg["start"],
        "end": seg["end"],
        "text": seg["text"].strip(),
        "speaker": speaker
    })

with open("/home/chris/projects/her/labeled_transcript.json", "w") as f:
    json.dump(labeled_segments, f, indent=2)

# Print speaker summary
from collections import Counter
speaker_counts = Counter(s["speaker"] for s in labeled_segments)
print("\nSpeaker line counts:")
for spk, count in speaker_counts.most_common():
    sample = next(s["text"][:60] for s in labeled_segments if s["speaker"] == spk)
    print(f"  {spk}: {count} lines (e.g. \"{sample}...\")")

print("\n--- Review the speaker labels above ---")
print("The labeled transcript is saved to labeled_transcript.json")
print("Once you identify which SPEAKER_XX is 'her', re-run with:")
print("  python3 export_from_transcript.py SPEAKER_XX")
