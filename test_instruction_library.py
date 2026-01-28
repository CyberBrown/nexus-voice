#!/usr/bin/env python3
"""
Test all instruction variants and find the best for each emotion.

Uses the existing TTS server API (port 8032) with the fine-tuned Samantha model.
"""

import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except ImportError:
    print("ERROR: soundfile required")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("ERROR: requests required")
    sys.exit(1)

# Add tts_quality module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_quality.tts_quality_metrics import compute_metrics
from emotion_instructions import EMOTION_LIBRARY, EMOTION_TEST_CORPUS, NEUTRAL_TEST_CORPUS

# Configuration
TTS_SERVER_URL = "http://localhost:8032"
OUTPUT_DIR = Path("/home/chris/spark-voice-pipeline/instruction_tests")
SPEAKER_NAME = "samantha"


def check_server():
    """Check if TTS server is running."""
    try:
        response = requests.get(f"{TTS_SERVER_URL}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"Server OK: {health.get('model', 'unknown')}, speaker: {health.get('speaker', 'unknown')}")
            return True
    except Exception as e:
        print(f"Server check failed: {e}")
    return False


def generate_audio(text: str, instruction: str) -> tuple[np.ndarray, int]:
    """Generate audio using the TTS server API."""
    params = {
        "text": text,
        "speaker": SPEAKER_NAME,
    }

    if instruction:
        params["instruct"] = instruction

    response = requests.post(
        f"{TTS_SERVER_URL}/synthesize",
        params=params,
        timeout=120,
    )

    if response.status_code != 200:
        raise RuntimeError(f"TTS failed: {response.status_code} - {response.text}")

    audio, sr = sf.read(io.BytesIO(response.content))
    return audio.astype(np.float32), sr


def test_instruction(instruction: str, sentences: list[str], emotion: str, variant_idx: int):
    """Generate audio for all sentences with given instruction and measure metrics."""

    segments = []
    total_duration = 0

    for i, sentence in enumerate(sentences):
        try:
            audio, sr = generate_audio(sentence, instruction)
            segments.append(audio)
            duration = len(audio) / sr
            total_duration += duration
        except Exception as e:
            print(f"      ERROR on sentence {i}: {e}")
            return None

    # Compute metrics
    try:
        metrics = compute_metrics(segments, sr)
    except Exception as e:
        print(f"      ERROR computing metrics: {e}")
        return None

    # Save concatenated audio for listening
    output_file = OUTPUT_DIR / f"{emotion}_variant{variant_idx}.wav"
    sf.write(output_file, np.concatenate(segments), sr)

    return {
        "emotion": emotion,
        "variant_idx": variant_idx,
        "instruction": instruction,
        "pitch_variance": metrics.pitch_variance,
        "pitch_contour_similarity": metrics.pitch_contour_similarity,
        "energy_variance": metrics.energy_variance,
        "speaking_rate_variance": metrics.speaking_rate_variance,
        "spectral_similarity": metrics.spectral_similarity,
        "speaker_embedding_drift": metrics.speaker_embedding_drift,
        "total_duration": total_duration,
        "audio_file": str(output_file),
    }


def run_full_test():
    """Test all emotions and all variants."""

    print("Checking TTS server...")
    if not check_server():
        print("ERROR: TTS server not available!")
        sys.exit(1)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    best_per_emotion = {}

    total_variants = sum(len(e["variants"]) for e in EMOTION_LIBRARY.values())
    tested = 0

    for emotion, config in EMOTION_LIBRARY.items():
        print(f"\n{'='*60}")
        print(f"Testing emotion: {emotion.upper()}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")

        # Use emotion-specific sentences if available, else neutral
        sentences = EMOTION_TEST_CORPUS.get(emotion, NEUTRAL_TEST_CORPUS)

        emotion_results = []

        for idx, instruction in enumerate(config["variants"]):
            tested += 1
            print(f"\n  [{tested}/{total_variants}] Variant {idx}:")
            print(f"    Instruction: {instruction[:60]}...")

            result = test_instruction(instruction, sentences, emotion, idx)

            if result:
                emotion_results.append(result)
                all_results.append(result)

                print(f"    Spectral similarity: {result['spectral_similarity']:.3f}")
                print(f"    Pitch variance: {result['pitch_variance']:.2f} Hz")
                print(f"    Energy variance: {result['energy_variance']:.4f}")
            else:
                print(f"    FAILED - skipping")

        # Find best variant for this emotion
        if emotion_results:
            best = max(emotion_results, key=lambda x: x["spectral_similarity"])
            best_per_emotion[emotion] = best
            print(f"\n  BEST for {emotion}: Variant {best['variant_idx']} ({best['spectral_similarity']:.3f})")

    return all_results, best_per_emotion


def generate_report(all_results, best_per_emotion):
    """Generate markdown report."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# Instruction Library Test Results

**Date:** {timestamp}
**Model:** Fine-tuned Samantha (via TTS server on port 8032)
**Target:** Spectral similarity > 0.85
**Emotions Tested:** {len(EMOTION_LIBRARY)}
**Total Variants:** {len(all_results)}

## Summary: Best Instruction Per Emotion

| Emotion | Best Variant | Spectral Sim | Pitch Var | Energy Var |
|---------|--------------|--------------|-----------|------------|
"""

    for emotion in EMOTION_LIBRARY.keys():
        if emotion in best_per_emotion:
            best = best_per_emotion[emotion]
            report += f"| {emotion} | {best['variant_idx']} | {best['spectral_similarity']:.3f} | {best['pitch_variance']:.2f} Hz | {best['energy_variance']:.4f} |\n"

    # Overall best
    if best_per_emotion:
        overall_best = max(best_per_emotion.values(), key=lambda x: x["spectral_similarity"])
        report += f"\n**Overall Best:** {overall_best['emotion']} variant {overall_best['variant_idx']} at {overall_best['spectral_similarity']:.3f}\n"

    # Check if any hit target
    hits_target = [r for r in all_results if r["spectral_similarity"] >= 0.85]
    if hits_target:
        report += f"\n## HIT TARGET (>0.85)\n"
        for r in sorted(hits_target, key=lambda x: x["spectral_similarity"], reverse=True):
            report += f"- {r['emotion']} variant {r['variant_idx']}: {r['spectral_similarity']:.3f}\n"
    else:
        if all_results:
            closest = max(all_results, key=lambda x: x["spectral_similarity"])
            gap = 0.85 - closest["spectral_similarity"]
            report += f"\n## No variants hit 0.85 target\n"
            report += f"Closest: {closest['emotion']} variant {closest['variant_idx']} at {closest['spectral_similarity']:.3f} (gap: {gap:.3f})\n"

    # Detailed results per emotion
    report += "\n## Detailed Results by Emotion\n"

    for emotion in EMOTION_LIBRARY.keys():
        emotion_results = [r for r in all_results if r["emotion"] == emotion]
        if not emotion_results:
            continue

        report += f"\n### {emotion.title()}\n\n"
        report += "| Variant | Instruction (truncated) | Spectral Sim | Pitch Var |\n"
        report += "|---------|------------------------|--------------|------------|\n"

        for r in sorted(emotion_results, key=lambda x: x["spectral_similarity"], reverse=True):
            instr_short = r["instruction"][:40] + "..."
            report += f"| {r['variant_idx']} | {instr_short} | {r['spectral_similarity']:.3f} | {r['pitch_variance']:.2f} Hz |\n"

    # Best instructions to use
    report += "\n## Recommended Instruction Library\n\n"
    report += "```python\nBEST_INSTRUCTIONS = {\n"
    for emotion in EMOTION_LIBRARY.keys():
        if emotion in best_per_emotion:
            best = best_per_emotion[emotion]
            # Escape quotes in instruction
            instr = best["instruction"].replace('"', '\\"')
            report += f'    "{emotion}": "{instr}",\n'
    report += "}\n```\n"

    # Instruction pattern analysis
    report += "\n## Instruction Pattern Analysis\n\n"
    report += "Analyzing which instruction styles perform best:\n\n"

    # Group by variant index (instruction style)
    style_names = ["basic", "physical_descriptors", "negative_constraints", "reference_style", "minimal"]
    style_scores = {name: [] for name in style_names}

    for r in all_results:
        idx = r["variant_idx"]
        if idx < len(style_names):
            style_scores[style_names[idx]].append(r["spectral_similarity"])

    report += "| Style | Avg Spectral Sim | Min | Max | Count |\n"
    report += "|-------|------------------|-----|-----|-------|\n"
    for style, scores in style_scores.items():
        if scores:
            avg = np.mean(scores)
            min_s = np.min(scores)
            max_s = np.max(scores)
            report += f"| {style} | {avg:.3f} | {min_s:.3f} | {max_s:.3f} | {len(scores)} |\n"

    # Find best style
    style_avgs = {s: np.mean(scores) for s, scores in style_scores.items() if scores}
    if style_avgs:
        best_style = max(style_avgs, key=style_avgs.get)
        report += f"\n**Best instruction style overall:** {best_style} (avg: {style_avgs[best_style]:.3f})\n"

    # Key insights
    report += "\n## Key Insights\n\n"

    if style_avgs:
        sorted_styles = sorted(style_avgs.items(), key=lambda x: x[1], reverse=True)
        report += "### Instruction Style Ranking (by avg spectral similarity)\n"
        for i, (style, avg) in enumerate(sorted_styles, 1):
            report += f"{i}. **{style}**: {avg:.3f}\n"

    # Variance analysis
    if all_results:
        low_pitch_var = [r for r in all_results if r["pitch_variance"] < 15]
        report += f"\n### Low Pitch Variance (<15 Hz): {len(low_pitch_var)} variants\n"
        if low_pitch_var:
            best_low_pitch = max(low_pitch_var, key=lambda x: x["spectral_similarity"])
            report += f"Best with low pitch variance: {best_low_pitch['emotion']} v{best_low_pitch['variant_idx']} ({best_low_pitch['spectral_similarity']:.3f})\n"

    return report


def main():
    print("=" * 60)
    print("INSTRUCTION LIBRARY TEST")
    print("Testing all emotion variants to find best instructions")
    print("=" * 60)

    start_time = time.time()

    all_results, best_per_emotion = run_full_test()

    elapsed = time.time() - start_time

    # Save raw results as JSON
    results_file = OUTPUT_DIR / "all_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate and save report
    report = generate_report(all_results, best_per_emotion)
    report_path = Path("/home/chris/spark-voice-pipeline/instruction_library_results.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Results: {results_file}")
    print(f"Report: {report_path}")
    print(f"Audio files: {OUTPUT_DIR}/*.wav")

    # Print summary
    if best_per_emotion:
        overall_best = max(best_per_emotion.values(), key=lambda x: x["spectral_similarity"])
        print(f"\nOverall best: {overall_best['emotion']} variant {overall_best['variant_idx']}")
        print(f"Spectral similarity: {overall_best['spectral_similarity']:.3f}")
        print(f"Target: 0.85 | Gap: {0.85 - overall_best['spectral_similarity']:.3f}")

        # Check if any hit target
        hits_target = [r for r in all_results if r["spectral_similarity"] >= 0.85]
        if hits_target:
            print(f"\n{len(hits_target)} variants hit the 0.85 target!")
        else:
            print(f"\nNo variants hit the 0.85 target yet.")


if __name__ == "__main__":
    main()
