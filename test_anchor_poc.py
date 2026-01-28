#!/usr/bin/env python3
"""
Emotion Anchor Proof of Concept Test

Tests whether combining the fine-tuned voice model with a zero-shot emotion anchor
improves sentence-to-sentence consistency.

Hypothesis:
- Fine-tuned model provides voice identity (WHO)
- Zero-shot anchor provides prosodic template (HOW)
- Using both together should give more consistent output than either alone

Usage:
    python test_anchor_poc.py [--server-url URL] [--lora-weights PATH]
"""

import argparse
import os
import sys
import io
import base64
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except ImportError:
    print("ERROR: soundfile required. Install with: pip install soundfile")
    sys.exit(1)

try:
    import requests
except ImportError:
    requests = None

try:
    import torch
    import torchaudio
except ImportError:
    torch = None
    torchaudio = None

# Add the tts_quality module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_quality.tts_quality_metrics import compute_metrics, compare_to_baseline, DEFAULT_THRESHOLDS


# Configuration
DEFAULT_SERVER_URL = "http://localhost:8032"
DEFAULT_VOICE_PROMPT = "/home/chris/ggml-org/voice samples/her audio samples.wav"
DEFAULT_LORA_WEIGHTS = "/home/chris/spark-voice-pipeline/lora_weights_step_04000.safetensors"
OUTPUT_DIR = "/home/chris/spark-voice-pipeline"

# Test corpus - diverse sentence types
TEST_SENTENCES = [
    "The solar installation is scheduled for next Tuesday.",
    "I've reviewed the documents you sent over.",
    "Would you like me to check on that for you?",
    "That's not what I meant. Let me clarify.",
    "I understand this is frustrating. Let's work through it.",
]

# Anchor configuration
ANCHOR_TEXT = "Let me look into that for you."
ANCHOR_INSTRUCTION = "Calm, even pace, professional neutrality, steady tone"
ANCHORED_INSTRUCTION = "Match the calm, steady tone of the reference"
BASELINE_INSTRUCTION = "Calm, even pace, professional neutrality"


class AnchorTTSClient:
    """
    Extended TTS client that supports dynamic reference audio for anchoring.

    Uses two endpoints:
    - /synthesize: Basic synthesis with fine-tuned voice (samantha)
    - /synthesize-clone: Voice cloning with reference audio for prosodic anchoring
    """

    def __init__(
        self,
        server_url: str = DEFAULT_SERVER_URL,
        voice_prompt_path: str = DEFAULT_VOICE_PROMPT,
        lora_weights_path: str = None,
    ):
        self.server_url = server_url
        self.voice_prompt_path = voice_prompt_path
        self.lora_weights_path = lora_weights_path
        self._base_voice_audio = None

    def _check_server_health(self) -> bool:
        """Check if TTS server is running."""
        if requests is None:
            return False
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def _load_base_voice(self) -> tuple[np.ndarray, int]:
        """Load the base voice reference audio."""
        if self._base_voice_audio is not None:
            return self._base_voice_audio

        if not os.path.exists(self.voice_prompt_path):
            raise FileNotFoundError(f"Voice prompt not found: {self.voice_prompt_path}")

        if torchaudio is not None:
            waveform, sample_rate = torchaudio.load(self.voice_prompt_path)
            audio = waveform.numpy().squeeze()
        else:
            audio, sample_rate = sf.read(self.voice_prompt_path)

        self._base_voice_audio = (audio, sample_rate)
        return self._base_voice_audio

    def generate_with_preset_voice(
        self,
        text: str,
        instruction: str = "",
        speaker: str = "samantha",
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio using preset fine-tuned voice via /synthesize endpoint.

        Args:
            text: Text to synthesize
            instruction: Style instruction (instruct parameter)
            speaker: Speaker name (default: samantha)
        """
        if requests is None:
            raise ImportError("requests required for API generation")

        params = {
            "text": text,
            "speaker": speaker,
        }

        if instruction:
            params["instruct"] = instruction

        response = requests.post(
            f"{self.server_url}/synthesize",
            params=params,
            timeout=120,
        )

        if response.status_code != 200:
            raise RuntimeError(f"TTS generation failed: {response.status_code} - {response.text}")

        audio, sample_rate = sf.read(io.BytesIO(response.content))
        return audio.astype(np.float32), sample_rate

    def generate_with_anchor(
        self,
        text: str,
        ref_audio: np.ndarray,
        ref_audio_sr: int,
        ref_text: str = "",
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio using /synthesize-clone endpoint with reference audio.

        This uses the anchor audio as a prosodic template for voice cloning.

        Args:
            text: Text to synthesize
            ref_audio: Reference audio array for voice/prosody cloning
            ref_audio_sr: Sample rate of reference audio
            ref_text: Text spoken in the reference audio (helps model understand prosody)
        """
        if requests is None:
            raise ImportError("requests required for API generation")

        # Convert reference audio to WAV bytes
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, ref_audio, ref_audio_sr, format='WAV')
        wav_buffer.seek(0)

        # Use multipart form for /synthesize-clone
        files = {
            "voice_file": ("anchor.wav", wav_buffer, "audio/wav"),
        }

        data = {
            "text": text,
        }

        if ref_text:
            data["ref_text"] = ref_text

        response = requests.post(
            f"{self.server_url}/synthesize-clone",
            files=files,
            data=data,
            timeout=120,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Clone synthesis failed: {response.status_code} - {response.text}")

        audio, sample_rate = sf.read(io.BytesIO(response.content))
        return audio.astype(np.float32), sample_rate

    def generate(
        self,
        text: str,
        instruction: str = "",
        ref_audio: np.ndarray = None,
        ref_audio_sr: int = None,
        ref_text: str = "",
        temperature: float = 0.7,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio - uses clone endpoint if ref_audio provided, else preset voice.

        Args:
            text: Text to synthesize
            instruction: Style instruction (for preset voice only)
            ref_audio: Optional reference audio for prosodic anchoring
            ref_audio_sr: Sample rate of reference audio
            ref_text: Text spoken in reference (for clone mode)
            temperature: Not used (server doesn't expose this)
        """
        if not self._check_server_health():
            raise RuntimeError(
                f"TTS server not available at {self.server_url}. "
                "Start the server with the fine-tuned model loaded."
            )

        if ref_audio is not None and ref_audio_sr is not None:
            # Use voice cloning with anchor
            return self.generate_with_anchor(
                text=text,
                ref_audio=ref_audio,
                ref_audio_sr=ref_audio_sr,
                ref_text=ref_text,
            )
        else:
            # Use preset fine-tuned voice
            return self.generate_with_preset_voice(
                text=text,
                instruction=instruction,
            )


def run_experiment(args):
    """Run the emotion anchor proof of concept experiment."""
    print("=" * 60)
    print("Emotion Anchor Proof of Concept")
    print("=" * 60)
    print(f"\nServer URL: {args.server_url}")
    print(f"Voice prompt: {args.voice_prompt}")
    print(f"LoRA weights: {args.lora_weights or 'None (using server default)'}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Initialize client
    client = AnchorTTSClient(
        server_url=args.server_url,
        voice_prompt_path=args.voice_prompt,
        lora_weights_path=args.lora_weights,
    )

    # Check server health
    if not client._check_server_health():
        print("ERROR: TTS server not available!")
        print(f"Please start the server at {args.server_url}")
        print("The server should have the fine-tuned model loaded with LoRA weights.")
        sys.exit(1)

    print("Server health check: OK")
    print()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # =========================================================================
    # Phase 1: Generate the Emotion Anchor
    # =========================================================================
    print("-" * 60)
    print("Phase 1: Generating Emotion Anchor")
    print("-" * 60)
    print(f"  Text: '{ANCHOR_TEXT}'")
    print(f"  Instruction: '{ANCHOR_INSTRUCTION}'")
    print(f"  Temperature: 0.5")
    print()

    try:
        anchor_audio, anchor_sr = client.generate(
            text=ANCHOR_TEXT,
            instruction=ANCHOR_INSTRUCTION,
        )
        anchor_duration = len(anchor_audio) / anchor_sr
        print(f"  Generated anchor: {anchor_duration:.2f}s at {anchor_sr}Hz")

        # Save anchor
        anchor_path = os.path.join(args.output_dir, "neutral_anchor.wav")
        sf.write(anchor_path, anchor_audio, anchor_sr)
        print(f"  Saved: {anchor_path}")

    except Exception as e:
        print(f"ERROR generating anchor: {e}")
        sys.exit(1)

    print()

    # =========================================================================
    # Phase 2: Generate Test Corpus WITH Anchor
    # =========================================================================
    print("-" * 60)
    print("Phase 2: Generating Test Corpus with Anchor")
    print("-" * 60)
    print(f"  Instruction: '{ANCHORED_INSTRUCTION}'")
    print(f"  Temperature: 0.7")
    print()

    anchored_segments = []
    anchor_fallback_used = False
    for i, sentence in enumerate(TEST_SENTENCES):
        print(f"  [{i+1}/{len(TEST_SENTENCES)}] {sentence[:50]}...")
        try:
            audio, sr = client.generate(
                text=sentence,
                ref_audio=anchor_audio,
                ref_audio_sr=anchor_sr,
                ref_text=ANCHOR_TEXT,  # Tell model what was said in anchor
            )
            anchored_segments.append(audio)
            print(f"         Generated: {len(audio)/sr:.2f}s")
        except Exception as e:
            print(f"         ERROR: {e}")
            # Try without ref_audio (server may not support clone endpoint)
            print("         Retrying with instruction-only (no clone)...")
            audio, sr = client.generate(
                text=sentence,
                instruction=ANCHORED_INSTRUCTION,
            )
            anchored_segments.append(audio)
            anchor_fallback_used = True
            print(f"         Generated (instruction only): {len(audio)/sr:.2f}s")

    if anchor_fallback_used:
        print("\n  WARNING: Anchor cloning failed, fell back to instruction-only mode")

    # Save concatenated output
    anchored_output_path = os.path.join(args.output_dir, "anchored_output.wav")
    anchored_concat = np.concatenate(anchored_segments)
    sf.write(anchored_output_path, anchored_concat, sr)
    print(f"\n  Saved concatenated: {anchored_output_path}")
    print()

    # =========================================================================
    # Phase 3: Generate Test Corpus WITHOUT Anchor (Baseline)
    # =========================================================================
    print("-" * 60)
    print("Phase 3: Generating Baseline (No Anchor)")
    print("-" * 60)
    print(f"  Instruction: '{BASELINE_INSTRUCTION}'")
    print(f"  Temperature: 0.7")
    print()

    baseline_segments = []
    for i, sentence in enumerate(TEST_SENTENCES):
        print(f"  [{i+1}/{len(TEST_SENTENCES)}] {sentence[:50]}...")
        try:
            audio, sr = client.generate(
                text=sentence,
                instruction=BASELINE_INSTRUCTION,
            )
            baseline_segments.append(audio)
            print(f"         Generated: {len(audio)/sr:.2f}s")
        except Exception as e:
            print(f"         ERROR: {e}")
            sys.exit(1)

    # Save concatenated output
    baseline_output_path = os.path.join(args.output_dir, "baseline_output.wav")
    baseline_concat = np.concatenate(baseline_segments)
    sf.write(baseline_output_path, baseline_concat, sr)
    print(f"\n  Saved concatenated: {baseline_output_path}")
    print()

    # =========================================================================
    # Phase 4: Compute Quality Metrics
    # =========================================================================
    print("-" * 60)
    print("Phase 4: Computing Quality Metrics")
    print("-" * 60)
    print()

    # Compute metrics for anchored output
    print("Computing metrics for ANCHORED output...")
    try:
        anchored_metrics = compute_metrics(anchored_segments, sr)
        print("  Done.")
    except Exception as e:
        print(f"  ERROR computing anchored metrics: {e}")
        anchored_metrics = None

    # Compute metrics for baseline output
    print("Computing metrics for BASELINE output...")
    try:
        baseline_metrics = compute_metrics(baseline_segments, sr)
        print("  Done.")
    except Exception as e:
        print(f"  ERROR computing baseline metrics: {e}")
        baseline_metrics = None

    print()

    # =========================================================================
    # Phase 5: Display Results & Save Report
    # =========================================================================
    print("-" * 60)
    print("Phase 5: Results")
    print("-" * 60)
    print()

    if anchored_metrics and baseline_metrics:
        print("=== ANCHORED (fine-tuned + zero-shot anchor) ===")
        print(f"  Pitch variance:          {anchored_metrics.pitch_variance:.2f} Hz")
        print(f"  Pitch contour sim:       {anchored_metrics.pitch_contour_similarity:.3f}")
        print(f"  Energy variance:         {anchored_metrics.energy_variance:.4f}")
        print(f"  Speaking rate variance:  {anchored_metrics.speaking_rate_variance:.3f}")
        print(f"  Spectral similarity:     {anchored_metrics.spectral_similarity:.3f}")
        print(f"  Speaker embed drift:     {anchored_metrics.speaker_embedding_drift:.4f}")
        print()

        print("=== BASELINE (fine-tuned only) ===")
        print(f"  Pitch variance:          {baseline_metrics.pitch_variance:.2f} Hz")
        print(f"  Pitch contour sim:       {baseline_metrics.pitch_contour_similarity:.3f}")
        print(f"  Energy variance:         {baseline_metrics.energy_variance:.4f}")
        print(f"  Speaking rate variance:  {baseline_metrics.speaking_rate_variance:.3f}")
        print(f"  Spectral similarity:     {baseline_metrics.spectral_similarity:.3f}")
        print(f"  Speaker embed drift:     {baseline_metrics.speaker_embedding_drift:.4f}")
        print()

        # Compute improvement percentages
        comparison = compare_to_baseline(anchored_metrics, baseline_metrics)

        print("=== IMPROVEMENT (anchored vs baseline) ===")
        print(f"  Pitch variance:          {comparison['pitch_variance']:+.1f}% (lower is better)")
        print(f"  Pitch contour sim:       {comparison['pitch_contour_similarity']:+.1f}% (higher is better)")
        print(f"  Energy variance:         {comparison['energy_variance']:+.1f}% (lower is better)")
        print(f"  Speaking rate variance:  {comparison['speaking_rate_variance']:+.1f}% (lower is better)")
        print(f"  Spectral similarity:     {comparison['spectral_similarity']:+.1f}% (higher is better)")
        print(f"  Speaker embed drift:     {comparison['speaker_embedding_drift']:+.1f}% (lower is better)")
        print()

        # Determine success
        spec_improvement = comparison['spectral_similarity']
        pitch_improvement = -comparison['pitch_variance']  # Negative is better for variance
        energy_improvement = -comparison['energy_variance']
        drift_improvement = -comparison['speaker_embedding_drift']

        # Check thresholds
        meets_spec_threshold = anchored_metrics.spectral_similarity >= DEFAULT_THRESHOLDS["spectral_similarity"]
        spec_better = spec_improvement > 5  # 5% improvement threshold

        if meets_spec_threshold or spec_better:
            conclusion = "SUCCESS - Anchoring improves consistency"
            next_steps = "Build full emotion library (10 emotions)"
        elif spec_improvement > 0:
            conclusion = "MARGINAL - Slight improvement, needs investigation"
            next_steps = "Try different anchor texts/instructions"
        else:
            conclusion = "NO IMPROVEMENT - Approach may not work as expected"
            next_steps = "Investigate alternative approaches"

        print("=== CONCLUSION ===")
        print(f"  {conclusion}")
        print(f"  Next steps: {next_steps}")
        print()

        # Generate report
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = f"""# Emotion Anchor Proof of Concept Results

**Date:** {timestamp}

## Test Configuration

- **Server URL:** {args.server_url}
- **Voice Prompt:** {args.voice_prompt}
- **LoRA Weights:** {args.lora_weights or 'Server default'}
- **Anchor Text:** "{ANCHOR_TEXT}"
- **Anchor Instruction:** "{ANCHOR_INSTRUCTION}"
- **Test Sentences:** {len(TEST_SENTENCES)}

## Test: Fine-tuned + Zero-shot Anchor vs Fine-tuned Only

### Anchored Output (with prosodic anchor)
| Metric | Value | Threshold |
|--------|-------|-----------|
| Pitch variance | {anchored_metrics.pitch_variance:.2f} Hz | < 20 Hz |
| Pitch contour similarity | {anchored_metrics.pitch_contour_similarity:.3f} | - |
| Energy variance | {anchored_metrics.energy_variance:.4f} | < 0.1 |
| Speaking rate variance | {anchored_metrics.speaking_rate_variance:.3f} | - |
| Spectral similarity | {anchored_metrics.spectral_similarity:.3f} | > 0.85 |
| Speaker embedding drift | {anchored_metrics.speaker_embedding_drift:.4f} | < 0.15 |

### Baseline (no anchor)
| Metric | Value | Threshold |
|--------|-------|-----------|
| Pitch variance | {baseline_metrics.pitch_variance:.2f} Hz | < 20 Hz |
| Pitch contour similarity | {baseline_metrics.pitch_contour_similarity:.3f} | - |
| Energy variance | {baseline_metrics.energy_variance:.4f} | < 0.1 |
| Speaking rate variance | {baseline_metrics.speaking_rate_variance:.3f} | - |
| Spectral similarity | {baseline_metrics.spectral_similarity:.3f} | > 0.85 |
| Speaker embedding drift | {baseline_metrics.speaker_embedding_drift:.4f} | < 0.15 |

### Improvement (anchored vs baseline)
| Metric | Change | Direction |
|--------|--------|-----------|
| Pitch variance | {comparison['pitch_variance']:+.1f}% | lower is better |
| Pitch contour similarity | {comparison['pitch_contour_similarity']:+.1f}% | higher is better |
| Energy variance | {comparison['energy_variance']:+.1f}% | lower is better |
| Speaking rate variance | {comparison['speaking_rate_variance']:+.1f}% | lower is better |
| Spectral similarity | {comparison['spectral_similarity']:+.1f}% | higher is better |
| Speaker embedding drift | {comparison['speaker_embedding_drift']:+.1f}% | lower is better |

## Conclusion

**{conclusion}**

### Threshold Check
- Spectral similarity meets threshold (>0.85): {'Yes' if meets_spec_threshold else 'No'} ({anchored_metrics.spectral_similarity:.3f})
- Spectral similarity improvement (>5%): {'Yes' if spec_better else 'No'} ({spec_improvement:+.1f}%)

### Next Steps
{next_steps}

## Output Files
- `neutral_anchor.wav` - The emotion anchor clip
- `anchored_output.wav` - Test corpus with anchoring
- `baseline_output.wav` - Test corpus without anchoring
"""

        # Save report
        report_path = os.path.join(args.output_dir, "anchor_poc_results.md")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Report saved: {report_path}")

        return {
            "anchored": anchored_metrics.to_dict(),
            "baseline": baseline_metrics.to_dict(),
            "comparison": comparison,
            "conclusion": conclusion,
            "report_path": report_path,
        }

    else:
        print("ERROR: Could not compute metrics for comparison")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Test emotion anchor approach for TTS consistency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help=f"TTS server URL (default: {DEFAULT_SERVER_URL})",
    )
    parser.add_argument(
        "--voice-prompt",
        default=DEFAULT_VOICE_PROMPT,
        help=f"Path to voice reference audio (default: {DEFAULT_VOICE_PROMPT})",
    )
    parser.add_argument(
        "--lora-weights",
        default=DEFAULT_LORA_WEIGHTS,
        help=f"Path to LoRA weights (default: {DEFAULT_LORA_WEIGHTS})",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )

    args = parser.parse_args()

    results = run_experiment(args)

    if results:
        print("\nExperiment completed successfully!")
        return 0
    else:
        print("\nExperiment failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
