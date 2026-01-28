#!/usr/bin/env python3
"""
Base Model vs Fine-Tuned Comparison Test

Tests whether base Qwen3-TTS model with zero-shot voice cloning + emotion anchors
produces better consistency than fine-tuned model with instructions only.

The Question:
Is base model + zero-shot cloning + anchors better than fine-tuned + instructions?

Usage:
    python test_base_model_anchor.py [--output-dir DIR]
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except ImportError:
    print("ERROR: soundfile required. Install with: pip install soundfile")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("ERROR: torch required")
    sys.exit(1)

# Add the tts_quality module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_quality.tts_quality_metrics import compute_metrics, compare_to_baseline, DEFAULT_THRESHOLDS

# Configuration
OUTPUT_DIR = "/home/chris/spark-voice-pipeline"
VOICE_REFERENCE = "/home/chris/ggml-org/voice samples/her audio samples.wav"
BASE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

# Test corpus
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

# Reference text for voice cloning (approximate transcript of reference audio)
VOICE_REF_TEXT = "Hello, I'm your assistant. How can I help you today?"

# Fine-tuned baseline from previous test
FINETUNE_SPECTRAL_SIMILARITY = 0.788


def load_base_model():
    """Load the base Qwen3-TTS model (NO fine-tuning)."""
    print(f"Loading base model: {BASE_MODEL_ID}")
    print("This may take a moment...")

    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError:
        print("ERROR: qwen_tts not found. Make sure you're in the voice-pipeline venv.")
        sys.exit(1)

    t0 = time.time()

    model = Qwen3TTSModel.from_pretrained(
        BASE_MODEL_ID,
        device_map="cuda:0",
        dtype=torch.bfloat16,  # Use dtype instead of torch_dtype
    )

    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # Try to get model type info
    model_type = "base"  # Assume base since we're loading the base model
    if hasattr(model, 'config'):
        model_type = getattr(model.config, 'tts_model_type', 'base')
    elif hasattr(model, 'model_type'):
        model_type = model.model_type

    print(f"Model type: {model_type}")

    # Check if voice cloning is supported
    has_clone = hasattr(model, 'generate_voice_clone')
    has_create_prompt = hasattr(model, 'create_voice_clone_prompt')
    print(f"Voice cloning support: generate_voice_clone={has_clone}, create_voice_clone_prompt={has_create_prompt}")

    if not has_clone:
        print("WARNING: Model doesn't have generate_voice_clone method!")
        print("Voice cloning may not work.")

    return model


def create_voice_prompt(model, ref_audio_path: str, ref_text: str = None):
    """Create a reusable voice clone prompt from reference audio."""
    print(f"\nCreating voice clone prompt from: {ref_audio_path}")

    if not os.path.exists(ref_audio_path):
        raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

    # Check if model has the method
    if not hasattr(model, 'create_voice_clone_prompt'):
        print("WARNING: Model doesn't have create_voice_clone_prompt method")
        print("Attempting direct generation with ref_audio parameter...")
        return None

    try:
        voice_prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio_path,
            ref_text=ref_text or "",
        )
        print("Voice clone prompt created successfully")
        return voice_prompt
    except Exception as e:
        print(f"ERROR creating voice prompt: {e}")
        return None


def generate_with_clone(
    model,
    text: str,
    voice_prompt=None,
    ref_audio_path: str = None,
    ref_text: str = None,
    temperature: float = 0.7,
    x_vector_only: bool = False,
) -> tuple[np.ndarray, int]:
    """
    Generate audio using voice cloning.

    Note: Qwen3-TTS voice cloning does NOT support instruction-based control.
    The voice style comes entirely from the reference audio.
    """

    # Method 1: Use pre-computed voice prompt (faster for repeated use)
    if voice_prompt is not None and hasattr(model, 'generate_voice_clone'):
        try:
            audio_list, sr = model.generate_voice_clone(
                text=text,
                voice_clone_prompt=voice_prompt,
                temperature=temperature,
            )
            # Handle list output
            if isinstance(audio_list, list):
                audio = np.concatenate(audio_list) if len(audio_list) > 0 else np.array([])
            else:
                audio = audio_list
            return audio.astype(np.float32), sr
        except Exception as e:
            print(f"  generate_voice_clone with prompt failed: {e}")

    # Method 2: Direct generation with ref_audio (for one-off use)
    if ref_audio_path and hasattr(model, 'generate_voice_clone'):
        try:
            audio_list, sr = model.generate_voice_clone(
                text=text,
                ref_audio=ref_audio_path,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only,
                temperature=temperature,
            )
            if isinstance(audio_list, list):
                audio = np.concatenate(audio_list) if len(audio_list) > 0 else np.array([])
            else:
                audio = audio_list
            return np.array(audio).astype(np.float32), sr
        except Exception as e:
            print(f"  generate_voice_clone with ref_audio failed: {e}")

    raise RuntimeError("No working generation method found")


def run_experiment(args):
    """Run the base model vs fine-tuned comparison experiment."""
    print("=" * 70)
    print("Base Model vs Fine-Tuned Comparison")
    print("=" * 70)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Voice reference: {VOICE_REFERENCE}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    # =========================================================================
    # Phase 1: Load Base Model
    # =========================================================================
    print("-" * 70)
    print("Phase 1: Loading Base Model")
    print("-" * 70)

    model = load_base_model()

    # =========================================================================
    # Phase 2: Create Voice Clone Prompt
    # =========================================================================
    print("-" * 70)
    print("Phase 2: Creating Voice Clone Prompt")
    print("-" * 70)

    voice_prompt = create_voice_prompt(model, VOICE_REFERENCE, VOICE_REF_TEXT)

    # Test voice cloning
    print("\nTesting voice cloning (ICL mode - uses ref_text)...")
    try:
        test_audio, sr = generate_with_clone(
            model,
            text="This is a test of the voice cloning system.",
            voice_prompt=voice_prompt,
            ref_audio_path=VOICE_REFERENCE,
            ref_text=VOICE_REF_TEXT,
            x_vector_only=False,  # ICL mode for better prosody
        )
        test_path = os.path.join(args.output_dir, "clone_test.wav")
        sf.write(test_path, test_audio, sr)
        print(f"  Voice clone test: {len(test_audio)/sr:.2f}s")
        print(f"  Saved: {test_path}")
        print("  Listen to verify it sounds like 'her'")
    except Exception as e:
        print(f"  ERROR: Voice cloning test failed: {e}")
        import traceback
        traceback.print_exc()
        print("\n  The base model may not support voice cloning in this configuration.")
        print("  Checking model capabilities...")

        # Debug: list available methods
        clone_methods = [m for m in dir(model) if 'clone' in m.lower() or 'voice' in m.lower()]
        print(f"  Available clone/voice methods: {clone_methods}")

        generate_methods = [m for m in dir(model) if 'generate' in m.lower() or 'synth' in m.lower()]
        print(f"  Available generate methods: {generate_methods}")

        return None

    print()

    # =========================================================================
    # Phase 3: Generate Emotion Anchor
    # =========================================================================
    print("-" * 70)
    print("Phase 3: Generating Emotion Anchor")
    print("-" * 70)
    print(f"  Text: '{ANCHOR_TEXT}'")
    print("  Note: Voice cloning doesn't support instruction-based style control.")
    print("  The voice style comes from the reference audio.")

    try:
        anchor_audio, sr = generate_with_clone(
            model,
            text=ANCHOR_TEXT,
            voice_prompt=voice_prompt,
            ref_audio_path=VOICE_REFERENCE,
            ref_text=VOICE_REF_TEXT,
            temperature=0.5,  # Lower for consistency
            x_vector_only=False,  # ICL mode
        )
        anchor_path = os.path.join(args.output_dir, "base_neutral_anchor.wav")
        sf.write(anchor_path, anchor_audio, sr)
        print(f"  Generated anchor: {len(anchor_audio)/sr:.2f}s")
        print(f"  Saved: {anchor_path}")
    except Exception as e:
        print(f"  ERROR generating anchor: {e}")
        import traceback
        traceback.print_exc()
        return None

    print()

    # =========================================================================
    # Phase 4: Generate Test Corpus
    # =========================================================================
    print("-" * 70)
    print("Phase 4: Generating Test Corpus")
    print("-" * 70)
    print("\nNote: Qwen3-TTS voice cloning does NOT support instruction-based control.")
    print("Testing two modes:")
    print("  - ICL mode: Uses ref_text for In-Context Learning (better prosody)")
    print("  - X-vector mode: Speaker embedding only (faster)")

    # Method A: ICL mode (In-Context Learning - uses ref_text)
    print("\n[A] Voice Clone - ICL Mode (ref_text for prosody)")
    icl_segments = []
    for i, sentence in enumerate(TEST_SENTENCES):
        print(f"  [{i+1}/{len(TEST_SENTENCES)}] {sentence[:50]}...")
        try:
            audio, sr = generate_with_clone(
                model,
                text=sentence,
                voice_prompt=voice_prompt,
                ref_audio_path=VOICE_REFERENCE,
                ref_text=VOICE_REF_TEXT,
                temperature=0.7,
                x_vector_only=False,  # ICL mode
            )
            icl_segments.append(audio)
            print(f"         Generated: {len(audio)/sr:.2f}s")
        except Exception as e:
            print(f"         ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None

    icl_path = os.path.join(args.output_dir, "base_icl_output.wav")
    sf.write(icl_path, np.concatenate(icl_segments), sr)
    print(f"  Saved: {icl_path}")

    # Method B: X-vector only mode (speaker embedding only)
    print("\n[B] Voice Clone - X-Vector Mode (speaker embedding only)")
    xvector_segments = []
    for i, sentence in enumerate(TEST_SENTENCES):
        print(f"  [{i+1}/{len(TEST_SENTENCES)}] {sentence[:50]}...")
        try:
            audio, sr = generate_with_clone(
                model,
                text=sentence,
                voice_prompt=voice_prompt,
                ref_audio_path=VOICE_REFERENCE,
                ref_text=None,
                temperature=0.7,
                x_vector_only=True,  # X-vector only mode
            )
            xvector_segments.append(audio)
            print(f"         Generated: {len(audio)/sr:.2f}s")
        except Exception as e:
            print(f"         ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None

    xvector_path = os.path.join(args.output_dir, "base_xvector_output.wav")
    sf.write(xvector_path, np.concatenate(xvector_segments), sr)
    print(f"  Saved: {xvector_path}")

    # Update variable names for metrics calculation
    instruction_segments = icl_segments  # ICL mode is the primary comparison
    anchored_segments = xvector_segments  # X-vector mode is secondary

    print()

    # =========================================================================
    # Phase 5: Compute Quality Metrics
    # =========================================================================
    print("-" * 70)
    print("Phase 5: Computing Quality Metrics")
    print("-" * 70)

    print("\nComputing metrics for ICL MODE output...")
    try:
        icl_metrics = compute_metrics(icl_segments, sr)
        print("  Done.")
    except Exception as e:
        print(f"  ERROR: {e}")
        icl_metrics = None

    print("Computing metrics for X-VECTOR MODE output...")
    try:
        xvector_metrics = compute_metrics(xvector_segments, sr)
        print("  Done.")
    except Exception as e:
        print(f"  ERROR: {e}")
        xvector_metrics = None

    # Alias for report generation
    instruction_metrics = icl_metrics
    anchored_metrics = xvector_metrics

    print()

    # =========================================================================
    # Phase 6: Display Results
    # =========================================================================
    print("-" * 70)
    print("Phase 6: Results")
    print("-" * 70)

    if icl_metrics and xvector_metrics:
        print("\n=== BASE MODEL - ICL MODE (In-Context Learning) ===")
        print(f"  Pitch variance:          {icl_metrics.pitch_variance:.2f} Hz (target: <20)")
        print(f"  Pitch contour sim:       {icl_metrics.pitch_contour_similarity:.3f}")
        print(f"  Energy variance:         {icl_metrics.energy_variance:.4f} (target: <0.1)")
        print(f"  Speaking rate variance:  {icl_metrics.speaking_rate_variance:.3f}")
        print(f"  Spectral similarity:     {icl_metrics.spectral_similarity:.3f} (target: >0.85)")
        print(f"  Speaker embed drift:     {icl_metrics.speaker_embedding_drift:.4f} (target: <0.15)")

        print("\n=== BASE MODEL - X-VECTOR MODE (Speaker Embedding Only) ===")
        print(f"  Pitch variance:          {xvector_metrics.pitch_variance:.2f} Hz")
        print(f"  Pitch contour sim:       {xvector_metrics.pitch_contour_similarity:.3f}")
        print(f"  Energy variance:         {xvector_metrics.energy_variance:.4f}")
        print(f"  Speaking rate variance:  {xvector_metrics.speaking_rate_variance:.3f}")
        print(f"  Spectral similarity:     {xvector_metrics.spectral_similarity:.3f}")
        print(f"  Speaker embed drift:     {xvector_metrics.speaker_embedding_drift:.4f}")

        print("\n=== COMPARISON VS FINE-TUNED (from previous test) ===")
        print(f"  Fine-tuned + instruction:  {FINETUNE_SPECTRAL_SIMILARITY:.3f}")
        print(f"  Base ICL mode:             {icl_metrics.spectral_similarity:.3f} ({(icl_metrics.spectral_similarity - FINETUNE_SPECTRAL_SIMILARITY) / FINETUNE_SPECTRAL_SIMILARITY * 100:+.1f}%)")
        print(f"  Base X-vector mode:        {xvector_metrics.spectral_similarity:.3f} ({(xvector_metrics.spectral_similarity - FINETUNE_SPECTRAL_SIMILARITY) / FINETUNE_SPECTRAL_SIMILARITY * 100:+.1f}%)")

        # Determine recommendation
        best_base = max(icl_metrics.spectral_similarity, xvector_metrics.spectral_similarity)
        best_mode = "ICL" if icl_metrics.spectral_similarity >= xvector_metrics.spectral_similarity else "X-vector"

        print("\n=== RECOMMENDATION ===")
        print(f"  Best base model mode: {best_mode} ({best_base:.3f})")
        if best_base >= 0.85:
            recommendation = f"USE BASE MODEL ({best_mode} mode) - Hits >0.85 target with voice cloning"
            print(f"  {recommendation}")
        elif best_base > FINETUNE_SPECTRAL_SIMILARITY:
            recommendation = f"BASE MODEL BETTER ({best_mode} mode) - But doesn't hit 0.85 target"
            print(f"  {recommendation}")
        elif best_base >= FINETUNE_SPECTRAL_SIMILARITY - 0.02:
            recommendation = "CLOSE CALL - Listen to outputs to decide"
            print(f"  {recommendation}")
        else:
            recommendation = "USE FINE-TUNED - Better consistency than voice cloning"
            print(f"  {recommendation}")

        # Generate report
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = f"""# Base Model vs Fine-Tuned Comparison

**Date:** {timestamp}

## The Question
Is base model + zero-shot voice cloning better than fine-tuned + instructions?

## Key Finding
**Voice cloning in Qwen3-TTS does NOT support instruction-based style control.**
The voice/prosody comes entirely from the reference audio.

## Test Configuration
- **Base Model:** {BASE_MODEL_ID}
- **Voice Reference:** {VOICE_REFERENCE}
- **Reference Text:** "{VOICE_REF_TEXT}"
- **Test Sentences:** {len(TEST_SENTENCES)}

## Voice Cloning Modes Tested

### ICL Mode (In-Context Learning)
Uses reference text to match prosody patterns. Better quality but requires accurate transcript.

### X-Vector Mode
Uses only speaker embedding. Faster but may lose prosodic nuances.

## Results

### Base Model - ICL Mode
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Pitch variance | {icl_metrics.pitch_variance:.2f} Hz | <20 Hz | {"PASS" if icl_metrics.pitch_variance < 20 else "FAIL"} |
| Energy variance | {icl_metrics.energy_variance:.4f} | <0.1 | {"PASS" if icl_metrics.energy_variance < 0.1 else "FAIL"} |
| Spectral similarity | {icl_metrics.spectral_similarity:.3f} | >0.85 | {"PASS" if icl_metrics.spectral_similarity > 0.85 else "FAIL"} |
| Pitch contour similarity | {icl_metrics.pitch_contour_similarity:.3f} | - | - |
| Speaking rate variance | {icl_metrics.speaking_rate_variance:.3f} | - | - |
| Speaker embedding drift | {icl_metrics.speaker_embedding_drift:.4f} | <0.15 | {"PASS" if icl_metrics.speaker_embedding_drift < 0.15 else "FAIL"} |

### Base Model - X-Vector Mode
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Pitch variance | {xvector_metrics.pitch_variance:.2f} Hz | <20 Hz | {"PASS" if xvector_metrics.pitch_variance < 20 else "FAIL"} |
| Energy variance | {xvector_metrics.energy_variance:.4f} | <0.1 | {"PASS" if xvector_metrics.energy_variance < 0.1 else "FAIL"} |
| Spectral similarity | {xvector_metrics.spectral_similarity:.3f} | >0.85 | {"PASS" if xvector_metrics.spectral_similarity > 0.85 else "FAIL"} |
| Pitch contour similarity | {xvector_metrics.pitch_contour_similarity:.3f} | - | - |
| Speaking rate variance | {xvector_metrics.speaking_rate_variance:.3f} | - | - |
| Speaker embedding drift | {xvector_metrics.speaker_embedding_drift:.4f} | <0.15 | {"PASS" if xvector_metrics.speaker_embedding_drift < 0.15 else "FAIL"} |

### Fine-Tuned + Instruction (Previous Test)
| Metric | Value |
|--------|-------|
| Spectral similarity | {FINETUNE_SPECTRAL_SIMILARITY:.3f} |

## Head-to-Head Comparison

| Approach | Spectral Sim | vs Fine-tuned | vs Target (0.85) |
|----------|--------------|---------------|------------------|
| Fine-tuned + instruction | {FINETUNE_SPECTRAL_SIMILARITY:.3f} | baseline | {(FINETUNE_SPECTRAL_SIMILARITY - 0.85) / 0.85 * 100:+.1f}% |
| Base ICL mode | {icl_metrics.spectral_similarity:.3f} | {(icl_metrics.spectral_similarity - FINETUNE_SPECTRAL_SIMILARITY) / FINETUNE_SPECTRAL_SIMILARITY * 100:+.1f}% | {(icl_metrics.spectral_similarity - 0.85) / 0.85 * 100:+.1f}% |
| Base X-vector mode | {xvector_metrics.spectral_similarity:.3f} | {(xvector_metrics.spectral_similarity - FINETUNE_SPECTRAL_SIMILARITY) / FINETUNE_SPECTRAL_SIMILARITY * 100:+.1f}% | {(xvector_metrics.spectral_similarity - 0.85) / 0.85 * 100:+.1f}% |

## Voice Quality Assessment

Listen to the output files and rate subjectively:
- [ ] `clone_test.wav` - Does it sound like "her"?
- [ ] `base_icl_output.wav` - ICL mode consistency?
- [ ] `base_xvector_output.wav` - X-vector mode consistency?
- [ ] Compare to fine-tuned outputs from previous test

## Recommendation

**{recommendation}**

### Rationale
- Best base model mode: {best_mode} ({best_base:.3f})
- Fine-tuned spectral similarity: {FINETUNE_SPECTRAL_SIMILARITY:.3f}
- Target: >0.85

## Architecture Trade-offs

| Approach | Voice Identity | Style Control | Consistency |
|----------|---------------|---------------|-------------|
| Fine-tuned + instruction | Excellent (trained) | Via instructions | Good |
| Base + ICL cloning | Good (from ref) | None (from ref prosody) | TBD |
| Base + X-vector | Moderate (embedding) | None | TBD |

## Next Steps
{"- Base model with cloning achieves target\n- Test with different reference audios for emotion variation\n- Build emotion library using different reference clips" if best_base >= 0.85 else "- Fine-tuned model is more consistent\n- Focus on instruction optimization\n- Consider using fine-tuned for production" if best_base < FINETUNE_SPECTRAL_SIMILARITY else "- Results are close - conduct subjective listening test\n- Test with more sentences\n- Consider hybrid approach"}

## Output Files
- `clone_test.wav` - Voice cloning verification
- `base_neutral_anchor.wav` - Anchor clip (for reference)
- `base_icl_output.wav` - Test corpus (ICL mode)
- `base_xvector_output.wav` - Test corpus (X-vector mode)

---
*Generated by nexus-voice base model test*
"""

        report_path = os.path.join(args.output_dir, "base_vs_finetune_results.md")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\nReport saved: {report_path}")

        return {
            "icl_metrics": icl_metrics.to_dict(),
            "xvector_metrics": xvector_metrics.to_dict(),
            "finetune_baseline": FINETUNE_SPECTRAL_SIMILARITY,
            "best_base_mode": best_mode,
            "best_base_spectral": best_base,
            "recommendation": recommendation,
            "report_path": report_path,
        }

    else:
        print("ERROR: Could not compute metrics")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Test base Qwen3-TTS model with voice cloning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )

    args = parser.parse_args()

    results = run_experiment(args)

    if results:
        print("\n" + "=" * 70)
        print("Experiment completed successfully!")
        print("=" * 70)
        return 0
    else:
        print("\nExperiment failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
