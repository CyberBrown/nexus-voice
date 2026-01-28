#!/usr/bin/env python3
"""
Main entry point for TTS Quality Tuning System.

Usage:
    # Quick test to verify setup
    python run_tuning.py --quick-test

    # Run full tuning with default settings
    python run_tuning.py

    # Run with specific strategy and iterations
    python run_tuning.py --strategy bayesian --max-iterations 100

    # Evaluate a specific configuration
    python run_tuning.py --evaluate --temp 0.5 --top-p 0.9 --instruction "calm"
"""

import argparse
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from tts_quality import (
    TEST_CORPUS,
    TUNING_PARAMETERS,
    compute_metrics,
    generate_audio,
    run_tuning_loop,
    log_iteration,
    create_nexus_note,
)
from tts_quality.qwen_tts_client import QwenTTSClient, TTSConfig
from tts_quality.tuning_harness import quick_test, run_single_evaluation, TuningState
from tts_quality.results_logger import log_progress


def verify_dependencies():
    """Check that all required dependencies are installed."""
    missing = []

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        import librosa
    except ImportError:
        missing.append("librosa")

    try:
        import parselmouth
    except ImportError:
        missing.append("parselmouth (praat-parselmouth)")

    try:
        import soundfile
    except ImportError:
        missing.append("soundfile")

    try:
        import requests
    except ImportError:
        missing.append("requests")

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except (ImportError, AttributeError, Exception) as e:
        # speechbrain has compatibility issues with newer torchaudio
        print(f"Warning: speechbrain unavailable ({e.__class__.__name__}), speaker embeddings will be skipped")

    if missing:
        print("Missing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with:")
        print("  pip install librosa praat-parselmouth soundfile requests speechbrain")
        return False

    return True


def test_tts_connection(server_url: str = "http://localhost:8032") -> bool:
    """Test connection to TTS server."""
    try:
        import requests
        response = requests.get(f"{server_url}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"TTS server not available: {e}")
        return False


def run_evaluation(args):
    """Run a single evaluation with specified parameters."""
    client = QwenTTSClient(
        server_url=args.server_url,
        voice_prompt_path=args.voice_prompt,
        lora_weights_path=args.lora_weights,
    )

    config = {
        "temperature": args.temp,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "instruction_prompt": args.instruction or "",
        "sentence_batching": args.batching,
    }

    log_progress(f"Running evaluation with config: {json.dumps(config, indent=2)}")

    metrics = run_single_evaluation(
        client=client,
        config_dict=config,
        corpus=TEST_CORPUS[:args.num_sentences] if args.num_sentences else TEST_CORPUS,
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Pitch Variance:        {metrics.pitch_variance:.2f} Hz")
    print(f"Pitch Contour Sim:     {metrics.pitch_contour_similarity:.4f}")
    print(f"Energy Variance:       {metrics.energy_variance:.4f}")
    print(f"Speaking Rate Var:     {metrics.speaking_rate_variance:.4f}")
    print(f"Spectral Similarity:   {metrics.spectral_similarity:.4f}")
    print(f"Speaker Drift:         {metrics.speaker_embedding_drift:.4f}")

    # Check against thresholds
    from tts_quality.tts_quality_metrics import DEFAULT_THRESHOLDS
    threshold_results = metrics.meets_thresholds(DEFAULT_THRESHOLDS)

    print("\n" + "-" * 60)
    print("THRESHOLD CHECK")
    print("-" * 60)
    for metric, passed in threshold_results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {metric}: {status}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="TTS Voice Quality Testing & Tuning System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test to verify setup
  python run_tuning.py --quick-test

  # Run full tuning
  python run_tuning.py --max-iterations 50 --strategy bayesian

  # Evaluate specific config
  python run_tuning.py --evaluate --temp 0.5 --instruction "calm and measured"

  # Grid search all combinations
  python run_tuning.py --strategy grid --max-iterations 100
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with 3 sentences to verify setup"
    )
    mode_group.add_argument(
        "--evaluate",
        action="store_true",
        help="Run single evaluation with specified parameters"
    )
    mode_group.add_argument(
        "--check-deps",
        action="store_true",
        help="Check if all dependencies are installed"
    )

    # Tuning parameters
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum tuning iterations (default: 50)"
    )
    parser.add_argument(
        "--strategy",
        choices=["grid", "random", "bayesian"],
        default="bayesian",
        help="Search strategy (default: bayesian)"
    )
    parser.add_argument(
        "--plateau-threshold",
        type=int,
        default=10,
        help="Stop after N iterations without improvement (default: 10)"
    )

    # TTS configuration
    parser.add_argument(
        "--server-url",
        default="http://localhost:8032",
        help="TTS server URL (default: http://localhost:8032)"
    )
    parser.add_argument(
        "--voice-prompt",
        default="/home/chris/ggml-org/voice samples/her audio samples.wav",
        help="Path to voice reference audio"
    )
    parser.add_argument(
        "--lora-weights",
        default=None,
        help="Path to LoRA fine-tuned weights"
    )

    # Single evaluation parameters
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p value")
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--instruction", type=str, default="", help="Style instruction")
    parser.add_argument(
        "--batching",
        choices=["single", "pairs", "all"],
        default="single",
        help="Sentence batching mode"
    )
    parser.add_argument(
        "--num-sentences",
        type=int,
        default=None,
        help="Number of test sentences (default: all)"
    )

    # Output options
    parser.add_argument(
        "--no-nexus",
        action="store_true",
        help="Skip creating Nexus notes"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)"
    )

    args = parser.parse_args()

    # Check dependencies
    if args.check_deps:
        if verify_dependencies():
            print("All dependencies installed!")
            # Also check TTS server
            if test_tts_connection(args.server_url):
                print(f"TTS server available at {args.server_url}")
            else:
                print(f"Warning: TTS server not available at {args.server_url}")
        sys.exit(0)

    # Verify dependencies before proceeding
    if not verify_dependencies():
        print("\nPlease install missing dependencies first.")
        sys.exit(1)

    # Run selected mode
    if args.quick_test:
        log_progress("Running quick test...")
        client = QwenTTSClient(
            server_url=args.server_url,
            voice_prompt_path=args.voice_prompt,
            lora_weights_path=args.lora_weights,
        )
        metrics = quick_test(client, num_sentences=3)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(metrics.to_dict(), f, indent=2)

    elif args.evaluate:
        metrics = run_evaluation(args)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(metrics.to_dict(), f, indent=2)

    else:
        # Full tuning loop
        log_progress("Starting full tuning loop...")

        client = QwenTTSClient(
            server_url=args.server_url,
            voice_prompt_path=args.voice_prompt,
            lora_weights_path=args.lora_weights,
        )

        state = run_tuning_loop(
            max_iterations=args.max_iterations,
            plateau_threshold=args.plateau_threshold,
            search_strategy=args.strategy,
            client=client,
            create_nexus_notes=not args.no_nexus,
        )

        if args.output:
            output_data = {
                "best_config": state.best_config,
                "best_metrics": state.best_metrics.to_dict() if state.best_metrics else None,
                "baseline_metrics": state.baseline_metrics.to_dict() if state.baseline_metrics else None,
                "total_iterations": state.iteration,
                "history": state.history,
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)

        print("\n" + "=" * 60)
        print("TUNING COMPLETE")
        print("=" * 60)
        print(f"Best configuration saved to: /home/chris/spark-voice-pipeline/tts_tuning_results.md")


if __name__ == "__main__":
    main()
