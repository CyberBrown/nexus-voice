"""
Main TTS quality tuning harness.

Implements automated parameter search to optimize voice consistency.
"""

import itertools
import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from .test_corpus import TEST_CORPUS
from .tts_quality_metrics import (
    MetricsResult,
    compute_metrics,
    compare_to_baseline,
    DEFAULT_THRESHOLDS,
)
from .qwen_tts_client import QwenTTSClient, TTSConfig
from .results_logger import (
    log_iteration,
    log_best_config,
    create_tuning_summary_note,
    log_progress,
)


# Parameter search space
TUNING_PARAMETERS = {
    "temperature": [0.3, 0.5, 0.7, 0.9],
    "top_p": [0.8, 0.9, 0.95, 1.0],
    "repetition_penalty": [1.0, 1.1, 1.2],
    "instruction_prompt": [
        "",  # no instruction
        "Speak in a calm, steady pace with professional tone.",
        "Warm and conversational, maintain consistent energy throughout.",
        "Measured and clear, neutral emotion, even pacing.",
    ],
    "sentence_batching": ["single", "pairs", "all"],
}


@dataclass
class TuningState:
    """State tracking for the tuning process."""
    iteration: int = 0
    best_config: dict = field(default_factory=dict)
    best_metrics: Optional[MetricsResult] = None
    best_score: float = float("inf")
    baseline_metrics: Optional[MetricsResult] = None
    plateau_count: int = 0
    history: list = field(default_factory=list)
    significant_findings: list = field(default_factory=list)


def compute_composite_score(metrics: MetricsResult, weights: dict = None) -> float:
    """
    Compute a single score from multiple metrics for optimization.

    Lower score is better.

    Args:
        metrics: Computed metrics
        weights: Optional custom weights for each metric

    Returns:
        float: Composite score (lower is better)
    """
    if weights is None:
        weights = {
            "pitch_variance": 0.25,
            "energy_variance": 0.20,
            "speaker_embedding_drift": 0.25,
            "spectral_similarity": 0.15,  # Inverted (higher is better)
            "speaking_rate_variance": 0.15,
        }

    score = 0.0

    # For variance metrics, lower is better
    score += weights["pitch_variance"] * (metrics.pitch_variance / 50.0)  # Normalize to ~0-1
    score += weights["energy_variance"] * (metrics.energy_variance / 0.2)
    score += weights["speaker_embedding_drift"] * (metrics.speaker_embedding_drift / 0.3)
    score += weights["speaking_rate_variance"] * (metrics.speaking_rate_variance / 2.0)

    # For similarity, higher is better, so invert
    score += weights["spectral_similarity"] * (1.0 - metrics.spectral_similarity)

    return score


def generate_test_audio(
    client: QwenTTSClient,
    config: TTSConfig,
    corpus: list[str],
    batching: str = "single",
) -> list[np.ndarray]:
    """
    Generate audio for the test corpus.

    Args:
        client: TTS client
        config: TTS configuration
        corpus: List of test sentences
        batching: "single" (one sentence at a time), "pairs", or "all"

    Returns:
        List of audio arrays
    """
    audio_segments = []

    if batching == "single":
        for i, text in enumerate(corpus):
            log_progress(f"Generating sentence {i+1}/{len(corpus)}")
            audio, sr = client.generate(text, config)
            audio_segments.append(audio)

    elif batching == "pairs":
        # Generate pairs of sentences together
        for i in range(0, len(corpus), 2):
            pair = corpus[i:i+2]
            combined_text = " ".join(pair)
            log_progress(f"Generating pair {i//2 + 1}/{(len(corpus)+1)//2}")
            audio, sr = client.generate(combined_text, config)
            audio_segments.append(audio)

    elif batching == "all":
        # Generate all sentences as one block
        combined_text = " ".join(corpus)
        log_progress("Generating full corpus as single block")
        audio, sr = client.generate(combined_text, config)
        audio_segments.append(audio)

    return audio_segments


def run_single_evaluation(
    client: QwenTTSClient,
    config_dict: dict,
    corpus: list[str] = None,
    sample_rate: int = 24000,
) -> MetricsResult:
    """
    Run a single evaluation with the given configuration.

    Args:
        client: TTS client
        config_dict: Configuration parameters
        corpus: Test sentences (default: TEST_CORPUS)
        sample_rate: Expected sample rate

    Returns:
        MetricsResult
    """
    if corpus is None:
        corpus = TEST_CORPUS

    # Create TTS config from dict
    tts_config = TTSConfig(
        temperature=config_dict.get("temperature", 0.7),
        top_p=config_dict.get("top_p", 0.95),
        repetition_penalty=config_dict.get("repetition_penalty", 1.0),
        instruction=config_dict.get("instruction_prompt", ""),
    )

    batching = config_dict.get("sentence_batching", "single")

    # Generate audio
    audio_segments = generate_test_audio(client, tts_config, corpus, batching)

    # Compute metrics
    metrics = compute_metrics(audio_segments, sample_rate)

    return metrics


def grid_search_configs(params: dict, max_configs: int = None) -> list[dict]:
    """
    Generate all combinations of parameters for grid search.

    Args:
        params: Parameter search space
        max_configs: Maximum number of configs to generate (random sample if exceeded)

    Returns:
        List of configuration dictionaries
    """
    keys = list(params.keys())
    values = [params[k] for k in keys]

    all_configs = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        all_configs.append(config)

    if max_configs and len(all_configs) > max_configs:
        random.shuffle(all_configs)
        all_configs = all_configs[:max_configs]

    return all_configs


def bayesian_next_config(
    state: TuningState,
    params: dict,
) -> dict:
    """
    Select next configuration using simple Bayesian-like approach.

    Narrows search around best parameters when improvements are found.

    Args:
        state: Current tuning state
        params: Parameter search space

    Returns:
        Next configuration to try
    """
    if state.best_config is None or len(state.history) < 5:
        # Initial exploration: random config
        config = {}
        for key, values in params.items():
            config[key] = random.choice(values)
        return config

    # Exploitation: perturb best config slightly
    config = state.best_config.copy()

    # Pick 1-2 parameters to modify
    num_changes = random.randint(1, 2)
    keys_to_change = random.sample(list(params.keys()), min(num_changes, len(params)))

    for key in keys_to_change:
        values = params[key]
        current_val = config.get(key)

        if current_val in values:
            current_idx = values.index(current_val)
            # Try adjacent values
            adjacent = []
            if current_idx > 0:
                adjacent.append(values[current_idx - 1])
            if current_idx < len(values) - 1:
                adjacent.append(values[current_idx + 1])
            if adjacent:
                config[key] = random.choice(adjacent)
        else:
            config[key] = random.choice(values)

    return config


def run_tuning_loop(
    max_iterations: int = 50,
    plateau_threshold: int = 10,
    search_strategy: str = "bayesian",
    client: QwenTTSClient = None,
    params: dict = None,
    corpus: list[str] = None,
    sample_rate: int = 24000,
    create_nexus_notes: bool = True,
) -> TuningState:
    """
    Main tuning loop for optimizing TTS parameters.

    Args:
        max_iterations: Maximum number of iterations
        plateau_threshold: Stop if no improvement for this many iterations
        search_strategy: "grid", "random", or "bayesian"
        client: TTS client (created if not provided)
        params: Parameter search space (default: TUNING_PARAMETERS)
        corpus: Test corpus (default: TEST_CORPUS)
        sample_rate: Expected sample rate
        create_nexus_notes: Whether to create Nexus notes for findings

    Returns:
        TuningState with best configuration found
    """
    if params is None:
        params = TUNING_PARAMETERS

    if corpus is None:
        corpus = TEST_CORPUS

    if client is None:
        client = QwenTTSClient()

    state = TuningState()

    log_progress("Starting TTS quality tuning")
    log_progress(f"Search strategy: {search_strategy}")
    log_progress(f"Max iterations: {max_iterations}")
    log_progress(f"Test corpus size: {len(corpus)} sentences")

    # Generate configs based on strategy
    if search_strategy == "grid":
        configs = grid_search_configs(params, max_configs=max_iterations)
        log_progress(f"Grid search: {len(configs)} configurations")
    elif search_strategy == "random":
        configs = grid_search_configs(params, max_configs=max_iterations)
        random.shuffle(configs)
    else:
        configs = None  # Will be generated per-iteration for bayesian

    # Run baseline first
    log_progress("Running baseline evaluation (default config)...")
    baseline_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "instruction_prompt": "",
        "sentence_batching": "single",
    }

    try:
        state.baseline_metrics = run_single_evaluation(
            client, baseline_config, corpus, sample_rate
        )
        state.best_metrics = state.baseline_metrics
        state.best_config = baseline_config
        state.best_score = compute_composite_score(state.baseline_metrics)

        log_progress(f"Baseline score: {state.best_score:.4f}")
        log_progress(f"Baseline pitch variance: {state.baseline_metrics.pitch_variance:.2f} Hz")
        log_progress(f"Baseline energy variance: {state.baseline_metrics.energy_variance:.4f}")

    except Exception as e:
        log_progress(f"Baseline evaluation failed: {e}", level="ERROR")
        raise

    # Main tuning loop
    for i in range(max_iterations):
        state.iteration = i + 1

        # Get next config
        if configs:
            if i >= len(configs):
                log_progress("Exhausted configuration space")
                break
            config = configs[i]
        else:
            config = bayesian_next_config(state, params)

        log_progress(f"\n=== Iteration {state.iteration}/{max_iterations} ===")
        log_progress(f"Config: temp={config.get('temperature')}, top_p={config.get('top_p')}")

        try:
            # Run evaluation
            metrics = run_single_evaluation(client, config, corpus, sample_rate)
            score = compute_composite_score(metrics)

            # Compare to baseline
            comparison = compare_to_baseline(metrics, state.baseline_metrics)

            # Log results
            log_iteration(
                iteration=state.iteration,
                config=config,
                metrics=metrics.to_dict(),
                baseline_comparison=comparison,
            )

            # Track history
            state.history.append({
                "iteration": state.iteration,
                "config": config,
                "score": score,
                "metrics": metrics.to_dict(),
            })

            # Check for improvement
            if score < state.best_score:
                improvement = (state.best_score - score) / state.best_score * 100
                log_progress(f"New best! Score: {score:.4f} ({improvement:.1f}% improvement)")

                state.best_score = score
                state.best_config = config
                state.best_metrics = metrics
                state.plateau_count = 0

                # Record significant finding
                finding = (
                    f"Iteration {state.iteration}: {improvement:.1f}% improvement with "
                    f"temp={config.get('temperature')}, instruction=\"{config.get('instruction_prompt', '')[:30]}...\""
                )
                state.significant_findings.append(finding)

            else:
                state.plateau_count += 1
                log_progress(f"No improvement. Score: {score:.4f} (plateau: {state.plateau_count})")

            # Check plateau condition
            if state.plateau_count >= plateau_threshold:
                log_progress(f"Plateau threshold reached ({plateau_threshold} iterations)")
                break

        except Exception as e:
            log_progress(f"Iteration failed: {e}", level="ERROR")
            continue

    # Log final results
    log_progress("\n" + "=" * 60)
    log_progress("TUNING COMPLETE")
    log_progress("=" * 60)
    log_progress(f"Total iterations: {state.iteration}")
    log_progress(f"Best score: {state.best_score:.4f}")
    log_progress(f"Best config: {json.dumps(state.best_config, indent=2)}")

    # Log best config to file
    log_best_config(
        config=state.best_config,
        metrics=state.best_metrics.to_dict() if state.best_metrics else {},
        total_iterations=state.iteration,
    )

    # Create Nexus note
    if create_nexus_notes and state.best_metrics and state.baseline_metrics:
        log_progress("Creating Nexus summary note...")
        create_tuning_summary_note(
            best_config=state.best_config,
            best_metrics=state.best_metrics.to_dict(),
            baseline_metrics=state.baseline_metrics.to_dict(),
            total_iterations=state.iteration,
            significant_findings=state.significant_findings,
        )

    return state


def quick_test(client: QwenTTSClient = None, num_sentences: int = 3) -> MetricsResult:
    """
    Run a quick test with minimal corpus for debugging.

    Args:
        client: TTS client
        num_sentences: Number of sentences to test

    Returns:
        MetricsResult
    """
    if client is None:
        client = QwenTTSClient()

    corpus = TEST_CORPUS[:num_sentences]

    log_progress(f"Running quick test with {num_sentences} sentences...")

    metrics = run_single_evaluation(
        client,
        config_dict={
            "temperature": 0.7,
            "top_p": 0.95,
            "repetition_penalty": 1.0,
            "instruction_prompt": "",
            "sentence_batching": "single",
        },
        corpus=corpus,
    )

    log_progress("Quick test results:")
    log_progress(f"  Pitch variance: {metrics.pitch_variance:.2f} Hz")
    log_progress(f"  Energy variance: {metrics.energy_variance:.4f}")
    log_progress(f"  Spectral similarity: {metrics.spectral_similarity:.4f}")
    log_progress(f"  Speaker drift: {metrics.speaker_embedding_drift:.4f}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TTS Quality Tuning")
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--strategy", choices=["grid", "random", "bayesian"], default="bayesian")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test only")
    parser.add_argument("--no-nexus", action="store_true", help="Skip Nexus note creation")

    args = parser.parse_args()

    if args.quick_test:
        quick_test()
    else:
        run_tuning_loop(
            max_iterations=args.max_iterations,
            search_strategy=args.strategy,
            create_nexus_notes=not args.no_nexus,
        )
