"""
Results logging to local markdown file and Nexus notes.
"""

import json
import os
from datetime import datetime
from typing import Optional

try:
    import requests
except ImportError:
    requests = None


RESULTS_FILE = "/home/chris/spark-voice-pipeline/tts_tuning_results.md"
NEXUS_MCP_URL = "https://nexus-mcp.solamp.workers.dev/mcp"
NEXUS_PASSPHRASE = "stale-coffee-44"


def ensure_results_dir():
    """Ensure the results directory exists."""
    results_dir = os.path.dirname(RESULTS_FILE)
    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)


def log_iteration(
    iteration: int,
    config: dict,
    metrics: dict,
    baseline_comparison: dict,
    notes: str = "",
) -> str:
    """
    Append results to markdown file with timestamp.

    Args:
        iteration: Iteration number
        config: Configuration used for this run
        metrics: Computed metrics
        baseline_comparison: Percentage change vs baseline
        notes: Optional notes about this iteration

    Returns:
        str: The formatted markdown entry
    """
    ensure_results_dir()

    timestamp = datetime.now().isoformat()

    # Format the entry
    entry = f"""
## Run {iteration}: {timestamp}

### Config
"""

    for key, value in config.items():
        if isinstance(value, str) and len(value) > 50:
            entry += f"- {key}: \"{value[:47]}...\"\n"
        else:
            entry += f"- {key}: {value}\n"

    entry += "\n### Metrics\n"
    entry += "| Metric | Value | vs Baseline |\n"
    entry += "|--------|-------|-------------|\n"

    # Define which metrics are "lower is better" vs "higher is better"
    lower_is_better = {
        "pitch_variance", "energy_variance", "speaker_embedding_drift",
        "speaking_rate_variance"
    }

    for metric_name, value in metrics.items():
        # Skip list values (per-segment data)
        if isinstance(value, list):
            continue

        # Format the value
        if metric_name == "pitch_variance":
            formatted_value = f"{value:.1f} Hz"
        elif metric_name in ("spectral_similarity", "pitch_contour_similarity"):
            formatted_value = f"{value:.3f}"
        else:
            formatted_value = f"{value:.4f}"

        # Get comparison value
        comparison = baseline_comparison.get(metric_name, 0)

        # Determine if this is an improvement
        if metric_name in lower_is_better:
            improved = comparison < 0
        else:
            improved = comparison > 0

        indicator = " ✓" if improved else " ✗" if abs(comparison) > 5 else ""
        sign = "+" if comparison > 0 else ""

        entry += f"| {metric_name} | {formatted_value} | {sign}{comparison:.1f}%{indicator} |\n"

    if notes:
        entry += f"\n### Notes\n{notes}\n"

    entry += "\n---\n"

    # Append to file
    file_exists = os.path.exists(RESULTS_FILE)

    with open(RESULTS_FILE, "a") as f:
        if not file_exists:
            f.write("# TTS Quality Tuning Results\n\n")
            f.write(f"Started: {timestamp}\n")
            f.write("=" * 60 + "\n")
        f.write(entry)

    return entry


def log_best_config(
    config: dict,
    metrics: dict,
    total_iterations: int,
):
    """Log the best configuration found."""
    ensure_results_dir()

    timestamp = datetime.now().isoformat()

    entry = f"""
# Best Configuration Found

Timestamp: {timestamp}
Total Iterations: {total_iterations}

## Configuration
```json
{json.dumps(config, indent=2)}
```

## Metrics Achieved
"""

    for metric_name, value in metrics.items():
        if isinstance(value, list):
            continue
        entry += f"- {metric_name}: {value}\n"

    entry += "\n" + "=" * 60 + "\n"

    with open(RESULTS_FILE, "a") as f:
        f.write(entry)


def create_nexus_note(title: str, content: str, category: str = "research") -> dict:
    """
    Create a note in Nexus via MCP.

    Args:
        title: Note title
        content: Note content (markdown supported)
        category: Note category (general, meeting, research, reference, idea, log)

    Returns:
        dict: Response from Nexus API
    """
    if requests is None:
        print("Warning: requests library not available, skipping Nexus note creation")
        return {"error": "requests not available"}

    # MCP JSON-RPC request format
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "nexus_create_note",
            "arguments": {
                "title": title,
                "content": content,
                "category": category,
                "passphrase": NEXUS_PASSPHRASE,
                "tags": json.dumps(["tts", "voice-quality", "tuning"]),
                "source_type": "claude_conversation",
            }
        }
    }

    try:
        response = requests.post(
            NEXUS_MCP_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            print(f"Created Nexus note: {title}")
            return result
        else:
            print(f"Failed to create Nexus note: {response.status_code} - {response.text}")
            return {"error": response.text}

    except Exception as e:
        print(f"Error creating Nexus note: {e}")
        return {"error": str(e)}


def create_tuning_summary_note(
    best_config: dict,
    best_metrics: dict,
    baseline_metrics: dict,
    total_iterations: int,
    significant_findings: list[str],
) -> dict:
    """
    Create a comprehensive Nexus note summarizing the tuning results.

    Args:
        best_config: Best configuration found
        best_metrics: Metrics achieved with best config
        baseline_metrics: Original baseline metrics
        total_iterations: Number of iterations run
        significant_findings: List of notable findings

    Returns:
        dict: Response from Nexus API
    """
    # Calculate improvements
    improvements = {}
    for key in best_metrics:
        if key in baseline_metrics and baseline_metrics[key] != 0:
            change = (best_metrics[key] - baseline_metrics[key]) / baseline_metrics[key] * 100
            improvements[key] = change

    content = f"""# TTS Voice Quality Tuning Summary

## Overview
- **Total Iterations**: {total_iterations}
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Best Configuration Found
```json
{json.dumps(best_config, indent=2)}
```

## Metrics Comparison

| Metric | Baseline | Best | Improvement |
|--------|----------|------|-------------|
"""

    lower_is_better = {
        "pitch_variance", "energy_variance", "speaker_embedding_drift",
        "speaking_rate_variance"
    }

    for key in best_metrics:
        if key not in baseline_metrics:
            continue

        baseline_val = baseline_metrics[key]
        best_val = best_metrics[key]
        improvement = improvements.get(key, 0)

        # Format improvement indicator
        if key in lower_is_better:
            improved = improvement < 0
        else:
            improved = improvement > 0

        indicator = "✓" if improved else "✗"
        sign = "+" if improvement > 0 else ""

        content += f"| {key} | {baseline_val:.4f} | {best_val:.4f} | {sign}{improvement:.1f}% {indicator} |\n"

    if significant_findings:
        content += "\n## Key Findings\n"
        for finding in significant_findings:
            content += f"- {finding}\n"

    content += """
## Recommendations
Based on the tuning results:
1. Use the best configuration above for production
2. Consider further fine-tuning around these parameters
3. Monitor voice consistency metrics in production
"""

    return create_nexus_note(
        title=f"TTS Tuning Results - {datetime.now().strftime('%Y-%m-%d')}",
        content=content,
        category="research",
    )


def log_progress(message: str, level: str = "INFO"):
    """Log progress message to console and optionally to file."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")
