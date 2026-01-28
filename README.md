# Nexus Voice Agent

Local voice agent stack for Nexus integration, running on DGX Spark (GB10).

## TTS Quality Testing & Tuning System

Automated testing harness for Qwen3-TTS voice consistency optimization.

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Quick test to verify setup
python run_tuning.py --quick-test

# Run full tuning
python run_tuning.py --max-iterations 50 --strategy bayesian

# Evaluate a specific configuration
python run_tuning.py --evaluate --temp 0.5 --top-p 0.9 --instruction "calm and measured"
```

### Metrics Computed

| Metric | Description | Target |
|--------|-------------|--------|
| `pitch_variance` | Std dev of mean F0 across segments | < 20 Hz |
| `energy_variance` | Std dev of RMS energy | < 0.1 |
| `speaker_embedding_drift` | Max cosine distance from reference | < 0.15 |
| `spectral_similarity` | Avg mel spectrogram similarity | > 0.85 |
| `speaking_rate_variance` | Std dev of syllable rate | Low |
| `pitch_contour_similarity` | F0 contour consistency | High |

### Project Structure

```
tts_quality/
├── __init__.py
├── test_corpus.py          # Fixed test sentences
├── tts_quality_metrics.py  # Metrics computation
├── qwen_tts_client.py      # TTS generation wrapper
├── results_logger.py       # Logging to markdown + Nexus
└── tuning_harness.py       # Main automation loop
```

### Configuration

The system searches over these parameters:
- `temperature`: [0.3, 0.5, 0.7, 0.9]
- `top_p`: [0.8, 0.9, 0.95, 1.0]
- `repetition_penalty`: [1.0, 1.1, 1.2]
- `instruction_prompt`: Various style prompts
- `sentence_batching`: single, pairs, all

### Results

Results are logged to:
- `/home/chris/spark-voice-pipeline/tts_tuning_results.md` (local)
- Nexus notes (for significant findings)

---

## Architecture

Built on NVIDIA's open models:
- **Nemotron Speech ASR** (0.6B params) - Streaming speech-to-text
- **Nemotron 3 Nano** (30B params, Q8) - LLM for conversation
- **Qwen3-TTS** - Text-to-speech with voice cloning

## Hardware Requirements

- NVIDIA Grace Blackwell GPU with 128GB unified VRAM
- CUDA 13+
- ~50GB disk space for models

## Voice Assets

- Reference audio: `/home/chris/ggml-org/voice samples/her audio samples.wav`
- Fine-tuned Samantha weights: `her audio samples.wav.d6ef30c7@1000.safetensors`
- Samantha v7-ep3: `lora_weights_step_04000.safetensors`

## Status

- [x] TTS Quality Testing System
- [ ] Docker build (nemotron-unified:cuda13)
- [ ] Model download (Nemotron-3-Nano-30B-A3B-GGUF)
- [ ] Local testing
- [ ] Nexus MCP integration

## License

MIT
