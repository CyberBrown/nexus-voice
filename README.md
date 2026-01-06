# Nexus Voice Agent

Local voice agent stack for Nexus integration, running on DGX Spark (GB10).

## Architecture

Built on NVIDIA's open models:
- **Nemotron Speech ASR** (0.6B params) - Streaming speech-to-text
- **Nemotron 3 Nano** (30B params, Q8) - LLM for conversation
- **Magpie TTS** (357M params) - Text-to-speech synthesis

## Hardware Requirements

- NVIDIA Grace Blackwell GPU with 128GB unified VRAM
- CUDA 13+
- ~50GB disk space for models

## Reference

Based on [pipecat-ai/nemotron-january-2026](https://github.com/pipecat-ai/nemotron-january-2026)

## Status

- [ ] Docker build (nemotron-unified:cuda13)
- [ ] Model download (Nemotron-3-Nano-30B-A3B-GGUF)
- [ ] Local testing
- [ ] Nexus MCP integration

## License

MIT
