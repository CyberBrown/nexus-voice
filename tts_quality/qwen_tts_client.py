"""
Qwen3-TTS Client for audio generation.

Supports both HTTP API (when server is running on port 8032) and direct model loading.
"""

import numpy as np
import io
import json
import os
from dataclasses import dataclass
from typing import Optional
import warnings

try:
    import requests
except ImportError:
    requests = None

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import torch
    import torchaudio
except ImportError:
    torch = None
    torchaudio = None


@dataclass
class TTSConfig:
    """Configuration for TTS generation."""
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.0
    max_new_tokens: int = 4096
    instruction: str = ""
    voice_prompt_path: str = "/home/chris/ggml-org/voice samples/her audio samples.wav"
    lora_weights_path: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for API requests."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "instruction": self.instruction,
        }


class QwenTTSClient:
    """Client for interacting with Qwen3-TTS."""

    DEFAULT_SERVER_URL = "http://localhost:8032"
    DEFAULT_VOICE_PROMPT = "/home/chris/ggml-org/voice samples/her audio samples.wav"
    DEFAULT_LORA_WEIGHTS = "/home/chris/ggml-org/voice samples/her audio samples.wav.d6ef30c7@1000.safetensors"
    ALT_LORA_WEIGHTS = "lora_weights_step_04000.safetensors"  # Samantha v7-ep3

    def __init__(
        self,
        server_url: str = None,
        voice_prompt_path: str = None,
        lora_weights_path: str = None,
        use_direct_model: bool = False,
        speaker_name: str = "her",
    ):
        """
        Initialize the TTS client.

        Args:
            server_url: URL of the TTS server (default: http://localhost:8032)
            voice_prompt_path: Path to reference audio for voice cloning
            lora_weights_path: Path to fine-tuned LoRA weights (optional)
            use_direct_model: If True, load model directly instead of using HTTP API
            speaker_name: Speaker/voice name for TTS server (default: "her")
        """
        self.server_url = server_url or self.DEFAULT_SERVER_URL
        self.voice_prompt_path = voice_prompt_path or self.DEFAULT_VOICE_PROMPT
        self.lora_weights_path = lora_weights_path
        self.use_direct_model = use_direct_model
        self.speaker_name = speaker_name

        self._model = None
        self._processor = None
        self._voice_prompt_audio = None

        if use_direct_model:
            self._load_model()

    def _load_model(self):
        """Load the Qwen3-TTS model directly."""
        if torch is None:
            raise ImportError("torch is required for direct model loading")

        try:
            from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration
        except ImportError:
            raise ImportError("transformers is required for direct model loading")

        print("Loading Qwen3-TTS model...")

        self._processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B",
            trust_remote_code=True,
        )

        self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B",
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )

        # Load LoRA weights if specified
        if self.lora_weights_path and os.path.exists(self.lora_weights_path):
            print(f"Loading LoRA weights from {self.lora_weights_path}")
            # Load safetensors
            from safetensors.torch import load_file
            lora_state = load_file(self.lora_weights_path)
            self._model.load_state_dict(lora_state, strict=False)

        print("Model loaded successfully")

    def _load_voice_prompt(self) -> tuple[np.ndarray, int]:
        """Load the voice prompt audio file."""
        if self._voice_prompt_audio is not None:
            return self._voice_prompt_audio

        if not os.path.exists(self.voice_prompt_path):
            raise FileNotFoundError(f"Voice prompt not found: {self.voice_prompt_path}")

        if torchaudio is not None:
            waveform, sample_rate = torchaudio.load(self.voice_prompt_path)
            audio = waveform.numpy().squeeze()
        elif sf is not None:
            audio, sample_rate = sf.read(self.voice_prompt_path)
        else:
            raise ImportError("torchaudio or soundfile required for audio loading")

        self._voice_prompt_audio = (audio, sample_rate)
        return self._voice_prompt_audio

    def _check_server_health(self) -> bool:
        """Check if the TTS server is running."""
        if requests is None:
            return False

        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def generate_via_api(
        self,
        text: str,
        config: TTSConfig = None,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio using the HTTP API.

        Args:
            text: Text to synthesize
            config: TTS configuration options

        Returns:
            tuple: (audio_array, sample_rate)
        """
        if requests is None:
            raise ImportError("requests is required for API-based generation")

        config = config or TTSConfig()

        # Use /synthesize endpoint with instruct param for style control
        params = {
            "text": text,
            "speaker": self.speaker_name or "her",
        }

        # Add instruction if provided
        if config.instruction:
            params["instruct"] = config.instruction

        # Make the request
        response = requests.post(
            f"{self.server_url}/synthesize",
            params=params,
            timeout=120,  # TTS can take a while
        )

        if response.status_code != 200:
            raise RuntimeError(f"TTS generation failed: {response.text}")

        # Parse the response - WAV audio bytes
        if sf is not None:
            audio, sample_rate = sf.read(io.BytesIO(response.content))
        else:
            raise ImportError("soundfile required for audio parsing")

        return audio.astype(np.float32), sample_rate

    def generate_direct(
        self,
        text: str,
        config: TTSConfig = None,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio using direct model inference.

        Args:
            text: Text to synthesize
            config: TTS configuration options

        Returns:
            tuple: (audio_array, sample_rate)
        """
        if self._model is None:
            self._load_model()

        config = config or TTSConfig()

        # Load voice prompt
        voice_audio, voice_sr = self._load_voice_prompt()

        # Prepare the conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": voice_audio},
                    {"type": "text", "text": f"Clone this voice and say: {text}"},
                ],
            }
        ]

        if config.instruction:
            conversation[0]["content"][1]["text"] = (
                f"{config.instruction} Clone this voice and say: {text}"
            )

        # Process inputs
        inputs = self._processor(
            conversation,
            return_tensors="pt",
            sampling_rate=voice_sr,
        ).to(self._model.device)

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                do_sample=True,
            )

        # Extract audio from outputs
        # This depends on the specific model architecture
        audio = outputs.audio.cpu().numpy().squeeze()
        sample_rate = 24000  # Typical for TTS models

        return audio.astype(np.float32), sample_rate

    def generate(
        self,
        text: str,
        config: TTSConfig = None,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio using the best available method.

        Tries HTTP API first, falls back to direct model loading.

        Args:
            text: Text to synthesize
            config: TTS configuration options

        Returns:
            tuple: (audio_array, sample_rate)
        """
        if self.use_direct_model:
            return self.generate_direct(text, config)

        # Try HTTP API first
        if self._check_server_health():
            return self.generate_via_api(text, config)

        # Fall back to direct model loading
        print("TTS server not available, loading model directly...")
        self.use_direct_model = True
        return self.generate_direct(text, config)

    def generate_batch(
        self,
        texts: list[str],
        config: TTSConfig = None,
    ) -> list[tuple[np.ndarray, int]]:
        """
        Generate audio for multiple texts.

        Args:
            texts: List of texts to synthesize
            config: TTS configuration options

        Returns:
            List of (audio_array, sample_rate) tuples
        """
        results = []
        for text in texts:
            audio, sr = self.generate(text, config)
            results.append((audio, sr))
        return results


def generate_audio(
    text: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
    instruction: str = "",
    voice_prompt_path: str = None,
    lora_weights_path: str = None,
) -> tuple[np.ndarray, int]:
    """
    Convenience function for generating audio.

    Args:
        text: Text to synthesize
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        instruction: Style instruction for generation
        voice_prompt_path: Path to reference audio
        lora_weights_path: Path to fine-tuned weights

    Returns:
        tuple: (audio_array, sample_rate)
    """
    client = QwenTTSClient(
        voice_prompt_path=voice_prompt_path,
        lora_weights_path=lora_weights_path,
    )

    config = TTSConfig(
        temperature=temperature,
        top_p=top_p,
        instruction=instruction,
    )

    return client.generate(text, config)


# Alternative API format for servers that use different endpoints
class QwenTTSClientAlt:
    """Alternative client for different TTS server API formats."""

    def __init__(self, server_url: str = "http://localhost:8032"):
        self.server_url = server_url

    def generate(
        self,
        text: str,
        speaker_audio_path: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        system_prompt: str = "",
    ) -> tuple[np.ndarray, int]:
        """Generate audio with speaker reference."""
        if requests is None:
            raise ImportError("requests required")

        # Read speaker audio and encode as base64
        import base64
        with open(speaker_audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        payload = {
            "text": text,
            "speaker_audio": audio_b64,
            "temperature": temperature,
            "top_p": top_p,
        }

        if system_prompt:
            payload["system_prompt"] = system_prompt

        response = requests.post(
            f"{self.server_url}/v1/audio/speech",
            json=payload,
            timeout=120,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Generation failed: {response.text}")

        audio, sr = sf.read(io.BytesIO(response.content))
        return audio.astype(np.float32), sr
