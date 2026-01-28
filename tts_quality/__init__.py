"""TTS Quality Testing and Tuning System for Qwen3-TTS."""

from .tts_quality_metrics import compute_metrics, MetricsResult
from .test_corpus import TEST_CORPUS
from .qwen_tts_client import generate_audio, QwenTTSClient
from .results_logger import log_iteration, create_nexus_note
from .tuning_harness import run_tuning_loop, TUNING_PARAMETERS

__all__ = [
    "compute_metrics",
    "MetricsResult",
    "TEST_CORPUS",
    "generate_audio",
    "QwenTTSClient",
    "log_iteration",
    "create_nexus_note",
    "run_tuning_loop",
    "TUNING_PARAMETERS",
]
