"""
TTS Quality Metrics Computation Module.

Uses librosa for audio analysis, parselmouth (Praat) for F0/pitch extraction,
and speechbrain for speaker embeddings.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import warnings

# Suppress warnings during import
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import librosa
except ImportError:
    librosa = None

try:
    import parselmouth
    from parselmouth.praat import call
except ImportError:
    parselmouth = None

try:
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    torch = None
    EncoderClassifier = None


@dataclass
class MetricsResult:
    """Container for computed quality metrics."""
    pitch_variance: float  # std dev of mean F0 across segments (Hz)
    pitch_contour_similarity: float  # avg cosine sim of F0 contours between consecutive segments
    energy_variance: float  # std dev of RMS energy across segments (normalized)
    speaking_rate_variance: float  # std dev of estimated speaking rate
    spectral_similarity: float  # avg mel spectrogram cosine sim between consecutive segments
    speaker_embedding_drift: float  # max cosine distance of x-vectors from first segment

    # Individual segment metrics for analysis
    segment_pitches: Optional[list[float]] = None
    segment_energies: Optional[list[float]] = None
    segment_rates: Optional[list[float]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "pitch_variance": self.pitch_variance,
            "pitch_contour_similarity": self.pitch_contour_similarity,
            "energy_variance": self.energy_variance,
            "speaking_rate_variance": self.speaking_rate_variance,
            "spectral_similarity": self.spectral_similarity,
            "speaker_embedding_drift": self.speaker_embedding_drift,
        }

    def meets_thresholds(self, thresholds: dict) -> dict[str, bool]:
        """Check if metrics meet specified thresholds."""
        results = {}
        if "pitch_variance" in thresholds:
            results["pitch_variance"] = self.pitch_variance < thresholds["pitch_variance"]
        if "energy_variance" in thresholds:
            results["energy_variance"] = self.energy_variance < thresholds["energy_variance"]
        if "speaker_embedding_drift" in thresholds:
            results["speaker_embedding_drift"] = self.speaker_embedding_drift < thresholds["speaker_embedding_drift"]
        if "spectral_similarity" in thresholds:
            results["spectral_similarity"] = self.spectral_similarity > thresholds["spectral_similarity"]
        return results


# Global speaker encoder (lazy loaded)
_speaker_encoder = None


def _get_speaker_encoder():
    """Lazy load the speaker encoder model."""
    global _speaker_encoder
    if _speaker_encoder is None and EncoderClassifier is not None:
        _speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/speechbrain_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
    return _speaker_encoder


def extract_pitch_contour(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
    """
    Extract F0 pitch contour using Parselmouth (Praat).

    Returns:
        tuple: (pitch_contour, mean_f0)
    """
    if parselmouth is None:
        raise ImportError("parselmouth is required for pitch extraction")

    # Convert to Praat Sound object
    sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sample_rate)

    # Extract pitch with appropriate settings for speech
    pitch = call(sound, "To Pitch", 0.0, 75, 600)  # time_step=0, min=75Hz, max=600Hz

    # Get pitch values
    pitch_values = pitch.selected_array["frequency"]

    # Filter out unvoiced frames (0 Hz)
    voiced_pitch = pitch_values[pitch_values > 0]

    if len(voiced_pitch) == 0:
        return np.array([]), 0.0

    mean_f0 = np.mean(voiced_pitch)

    return voiced_pitch, mean_f0


def extract_rms_energy(audio: np.ndarray, sample_rate: int) -> float:
    """Extract normalized RMS energy from audio."""
    if librosa is None:
        # Fallback to numpy
        rms = np.sqrt(np.mean(audio ** 2))
    else:
        rms_frames = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        rms = np.mean(rms_frames)

    # Normalize to 0-1 range (assuming 16-bit audio normalized to -1,1)
    return float(rms)


def estimate_speaking_rate(audio: np.ndarray, sample_rate: int, text: str = None) -> float:
    """
    Estimate speaking rate in syllables per second.

    Uses onset detection as a proxy for syllable boundaries when text is not available.
    """
    if librosa is None:
        raise ImportError("librosa is required for speaking rate estimation")

    # Use onset detection as proxy for syllables
    onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sample_rate,
        backtrack=False,
        units="time"
    )

    # Calculate duration
    duration = len(audio) / sample_rate

    if duration < 0.1:
        return 0.0

    # Syllables per second (rough estimate)
    syllable_rate = len(onsets) / duration

    return float(syllable_rate)


def compute_mel_spectrogram(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Compute mel spectrogram for spectral similarity comparison."""
    if librosa is None:
        raise ImportError("librosa is required for mel spectrogram computation")

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=80,
        fmax=8000,
        n_fft=1024,
        hop_length=256
    )

    # Convert to log scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db


def compute_speaker_embedding(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Compute speaker embedding using SpeechBrain ECAPA-TDNN."""
    encoder = _get_speaker_encoder()

    if encoder is None:
        raise ImportError("speechbrain is required for speaker embeddings")

    # Resample to 16kHz if needed (model expects 16kHz)
    if sample_rate != 16000 and librosa is not None:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

    # Convert to tensor
    audio_tensor = torch.tensor(audio).unsqueeze(0).float()

    # Get embedding
    with torch.no_grad():
        embedding = encoder.encode_batch(audio_tensor)

    return embedding.squeeze().cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance (1 - similarity) between two vectors."""
    return 1.0 - cosine_similarity(a, b)


def pad_or_truncate(arr: np.ndarray, target_length: int) -> np.ndarray:
    """Pad or truncate array to target length for comparison."""
    if len(arr) >= target_length:
        return arr[:target_length]
    else:
        return np.pad(arr, (0, target_length - len(arr)), mode='constant')


def compute_metrics(audio_segments: list[np.ndarray], sample_rate: int) -> MetricsResult:
    """
    Compute consistency metrics across audio segments.

    Args:
        audio_segments: List of audio arrays (one per generated sentence)
        sample_rate: Sample rate of the audio

    Returns:
        MetricsResult with:
        - pitch_variance: std dev of mean F0 across segments
        - pitch_contour_similarity: avg cosine sim of F0 contours between consecutive segments
        - energy_variance: std dev of RMS energy across segments
        - speaking_rate_variance: std dev of estimated speaking rate
        - spectral_similarity: avg mel spectrogram cosine sim between consecutive segments
        - speaker_embedding_drift: max cosine distance of x-vectors from first segment
    """
    if len(audio_segments) == 0:
        raise ValueError("No audio segments provided")

    if len(audio_segments) == 1:
        # Single segment - return perfect consistency scores
        return MetricsResult(
            pitch_variance=0.0,
            pitch_contour_similarity=1.0,
            energy_variance=0.0,
            speaking_rate_variance=0.0,
            spectral_similarity=1.0,
            speaker_embedding_drift=0.0,
        )

    # Extract per-segment metrics
    segment_pitches = []
    pitch_contours = []
    segment_energies = []
    segment_rates = []
    mel_specs = []
    embeddings = []

    for i, audio in enumerate(audio_segments):
        # Ensure float32 and normalize
        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / 32768.0  # Assume 16-bit

        # Pitch extraction
        try:
            contour, mean_f0 = extract_pitch_contour(audio, sample_rate)
            segment_pitches.append(mean_f0)
            pitch_contours.append(contour)
        except Exception as e:
            print(f"Warning: Pitch extraction failed for segment {i}: {e}")
            segment_pitches.append(0.0)
            pitch_contours.append(np.array([]))

        # Energy extraction
        try:
            energy = extract_rms_energy(audio, sample_rate)
            segment_energies.append(energy)
        except Exception as e:
            print(f"Warning: Energy extraction failed for segment {i}: {e}")
            segment_energies.append(0.0)

        # Speaking rate estimation
        try:
            rate = estimate_speaking_rate(audio, sample_rate)
            segment_rates.append(rate)
        except Exception as e:
            print(f"Warning: Speaking rate estimation failed for segment {i}: {e}")
            segment_rates.append(0.0)

        # Mel spectrogram
        try:
            mel_spec = compute_mel_spectrogram(audio, sample_rate)
            mel_specs.append(mel_spec)
        except Exception as e:
            print(f"Warning: Mel spectrogram computation failed for segment {i}: {e}")
            mel_specs.append(None)

        # Speaker embedding
        try:
            embedding = compute_speaker_embedding(audio, sample_rate)
            embeddings.append(embedding)
        except Exception as e:
            print(f"Warning: Speaker embedding computation failed for segment {i}: {e}")
            embeddings.append(None)

    # Compute aggregate metrics

    # 1. Pitch variance (std dev of mean F0 across segments)
    valid_pitches = [p for p in segment_pitches if p > 0]
    pitch_variance = np.std(valid_pitches) if len(valid_pitches) > 1 else 0.0

    # 2. Pitch contour similarity (avg cosine sim between consecutive segments)
    pitch_similarities = []
    for i in range(len(pitch_contours) - 1):
        if len(pitch_contours[i]) > 0 and len(pitch_contours[i + 1]) > 0:
            # Pad to same length for comparison
            max_len = max(len(pitch_contours[i]), len(pitch_contours[i + 1]))
            c1 = pad_or_truncate(pitch_contours[i], max_len)
            c2 = pad_or_truncate(pitch_contours[i + 1], max_len)
            sim = cosine_similarity(c1, c2)
            pitch_similarities.append(sim)
    pitch_contour_similarity = np.mean(pitch_similarities) if pitch_similarities else 0.0

    # 3. Energy variance (std dev of RMS energy)
    energy_variance = np.std(segment_energies) if len(segment_energies) > 1 else 0.0

    # 4. Speaking rate variance
    valid_rates = [r for r in segment_rates if r > 0]
    speaking_rate_variance = np.std(valid_rates) if len(valid_rates) > 1 else 0.0

    # 5. Spectral similarity (avg mel cosine sim between consecutive segments)
    spectral_similarities = []
    for i in range(len(mel_specs) - 1):
        if mel_specs[i] is not None and mel_specs[i + 1] is not None:
            # Flatten and pad to same length
            m1 = mel_specs[i].flatten()
            m2 = mel_specs[i + 1].flatten()
            max_len = max(len(m1), len(m2))
            m1 = pad_or_truncate(m1, max_len)
            m2 = pad_or_truncate(m2, max_len)
            sim = cosine_similarity(m1, m2)
            spectral_similarities.append(sim)
    spectral_similarity = np.mean(spectral_similarities) if spectral_similarities else 0.0

    # 6. Speaker embedding drift (max cosine distance from first segment)
    embedding_drifts = []
    valid_embeddings = [e for e in embeddings if e is not None]
    if len(valid_embeddings) > 1:
        reference = valid_embeddings[0]
        for emb in valid_embeddings[1:]:
            drift = cosine_distance(reference, emb)
            embedding_drifts.append(drift)
    speaker_embedding_drift = max(embedding_drifts) if embedding_drifts else 0.0

    return MetricsResult(
        pitch_variance=float(pitch_variance),
        pitch_contour_similarity=float(pitch_contour_similarity),
        energy_variance=float(energy_variance),
        speaking_rate_variance=float(speaking_rate_variance),
        spectral_similarity=float(spectral_similarity),
        speaker_embedding_drift=float(speaker_embedding_drift),
        segment_pitches=segment_pitches,
        segment_energies=segment_energies,
        segment_rates=segment_rates,
    )


def compare_to_baseline(metrics: MetricsResult, baseline: MetricsResult) -> dict:
    """
    Compare metrics to baseline, returning percentage change for each metric.

    Negative values indicate improvement for variance metrics.
    Positive values indicate improvement for similarity metrics.
    """
    comparisons = {}

    if baseline.pitch_variance > 0:
        comparisons["pitch_variance"] = (
            (metrics.pitch_variance - baseline.pitch_variance) / baseline.pitch_variance * 100
        )
    else:
        comparisons["pitch_variance"] = 0.0

    if baseline.energy_variance > 0:
        comparisons["energy_variance"] = (
            (metrics.energy_variance - baseline.energy_variance) / baseline.energy_variance * 100
        )
    else:
        comparisons["energy_variance"] = 0.0

    if baseline.speaker_embedding_drift > 0:
        comparisons["speaker_embedding_drift"] = (
            (metrics.speaker_embedding_drift - baseline.speaker_embedding_drift) / baseline.speaker_embedding_drift * 100
        )
    else:
        comparisons["speaker_embedding_drift"] = 0.0

    if baseline.spectral_similarity > 0:
        comparisons["spectral_similarity"] = (
            (metrics.spectral_similarity - baseline.spectral_similarity) / baseline.spectral_similarity * 100
        )
    else:
        comparisons["spectral_similarity"] = 0.0

    if baseline.pitch_contour_similarity > 0:
        comparisons["pitch_contour_similarity"] = (
            (metrics.pitch_contour_similarity - baseline.pitch_contour_similarity) / baseline.pitch_contour_similarity * 100
        )
    else:
        comparisons["pitch_contour_similarity"] = 0.0

    if baseline.speaking_rate_variance > 0:
        comparisons["speaking_rate_variance"] = (
            (metrics.speaking_rate_variance - baseline.speaking_rate_variance) / baseline.speaking_rate_variance * 100
        )
    else:
        comparisons["speaking_rate_variance"] = 0.0

    return comparisons


# Default thresholds for quality evaluation
DEFAULT_THRESHOLDS = {
    "pitch_variance": 20.0,  # Hz std dev
    "energy_variance": 0.1,  # normalized RMS
    "speaker_embedding_drift": 0.15,  # cosine distance
    "spectral_similarity": 0.85,  # cosine similarity
}
