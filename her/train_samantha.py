#!/usr/bin/env python3
"""
Fine-tune XTTS v2 on Samantha voice using the official recipe.

This uses the Coqui TTS XTTS fine-tuning example which is well-tested.
"""

import json
import os
import sys
from pathlib import Path

# Configuration
HER_DIR = Path("/home/chris/projects/her")
DATA_DIR = HER_DIR / "her_lines"
OUTPUT_DIR = HER_DIR / "xtts_samantha"
MANIFEST_FILE = HER_DIR / "train.jsonl"

# Training hyperparameters
BATCH_SIZE = 2  # Small due to XTTS memory requirements
GRAD_ACCUM = 4  # Effective batch = 8
EPOCHS = 10
LR = 5e-6  # Low LR for fine-tuning


def check_environment():
    """Check if all required packages are installed."""
    missing = []

    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        missing.append("torch")

    try:
        import TTS
        print(f"TTS (Coqui): {TTS.__version__}")
    except ImportError:
        missing.append("TTS")

    if missing:
        print(f"\nMissing packages: {missing}")
        print("Install with: pip install TTS torch")
        return False

    return True


def prepare_xtts_dataset():
    """Convert train.jsonl to XTTS-compatible dataset structure."""
    import shutil

    output_dir = OUTPUT_DIR / "dataset"
    wavs_dir = output_dir / "wavs"

    output_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir.mkdir(exist_ok=True)

    metadata_lines = []

    with open(MANIFEST_FILE, "r") as f:
        for idx, line in enumerate(f):
            entry = json.loads(line)
            audio_path = entry["audio"]
            if audio_path.startswith("./"):
                audio_path = audio_path[2:]

            src_path = HER_DIR / audio_path
            filename = Path(audio_path).name
            dst_path = wavs_dir / filename

            # Copy if not already there
            if src_path.exists() and not dst_path.exists():
                shutil.copy(src_path, dst_path)

            text = entry["text"].strip()
            # Remove problematic characters
            text = text.replace('"', "'").replace("\n", " ")

            # Format: filename|speaker|language|text
            metadata_lines.append(f"wavs/{filename}|samantha|en|{text}")

    # Write metadata files
    meta_train = output_dir / "metadata_train.csv"
    meta_eval = output_dir / "metadata_eval.csv"

    # Use 90/10 split
    split_idx = int(len(metadata_lines) * 0.9)

    with open(meta_train, "w") as f:
        f.write("\n".join(metadata_lines[:split_idx]))

    with open(meta_eval, "w") as f:
        f.write("\n".join(metadata_lines[split_idx:]))

    print(f"Dataset prepared: {len(metadata_lines)} total samples")
    print(f"  Train: {split_idx}")
    print(f"  Eval: {len(metadata_lines) - split_idx}")
    print(f"  Location: {output_dir}")

    return output_dir, meta_train, meta_eval


def run_training():
    """Run XTTS fine-tuning."""
    import torch
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsArgs, Xtts
    from TTS.utils.manage import ModelManager

    # Prepare dataset
    print("\n[1/4] Preparing dataset...")
    dataset_dir, meta_train, meta_eval = prepare_xtts_dataset()

    # Download base model if needed
    print("\n[2/4] Loading base XTTS v2 model...")
    manager = ModelManager()
    model_path, config_path, _ = manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")

    print(f"  Model path: {model_path}")
    print(f"  Config path: {config_path}")

    # Load config
    config = XttsConfig()
    config.load_json(config_path)

    # Update for fine-tuning
    config.batch_size = BATCH_SIZE
    config.eval_batch_size = BATCH_SIZE
    config.num_loader_workers = 2
    config.num_eval_loader_workers = 1
    config.run_eval = True
    config.test_delay_epochs = 0
    config.epochs = EPOCHS
    config.lr = LR
    config.output_path = str(OUTPUT_DIR / "run")

    # Override dataset config
    config.datasets = [
        {
            "formatter": "ljspeech",
            "dataset_name": "samantha",
            "path": str(dataset_dir),
            "meta_file_train": str(meta_train),
            "meta_file_val": str(meta_eval),
            "language": "en",
        }
    ]

    # Initialize model
    print("\n[3/4] Initializing model...")
    model = Xtts.init_from_config(config)

    # Load pretrained checkpoint
    model.load_checkpoint(
        config,
        checkpoint_path=model_path,
        eval=False,
    )

    print(f"\n[4/4] Starting training...")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM})")
    print(f"  Learning rate: {LR}")
    print(f"  Output: {OUTPUT_DIR / 'run'}")
    print("=" * 60)

    # Use trainer
    from trainer import Trainer, TrainerArgs

    trainer_args = TrainerArgs(
        grad_accum_steps=GRAD_ACCUM,
    )

    # Get training samples
    from TTS.tts.datasets import load_tts_samples

    train_samples, eval_samples = load_tts_samples(
        config.datasets[0],
        eval_split=True,
    )

    trainer = Trainer(
        trainer_args,
        config,
        output_path=str(OUTPUT_DIR / "run"),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    trainer.fit()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {OUTPUT_DIR / 'run'}")
    print("=" * 60)


def main():
    print("=" * 60)
    print("Samantha Voice - XTTS v2 Fine-tuning")
    print("=" * 60)

    if not check_environment():
        sys.exit(1)

    run_training()


if __name__ == "__main__":
    main()
