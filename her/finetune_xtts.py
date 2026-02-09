#!/usr/bin/env python3
"""
Fine-tune XTTS v2 on Samantha voice clips.

XTTS is specifically designed for voice cloning and works well with small datasets.
This script fine-tunes the model on the extracted Her movie voice clips.
"""

import json
import os
import shutil
from pathlib import Path

# Training settings
DATA_DIR = Path("/home/chris/projects/her/her_lines")
MANIFEST_FILE = Path("/home/chris/projects/her/train.jsonl")
OUTPUT_DIR = Path("/home/chris/projects/her/xtts_samantha")
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-5

def prepare_metadata():
    """Convert train.jsonl to XTTS metadata format."""
    metadata_path = OUTPUT_DIR / "metadata.txt"
    wavs_dir = OUTPUT_DIR / "wavs"

    # Create output dirs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wavs_dir.mkdir(exist_ok=True)

    lines = []
    with open(MANIFEST_FILE, "r") as f:
        for line in f:
            entry = json.loads(line)
            # Extract filename from path
            audio_path = entry["audio"]
            if audio_path.startswith("./"):
                audio_path = audio_path[2:]

            src_path = Path("/home/chris/projects/her") / audio_path
            filename = Path(audio_path).name

            # Copy wav to output dir
            dst_path = wavs_dir / filename
            if src_path.exists() and not dst_path.exists():
                shutil.copy(src_path, dst_path)

            # XTTS format: audio_file|text|speaker_name
            text = entry["text"]
            lines.append(f"wavs/{filename}|{text}|samantha")

    with open(metadata_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Prepared {len(lines)} samples in {metadata_path}")
    return metadata_path


def main():
    print("="*60)
    print("XTTS v2 Fine-tuning for Samantha Voice")
    print("="*60)

    # Prepare data
    print("\nStep 1: Preparing metadata...")
    metadata_path = prepare_metadata()

    # Check if TTS is available
    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
    except ImportError:
        print("\nERROR: TTS (Coqui) not installed!")
        print("Install with: pip install TTS")
        return

    print("\nStep 2: Loading XTTS v2 base model...")

    # Import training utilities
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.datasets import load_tts_samples
    from trainer import Trainer, TrainerArgs

    # Configure dataset
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train=str(metadata_path),
        path=str(OUTPUT_DIR),
        language="en",
    )

    # Load base XTTS config
    config = XttsConfig()
    config.load_json("XTTS-v2/config.json")  # Will download if needed

    # Update config for fine-tuning
    config.output_path = str(OUTPUT_DIR / "checkpoints")
    config.batch_size = BATCH_SIZE
    config.eval_batch_size = BATCH_SIZE
    config.num_loader_workers = 4
    config.num_eval_loader_workers = 2
    config.run_eval = True
    config.test_delay_epochs = 1
    config.epochs = EPOCHS
    config.lr = LEARNING_RATE

    # Initialize model
    model = Xtts.init_from_config(config)

    # Load pretrained weights
    print("\nStep 3: Loading pretrained XTTS v2 weights...")
    model.load_checkpoint(
        config,
        checkpoint_dir="XTTS-v2",
        eval=False,
    )

    # Load training samples
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=50,
        eval_split_size=0.1,
    )

    print(f"\nTrain samples: {len(train_samples)}")
    print(f"Eval samples: {len(eval_samples)}")

    # Configure trainer
    trainer_args = TrainerArgs(
        restore_path=None,
        skip_train_epoch=False,
        start_with_eval=True,
        grad_accum_steps=4,
    )

    # Create trainer
    trainer = Trainer(
        trainer_args,
        config,
        str(OUTPUT_DIR / "checkpoints"),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    # Start training
    print("\nStep 4: Starting fine-tuning...")
    print(f"Output: {OUTPUT_DIR / 'checkpoints'}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("="*60)

    trainer.fit()

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Checkpoints saved to: {OUTPUT_DIR / 'checkpoints'}")
    print("="*60)


if __name__ == "__main__":
    main()
