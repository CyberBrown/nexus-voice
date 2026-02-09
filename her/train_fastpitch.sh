#!/bin/bash
set -e

echo "=== FastPitch Fine-tuning for Samantha Voice ==="
mkdir -p /data/her/fastpitch_samantha

echo "=== Preparing manifest ==="
python3 /data/her/prepare_manifest.py

echo "=== Starting Training ==="
python3 /data/her/finetune_fastpitch.py

echo "=== Done! ==="
