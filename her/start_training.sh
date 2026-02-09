#!/bin/bash
#
# Start XTTS fine-tuning for Samantha voice
# Run overnight on Spark server
#

set -e

cd /home/chris/projects/her

echo "========================================"
echo "Samantha Voice Training Setup"
echo "========================================"
echo "Time: $(date)"
echo ""

# Create virtual environment with system packages (to use existing torch)
if [ ! -d "training-venv" ]; then
    echo "[1/4] Creating virtual environment with system packages..."
    python3 -m venv --system-site-packages training-venv
fi

echo "[2/4] Activating environment..."
source training-venv/bin/activate

echo "[3/4] Installing TTS (Coqui)..."

# Verify torch is available
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Install TTS and requirements
pip install --upgrade pip -q
pip install TTS trainer -q

# Verify TTS installation
python3 -c "import TTS; print(f'TTS: {TTS.__version__}')"
echo ""

# Start training
echo "[4/4] Starting training in background..."
echo ""
echo "Log file: /home/chris/projects/her/training.log"
echo ""

nohup python3 train_samantha.py > training.log 2>&1 &
TRAIN_PID=$!

sleep 3

if ps -p $TRAIN_PID > /dev/null 2>&1; then
    echo "Training started successfully!"
    echo "PID: $TRAIN_PID"
    echo ""
    echo "Monitor progress:"
    echo "  tail -f /home/chris/projects/her/training.log"
    echo ""
    echo "Stop training:"
    echo "  kill $TRAIN_PID"
else
    echo "Training failed to start. Check training.log for errors:"
    tail -30 training.log
fi
